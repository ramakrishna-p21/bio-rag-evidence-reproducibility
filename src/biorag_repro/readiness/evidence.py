"""Evidence adapters for the 13 readiness gates.

Each adapter reads real artifacts and returns what it found as `Gathered`; `gather.to_result`
turns that and the policy into a gate result. An adapter never reads the policy and never
decides approval. Locators of evidence are logical: `data/...` below the data root,
`repo/...` below the checkout, `snapshot/<sha>.json` for a stored observation.

Identity subjects (what a future policy slot would approve) are defined here, next to the
check that produces them:

* G01 corpus: corpus version, release record, content and semantic hashes, manifest CSV and
  the (artifact id, SHA-256) of every stored JATS and PDF source.
* G02 benchmark: release record, items, split and corpus hashes plus the derived counts.
* G06 Python environment: the SHA-256 of an approved environment lock.
* G09 source: Git blob identities of the tracked scientific scope and the SHA-256 of each
  frozen specification file. The specifications are Git-ignored local files, so Git cleanliness
  says nothing about them; their bytes are hashed here and compared only with an approved
  reference.
* G03/G04/G05: the prospective production and retained qualification readers live in
  `readiness/production.py` and are called by `evaluate.gather_all`.

G08, G10, G11 and G12 retain their own prerequisite states. G13 is the one adapter that reads the policy: it takes the
G11 storage-authorization slot as the source of the expected root identity and verifies the
final-test root against that approved artifact (`finaltest.inventory.verify_root`). With the
slot unset, or without the artifact it names, G13 is `unavailable`; nothing found on disk and
nothing a caller passes can establish it.
"""

import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import StringConstraints, ValidationError, model_validator

from biorag_repro.benchmark.freeze import verify_release_inputs
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.benchmark.seeds import BenchmarkSeedRegistry
from biorag_repro.benchmark.split import PaperSplit
from biorag_repro.benchmark.storage import decode_jsonl
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.corpus.manifest import (
    ReleaseRecord,
    corpus_content_hash,
    manifest_semantic_hash,
    parse_csv,
    validate_corpus,
    verify_source_bytes,
)
from biorag_repro.evaluation.seeds import (
    APPROVED_REGISTRY_SHA256,
    REGISTRY_PATH,
    read_analysis_seed,
)
from biorag_repro.finaltest import layout as root_layout
from biorag_repro.finaltest.inventory import (
    INCOMPLETE,
    AuthorityNotEstablished,
    verify_root,
)
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.environment_lock import (
    EnvironmentInventory,
    EnvironmentLockError,
    EnvironmentVerification,
    parse_spec,
    verify_environment,
)
from biorag_repro.readiness import git_state
from biorag_repro.readiness.files import FileMissing, FileProblem, loads_strict, read_regular
from biorag_repro.readiness.gather import Gathered, file_ref
from biorag_repro.readiness.models import HISTORY_FAIL_CODES, EvidenceRef
from biorag_repro.readiness.policy import ApprovalPolicy

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
BENCHMARK_RELEASE = "benchmark/releases/benchmark_v1/release_record.json"
GPU_ATTESTATION = "readiness/attestations/g12_researcher_guidance_v1.json"
GPU_OPERATIONAL = "readiness/attestations/g12_operational_v1.json"
GPU_CAMPAIGN_DECISION = "readiness/decisions/campaign_storage_decision_2026_11_20.json"
GPU_AMENDMENT = "docs/frozen/G12_COLAB_CAMPAIGN_AMENDMENT_V1.md"
GPU_AMENDMENT_SHA256 = "a648376654cfa3925f3ee6acb6ffadc12e14130e266cb49f7924e540cb67d6f8"
GPU_G11_SHA256 = "0f7d387ac1d7d62e34a846ab3fb32dc8de8d6517ee7019a44a382af56c6922af"
GPU_CAMPAIGN_DECISION_SHA256 = "508c5f0c4431131b97eb80f00b13a8bcf7769c7ec5ae8dddf738477c64dba9ce"
FROZEN_SPECIFICATIONS = (
    "SCIENTIFIC_CONTRACT.md",
    "ARCHITECTURE.md",
    "BENCHMARK_SPEC.md",
    "EXPERIMENT_PLAN.md",
    "REPRODUCIBILITY_SPEC.md",
    "CORPUS_MANIFEST_SPEC.md",
    "IMPLEMENTATION_PLAN.md",
    "RESEARCHER_AUTHORITY_AMENDMENT_V1.md",
    "AMENDMENT_STAGE_A_RETROSPECTIVE_ACCOUNTING_V2.md",
    "AMENDMENT_STAGE_B_HISTORICAL_CONTEXT_READBACK_V1.md",
    "AMENDMENT_STAGE_B_HISTORICAL_PROVENANCE_V1.md",
    "DRIVE_CAMPAIGN_CUSTODY_AMENDMENT_V1.md",
    "G12_COLAB_CAMPAIGN_AMENDMENT_V1.md",
)
ENVIRONMENT_SPEC = "configs/environment.toml"
RENV_LOCK = "analysis/renv.lock"
R_SOURCES = ("analysis/run_analysis.R",)
R_PROBE_TIMEOUT_S = 300


@dataclass
class Probes:
    """The only live observations that are not plain file reads; tests replace them."""

    inventory: Callable[[], EnvironmentInventory]
    r_runtime: Callable[[Path, list[str]], str]


@dataclass
class Context:
    repo: Path
    data_root: Path
    probes: Probes
    warnings: list[str] = field(default_factory=list)


def _data_ref(kind: str, root: Path, ref: ArtifactRef) -> tuple[EvidenceRef, bytes]:
    """Read a referenced data-root artifact and require the bytes to match the reference."""
    data = read_regular(root, ref.logical_ref)
    if sha256_bytes(data) != ref.sha256:
        raise FileProblem(f"{ref.logical_ref} differs from its recorded SHA-256")
    return file_ref(kind, "data/" + ref.logical_ref, data), data


def _json_model(data: bytes, model: type[Any]) -> Any:
    return model.model_validate(loads_strict(data), strict=True)


# ---- G01 / G02 ---------------------------------------------------------------------------


def read_benchmark_record(ctx: Context) -> tuple[BenchmarkRelease, bytes]:
    data = read_regular(ctx.data_root, BENCHMARK_RELEASE)
    return _json_model(data, BenchmarkRelease), data


def corpus_sources(ctx: Context, release: BenchmarkRelease | None, problem: str | None) -> Gathered:
    scope = "data root: corpus release cited by the frozen benchmark"
    if release is None:
        return Gathered("unavailable", "corpus_release_reference_unavailable", scope, problem)
    out = Gathered("ok", "checks_passed", scope)
    try:
        record_ref, record_bytes = _data_ref(
            "corpus_release_record", ctx.data_root, release.corpus_release
        )
        record = _json_model(record_bytes, ReleaseRecord)
        csv_ref, csv_bytes = _data_ref(
            "corpus_manifest_csv", ctx.data_root, record.manifest_artifact
        )
        report_ref, _ = _data_ref(
            "corpus_validation_report", ctx.data_root, record.validation_report_artifact
        )
        out.refs += [record_ref, csv_ref, report_ref]
        manifest = parse_csv(csv_bytes)
        if (
            record.corpus_version != release.corpus_version
            or record.corpus_content_sha256 != release.corpus_content_sha256
            or corpus_content_hash(manifest) != record.corpus_content_sha256
            or manifest_semantic_hash(manifest) != record.manifest_semantic_sha256
        ):
            return Gathered("failed", "corpus_identity_differs", scope, "hash or version mismatch")
        hard = [f for f in validate_corpus(manifest).findings if f.severity == "hard_error"]
        if hard:
            return Gathered("failed", "corpus_validation_failed", scope, f"{len(hard)} hard errors")
        sources: list[list[str]] = []
        for paper in sorted(manifest.papers, key=lambda p: p.paper_id):
            for kind, source in (("jats", paper.jats), ("pdf", paper.pdf)):
                logical = "corpus/raw/" + source.logical_ref
                data = read_regular(ctx.data_root, logical)
                if sha256_bytes(data) != source.sha256:
                    return Gathered(
                        "failed", "source_bytes_differ", scope, f"{paper.paper_id} {kind}"
                    )
                verify_source_bytes(paper, kind, ctx.data_root / logical)  # type: ignore[arg-type]
                out.refs.append(file_ref(f"corpus_source_{kind}", "data/" + logical, data))
                sources.append([source.artifact_id, source.sha256])
    except FileMissing as error:
        return Gathered("failed", "corpus_artifact_missing", scope, str(error))
    except (FileProblem, ValueError, ValidationError, OSError) as error:
        return Gathered("failed", "corpus_artifact_invalid", scope, str(error)[:300])
    out.subject = {
        "kind": "corpus_sources_v1",
        "corpus_version": record.corpus_version,
        "release_record_sha256": sha256_bytes(record_bytes),
        "corpus_content_sha256": record.corpus_content_sha256,
        "manifest_semantic_sha256": record.manifest_semantic_sha256,
        "manifest_csv_sha256": record.csv_sha256,
        "sources": sources,
    }
    out.facts = {"corpus_content_sha256": record.corpus_content_sha256}
    out.detail = f"{len(manifest.papers)} papers, {len(sources)} source files"
    return out


def benchmark_freeze(
    ctx: Context,
    release: BenchmarkRelease | None,
    release_bytes: bytes | None,
    problem: str | None,
    *,
    hash_only: bool = False,
) -> Gathered:
    scope = "data root: frozen benchmark release record"
    if release is None or release_bytes is None:
        return Gathered("unavailable", "benchmark_release_unavailable", scope, problem)
    out = Gathered("ok", "checks_passed", scope)
    out.refs.append(
        file_ref("benchmark_release_record", "data/" + BENCHMARK_RELEASE, release_bytes)
    )
    try:
        data: dict[str, bytes] = {}
        for name in (
            "seed_registry",
            "relatedness",
            "split",
            "items",
            "decisions",
            "dispositions",
            "pre_audit_population",
            "audit",
            "validation_report",
        ):
            ref, data[name] = _data_ref("benchmark_" + name, ctx.data_root, getattr(release, name))
            out.refs.append(ref)
        split = _json_model(data["split"], PaperSplit)
        if hash_only:
            # The separate witness is a researcher approval of counts from the completed
            # freeze. This boundary must not parse the protected item stream.
            witness_bytes = read_regular(ctx.repo, "approvals/benchmark_count_witness_v1.json")
            witness = loads_strict(witness_bytes)
            expected = {
                "schema_version": "benchmark_count_witness_v1",
                "researcher": "Rama Krishna Pudota",
                "approved": True,
                "benchmark_version": release.benchmark_version,
                "release_record_sha256": sha256_bytes(release_bytes),
                "items_sha256": release.items.sha256,
                "split_sha256": release.split.sha256,
                "validation_report_sha256": release.validation_report.sha256,
                "item_count": release.item_count,
            }
            if not isinstance(witness, dict) or any(
                witness.get(k) != v for k, v in expected.items()
            ):
                return Gathered("failed", "benchmark_count_witness_invalid", scope)
            development = witness.get("development_items")
            final = witness.get("final_test_items")
            if (
                not all(type(n) is int and n >= 0 for n in (development, final))
                or development + final != release.item_count
            ):
                return Gathered("failed", "benchmark_counts_differ", scope)
            out.refs.append(
                file_ref(
                    "benchmark_count_witness",
                    "repo/approvals/benchmark_count_witness_v1.json",
                    witness_bytes,
                )
            )
            item_count = release.item_count
        else:
            items = decode_jsonl(data["items"], BenchmarkItem)
            development = sum(1 for i in items if i.paper_id in set(split.development_ids))
            final = sum(1 for i in items if i.paper_id in set(split.final_test_ids))
            if len(items) != release.item_count or development + final != len(items):
                return Gathered("failed", "benchmark_counts_differ", scope, "counts do not agree")
            verify_release_inputs(release, ctx.data_root)
            item_count = len(items)
        if not git_state.is_ancestor(ctx.repo, release.implementation_git_sha):
            return Gathered("failed", "implementation_commit_not_in_history", scope)
    except FileMissing as error:
        return Gathered(
            "failed",
            "benchmark_count_witness_missing"
            if hash_only and "witness" in str(error)
            else "benchmark_artifact_missing",
            scope,
            str(error),
        )
    except ModuleNotFoundError as error:
        return Gathered("unavailable", "benchmark_validation_dependency_missing", scope, str(error))
    except git_state.GitUnavailable as error:
        return Gathered("unavailable", "git_unavailable", scope, str(error))
    except (FileProblem, ValueError, ValidationError, OSError, json.JSONDecodeError) as error:
        return Gathered("failed", "benchmark_release_verification_failed", scope, str(error)[:300])
    out.subject = {
        "kind": "benchmark_freeze_v1",
        "release_record_sha256": sha256_bytes(release_bytes),
        "items_sha256": release.items.sha256,
        "split_sha256": release.split.sha256,
        "corpus_content_sha256": release.corpus_content_sha256,
        "item_count": item_count,
        "development_items": development,
        "final_test_items": final,
        "development_papers": release.development_paper_count,
        "final_test_papers": release.final_test_paper_count,
    }
    out.facts = {
        "benchmark_release_sha256": sha256_bytes(release_bytes),
        "benchmark_seed_registry_sha256": release.seed_registry.sha256,
    }
    out.detail = f"{item_count} items: {development} development, {final} final test"
    return out


# ---- G03, G04, G05, G10, G11, G12: prerequisites that do not exist yet ---------------------


def _absent(reason: str, scope: str, detail: str) -> Gathered:
    return Gathered("absent", reason, scope, detail)


class FinalTestFreeze(Record):
    """Shape of a future freeze record: the identity of every frozen scientific setting."""

    schema_version: Literal["1"]
    settings_sha256: dict[Nonblank, Sha256]

    @model_validator(mode="after")
    def not_empty(self) -> Self:
        if not self.settings_sha256:
            raise ValueError("a freeze record binds at least one setting")
        return self


def final_test_freeze(record: bytes | None = None) -> Gathered:
    """No freeze has been approved, so the evaluator calls this without a record.

    Only the reader exists: a later, policy-pinned record is verified for shape and its
    canonical content becomes the identity subject. Nothing here creates or approves one.
    """
    scope = "FinalTestFreeze record"
    if record is None:
        return _absent("final_test_freeze_absent", scope, "no freeze record exists to verify")
    try:
        value = FinalTestFreeze.model_validate(loads_strict(record), strict=True)
    except (FileProblem, ValueError) as error:
        return Gathered("failed", "final_test_freeze_invalid", scope, str(error)[:300])
    return Gathered("ok", "checks_passed", scope, subject=value.model_dump(mode="json"))


def generator_qualification() -> Gathered:
    return _absent(
        "independent_phase7_evidence_absent",
        "Phase 7 qualification",
        "no published machine-verifiable gate subject binds the archived Phase 7 "
        "fit, boundary and replay evidence; reports and scientific checkpoint approval "
        "alone are not gate evidence",
    )


def model_parser_identities() -> Gathered:
    return _absent(
        "checkpoint_and_parser_identities_not_approved",
        "model, tokenizer, retrieval model and parser identities",
        f"{len(generator_config.APPROVED_CHECKPOINTS)} approved checkpoint manifests; "
        "no weights are read",
    )


def restart_recovery() -> Gathered:
    return _absent(
        "official_restart_evidence_absent",
        "development restart and recovery",
        "no official development run with recorded restart or retry behaviour has been accepted",
    )


def storage_authorization(
    ctx: Context | None = None, policy: ApprovalPolicy | None = None
) -> Gathered:
    scope = "approved final-test storage and frozen corpus/benchmark identity"
    if ctx is None or policy is None or policy.slots.storage_authorization is None:
        return _absent("storage_authorization_absent", scope, "G11 policy slot is unset")
    try:
        authorization, path = _approved_storage(ctx, policy)
    except AuthorityNotEstablished as error:
        return Gathered("failed", error.cause, scope, str(error)[:300])
    try:
        evidence_bytes = read_regular(ctx.data_root, path)
    except FileProblem as error:
        return Gathered("failed", "storage_approval_unusable", scope, str(error)[:300])
    if sha256_bytes(evidence_bytes) != policy.slots.storage_authorization:
        return Gathered("failed", "storage_approval_changed", scope)
    out = Gathered("ok", "checks_passed", scope, subject=authorization.model_dump(mode="json"))
    out.refs.append(file_ref("storage_authorization", "data/" + path, evidence_bytes))
    return out


def shared_gpu_authorization(ctx: Context | None = None) -> Gathered:
    scope = "amended Colab L4 campaign GPU and material authorization"
    if ctx is None:
        return _absent("gpu_authorization_absent", scope, "no attestation context")
    try:
        amendment = read_regular(ctx.repo, GPU_AMENDMENT, max_bytes=16_384)
        data = read_regular(ctx.data_root, GPU_ATTESTATION, max_bytes=16_384)
    except FileMissing:
        return _absent(
            "gpu_authorization_absent", scope, "amendment or advisor attestation is missing"
        )
    except FileProblem as error:
        return Gathered("failed", "gpu_attestation_unusable", scope, str(error)[:300])
    if sha256_bytes(amendment) != GPU_AMENDMENT_SHA256:
        return Gathered("failed", "gpu_amendment_changed", scope)
    try:
        value = loads_strict(data)
        required = {
            "schema_version",
            "researcher",
            "attested_on",
            "advisor",
            "source",
            "guidance",
            "condition",
            "colab_gpu_use",
            "restartable_batches",
            "study_material_placement",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("attestation fields differ")
        if (
            value["schema_version"] != "g12_researcher_guidance_v1"
            or value["researcher"] != "Rama Krishna Pudota"
            or value["advisor"] != "Professor William Andreopoulos"
            or value["source"] != "researcher_reported_verbal_guidance"
            or value["guidance"] != "conditional_support_for_colab"
            or value["condition"] != "scientifically_valid_and_defensible"
            or type(value["attested_on"]) is not str
            or date.fromisoformat(value["attested_on"]).isoformat() != value["attested_on"]
            or date.fromisoformat(value["attested_on"]) > datetime.now(UTC).date()
            or value["colab_gpu_use"] is not True
            or value["restartable_batches"] is not None
            or value["study_material_placement"] is not None
        ):
            raise ValueError("attestation identity or scope is invalid")
        if data != canonical_json_bytes(value):
            raise ValueError("attestation is not canonical JSON")
    except (ValueError, TypeError, KeyError) as error:
        return Gathered("failed", "gpu_attestation_invalid", scope, str(error)[:300])
    refs = [
        file_ref("gpu_campaign_amendment", "repo/" + GPU_AMENDMENT, amendment),
        file_ref("gpu_researcher_attestation", "data/" + GPU_ATTESTATION, data),
    ]
    try:
        operational_bytes = read_regular(ctx.data_root, GPU_OPERATIONAL, max_bytes=16_384)
    except FileMissing:
        return Gathered("absent", "gpu_operational_authorization_absent", scope, refs=refs)
    except FileProblem as error:
        return Gathered(
            "failed", "gpu_operational_authorization_unusable", scope, str(error)[:300], refs=refs
        )
    refs.append(
        file_ref(
            "gpu_researcher_operational_decision", "data/" + GPU_OPERATIONAL, operational_bytes
        )
    )
    try:
        campaign_bytes = read_regular(ctx.data_root, GPU_CAMPAIGN_DECISION, max_bytes=16_384)
        storage_bytes = read_regular(
            ctx.data_root, f"{STORAGE_APPROVALS}/{GPU_G11_SHA256}.json", max_bytes=16_384
        )
        campaign = loads_strict(campaign_bytes)
        if (
            not isinstance(campaign, dict)
            or campaign_bytes != canonical_json_bytes(campaign)
            or sha256_bytes(campaign_bytes) != GPU_CAMPAIGN_DECISION_SHA256
            or sha256_bytes(storage_bytes) != GPU_G11_SHA256
            or campaign.get("campaign_id") != "rk_capstone_final_test_2026"
            or campaign.get("drive_root") != "/content/drive/MyDrive/RK_Capstone_Final_2026"
            or campaign.get("execution_root") != "/content/rk_capstone_final_2026"
            or campaign.get("custody_root")
            != "/home/ramakrishnap/colab_transfer/official_custody_heads"
            or campaign.get("g11_authorization_sha256") != GPU_G11_SHA256
        ):
            raise ValueError("campaign or storage decision changed")
        operational = loads_strict(operational_bytes)
        required = {
            "schema_version",
            "researcher",
            "attested_on",
            "campaign_id",
            "platform",
            "gpu_model",
            "execution_root",
            "drive_root",
            "custody_root",
            "g11_authorization_sha256",
            "campaign_storage_decision_sha256",
            "restartable_batches_authorized",
            "restricted_material_authorized",
            "external_restrictions_status",
        }
        if not isinstance(operational, dict) or set(operational) != required:
            raise ValueError("operational fields differ")
        if (
            operational["schema_version"] != "g12_operational_v1"
            or operational["researcher"] != "Rama Krishna Pudota"
            or type(operational["attested_on"]) is not str
            or date.fromisoformat(operational["attested_on"]).isoformat()
            != operational["attested_on"]
            or date.fromisoformat(operational["attested_on"]) > datetime.now(UTC).date()
            or operational["campaign_id"] != "rk_capstone_final_test_2026"
            or operational["platform"] != "researcher_controlled_colab_pro"
            or operational["gpu_model"] != "NVIDIA L4"
            or operational["execution_root"] != "/content/rk_capstone_final_2026"
            or operational["drive_root"] != "/content/drive/MyDrive/RK_Capstone_Final_2026"
            or operational["custody_root"]
            != "/home/ramakrishnap/colab_transfer/official_custody_heads"
            or operational["g11_authorization_sha256"] != GPU_G11_SHA256
            or operational["campaign_storage_decision_sha256"] != GPU_CAMPAIGN_DECISION_SHA256
            or operational["restartable_batches_authorized"] is not True
            or operational["restricted_material_authorized"] is not True
            or operational["external_restrictions_status"]
            not in ("pending_account_specific_review", "confirmed_compliant", "conflict")
            or operational_bytes != canonical_json_bytes(operational)
        ):
            raise ValueError("operational identity or scope is invalid")
    except (FileProblem, ValueError, TypeError, KeyError) as error:
        return Gathered(
            "failed", "gpu_operational_authorization_invalid", scope, str(error)[:300], refs=refs
        )
    refs.extend(
        [
            file_ref(
                "gpu_campaign_storage_decision", "data/" + GPU_CAMPAIGN_DECISION, campaign_bytes
            ),
            file_ref(
                "gpu_storage_authorization",
                "data/" + f"{STORAGE_APPROVALS}/{GPU_G11_SHA256}.json",
                storage_bytes,
            ),
        ]
    )
    if operational["external_restrictions_status"] == "conflict":
        return Gathered("failed", "gpu_external_restriction_conflict", scope, refs=refs)
    if operational["external_restrictions_status"] != "confirmed_compliant":
        return Gathered("absent", "gpu_external_restrictions_unconfirmed", scope, refs=refs)
    return Gathered(
        "ok",
        "checks_passed",
        scope,
        subject={
            "campaign_id": operational["campaign_id"],
            "amendment_sha256": sha256_bytes(amendment),
            "advisor_attestation_sha256": sha256_bytes(data),
            "researcher_operational_sha256": sha256_bytes(operational_bytes),
            "g11_authorization_sha256": GPU_G11_SHA256,
            "campaign_storage_decision_sha256": GPU_CAMPAIGN_DECISION_SHA256,
        },
        refs=refs,
    )


NO_AUTHORITY = "final_test_inventory_authority_not_established"
STORAGE_APPROVALS = "readiness/approvals/storage_authorization"
ROOT_ANCHORS = "readiness/final_test_root"


def _approved_storage(
    ctx: Context, policy: ApprovalPolicy
) -> tuple[root_layout.StorageAuthorization, str]:
    """The storage authorization that policy slot G11 approves, and its file path in the data
    root. Raises `AuthorityNotEstablished` unless it exists, is canonical, hashes to the slot
    and names the final-test kind and the frozen benchmark."""
    slot = policy.slots.storage_authorization
    if slot is None:
        raise AuthorityNotEstablished("no_storage_approval", "policy slot G11 is unset")
    path = f"{STORAGE_APPROVALS}/{slot}.json"
    try:
        data = read_regular(ctx.data_root, path)
        authorization = _json_model(data, root_layout.StorageAuthorization)
        release, release_bytes = read_benchmark_record(ctx)
    except (FileProblem, ValueError) as error:
        raise AuthorityNotEstablished("storage_approval_unusable", str(error)[:200]) from error
    canonical = canonical_json_bytes(authorization.model_dump(mode="json"))
    if data != canonical or sha256_bytes(data) != slot:
        raise AuthorityNotEstablished("storage_approval_unusable", "bytes differ from the slot")
    if authorization.root_kind != "final_test":
        raise AuthorityNotEstablished("root_kind", "only a final_test root can satisfy G13")
    if (authorization.corpus_content_sha256, authorization.benchmark_release_sha256) != (
        release.corpus_content_sha256,
        sha256_bytes(release_bytes),
    ):
        raise AuthorityNotEstablished("scope", "the approval names another corpus or benchmark")
    return authorization, path


def history_integrity(ctx: Context | None = None, policy: ApprovalPolicy | None = None) -> Gathered:
    """G13. The final-test root must be the one the G11 approval describes."""
    scope = "final-test history"
    if ctx is None or policy is None:
        return Gathered("unavailable", NO_AUTHORITY, scope, "no approval to verify against")
    try:
        authorization, approval_path = _approved_storage(ctx, policy)
        verdict = verify_root(
            Path(authorization.authorized_root), authorization, ctx.data_root / ROOT_ANCHORS
        )
    except AuthorityNotEstablished as error:
        return Gathered("unavailable", NO_AUTHORITY, scope, error.cause)
    if verdict.code == INCOMPLETE:
        return Gathered("unavailable", INCOMPLETE, scope, "; ".join(verdict.problems)[:300])
    if verdict.code in HISTORY_FAIL_CODES:
        return Gathered("failed", verdict.code, scope, "; ".join(verdict.problems)[:300])
    out = Gathered("ok", verdict.code, scope)
    out.refs.append(
        EvidenceRef(
            kind="storage_authorization",
            locator="data/" + approval_path,
            sha256=root_layout.authorization_sha256(authorization),
            subject_kind="stored_file_bytes",
        )
    )
    if verdict.anchor_sha256 is not None:
        out.refs.append(
            EvidenceRef(
                kind="final_test_root_anchor",
                locator=f"data/{ROOT_ANCHORS}/{verdict.root_id}{root_layout.ANCHOR_SUFFIX}",
                sha256=verdict.anchor_sha256,
                subject_kind="stored_file_bytes",
            )
        )
    out.add_snapshot("final_test_inventory", verdict.inventory)
    return out


# ---- G06 Python environment ---------------------------------------------------------------


def python_environment(ctx: Context) -> Gathered:
    scope = "intended lock, installed packages of the evaluating host"
    out = Gathered("ok", "checks_passed", scope)
    try:
        spec_bytes = read_regular(ctx.repo, ENVIRONMENT_SPEC)
        spec = parse_spec(spec_bytes)
    except FileMissing as error:
        return Gathered("unavailable", "environment_spec_missing", scope, str(error))
    except (FileProblem, EnvironmentLockError) as error:
        return Gathered("failed", "environment_spec_unusable", scope, str(error)[:300])
    spec_sha = sha256_bytes(spec_bytes)
    out.refs.append(file_ref("environment_spec", "repo/" + ENVIRONMENT_SPEC, spec_bytes))
    try:
        inventory = ctx.probes.inventory()
    except Exception as error:  # noqa: BLE001 - any probe failure means "not inspected"
        return Gathered("unavailable", "environment_inventory_unavailable", scope, str(error)[:300])
    approved = spec_sha in generator_config.APPROVED_ENVIRONMENT_LOCKS
    report = verify_environment(
        spec_bytes,
        inventory,
        expected_spec_sha256=spec_sha if approved else None,
        inventory_source="live",
    )
    out.add_snapshot("python_inventory", inventory.model_dump(mode="json"))
    out.add_snapshot("environment_verification", _verification(report))
    out.facts = {"python_lock_sha256": spec_sha if spec.status == "approved_lock" else None}
    out.detail = (
        f"lock status {spec.status}, outcome {report.outcome}; the inventory is of the "
        "evaluating host, which is not necessarily the execution host"
    )
    if not approved or report.outcome == "not_approved":
        return _carry(out, "absent", "environment_lock_not_approved")
    if report.outcome == "differs_from_approved_lock":
        return _carry(out, "failed", "installed_environment_differs_from_approved_lock")
    if report.outcome != "matches_approved_lock":
        return _carry(out, "unavailable", "environment_inventory_incomplete")
    out.subject = {"kind": "python_environment_lock_v1", "lock_sha256": spec_sha}
    return out


def _verification(report: EnvironmentVerification) -> dict[str, Any]:
    value: dict[str, Any] = report.model_dump(mode="json")
    return value


def _carry(out: Gathered, state: Any, reason: str) -> Gathered:
    out.state, out.reason = state, reason
    return out


# ---- G07 R analysis environment -----------------------------------------------------------

_R_SCRIPT = r"""
a <- commandArgs(trailingOnly = TRUE)
cat("BIORAG\tlibpaths\t", paste(.libPaths(), collapse = "\t"), "\n", sep = "")
cat("BIORAG\trversion\t", as.character(getRversion()), "\n", sep = "")
for (p in a) {
  d <- find.package(p, quiet = TRUE)
  v <- if (length(d)) packageDescription(p, lib.loc = dirname(d[1]))$Version else "-"
  i <- if (length(d)) match(dirname(d[1]), .libPaths()) else NA
  cat("BIORAG\tpkg\t", p, "\t", v, "\t", i, "\n", sep = "")
}
"""


def run_r_runtime(analysis_dir: Path, packages: list[str]) -> str:
    """Observe the R session the analysis project resolves. Read-only; nothing is installed."""
    done = subprocess.run(
        ["Rscript", "-e", _R_SCRIPT, *packages],
        cwd=analysis_dir,
        capture_output=True,
        text=True,
        timeout=R_PROBE_TIMEOUT_S,
        check=True,
    )
    return done.stdout


def parse_r_runtime(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {"libpaths": [], "rversion": None, "packages": {}}
    for line in text.splitlines():
        parts = line.split("\t")
        if parts[0] != "BIORAG" or len(parts) < 3:
            continue
        if parts[1] == "libpaths":
            result["libpaths"] = [p for p in parts[2:] if p]
        elif parts[1] == "rversion":
            result["rversion"] = parts[2]
        elif parts[1] == "pkg" and len(parts) == 5:
            result["packages"][parts[2]] = {"version": parts[3], "library_index": parts[4]}
    if not result["libpaths"] or result["rversion"] is None:
        raise ValueError("R probe output is incomplete")
    return result


def r_environment(ctx: Context) -> Gathered:
    scope = "analysis lock and the R session the project resolves on the evaluating host"
    out = Gathered("absent", "analysis_host_not_designated", scope)
    try:
        lock_bytes = read_regular(ctx.repo, RENV_LOCK)
        lock = loads_strict(lock_bytes)
        locked = {n: p["Version"] for n, p in lock["Packages"].items()}
        locked_r = lock["R"]["Version"]
        sources = {
            p: sha256_bytes(read_regular(ctx.repo, p))
            for p in [*sorted(_r_files(ctx.repo)), *R_SOURCES]
        }
    except FileMissing as error:
        return Gathered("unavailable", "analysis_lock_missing", scope, str(error))
    except (FileProblem, KeyError, TypeError, AttributeError) as error:
        return Gathered("failed", "analysis_lock_unusable", scope, str(error)[:300])
    out.refs.append(file_ref("renv_lock", "repo/" + RENV_LOCK, lock_bytes))
    out.facts = {"renv_lock_sha256": sha256_bytes(lock_bytes)}
    out.add_snapshot("analysis_sources", {"kind": "analysis_sources", "sha256": sources})
    try:
        observed = parse_r_runtime(ctx.probes.r_runtime(ctx.repo / "analysis", sorted(locked)))
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        out.add_snapshot(
            "r_inspection",
            {"kind": "r_inspection", "runtime_probe": "unavailable", "locked": locked},
        )
        out.state, out.reason = "unavailable", "r_runtime_probe_unavailable"
        out.detail = (
            "R could not be run, so package resolution is not verified; "
            f"no static result is claimed as runtime conformity ({str(error)[:200]})"
        )
        return out
    project_library = observed["libpaths"][0].startswith(
        str((ctx.repo / "analysis/renv/library").resolve())
    )
    differing = sorted(
        n for n, v in locked.items() if observed["packages"].get(n, {}).get("version") != v
    )
    out.add_snapshot(
        "r_inspection",
        {
            "kind": "r_inspection",
            "runtime_probe": "run",
            "project_library_first": project_library,
            "r_version": observed["rversion"],
            "locked_r_version": locked_r,
            "packages": observed["packages"],
            "differing": differing,
        },
    )
    if not project_library:
        out.state, out.reason = "unavailable", "r_project_library_not_active"
    elif observed["rversion"] != locked_r:
        out.state, out.reason = "failed", "r_version_differs_from_lock"
    elif differing:
        out.state, out.reason = "failed", "installed_r_packages_differ_from_lock"
        out.detail = f"{len(differing)} packages differ"
    else:
        from biorag_repro.readiness.production import host_designation

        try:
            designation = host_designation(ctx, sources, sha256_bytes(lock_bytes))
        except FileMissing as error:
            out.state, out.reason, out.detail = "absent", "analysis_designation_missing", str(error)
        except (FileProblem, ValueError, KeyError, OSError) as error:
            out.state, out.reason, out.detail = (
                "failed",
                "analysis_designation_invalid",
                str(error)[:300],
            )
        else:
            designation_bytes = read_regular(
                ctx.repo, "docs/reports/g07_researcher_designation_evidence_2026_11_20.json"
            )
            out.refs.append(
                file_ref(
                    "analysis_designation",
                    "repo/docs/reports/g07_researcher_designation_evidence_2026_11_20.json",
                    designation_bytes,
                )
            )
            out.state, out.reason = "ok", "checks_passed"
            out.subject = {
                "kind": "designated_analysis_environment_v1",
                "designation_sha256": sha256_bytes(designation_bytes),
                "host": designation["designation"]["host"],
                "renv_lock_sha256": sha256_bytes(lock_bytes),
                "analysis_sources_sha256": sources,
                "r_version": observed["rversion"],
                "packages": observed["packages"],
                "blas_execution_pin": {"OPENBLAS_CORETYPE": "HASWELL", "OPENBLAS_NUM_THREADS": "1"},
            }
            out.detail = (
                "designated host and locked runtime match; no official analysis execution claimed"
            )
    return out


def _r_files(repo: Path) -> list[str]:
    directory = repo / "analysis" / "R"
    return [f"analysis/R/{p.name}" for p in directory.glob("*.R")] if directory.is_dir() else []


# ---- G08 seeds -----------------------------------------------------------------------------


def seed_registry(ctx: Context, release: BenchmarkRelease | None) -> Gathered:
    scope = "benchmark seed registry and the production analysis seed"
    if release is None:
        return Gathered("unavailable", "benchmark_release_unavailable", scope)
    try:
        ref, data = _data_ref("benchmark_seed_registry", ctx.data_root, release.seed_registry)
        _json_model(data, BenchmarkSeedRegistry)
    except FileMissing as error:
        return Gathered("failed", "seed_registry_missing", scope, str(error))
    except (FileProblem, ValueError, ValidationError) as error:
        return Gathered("failed", "seed_registry_invalid", scope, str(error)[:300])
    try:
        seed = read_analysis_seed(ctx.repo)
    except FileMissing:
        out = Gathered("absent", "production_analysis_seed_unapproved", scope)
    except (FileProblem, ValueError) as error:
        out = Gathered("failed", "production_analysis_seed_invalid", scope, str(error)[:300])
    else:
        out = Gathered(
            "ok",
            "checks_passed",
            scope,
            subject={
                "kind": "seed_registries_v1",
                "benchmark_seed_registry_sha256": release.seed_registry.sha256,
                "analysis_seed_registry_sha256": APPROVED_REGISTRY_SHA256,
            },
        )
        out.refs.append(
            EvidenceRef(
                kind="analysis_seed_registry",
                locator="repo/" + REGISTRY_PATH,
                sha256=APPROVED_REGISTRY_SHA256,
                subject_kind="stored_file_bytes",
            )
        )
        out.add_snapshot("registered_analysis_seed", seed.model_dump(mode="json"))
    out.refs.append(ref)
    return out


# ---- G09 scientific source -----------------------------------------------------------------


def scientific_source(ctx: Context) -> Gathered:
    scope = "tracked scientific source (Git) and frozen specification files (ignored, hashed)"
    try:
        state = git_state.probe(ctx.repo)
    except git_state.GitUnavailable as error:
        return Gathered("unavailable", "git_unavailable", scope, str(error))
    out = Gathered("ok", "checks_passed", scope, facts={"head_sha": state.head})
    out.add_snapshot("git_scope_state", json.loads(state.snapshot_bytes()))
    if state.untracked_outside_scope:
        ctx.warnings.append(
            f"{state.untracked_outside_scope} untracked files lie outside the scientific scope"
        )
    if state.modified or state.untracked or state.approval_blob is None:
        out.state, out.reason = "failed", "scientific_source_not_clean"
        out.detail = f"{len(state.modified)} modified, {len(state.untracked)} untracked in scope"
        if state.approval_blob is None:
            out.detail += f"; {git_state.APPROVAL_FILE} is not tracked by Git"
        return out
    specs: dict[str, str] = {}
    try:
        for name in FROZEN_SPECIFICATIONS:
            data = read_regular(ctx.repo, "docs/frozen/" + name)
            specs[name] = sha256_bytes(data)
            out.refs.append(file_ref("frozen_specification", "repo/docs/frozen/" + name, data))
    except FileProblem as error:
        out.state, out.reason, out.detail = (
            "unavailable",
            "frozen_specification_unavailable",
            str(error),
        )
        return out
    out.subject = {
        "kind": "scientific_source_v1",
        "tracked_scope": [list(e) for e in state.entries],
        "frozen_specifications": specs,
    }
    out.detail = (
        f"{len(state.entries)} tracked files; frozen specifications are mutable local "
        "files whose bytes are hashed here and are not retained"
    )
    return out
