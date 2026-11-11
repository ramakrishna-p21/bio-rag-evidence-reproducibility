"""Scientific candidate checks; human authorization has no production provider."""

import json
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from biorag_repro.core.hashing import (
    read_verified_bytes,
    sha256_bytes,
    verify_file_sha256,
)
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.documents.ingestion import (
    Arm,
    JatsBuild,
    Phase4GateIndex,
    Phase4Validation,
    evidence_ref,
    frozen_source,
    load_frozen_paper,
    validate_build,
)
from biorag_repro.documents.jats import TRANSFORMATION_ID as JATS_TRANSFORMATION
from biorag_repro.documents.lineage import check_manifest_contract, runtime_settings
from biorag_repro.documents.models import CanonicalDocument, Record
from biorag_repro.documents.pdf import TRANSFORMATION_ID as PDF_TRANSFORMATION
from biorag_repro.documents.pdf_ingest import (
    NATIVE_TRANSFORMATION,
    ConversionAttempt,
    PdfCanonicalBuild,
    canonical_build_ref,
)

# Frozen scientific selection; neither value is human publication approval.
PHASE2H_SHA256 = "70c365230fb05041e6466b3686fb3e3bf1e93f20bb3867172163df37f3e9c064"
QUALIFIED_ENVIRONMENT = "867f856314c41ed313fd4e697be20854e789b14fd4ecffdcae75ac7877148a42"


@dataclass(frozen=True)
class ScientificAuthority:
    """Independently selected inputs, never deserialized from a candidate attestation.

    This is scientific selection context, not a human approval capability.
    """

    release_path: Path
    release_sha256: str
    native_attempt: ArtifactRef
    jats_configuration_sha256: str
    pdf_configuration_sha256: str
    environment_sha256: str
    asset_inventory_sha256: str


class ReproductionCandidate(Record):
    schema_version: Literal["1"] = "1"
    status: Literal["validated_candidate_pending_human_review"] = (
        "validated_candidate_pending_human_review"
    )
    claim: Literal["deterministic_reproduction_not_historical_execution"] = (
        "deterministic_reproduction_not_historical_execution"
    )
    historical_producer_authenticated: Literal[False] = False
    human_authorized: Literal[False] = False
    reproduction_git_sha: str
    repository_state: Literal["clean"] = "clean"
    started_utc: str
    ended_utc: str
    runtime_inventory: ArtifactRef
    corpus_version: str
    paper_id: str
    corpus_release_sha256: str
    jats_validation: ArtifactRef
    pdf_validation: ArtifactRef


def candidate_ref(value: ReproductionCandidate) -> ArtifactRef:
    content = serialize_project_json(value, ReproductionCandidate)
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind="phase4_reproduction_candidate",
        inputs=[
            ArtifactInput(role, ref.artifact_id, ref.sha256)
            for role, ref in (
                ("jats_validation", value.jats_validation),
                ("pdf_validation", value.pdf_validation),
                ("runtime_inventory", value.runtime_inventory),
            )
        ],
        transformation_id="phase4_reproduction_candidate_v1",
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/phase4_candidates/{artifact_id}.json",
        sha256=digest,
    )


def validation_ref(value: Phase4Validation) -> ArtifactRef:
    return evidence_ref(
        value,
        Phase4Validation,
        "phase4_validation",
        [
            ArtifactInput(
                "canonical", value.canonical_artifact.artifact_id, value.canonical_artifact.sha256
            )
        ],
    )


def validate_scientific_pair(
    value: Phase4GateIndex,
    root: Path,
    corpus: Path,
    authority: ScientificAuthority,
    expected_revision: str,
) -> None:
    """Recompute scientific claims using selection independent of the submitted pair.

    expected_revision is the separately observed reproduction revision, not approval.
    This check does not authenticate that observation or historical execution.
    """
    if not re.fullmatch("[0-9a-f]{40}", expected_revision) or expected_revision == "0" * 40:
        raise ValueError("invalid independently observed reproduction revision")
    if value.implementation_git_sha != expected_revision:
        raise ValueError("reproduction revision differs from independently observed revision")
    if authority.release_sha256 != PHASE2H_SHA256:
        raise ValueError("authorized frozen release selection differs from Phase 2H")
    if value.corpus_release_sha256 != authority.release_sha256:
        raise ValueError("authorized frozen release mismatch")
    release, paper = load_frozen_paper(
        corpus, authority.release_path, authority.release_sha256, "corpus_v1", value.paper_id
    )
    if value.corpus_version != release.corpus_version:
        raise ValueError("authorized corpus version mismatch")
    arms: tuple[Arm, Arm] = ("JATS", "PDF_DOCLING")
    sources = {arm: frozen_source(paper, arm, corpus)[1] for arm in arms}
    attempt = read_project_json(
        root / authority.native_attempt.logical_ref,
        authority.native_attempt.sha256,
        ConversionAttempt,
    )
    if (
        authority.native_attempt.artifact_id != "attempt_" + attempt.attempt_id
        or authority.native_attempt.logical_ref != attempt.attempt_logical_ref
        or attempt.paper_id != value.paper_id
        or attempt.source != sources["PDF_DOCLING"]
        or attempt.configuration_sha256 != authority.pdf_configuration_sha256
        or attempt.environment_sha256 != authority.environment_sha256
        or attempt.asset_inventory_sha256 != authority.asset_inventory_sha256
        or attempt.status != "success"
        or attempt.build_status != "complete"
        or attempt.errors
        or attempt.native_artifact is None
        or attempt.native_build_manifest is None
    ):
        raise ValueError("qualified native attempt authority mismatch")
    settings = runtime_settings(
        authority.pdf_configuration_sha256,
        authority.environment_sha256,
        authority.asset_inventory_sha256,
    )
    native = attempt.native_artifact
    native_data = json.loads(read_verified_bytes(root / native.logical_ref, native.sha256))
    if native_data.get("schema_name") != "DoclingDocument" or not native_data.get("version"):
        raise ValueError("invalid qualified native schema")
    check_manifest_contract(
        attempt.native_build_manifest,
        root,
        native,
        [sources["PDF_DOCLING"]],
        ["raw_pdf"],
        "docling_native",
        NATIVE_TRANSFORMATION,
        settings,
        native_data["version"],
    )
    for arm, ref in (("JATS", value.jats_validation), ("PDF_DOCLING", value.pdf_validation)):
        report = read_project_json(root / ref.logical_ref, ref.sha256, Phase4Validation)
        if validation_ref(report) != ref:
            raise ValueError("validation reference identity mismatch")
        if (
            report.paper_id != value.paper_id
            or report.corpus_version != value.corpus_version
            or report.representation != arm
            or report.source != sources[arm]
        ):
            raise ValueError("authorized frozen source or validation arm mismatch")
        config = (
            authority.jats_configuration_sha256
            if arm == "JATS"
            else authority.pdf_configuration_sha256
        )
        if report.configuration_sha256 != config:
            raise ValueError("authorized configuration mismatch")
        doc = read_project_json(
            root / report.canonical_artifact.logical_ref,
            report.canonical_artifact.sha256,
            CanonicalDocument,
        )
        transform = JATS_TRANSFORMATION if arm == "JATS" else PDF_TRANSFORMATION
        check_manifest_contract(
            report.build_manifest,
            root,
            report.canonical_artifact,
            [sources[arm]] if arm == "JATS" else [sources[arm], native],
            ["raw_jats"] if arm == "JATS" else ["raw_pdf", "docling_native"],
            "canonical_jats" if arm == "JATS" else "canonical_pdf_docling",
            transform,
            config if arm == "JATS" else settings,
            "4",
        )
        if doc.schema_version != "4" or doc.transformation_id != transform:
            raise ValueError("canonical schema or adapter mismatch")
        build: JatsBuild | ConversionAttempt | PdfCanonicalBuild
        if arm == "JATS":
            build = JatsBuild(
                paper_id=value.paper_id,
                source=report.source,
                configuration_sha256=config,
                canonical_artifact=report.canonical_artifact,
                build_manifest=report.build_manifest,
            )
        else:
            if (
                report.attempt_artifact != authority.native_attempt
                or report.native_artifact != native
                or report.native_build_manifest != attempt.native_build_manifest
                or report.environment_sha256 != authority.environment_sha256
                or report.asset_inventory_sha256 != authority.asset_inventory_sha256
            ):
                raise ValueError("qualified native lineage mismatch")
            if report.canonical_build_artifact is None:
                build = attempt
            else:
                build = read_project_json(
                    root / report.canonical_build_artifact.logical_ref,
                    report.canonical_build_artifact.sha256,
                    PdfCanonicalBuild,
                )
                if (
                    canonical_build_ref(build) != report.canonical_build_artifact
                    or build.original_attempt_artifact != authority.native_attempt
                    or build.native_artifact != native
                    or build.source != sources[arm]
                    or build.native_build_manifest != attempt.native_build_manifest
                    or build.environment_sha256 != authority.environment_sha256
                    or build.asset_inventory_sha256 != authority.asset_inventory_sha256
                ):
                    raise ValueError("saved native reuse lineage mismatch")
        actual = validate_build(
            build, root, corpus_root=corpus, corpus_version=release.corpus_version
        )
        if actual != report:
            raise ValueError("validation reproduction differs, including warnings and coverage")


def validate_candidate(
    value: ReproductionCandidate,
    root: Path,
    corpus: Path,
    authority: ScientificAuthority,
    expected_revision: str,
) -> None:
    value = ReproductionCandidate.model_validate(value.model_dump())
    for stamp in (value.started_utc, value.ended_utc):
        parsed = datetime.fromisoformat(stamp)
        if parsed.utcoffset() != UTC.utcoffset(parsed):
            raise ValueError("reproduction timestamps must be UTC")
    if datetime.fromisoformat(value.ended_utc) < datetime.fromisoformat(value.started_utc):
        raise ValueError("reproduction end precedes start")
    if value.runtime_inventory.artifact_id != "runtime_" + value.runtime_inventory.sha256:
        raise ValueError("runtime identity mismatch")
    verify_file_sha256(root / value.runtime_inventory.logical_ref, value.runtime_inventory.sha256)
    validate_scientific_pair(
        Phase4GateIndex(
            corpus_version=value.corpus_version,
            paper_id=value.paper_id,
            corpus_release_sha256=value.corpus_release_sha256,
            implementation_git_sha=value.reproduction_git_sha,
            jats_validation=value.jats_validation,
            pdf_validation=value.pdf_validation,
        ),
        root,
        corpus,
        authority,
        expected_revision,
    )


def publish_candidate(
    value: ReproductionCandidate,
    root: Path,
    corpus: Path,
    authority: ScientificAuthority,
    expected_revision: str,
) -> ArtifactRef:
    validate_candidate(value, root, corpus, authority, expected_revision)
    ref = candidate_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, ReproductionCandidate),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, ReproductionCandidate),
    )
    return ref


def publish_complete_gate(
    value: ReproductionCandidate,
    root: Path,
    corpus: Path,
    authority: ScientificAuthority,
    expected_revision: str,
) -> ArtifactRef:
    validate_candidate(value, root, corpus, authority, expected_revision)
    raise ValueError("independent human authorization is unavailable in production")


def observed_clean_revision(repository: Path) -> str:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        if subprocess.run(["git", *args], cwd=repository, check=False).returncode:
            raise ValueError("reproduction requires clean tracked implementation")
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=repository
    ):
        raise ValueError("reproduction requires no untracked implementation files")
    return revision
