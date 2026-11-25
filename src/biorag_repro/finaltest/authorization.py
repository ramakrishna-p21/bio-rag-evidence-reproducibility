"""Prospective four-check authorization for final-test execution.

Historical G01–G13 reports retain their original meaning. This reader groups their live,
verified evidence for execution and requires G07 under the frozen start contract.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.records import Record
from biorag_repro.finaltest.history_v2 import HistoryRefused, _verify_history, verify_history
from biorag_repro.finaltest.inventory import AuthorityNotEstablished
from biorag_repro.finaltest.layout import (
    Sha256,
    StorageAuthorization,
    authorization_sha256,
    root_id_for,
)
from biorag_repro.generation.environment import check_utc
from biorag_repro.readiness import evidence
from biorag_repro.readiness.approvals import load_policy
from biorag_repro.readiness.evaluate import evaluate, system_context
from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular
from biorag_repro.readiness.git_state import GitUnavailable, probe, verify_execution_revision
from biorag_repro.readiness.models import GitRevision

GROUPS = {
    "scientific_integrity": (
        "G01_corpus_sources",
        "G02_benchmark_freeze",
        "G03_final_test_configuration_freeze",
        "G08_seed_registry",
        "G09_scientific_source_revision",
    ),
    "model_environment": (
        "G04_generator_qualification",
        "G05_model_parser_identities",
        "G06_python_environment",
        "G07_r_analysis_environment",
    ),
    "authorization": ("G11_storage_authorization",),
    "execution_integrity": ("G10_development_restart_recovery",),
}
CAMPAIGN_ID = "rk_capstone_final_test_2026"
DRIVE_ROOT = "/content/drive/MyDrive/RK_Capstone_Final_2026"


class ExecutionRefused(ValueError):
    """A prospective execution check lacks genuine verified evidence."""


class CampaignDecision(Record):
    """Researcher's exact prospective authorization for this final-test campaign."""

    schema_version: Literal["final_test_execution_v2"]
    researcher: Literal["Rama Krishna Pudota"]
    source_revision: GitRevision
    private_storage_and_transfer_approved: Literal[True]
    frozen_configuration_sha256: Sha256
    storage_authorization_sha256: Sha256
    storage_root: str
    root_id: Sha256
    genesis_created_utc: str
    genesis_anchor_sha256: Sha256
    custody_mode: Literal["amended_drive_v1"] | None = None
    campaign_id: Literal["rk_capstone_final_test_2026"] | None = None
    drive_root: str | None = None

    _utc = field_validator("genesis_created_utc")(check_utc)

    @model_validator(mode="after")
    def exact_custody_selection(self):
        values = (self.custody_mode, self.campaign_id, self.drive_root)
        if any(value is not None for value in values) and values != (
            "amended_drive_v1",
            CAMPAIGN_ID,
            DRIVE_ROOT,
        ):
            raise ValueError("amended custody fields do not name the approved campaign")
        if not Path(self.storage_root).is_absolute():
            raise ValueError("storage_root must be absolute")
        return self


@dataclass(frozen=True)
class ExecutionAuthorization:
    source_revision: str
    policy_sha256: str
    binding_sha256: str
    evidence_sha256: str
    history_head_sha256: str
    storage_root: Path
    anchor_dir: Path
    storage_authorization: StorageAuthorization
    custody_mode: str = "ordinary"
    campaign_id: str | None = None
    drive_root: Path | None = None
    pending_head_sha256: str | None = None


@dataclass(frozen=True)
class CampaignPreflight:
    report: object
    storage: StorageAuthorization
    decision: CampaignDecision
    decision_bytes: bytes


def _load_decision(repo: Path) -> tuple[CampaignDecision, bytes]:
    try:
        data = read_regular(repo, "approvals/final_test_execution.json")
        decision = CampaignDecision.model_validate(loads_strict(data), strict=True)
    except (FileProblem, ValueError) as error:
        raise ExecutionRefused("authorization: campaign approval is missing or invalid") from error
    if canonical_json_bytes(decision.model_dump(mode="json", exclude_none=True)) != data:
        raise ExecutionRefused("authorization: campaign decision is not canonical JSON")
    return decision, data


def _validate_decision(
    repo: Path,
    report,
    storage: StorageAuthorization,
    *,
    custody_mode: str,
) -> tuple[CampaignDecision, bytes]:
    decision, data = _load_decision(repo)
    try:
        state = probe(repo)
        if state.head != report.candidate.head_sha or state.modified or state.untracked:
            raise GitUnavailable("live scientific source is not the evaluated clean HEAD")
        if decision.source_revision == state.head:
            raise GitUnavailable("source revision must precede the decision commit")
        verify_execution_revision(repo, state, decision.source_revision)
    except GitUnavailable as error:
        raise ExecutionRefused(f"authorization: {error}") from error
    if (
        decision.frozen_configuration_sha256 != report.candidate.freeze_record_sha256
        or decision.storage_authorization_sha256 != authorization_sha256(storage)
        or decision.storage_root != storage.authorized_root
        or decision.root_id != root_id_for(storage)
    ):
        raise ExecutionRefused("authorization: campaign decision differs from live evidence")
    if custody_mode not in ("ordinary", "amended_drive"):
        raise ExecutionRefused("unknown custody mode")
    if custody_mode == "amended_drive" and decision.custody_mode != "amended_drive_v1":
        raise ExecutionRefused(
            "amended campaign custody gate: exact campaign and Drive approval missing"
        )
    return decision, data


def campaign_preflight(
    repo: Path, data_root: Path, *, custody_mode: str = "ordinary"
) -> CampaignPreflight:
    """Verify every prospective campaign requirement except the not-yet-created G13 root."""
    policy = load_policy(repo)
    ctx = system_context(repo, data_root)
    report = evaluate(ctx, policy, g02_hash_only=True).report
    by_gate = {gate.gate_id: gate for gate in report.gates}
    for group, names in GROUPS.items():
        for name in names:
            gate = by_gate[name]
            if gate.status != "passed" or gate.verification != "verified":
                raise ExecutionRefused(f"{group}: {name}: {gate.status} ({gate.reason_code})")
    try:
        storage, _ = evidence._approved_storage(ctx, policy)
    except (AuthorityNotEstablished, ValueError, OSError) as error:
        raise ExecutionRefused(f"authorization: G11: {error}") from error
    decision, decision_bytes = _validate_decision(repo, report, storage, custody_mode=custody_mode)
    return CampaignPreflight(report, storage, decision, decision_bytes)


def authorize(
    repo: Path,
    data_root: Path,
    expected_head_sha256: str,
    *,
    custody_mode: str = "ordinary",
    pending_batch_id: str | None = None,
) -> ExecutionAuthorization:
    """Check all four groups from fresh evidence; never accept a caller's READY claim."""
    preflight = campaign_preflight(repo, data_root, custody_mode=custody_mode)
    report = preflight.report
    storage = preflight.storage
    try:
        pending_head = None
        try:
            verify_history(
                Path(storage.authorized_root),
                storage,
                data_root / evidence.ROOT_ANCHORS,
                expected_head_sha256,
            )
        except HistoryRefused:
            if custody_mode != "amended_drive" or pending_batch_id is None:
                raise
            anchors = data_root / evidence.ROOT_ANCHORS
            try:
                _verify_history(
                    Path(storage.authorized_root),
                    storage,
                    anchors,
                    expected_head_sha256,
                    pending_batch_id,
                )
            except HistoryRefused:
                names = sorted(
                    anchors.glob(f"{root_id_for(storage)}.[0-9][0-9][0-9][0-9][0-9][0-9].json")
                )
                if not names:
                    raise
                latest = loads_strict(names[-1].read_bytes())
                if (
                    latest.get("previous_anchor_sha256") != expected_head_sha256
                    or latest.get("batch_id") != pending_batch_id
                ):
                    raise
                pending_head = sha256_file(names[-1])
                verify_history(Path(storage.authorized_root), storage, anchors, pending_head)
    except (AuthorityNotEstablished, ValueError, OSError) as error:
        raise ExecutionRefused(f"execution_integrity: G13: {error}") from error
    decision, approval = preflight.decision, preflight.decision_bytes
    return ExecutionAuthorization(
        source_revision=decision.source_revision,
        policy_sha256=report.policy_sha256,
        binding_sha256=report.binding_sha256,
        evidence_sha256=sha256_bytes(approval),
        history_head_sha256=expected_head_sha256,
        storage_root=Path(storage.authorized_root),
        anchor_dir=data_root / evidence.ROOT_ANCHORS,
        storage_authorization=storage,
        custody_mode=custody_mode,
        campaign_id=decision.campaign_id if custody_mode == "amended_drive" else None,
        drive_root=Path(decision.drive_root) if custody_mode == "amended_drive" else None,
        pending_head_sha256=pending_head,
    )
