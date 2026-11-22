"""Prospective four-check authorization for final-test execution.

Historical G01–G13 reports retain their original meaning. This reader groups their live,
verified evidence for execution and requires G07 under the frozen start contract.
"""

from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes, sha256_file
from biorag_repro.finaltest.history_v2 import HistoryRefused, _verify_history, verify_history
from biorag_repro.finaltest.inventory import AuthorityNotEstablished
from biorag_repro.finaltest.layout import StorageAuthorization, root_id_for
from biorag_repro.readiness import evidence
from biorag_repro.readiness.approvals import load_policy
from biorag_repro.readiness.evaluate import evaluate, system_context
from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular

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


class ExecutionRefused(ValueError):
    """A prospective execution check lacks genuine verified evidence."""


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


def authorize(
    repo: Path,
    data_root: Path,
    expected_head_sha256: str,
    *,
    custody_mode: str = "ordinary",
    pending_batch_id: str | None = None,
) -> ExecutionAuthorization:
    """Check all four groups from fresh evidence; never accept a caller's READY claim."""
    policy = load_policy(repo)
    ctx = system_context(repo, data_root)
    evaluation = evaluate(ctx, policy, g02_hash_only=True)
    report = evaluation.report
    by_gate = {gate.gate_id: gate for gate in report.gates}
    for group, names in GROUPS.items():
        for name in names:
            gate = by_gate[name]
            if gate.status != "passed" or gate.verification != "verified":
                raise ExecutionRefused(f"{group}: {name}: {gate.status} ({gate.reason_code})")
    try:
        storage, _ = evidence._approved_storage(ctx, policy)
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
    # The amendment authorizes code development. Campaign execution still requires a
    # separately recorded researcher decision with exact source and transfer scope.
    try:
        approval = read_regular(repo, "approvals/final_test_execution.json")
    except FileProblem as error:
        raise ExecutionRefused("authorization: campaign approval is missing") from error
    decision = loads_strict(approval)
    if (
        decision.get("schema_version") != "final_test_execution_v2"
        or decision.get("researcher") != "Rama Krishna Pudota"
        or decision.get("source_revision") != report.candidate.head_sha
        or decision.get("private_storage_and_transfer_approved") is not True
        or decision.get("frozen_configuration_sha256") != report.candidate.freeze_record_sha256
    ):
        raise ExecutionRefused("authorization: campaign decision differs from live evidence")
    if custody_mode not in ("ordinary", "amended_drive"):
        raise ExecutionRefused("unknown custody mode")
    if custody_mode == "amended_drive" and (
        decision.get("custody_mode") != "amended_drive_v1"
        or decision.get("campaign_id") != "rk_capstone_final_test_2026"
        or not isinstance(decision.get("drive_root"), str)
        or not Path(decision["drive_root"]).is_absolute()
    ):
        raise ExecutionRefused(
            "amended campaign custody gate: exact campaign and Drive approval missing"
        )
    return ExecutionAuthorization(
        source_revision=decision["source_revision"],
        policy_sha256=report.policy_sha256,
        binding_sha256=report.binding_sha256,
        evidence_sha256=sha256_bytes(approval),
        history_head_sha256=expected_head_sha256,
        storage_root=Path(storage.authorized_root),
        anchor_dir=data_root / evidence.ROOT_ANCHORS,
        storage_authorization=storage,
        custody_mode=custody_mode,
        campaign_id=decision.get("campaign_id") if custody_mode == "amended_drive" else None,
        drive_root=Path(decision["drive_root"]) if custody_mode == "amended_drive" else None,
        pending_head_sha256=pending_head,
    )
