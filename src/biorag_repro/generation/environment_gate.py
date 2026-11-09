"""Environment evidence an official run must carry.

An official run needs an approved environment lock and a verification of it against a live
inventory of the machine that runs the model. The lock is approved only if its SHA-256 is in
`generation.config.APPROVED_ENVIRONMENT_LOCKS`, which is edited by hand outside the lock and
outside any run; a status written inside the lock, or a hash a caller supplies, is not
approval. The lock and its verification report are stored with the run, so the completion
inventory and the retained anchor bind them to it.

What a match shows: the pinned Python, packages, torch build and GPU were present when the
report was made. It does not show numerical determinism of generation. Every running event
of an official run cites the SHA-256 of the stored verification, so the verification was in
the run directory before the first attempt event could be written and cannot be replaced
afterwards. That is integrity and recorded-event ordering: the event timestamps are written
by the producing process and are not an independent clock, and a producer could still store a
fabricated report that matches an approved lock.
"""

from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.attempts import (
    EVIDENCE_DIR,
    LOCK_FILE,
    VERIFICATION_FILE,
    CorruptRecord,
    LifecycleError,
    NonOfficialBackend,
)
from biorag_repro.generation.completion import CompletedRun
from biorag_repro.generation.environment import EnvironmentRecord
from biorag_repro.generation.environment_lock import (
    EnvironmentLockError,
    EnvironmentSpec,
    EnvironmentVerification,
    normalise,
    parse_spec,
)


class EnvironmentNotApproved(LifecycleError):
    """The environment evidence is not an approved lock matching a live inventory."""


@dataclass(frozen=True)
class EnvironmentEvidence:
    lock_bytes: bytes
    report: EnvironmentVerification


def check_approved(evidence: EnvironmentEvidence) -> None:
    digest = sha256_bytes(evidence.lock_bytes)
    report = evidence.report
    if report.spec_sha256 != digest:
        raise EnvironmentNotApproved("the verification describes another lock")
    if digest not in generator_config.APPROVED_ENVIRONMENT_LOCKS:
        raise EnvironmentNotApproved(f"lock {digest} is not an approved lock")
    try:
        status = parse_spec(evidence.lock_bytes).status
    except EnvironmentLockError as error:
        raise EnvironmentNotApproved(f"the approved file is unusable: {error}") from error
    if status != "approved_lock":
        raise EnvironmentNotApproved("the approved file is not an approved_lock specification")
    if report.outcome != "matches_approved_lock":
        raise EnvironmentNotApproved(f"outcome is {report.outcome}, not matches_approved_lock")
    if report.inventory_source != "live":
        raise EnvironmentNotApproved("the inventory was not captured live on the execution machine")
    if not report.checks or "spec_sha256" not in {c.name for c in report.checks}:
        raise EnvironmentNotApproved("the verification lists no checks of the lock hash")
    # The outcome is stored data; the checks it was derived from must agree with it.
    for check in report.checks:
        if check.status != "pass":
            raise EnvironmentNotApproved(f"check {check.name} is {check.status}")


def publish_environment_evidence(run_dir: Path, evidence: EnvironmentEvidence) -> None:
    """Store the evidence with the run. The same evidence may be stored again."""
    files = {
        LOCK_FILE: evidence.lock_bytes,
        VERIFICATION_FILE: serialize_project_json(evidence.report, EnvironmentVerification),
    }
    (run_dir / EVIDENCE_DIR).mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        target = run_dir / name
        try:
            publish_bytes(target, content, sha256_bytes(content))
        except FileExistsError:
            # A resumed run keeps its first verification (a new capture differs in its
            # hashes); the lock itself must be the same file.
            if name == LOCK_FILE and target.read_bytes() != content:
                raise CorruptRecord("a different environment lock is already stored") from None


def _disagreements(spec: EnvironmentSpec, record: EnvironmentRecord) -> list[str]:
    """Where an attempt's own environment record departs from the approved lock."""
    problems = []
    if (record.python_version, record.python_implementation) != (
        spec.python,
        spec.python_implementation,
    ):
        problems.append(f"python {record.python_implementation} {record.python_version}")
    if (record.operating_system, record.machine) != (spec.system, spec.machine):
        problems.append(f"platform {record.operating_system} {record.machine}")
    pins = {normalise(name): version for name, version in spec.packages.items()}
    for name, version in record.packages.items():
        if pins.get(normalise(name)) != version:
            problems.append(f"package {name} {version}")
    if record.cuda_runtime != spec.torch_cuda:
        problems.append(f"cuda runtime {record.cuda_runtime}")
    if not record.gpus:
        problems.append("gpu none recorded")
    for gpu in record.gpus:
        if spec.gpu.name_contains not in gpu.name:
            problems.append(f"gpu {gpu.name}")
        if gpu.driver_version != spec.gpu.nvidia_driver:
            problems.append(f"driver {gpu.driver_version}")
    execution = record.execution
    if execution is None:
        problems.append("device none recorded")
    elif execution.device_name is None or spec.gpu.name_contains not in execution.device_name:
        problems.append(f"device {execution.device_name}")
    return problems


def read_environment_evidence(completed: CompletedRun) -> EnvironmentEvidence:
    """The evidence stored with a run whose records already agree with the retained anchor."""
    files = completed.completion.files
    if LOCK_FILE not in files or VERIFICATION_FILE not in files:
        raise NonOfficialBackend("no environment evidence is stored with the run")
    run_dir = completed.store.run_dir
    lock = (run_dir / LOCK_FILE).read_bytes()
    if sha256_bytes(lock) != files[LOCK_FILE]:
        raise CorruptRecord("the stored environment lock differs from the completion record")
    report = read_project_json(
        run_dir / VERIFICATION_FILE, files[VERIFICATION_FILE], EnvironmentVerification
    )
    return EnvironmentEvidence(lock, report)


def verify_run_environment(
    completed: CompletedRun, environments: list[EnvironmentRecord]
) -> EnvironmentEvidence:
    """Missing, unapproved, non-matching or incompatible evidence is refused."""
    evidence = read_environment_evidence(completed)
    check_approved(evidence)
    spec = parse_spec(evidence.lock_bytes)
    for record in environments:
        if problems := _disagreements(spec, record):
            raise NonOfficialBackend(
                "an attempt's environment differs from the approved lock: " + "; ".join(problems)
            )
    return evidence
