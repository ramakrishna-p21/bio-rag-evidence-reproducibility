"""Descriptive qualification report over the stored records of one decided generation run.

Every figure is read or counted from records that already exist (attempt events, execution
receipts, environment records, the roster and the completion anchor). Nothing is measured
here and nothing is inferred. Each series says whether it was observed, derived from
recorded observations, unavailable, or not applicable, and each rate keeps its denominator.

The report describes how a run behaved. It does not score answers, does not say whether any
answer is correct, and never says that a Phase 7 gate is met. Failed attempts are operational
facts, kept apart from answers that parsed but may be scientifically wrong.

Two durations are kept apart. The attempt duration is the difference between the wall-clock
`at_utc` of an attempt's start and closing events: it includes the cost of writing those
records, is not monotonic, and a system clock adjustment between two events would change it.
An interrupted attempt has no trustworthy end and is not timed. The generation duration is
the backend's monotonic reading around one `model.generate` call, carried by the receipt (see
receipt.ExecutionMeasurements for the exact boundaries). Process memory peaks and the
device-wide snapshot are likewise read from the receipt; a receipt without them gives an
unavailable series, never a zero. All measurements are the executing process's own account.
"""

from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import (
    publish_bytes,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.generation.attempts import (
    AttemptEvent,
    CorruptRecord,
    NonOfficialBackend,
    ObservationState,
)
from biorag_repro.generation.completion import (
    CompletedRun,
    RunCompletion,
    verify_completed_run,
)
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import BackendKind, EnvironmentRecord
from biorag_repro.generation.environment_lock import EnvironmentVerification
from biorag_repro.generation.official import VerifiedOfficialRun

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Provenance = Literal["observed", "derived", "unavailable", "not_applicable"]

# The class of evidence follows from the backend of the run and nothing else.
EVIDENCE_CLASS: dict[str, str] = {
    "mock": "synthetic_engineering",
    "smoke": "smoke_engineering",
    "official": "official_development",
}
_CLOSING = ("complete", "retryable_failure", "terminal_failure")
LIMITATIONS = [
    "Descriptive qualification evidence; it does not evaluate any scientific gate.",
    (
        "Attempt durations are differences between producer-written event timestamps: they "
        "include record writes and are not monotonic. Generation durations are monotonic "
        "readings around one generate call and exclude tokenization, decoding and record writing."
    ),
    (
        "Memory peaks are torch counters of the generating process. Device-wide used memory is a "
        "snapshot that includes other GPU users. A missing reading is reported unavailable, "
        "never inferred."
    ),
    "Measurements are the executing process's own account and are not independently attested.",
    "Answers are not scored here; a parsed answer may be wrong and a failed attempt has no answer.",
]


class Fraction(Record):
    """A count with its denominator; the ratio is left to the reader."""

    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @model_validator(mode="after")
    def within(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("numerator exceeds its denominator")
        return self


class Series(Record):
    """Integer values, one per unit counted, summarised exactly."""

    status: Provenance
    reason: str | None
    n: int = Field(ge=0)
    total: int | None
    minimum: int | None
    maximum: int | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        measured = self.status in ("observed", "derived")
        values = (self.total, self.minimum, self.maximum)
        if measured != (self.n > 0) or measured != all(v is not None for v in values):
            raise ValueError("values are present exactly when the series is measured")
        if not measured and (self.reason is None or any(v is not None for v in values)):
            raise ValueError("an unmeasured series states why and holds no value")
        return self


class Throughput(Record):
    """Output tokens over a duration in nanoseconds, kept as the two integers they come from."""

    status: Provenance
    reason: str | None
    attempts: int = Field(ge=0)
    output_tokens: int | None
    duration_ns: int | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        measured = self.status in ("observed", "derived")
        if measured != (self.attempts > 0) or measured != (self.duration_ns is not None):
            raise ValueError("throughput values are present exactly when it is measured")
        if measured and (self.output_tokens is None or not self.duration_ns):
            raise ValueError("throughput needs tokens and a positive duration")
        if not measured and (self.reason is None or self.output_tokens is not None):
            raise ValueError("an unmeasured throughput states why and holds no value")
        return self


class RunIdentity(Record):
    run_id: str
    run_record_sha256: Sha256
    resolved_config_sha256: Sha256
    raw_config_sha256: Sha256
    roster_sha256: Sha256
    completion_sha256: Sha256
    completion_manifest_sha256: Sha256  # the retained anchor the records were checked against
    matrix_arm: str
    backend_kind: BackendKind
    purpose: str
    phase: Literal["development", "qualification"]


class EnvironmentUse(Record):
    """What an environment record of the run says; the record is cited by hash."""

    environment_sha256: Sha256
    backend_kind: BackendKind
    backend_name: str
    backend_version: str | None
    model_repository: str
    model_revision: str
    checkpoint_manifest_sha256: Sha256 | None
    device: str | None
    device_name: str | None
    dtype: str | None
    gpu_status: Literal["ok", "no_nvidia_smi", "failed"]


class EnvironmentEvidenceRef(Record):
    status: Literal["verified", "not_applicable"]
    verification_sha256: Sha256 | None
    lock_sha256: Sha256 | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.status == "verified") != (
            self.verification_sha256 is not None and self.lock_sha256 is not None
        ):
            raise ValueError("a verified reference names the verification and the lock")
        return self


class ObservationCounts(Record):
    planned: int = Field(ge=0)
    attempted: int = Field(ge=0)
    completed: int = Field(ge=0)
    missing: int = Field(ge=0)  # planned without a completed answer
    terminal_failure: int = Field(ge=0)
    retries_exhausted: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.missing != self.planned - self.completed:
            raise ValueError("missing must equal planned minus completed")
        if self.missing != self.terminal_failure + self.retries_exhausted:
            raise ValueError("a decided run misses an observation only by failure")
        return self


class AttemptCounts(Record):
    total: int = Field(ge=0)
    observations_with_retries: int = Field(ge=0)
    retryable_failure_events: int = Field(ge=0)
    terminal_failure_events: int = Field(ge=0)
    interrupted_events: int = Field(ge=0)
    untimed_attempts: int = Field(ge=0)
    failure_kinds: dict[str, int]


class ParseCounts(Record):
    parsed: Fraction  # of completed observations
    unparseable: Fraction  # of completed observations
    abstained: Fraction  # of parsed outputs; undefined for unparseable ones


class QualificationReport(Record):
    schema_version: Literal["2"] = "2"
    report_kind: Literal["descriptive_qualification_evidence"] = (
        "descriptive_qualification_evidence"
    )
    evidence_class: Literal["synthetic_engineering", "smoke_engineering", "official_development"]
    scientific_gate: Literal["not_evaluated"] = "not_evaluated"
    run: RunIdentity
    environments: list[EnvironmentUse]
    environment_verification: EnvironmentEvidenceRef
    observations: ObservationCounts
    attempts: AttemptCounts
    parse: ParseCounts
    receipts: Fraction  # completed observations that carry an execution receipt
    planned_input_tokens: Series
    input_tokens_received: Series
    output_tokens: Series
    stop_reasons: dict[str, int]  # over observations with a receipt
    attempt_duration_us: Series  # wall-clock event differences
    attempt_throughput: Throughput  # output tokens over attempt durations
    measurements: Fraction  # receipts that carry execution measurements
    memory_readings: Fraction  # receipts whose measurements include a CUDA memory reading
    memory_device_indices: list[int]  # devices the memory readings came from
    generation_duration_ns: Series  # monotonic, around one generate call
    generation_throughput: Throughput  # output tokens over generation durations
    peak_allocated_bytes: Series  # process counter
    peak_reserved_bytes: Series  # process counter
    device_used_after_bytes: Series  # device-wide snapshot, includes other GPU users
    limitations: list[str]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        kind = self.run.backend_kind
        if self.evidence_class != EVIDENCE_CLASS[kind]:
            raise ValueError(f"a {kind} run cannot yield {self.evidence_class} evidence")
        if (self.environment_verification.status == "verified") != (kind == "official"):
            raise ValueError("only an official run cites a verified environment")
        if self.limitations != LIMITATIONS:
            raise ValueError("limitations are fixed text")
        return self


def report_sha256(report: QualificationReport) -> str:
    return sha256_bytes(serialize_project_json(report, QualificationReport))


def publish_report(path: Path, report: QualificationReport) -> str:
    content = serialize_project_json(report, QualificationReport)
    path.parent.mkdir(parents=True, exist_ok=True)
    return publish_bytes(path, content, sha256_bytes(content))


def read_report(path: Path, expected_sha256: str) -> QualificationReport:
    return read_project_json(path, expected_sha256, QualificationReport)


def _series(
    values: list[int],
    measured: Provenance,
    empty: Provenance,
    why_not: str,
    reason: str | None,
) -> Series:
    """`measured` labels a series that has values; `empty` labels one that has none."""
    if not values:
        return Series(status=empty, reason=why_not, n=0, total=None, minimum=None, maximum=None)
    return Series(
        status=measured,
        reason=reason,
        n=len(values),
        total=sum(values),
        minimum=min(values),
        maximum=max(values),
    )


def _attempts(events: list[AttemptEvent]) -> list[tuple[AttemptEvent, AttemptEvent]]:
    """(start, closing) pairs. The state machine guarantees a closing event follows a start."""
    pairs, start = [], None
    for event in events:
        if event.state == "running":
            start = event
        elif event.state in _CLOSING or event.state == "interrupted":
            if start is None:
                raise CorruptRecord("an attempt closes without having started")
            pairs.append((start, event))
            start = None
    return pairs


def _microseconds(start: AttemptEvent, end: AttemptEvent) -> int:
    delta = datetime.fromisoformat(end.at_utc) - datetime.fromisoformat(start.at_utc)
    return delta // timedelta(microseconds=1)


def _throughput(
    pairs: list[tuple[int, int]], unmeasured: Provenance, why_none: str, why_zero: str
) -> Throughput:
    """Pairs of (output tokens, nanoseconds); only the sums are kept."""
    tokens, nanoseconds = sum(t for t, _ in pairs), sum(n for _, n in pairs)
    if pairs and nanoseconds > 0:
        return Throughput(
            status="derived",
            reason=None,
            attempts=len(pairs),
            output_tokens=tokens,
            duration_ns=nanoseconds,
        )
    return Throughput(
        status=unmeasured,
        reason=why_none if not pairs else why_zero,
        attempts=0,
        output_tokens=None,
        duration_ns=None,
    )


def _environments(completed: CompletedRun) -> list[EnvironmentUse]:
    uses = []
    for digest in completed.completion.environments:
        record = read_project_json(
            completed.store.run_dir / "environments" / f"{digest}.json", digest, EnvironmentRecord
        )
        run = record.execution
        uses.append(
            EnvironmentUse(
                environment_sha256=digest,
                backend_kind=record.backend.kind,
                backend_name=record.backend.name,
                backend_version=record.backend.version,
                model_repository=record.model_repository,
                model_revision=record.model_revision,
                checkpoint_manifest_sha256=run.snapshot.manifest_sha256 if run else None,
                device=run.device if run else None,
                device_name=run.device_name if run else None,
                dtype=run.dtype if run else None,
                gpu_status=record.gpu_status,
            )
        )
    return sorted(uses, key=lambda u: u.environment_sha256)


def _report(completed: CompletedRun, environment: EnvironmentEvidenceRef) -> QualificationReport:
    record = completed.store.record
    states: list[ObservationState] = completed.states
    mock = record.backend_kind == "mock"
    unmeasured: Provenance = "not_applicable" if mock else "unavailable"
    no_model = "a mock backend runs no model" if mock else "no execution receipt was recorded"

    pairs = [(s, p) for s in states for p in _attempts(s.events)]
    closing = Counter(close.state for _, (_, close) in pairs)
    kinds = Counter(
        close.failure_kind
        for _, (_, close) in pairs
        if close.state in ("retryable_failure", "terminal_failure") and close.failure_kind
    )
    timed = [
        (s, start, close)
        for s, (start, close) in pairs
        if close.state != "interrupted"  # its end was written at recovery, not at the stop
    ]

    finals = [s for s in states if s.status == "complete"]
    receipts = {s.key: completed.store.receipt_of(s) for s in finals}
    held = {key: r for key, r in receipts.items() if r is not None}
    durations = [_microseconds(start, close) for _, start, close in timed]
    throughput_pairs = [
        (len(held[s.key].facts.output_token_ids), _microseconds(start, close))
        for s, start, close in timed
        if close.state == "complete" and s.key in held
    ]
    attempt_throughput = _throughput(
        [(tokens, micros * 1000) for tokens, micros in throughput_pairs],
        unmeasured,
        no_model,
        "the timed attempts took no measurable time",
    )

    measured = {key: r.facts.measurements for key, r in held.items() if r.facts.measurements}
    no_measure = (
        "a mock backend runs no model" if mock else "no execution receipt carries measurements"
    )
    generation_throughput = _throughput(
        [(len(held[key].facts.output_token_ids), m.generation_ns) for key, m in measured.items()],
        unmeasured,
        no_measure,
        "the measured generations took no measurable time",
    )
    readings = [m.cuda_memory for m in measured.values()]
    taken = [r for r in readings if r.status == "measured"]
    memory_why = no_measure
    if measured and not taken:
        why = sorted({r.reason for r in readings if r.status == "unavailable"})
        memory_why = "the backend took no CUDA memory reading: " + "; ".join(why)

    def memory(values: list[int]) -> Series:
        return _series(values, "observed", unmeasured, memory_why, None)

    parsed = sum(1 for s in finals if s.events[-1].parse_status == "parsed")
    abstained = sum(1 for s in finals if s.events[-1].abstained)
    statuses = Counter(s.status for s in states)
    completion = completed.completion
    run = RunIdentity(
        run_id=completed.run_id,
        run_record_sha256=completion.run_record_sha256,
        resolved_config_sha256=completion.resolved_config_sha256,
        raw_config_sha256=completion.raw_config_sha256,
        roster_sha256=completion.roster_sha256,
        completion_sha256=sha256_bytes(serialize_project_json(completion, RunCompletion)),
        completion_manifest_sha256=completed.manifest_sha256,
        matrix_arm=completion.matrix_arm,
        backend_kind=record.backend_kind,
        purpose=completion.purpose,
        phase=record.phase,
    )
    return QualificationReport(
        evidence_class=EVIDENCE_CLASS[record.backend_kind],  # type: ignore[arg-type]
        run=run,
        environments=_environments(completed),
        environment_verification=environment,
        observations=ObservationCounts(
            planned=len(completed.roster.rows),
            attempted=sum(1 for s in states if any(e.state == "running" for e in s.events)),
            completed=len(finals),
            missing=len(completed.roster.rows) - len(finals),
            terminal_failure=statuses["terminal_failure"],
            retries_exhausted=statuses["retries_exhausted"],
        ),
        attempts=AttemptCounts(
            total=len(pairs),
            observations_with_retries=sum(1 for s in states if s.attempts > 1),
            retryable_failure_events=closing["retryable_failure"],
            terminal_failure_events=closing["terminal_failure"],
            interrupted_events=closing["interrupted"],
            untimed_attempts=closing["interrupted"],
            failure_kinds=dict(sorted(kinds.items())),
        ),
        parse=ParseCounts(
            parsed=Fraction(numerator=parsed, denominator=len(finals)),
            unparseable=Fraction(numerator=len(finals) - parsed, denominator=len(finals)),
            abstained=Fraction(numerator=abstained, denominator=parsed),
        ),
        receipts=Fraction(numerator=len(held), denominator=len(finals)),
        planned_input_tokens=_series(
            [row.input_tokens for row in completed.roster.rows],
            "derived",
            "unavailable",
            "the roster lists no rows",
            "token counts the roster recorded when each input was rendered",
        ),
        input_tokens_received=_series(
            [r.facts.input_token_count for r in held.values()],
            "observed",
            unmeasured,
            no_model,
            None,
        ),
        output_tokens=_series(
            [len(r.facts.output_token_ids) for r in held.values()],
            "observed",
            unmeasured,
            no_model,
            None,
        ),
        stop_reasons=dict(sorted(Counter(r.facts.stop_reason for r in held.values()).items())),
        attempt_duration_us=_series(
            durations,
            "derived",
            "unavailable",
            "no attempt has a start and a trustworthy end",
            "wall-clock difference between event timestamps; includes record writes",
        ),
        attempt_throughput=attempt_throughput,
        measurements=Fraction(numerator=len(measured), denominator=len(held)),
        memory_readings=Fraction(numerator=len(taken), denominator=len(held)),
        memory_device_indices=sorted({r.device_index for r in taken}),
        generation_duration_ns=_series(
            [m.generation_ns for m in measured.values()],
            "observed",
            unmeasured,
            no_measure,
            None,
        ),
        generation_throughput=generation_throughput,
        peak_allocated_bytes=memory([r.peak_allocated_bytes for r in taken]),
        peak_reserved_bytes=memory([r.peak_reserved_bytes for r in taken]),
        device_used_after_bytes=memory([r.device_used_after_bytes for r in taken]),
        limitations=list(LIMITATIONS),
    )


def qualification_report(
    root: Path,
    run_id: str,
    manifest_sha256: str,
    *,
    config: ResolvedConfiguration[GeneratorConfiguration],
    backend_kind: BackendKind,
) -> QualificationReport:
    """Report on a mock or smoke run, after re-verifying it against its retained anchor.

    A run on the official backend is refused: its report comes from a verified official run.
    """
    completed = verify_completed_run(
        root, run_id, manifest_sha256, config=config, backend_kind=backend_kind
    )
    if completed.store.record.official:
        raise NonOfficialBackend(
            "an official run is reported through generation.official verification"
        )
    return _report(
        completed,
        EnvironmentEvidenceRef(status="not_applicable", verification_sha256=None, lock_sha256=None),
    )


def official_qualification_report(verified: VerifiedOfficialRun) -> QualificationReport:
    """Report on an official run that passed every official check, including its environment."""
    evidence = verified.environment
    return _report(
        verified.completed,
        EnvironmentEvidenceRef(
            status="verified",
            verification_sha256=sha256_bytes(
                serialize_project_json(evidence.report, EnvironmentVerification)
            ),
            lock_sha256=sha256_bytes(evidence.lock_bytes),
        ),
    )
