"""Run and attempt records for generation, with restart and retry.

One run directory holds, for each planned observation, an append-only log of events
(planned, running, complete, interrupted, retryable_failure, terminal_failure). Each
event is a separate file published without replacing anything, and each cites the hash of
the previous event, so a missing, edited or reordered record is found on read-back. Raw
output, the parsed fields and the attempt status are kept apart: the parsed fields are
recomputed from the stored raw text whenever a record is read.

Nothing here runs a model. Mock and smoke runs are valid records of themselves, but only
a run on the official backend with a development-candidate configuration can yield
official development observations; that is checked by validation, not by naming.
"""

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes, validate_sha256
from biorag_repro.core.identity import (
    ArtifactInput,
    derived_artifact_id,
    new_attempt_id,
    new_run_id,
    observation_key,
)
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import (
    BackendKind,
    EnvironmentRecord,
    check_utc,
    environment_sha256,
    utc_now,
)
from biorag_repro.generation.inputs import (
    CorruptInput,
    GeneratorInputRecord,
    publish_input_record,
    read_input_record,
)
from biorag_repro.generation.parsing import parse_output
from biorag_repro.generation.receipt import ExecutionFacts, ExecutionReceipt, token_ids_sha256
from biorag_repro.generation.seeds import MAX_SEED

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
OpaqueId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
State = Literal[
    "planned", "running", "complete", "interrupted", "retryable_failure", "terminal_failure"
]

Status = Literal[
    "planned", "running", "complete", "retryable", "retries_exhausted", "terminal_failure"
]

_FINAL = ("complete", "terminal_failure", "retries_exhausted")

_NEXT: dict[str | None, set[str]] = {
    None: {"planned"},
    "planned": {"running"},
    "running": {"complete", "interrupted", "retryable_failure", "terminal_failure"},
    "interrupted": {"running"},
    "retryable_failure": {"running"},
    "complete": set(),
    "terminal_failure": set(),
}


class LifecycleError(ValueError):
    """Base class for refusals and detected problems."""


class InvalidTransition(LifecycleError): ...


class PublicationConflict(LifecycleError): ...


class CorruptRecord(LifecycleError): ...


class DuplicateCompletion(LifecycleError): ...


class UnsafeResume(LifecycleError): ...


class RunMismatch(LifecycleError): ...


class NonOfficialBackend(LifecycleError): ...


class NotPlanned(LifecycleError): ...


class InvalidReceipt(LifecycleError):
    """An execution receipt disagrees with the attempt, the input record or the configuration."""


class RetryPolicy(Record):
    """The retry rule as recorded on a run. Its source is the hashed generator configuration."""

    max_attempts: int = Field(ge=1)
    retryable_failures: list[Nonblank]

    @model_validator(mode="after")
    def distinct(self) -> Self:
        if len(set(self.retryable_failures)) != len(self.retryable_failures):
            raise ValueError("duplicate retryable failure kind")
        return self


def retry_policy_of(settings: GeneratorConfiguration) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=settings.retry_max_attempts, retryable_failures=settings.retryable_failures
    )


def _check_backend_purpose(kind: str, purpose: str) -> None:
    if kind == "official" and purpose != "development_candidate":
        raise NonOfficialBackend(
            "backend 'official' requires a development-candidate configuration"
        )
    if kind == "mock" and purpose != "synthetic_test":
        raise NonOfficialBackend(
            "backend 'mock' cannot carry a development-candidate configuration"
        )


class RunRecord(Record):
    schema_version: Literal["1"] = "1"
    run_id: OpaqueId
    created_utc: str
    phase: Literal["development"]  # qualification and final_test are not authorized here
    benchmark_version: Nonblank
    resolved_config_sha256: Sha256
    raw_config_sha256: Sha256
    purpose: Literal["synthetic_test", "development_candidate"]
    matrix_arm: Nonblank
    backend_kind: BackendKind
    retry: RetryPolicy

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def backend_matches_purpose(self) -> Self:
        try:
            _check_backend_purpose(self.backend_kind, self.purpose)
        except NonOfficialBackend as error:
            raise ValueError(str(error)) from error
        return self

    @property
    def official(self) -> bool:
        return self.backend_kind == "official" and self.purpose == "development_candidate"


class Coordinates(Record):
    item_id: Nonblank
    paper_id: Nonblank
    matrix_arm: Nonblank
    representation_or_condition: Nonblank
    repeat_index: int = Field(ge=0)
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)


class AttemptEvent(Record):
    schema_version: Literal["1"] = "1"
    run_id: OpaqueId
    run_record_sha256: Sha256
    observation_key: Annotated[str, StringConstraints(pattern=r"^obs_[0-9a-f]{64}$")]
    sequence: int = Field(ge=1)
    previous_event_sha256: Sha256 | None
    state: State
    at_utc: str
    coordinates: Coordinates | None = None
    attempt_id: OpaqueId | None = None
    attempt_number: int | None = Field(default=None, ge=1)
    backend_kind: BackendKind | None = None
    environment_sha256: Sha256 | None = None
    rendered_input_sha256: Sha256 | None = None
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    raw_output: ArtifactRef | None = None
    parse_status: Literal["parsed", "unparseable"] | None = None
    answer: str | list[str] | None = None
    abstained: bool | None = None
    parse_reason: str | None = None
    execution_receipt_sha256: Sha256 | None = None
    failure_kind: Nonblank | None = None
    failure_detail: Nonblank | None = None

    _utc = field_validator("at_utc")(check_utc)

    @model_validator(mode="after")
    def fields_match_state(self) -> Self:
        required = {
            "planned": {"coordinates"},
            "running": {"backend_kind", "environment_sha256", "rendered_input_sha256"},
            "complete": {"raw_output", "parse_status"},
            "retryable_failure": {"failure_kind", "failure_detail"},
            "terminal_failure": {"failure_kind", "failure_detail"},
            "interrupted": set(),
        }[self.state]
        stateful = {
            "coordinates", "backend_kind", "environment_sha256", "rendered_input_sha256",
            "raw_output", "parse_status", "failure_kind", "failure_detail",
        }  # fmt: skip
        for name in stateful:
            if (getattr(self, name) is not None) != (name in required):
                raise ValueError(
                    f"{self.state} event: {name} is {'missing' if name in required else 'not allowed'}"
                )
        if self.state == "planned":
            if self.sequence != 1 or self.previous_event_sha256 is not None:
                raise ValueError("the planned event starts the log")
            if any(v is not None for v in (self.attempt_id, self.attempt_number)):
                raise ValueError("a planned event belongs to no attempt")
            return self
        if self.sequence < 2 or self.previous_event_sha256 is None:
            raise ValueError("only the planned event may be first")
        if self.attempt_id is None or self.attempt_number is None:
            raise ValueError(f"{self.state} event needs its attempt")
        if self.state != "running" and self.seed is not None:
            raise ValueError("the seed is recorded when the attempt starts")
        if self.execution_receipt_sha256 is not None and self.state != "complete":
            raise ValueError("only a complete attempt cites an execution receipt")
        parsed_fields = (self.answer, self.abstained, self.parse_reason)
        if self.state != "complete":
            if any(v is not None for v in parsed_fields):
                raise ValueError("parsed fields belong to a complete attempt")
        elif self.parse_status == "unparseable":
            if self.answer is not None or self.abstained is not None or not self.parse_reason:
                raise ValueError("an unparseable output carries a reason and no parsed fields")
        elif self.abstained is None or self.parse_reason is not None:
            raise ValueError("a parsed output needs an abstention indicator and no reason")
        return self


@dataclass(frozen=True)
class ObservationState:
    key: str
    coordinates: Coordinates
    events: list[AttemptEvent]
    last_event_sha256: str
    status: Status
    attempts: int
    first_event_sha256: str


def _load_json(path: Path, model: Any, what: str) -> tuple[Any, str]:
    content = path.read_bytes()
    try:
        value = model.model_validate(json.loads(content.decode("utf-8")), strict=True)
    except ValueError as error:  # includes JSON, Unicode and validation errors
        raise CorruptRecord(f"unreadable {what} {path.name}: {error}") from error
    return value, sha256_bytes(content)


class RunStore:
    """Append-only records of one execution batch under `root/runs/<run_id>`."""

    def __init__(
        self,
        root: Path,
        record: RunRecord,
        run_sha256: str,
        config: ResolvedConfiguration[GeneratorConfiguration],
        now: Callable[[], str],
    ) -> None:
        self.root = root
        self.record = record
        self.run_sha256 = run_sha256
        self.config = config
        self.now = now

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.record.run_id

    @property
    def retry_policy(self) -> RetryPolicy:
        return self.record.retry

    # --- creation and resume ---------------------------------------------------------

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        config: ResolvedConfiguration[GeneratorConfiguration],
        phase: Literal["development"],
        benchmark_version: str,
        backend_kind: BackendKind,
        retry_policy: RetryPolicy | None = None,
        now: Callable[[], str] = utc_now,
    ) -> Self:
        settings = config.settings
        _check_backend_purpose(backend_kind, settings.purpose)
        derived = retry_policy_of(settings)
        if retry_policy is not None and retry_policy != derived:
            raise RunMismatch("retry policy contradicts the retry settings of the configuration")
        record = RunRecord(
            run_id=new_run_id(),
            created_utc=now(),
            phase=phase,
            benchmark_version=benchmark_version,
            resolved_config_sha256=config.configuration_sha256,
            raw_config_sha256=config.raw_toml_sha256,
            purpose=settings.purpose,
            matrix_arm=settings.matrix_arm,
            backend_kind=backend_kind,
            retry=derived,
        )
        run_dir = root / "runs" / record.run_id
        run_dir.mkdir(parents=True)
        publish_bytes(
            run_dir / "config.canonical.json", config.canonical_json, config.configuration_sha256
        )
        publish_bytes(run_dir / "config.toml", config.raw_toml, config.raw_toml_sha256)
        content = serialize_project_json(record, RunRecord)
        digest = sha256_bytes(content)
        publish_bytes(
            run_dir / "run.json",
            content,
            digest,
            validate_temp=lambda path: read_project_json(path, digest, RunRecord),
        )
        return cls(root, record, digest, config, now)

    @classmethod
    def open(
        cls,
        root: Path,
        run_id: str,
        *,
        config: ResolvedConfiguration[GeneratorConfiguration],
        backend_kind: BackendKind,
        now: Callable[[], str] = utc_now,
    ) -> Self:
        """Resume a run under its original identity after validating every stored record."""
        run_path = root / "runs" / run_id / "run.json"
        if not run_path.is_file():
            raise LifecycleError(f"no such run {run_id}")
        record, digest = _load_json(run_path, RunRecord, "run record")
        if record.run_id != run_id:
            raise CorruptRecord("run record names a different run")
        if record.resolved_config_sha256 != config.configuration_sha256:
            raise RunMismatch("resolved configuration differs from the run's configuration")
        if record.backend_kind != backend_kind:
            raise RunMismatch(
                f"backend is {backend_kind}, the run was created on {record.backend_kind}"
            )
        if (record.purpose, record.matrix_arm) != (
            config.settings.purpose,
            config.settings.matrix_arm,
        ):
            raise CorruptRecord("run record disagrees with its configuration")
        if record.retry != retry_policy_of(config.settings):
            raise RunMismatch(
                "run retry policy contradicts the retry settings of the configuration"
            )
        store = cls(root, record, digest, config, now)
        for name, expected in (
            ("config.canonical.json", record.resolved_config_sha256),
            ("config.toml", record.raw_config_sha256),
        ):
            path = store.run_dir / name
            if not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
                raise CorruptRecord(
                    f"stored configuration file {name} does not match the run record"
                )
        for state in store.verify():
            if state.status in _FINAL and not store._seal_path(state.key).exists():
                store._seal(state)  # completes a publication interrupted after the final event
        return store

    # --- reading ------------------------------------------------------------------------

    def observation_keys(self) -> list[str]:
        base = self.run_dir / "observations"
        if not base.is_dir():
            return []
        # Any event file makes it an observation: a lost first event must surface as corruption
        # when the log is read, not hide the observation.
        return sorted(p.name for p in base.iterdir() if any((p / "events").glob("0*.json")))

    def _load(self, key: str) -> ObservationState:
        events_dir = self.run_dir / "observations" / key / "events"
        if not events_dir.is_dir() or not any(events_dir.glob("0*.json")):
            raise NotPlanned(f"observation {key} is not planned")
        names = sorted(p.name for p in events_dir.iterdir() if not p.name.startswith("."))
        for name in names:
            if re.fullmatch(r"\d{6}\.json", name) is None:
                raise CorruptRecord(f"unexpected file {name} among events of {key}")
        if names != [f"{i:06d}.json" for i in range(1, len(names) + 1)]:
            raise CorruptRecord(f"event sequence of {key} has a gap or does not start at 1")

        events: list[AttemptEvent] = []
        previous_sha: str | None = None
        first_sha: str | None = None
        for name in names:
            event, digest = _load_json(events_dir / name, AttemptEvent, "event")
            self._check_event(key, event, events, previous_sha)
            events.append(event)
            previous_sha = digest
            first_sha = first_sha or digest
        assert previous_sha is not None and first_sha is not None
        state = self._state(key, events, previous_sha, first_sha)
        self._check_seal(state)
        return state

    def _check_event(
        self, key: str, event: AttemptEvent, before: list[AttemptEvent], previous_sha: str | None
    ) -> None:
        last = before[-1] if before else None
        if event.sequence != len(before) + 1:
            raise CorruptRecord(f"event sequence of {key} is inconsistent")
        if event.run_id != self.run_id or event.run_record_sha256 != self.run_sha256:
            raise CorruptRecord(
                f"event {event.sequence} of {key} does not belong to this run record"
            )
        if event.observation_key != key:
            raise CorruptRecord(
                f"event {event.sequence} belongs to {event.observation_key}, not {key}"
            )
        if event.previous_event_sha256 != previous_sha:
            raise CorruptRecord(f"hash chain of {key} is broken at event {event.sequence}")
        if last is not None and datetime.fromisoformat(event.at_utc) < datetime.fromisoformat(
            last.at_utc
        ):
            raise CorruptRecord(f"event {event.sequence} of {key} precedes the previous event")
        if last is not None and last.state == "complete" and event.state == "complete":
            raise DuplicateCompletion(f"observation {key} has more than one completion")
        if event.state not in _NEXT[last.state if last else None]:
            raise CorruptRecord(
                f"illegal transition {last.state if last else None} -> {event.state} in {key}"
            )
        try:
            self._check_content(key, event, before)
        except LifecycleError as error:
            raise CorruptRecord(str(error)) from error

    def _check_content(self, key: str, event: AttemptEvent, before: list[AttemptEvent]) -> None:
        policy = self.retry_policy
        if event.state == "planned":
            assert event.coordinates is not None
            self._check_coordinates(event.coordinates)
            if self.key_for(event.coordinates) != key:
                raise LifecycleError(f"coordinates do not give observation key {key}")
            return
        started = [e for e in before if e.state == "running"]
        planned = before[0].coordinates
        assert planned is not None
        if event.state == "running":
            assert event.environment_sha256 and event.rendered_input_sha256
            if event.attempt_number != len(started) + 1 or event.attempt_id in {
                e.attempt_id for e in started
            }:
                raise LifecycleError(f"attempt numbering of {key} is inconsistent")
            if event.attempt_number > policy.max_attempts:
                raise LifecycleError(f"{key} exceeds the retry limit of {policy.max_attempts}")
            if event.backend_kind != self.record.backend_kind:
                raise LifecycleError(f"attempt backend differs from the run backend in {key}")
            if started and event.rendered_input_sha256 != started[0].rendered_input_sha256:
                raise LifecycleError(f"retry of {key} does not use the identical rendered input")
            if event.seed != planned.seed:
                raise LifecycleError(f"attempt seed differs from the planned seed in {key}")
            self._check_environment_file(event.environment_sha256)
            return
        current = started[-1]
        if (event.attempt_id, event.attempt_number) != (current.attempt_id, current.attempt_number):
            raise LifecycleError(f"event closes a different attempt than the running one in {key}")
        if (
            event.state == "retryable_failure"
            and event.failure_kind not in policy.retryable_failures
        ):
            raise LifecycleError(f"failure kind {event.failure_kind!r} is not retryable")
        if event.state == "complete":
            self._check_completion(key, event, current)

    def _check_completion(self, key: str, event: AttemptEvent, current: AttemptEvent) -> None:
        ref = event.raw_output
        assert ref is not None and current.rendered_input_sha256 is not None
        path = self.root / ref.logical_ref
        if not path.is_file() or sha256_bytes(path.read_bytes()) != ref.sha256:
            raise LifecycleError(f"raw output of attempt {event.attempt_id} is missing or altered")
        raw = path.read_bytes().decode("utf-8")
        if ref != self._raw_ref(
            key, event.attempt_id or "", current.rendered_input_sha256, ref.sha256
        ):
            raise LifecycleError("raw output reference does not match its identity")
        parsed = parse_output(raw, self.config.settings.parser_version)
        if (parsed.parse_status, parsed.answer, parsed.abstained, parsed.reason) != (
            event.parse_status,
            event.answer,
            event.abstained,
            event.parse_reason,
        ):
            raise LifecycleError(f"parsed fields of {key} differ from its raw output")
        if event.execution_receipt_sha256 is not None:
            self._check_stored_receipt(key, event, current)

    def _receipt_path(self, key: str, attempt_id: str) -> Path:
        return self.run_dir / "observations" / key / "receipts" / f"{attempt_id}.json"

    def _check_stored_receipt(self, key: str, event: AttemptEvent, current: AttemptEvent) -> None:
        assert event.attempt_id and event.execution_receipt_sha256
        path = self._receipt_path(key, event.attempt_id)
        try:
            receipt = read_project_json(path, event.execution_receipt_sha256, ExecutionReceipt)
        except (OSError, ValueError) as error:
            raise LifecycleError(
                f"execution receipt of {key} is missing or altered: {error}"
            ) from error
        self._check_receipt(key, receipt, current)

    def _check_receipt(self, key: str, receipt: ExecutionReceipt, running: AttemptEvent) -> None:
        """Everything a receipt asserts must agree with records that exist independently of it."""
        settings = self.config.settings
        if (receipt.run_id, receipt.observation_key, receipt.attempt_id) != (
            self.run_id,
            key,
            running.attempt_id,
        ):
            raise InvalidReceipt("receipt belongs to another run, observation or attempt")
        if (receipt.environment_sha256, receipt.rendered_input_sha256) != (
            running.environment_sha256,
            running.rendered_input_sha256,
        ):
            raise InvalidReceipt("receipt names another environment or rendered input")
        try:
            record = read_input_record(self.run_dir, receipt.input_record_sha256)
        except CorruptInput as error:
            raise InvalidReceipt(f"input record cannot support the receipt: {error}") from error
        rendered = record.rendered
        if record.observation_key != key:
            raise InvalidReceipt("input record belongs to another observation")
        if rendered.rendered_sha256 != running.rendered_input_sha256:
            raise InvalidReceipt(
                "input record holds other rendered text than the attempt started with"
            )
        if rendered.resolved_config_sha256 != self.record.resolved_config_sha256:
            raise InvalidReceipt("input record was rendered under another configuration")
        if rendered.prompt_sha256 != settings.prompt_sha256:
            raise InvalidReceipt("input record used another prompt than the configuration")
        if record.tokenizer != settings.tokenizer:
            raise InvalidReceipt("input record used another tokenizer than the configuration")
        facts = receipt.facts
        if (facts.input_token_ids_sha256, facts.input_token_count) != (
            rendered.input_token_ids_sha256,
            rendered.input_tokens,
        ):
            raise InvalidReceipt("token ids passed to the model differ from the input record")
        g, decoding = facts.generation, settings.decoding
        if (g.max_new_tokens, g.do_sample, g.temperature, g.top_p, g.top_k) != (
            settings.budgets.max_new_tokens,
            decoding.do_sample,
            decoding.temperature,
            decoding.top_p,
            decoding.top_k,
        ):
            raise InvalidReceipt("generation settings in the receipt differ from the configuration")
        if g.seed != running.seed:
            raise InvalidReceipt("seed in the receipt differs from the planned seed")

    def _check_environment_file(self, digest: str) -> EnvironmentRecord:
        path = self.run_dir / "environments" / f"{digest}.json"
        if not path.is_file():
            raise LifecycleError(f"environment record {digest} is missing")
        content = path.read_bytes()
        try:
            record = EnvironmentRecord.model_validate(json.loads(content), strict=True)
        except ValueError as error:
            raise LifecycleError(f"environment record {digest} is unreadable") from error
        if sha256_bytes(content) != digest:
            raise LifecycleError(f"environment record {digest} was altered")
        self._check_environment(record)
        return record

    def _check_environment(self, environment: EnvironmentRecord) -> None:
        if environment.backend.kind != self.record.backend_kind:
            raise NonOfficialBackend(
                f"environment backend {environment.backend.kind} differs from run backend "
                f"{self.record.backend_kind}"
            )
        if self.record.official:
            settings = self.config.settings
            if (environment.model_repository, environment.model_revision) != (
                settings.model.repository,
                settings.model.revision,
            ) or (environment.tokenizer_repository, environment.tokenizer_revision) != (
                settings.tokenizer.repository,
                settings.tokenizer.revision,
            ):
                raise NonOfficialBackend(
                    "environment model or tokenizer differs from the configuration"
                )

    def _check_coordinates(self, coordinates: Coordinates) -> None:
        settings = self.config.settings
        if coordinates.matrix_arm != settings.matrix_arm:
            raise LifecycleError("matrix arm differs from the configuration")
        if settings.decoding.do_sample != (coordinates.seed is not None):
            raise LifecycleError("a seed is recorded exactly when decoding samples")

    def key_for(self, coordinates: Coordinates) -> str:
        return observation_key(
            phase=self.record.phase,
            benchmark_version=self.record.benchmark_version,
            item_id=coordinates.item_id,
            matrix_arm=coordinates.matrix_arm,
            representation_or_condition=coordinates.representation_or_condition,
            repeat_index=coordinates.repeat_index,
            resolved_config_sha256=self.record.resolved_config_sha256,
        )

    def _state(
        self, key: str, events: list[AttemptEvent], last_sha: str, first_sha: str
    ) -> ObservationState:
        attempts = sum(e.state == "running" for e in events)
        last = events[-1].state
        status: Status
        if last in ("retryable_failure", "interrupted"):
            status = (
                "retryable" if attempts < self.retry_policy.max_attempts else "retries_exhausted"
            )
        elif last == "planned" or last == "running" or last == "complete":
            status = last
        else:
            status = "terminal_failure"
        coordinates = events[0].coordinates
        assert coordinates is not None
        return ObservationState(key, coordinates, events, last_sha, status, attempts, first_sha)

    # --- seal -----------------------------------------------------------------------------
    #
    # A log is hash-chained, so only its newest event has no later record that cites it.
    # When an observation reaches a final status (complete, terminal failure, retries
    # exhausted) a build manifest, the project's existing lineage record, is published for
    # that newest event. A sealed log can then not lose or change its final event unnoticed.
    # Open logs stay unsealed: nothing yet exists that could anchor their newest event.

    def _seal_path(self, key: str) -> Path:
        return self.run_dir / "observations" / key / "final.build.json"

    def _seal_manifest(self, state: ObservationState) -> BuildManifest:
        key = state.key
        base = f"runs/{self.run_id}/observations/{key}/events"
        final = state.events[-1]
        inputs = [
            BuildInput(
                role="planned_event",
                artifact=ArtifactRef(
                    artifact_id=f"{key}:000001",
                    logical_ref=f"{base}/000001.json",
                    sha256=state.first_event_sha256,
                ),
            )
        ]
        if final.raw_output is not None:
            inputs.append(BuildInput(role="raw_output", artifact=final.raw_output))
        lineage = [ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs]
        settings = self.record.resolved_config_sha256
        artifact_id = derived_artifact_id(
            artifact_kind="generation_final_event",
            inputs=lineage,
            transformation_id="generation_observation_seal_v1",
            transformation_settings_sha256=settings,
            output_sha256=state.last_event_sha256,
            output_format="json",
            serialization_version="1",
        )
        return BuildManifest(
            schema_version="1",
            inputs=inputs,
            transformation_id="generation_observation_seal_v1",
            transformation_settings_sha256=settings,
            outputs=[
                DerivedOutput(
                    artifact=ArtifactRef(
                        artifact_id=artifact_id,
                        logical_ref=f"{base}/{final.sequence:06d}.json",
                        sha256=state.last_event_sha256,
                    ),
                    artifact_kind="generation_final_event",
                    output_format="json",
                    serialization_version="1",
                    identity_version="1",
                )
            ],
            status="complete",
        )

    def _check_seal(self, state: ObservationState) -> None:
        path = self._seal_path(state.key)
        if not path.exists():
            return
        try:
            if state.status not in _FINAL:
                raise ValueError("the observation is not in a final status")
            stored = BuildManifest.model_validate_json(path.read_bytes())
            if stored != self._seal_manifest(state):
                raise ValueError("the sealed final event is not the newest event")
        except ValueError as error:
            raise CorruptRecord(
                f"seal of {state.key} does not match its event log: {error}"
            ) from error

    def _seal(self, state: ObservationState) -> None:
        final_path = self.run_dir / "observations" / state.key / "events"
        final_path = final_path / f"{state.events[-1].sequence:06d}.json"
        manifest = self._seal_manifest(state)
        logical = manifest.outputs[0].artifact.logical_ref
        try:
            publish_manifest(manifest, self._seal_path(state.key), {logical: final_path})
        except FileExistsError:
            pass  # sealed concurrently; the next read compares it with the log

    def observation(self, key: str) -> ObservationState:
        return self._load(key)

    def verify(self) -> list[ObservationState]:
        return [self._load(key) for key in self.observation_keys()]

    def pending(self) -> list[str]:
        """Observations that can start an attempt, in planning order."""
        states = [s for s in self.verify() if s.status in ("planned", "retryable")]
        states.sort(key=lambda s: (s.events[0].at_utc, s.key))
        return [s.key for s in states]

    # --- writing ------------------------------------------------------------------------

    def _append(self, key: str, prior: ObservationState | None, **fields: Any) -> AttemptEvent:
        previous = prior.events[-1].sequence if prior else 0
        event = AttemptEvent(
            run_id=self.run_id,
            run_record_sha256=self.run_sha256,
            observation_key=key,
            sequence=previous + 1,
            previous_event_sha256=prior.last_event_sha256 if prior else None,
            at_utc=self.now(),
            **fields,
        )
        if prior and event.at_utc < prior.events[-1].at_utc:
            # Checked before writing: a published event cannot be removed from an append-only log.
            raise LifecycleError(
                f"clock moved backwards: {event.at_utc} precedes {prior.events[-1].at_utc}"
            )
        content = serialize_project_json(event, AttemptEvent)
        digest = sha256_bytes(content)
        target = self.run_dir / "observations" / key / "events" / f"{event.sequence:06d}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            publish_bytes(
                target,
                content,
                digest,
                validate_temp=lambda path: read_project_json(path, digest, AttemptEvent),
            )
        except FileExistsError as error:
            raise PublicationConflict(f"event {event.sequence} of {key} already exists") from error
        current = self._load(key)
        if current.status in _FINAL:
            self._seal(current)
        return event

    def plan(self, coordinates: Coordinates) -> str:
        """Register an observation. Planning the same coordinates again changes nothing."""
        self._check_coordinates(coordinates)
        key = self.key_for(coordinates)
        try:
            existing = self._load(key)
        except NotPlanned:
            try:
                self._append(key, None, state="planned", coordinates=coordinates)
                return key
            except PublicationConflict:
                existing = self._load(key)
        if existing.coordinates != coordinates:
            raise PublicationConflict(f"{key} is already planned with different coordinates")
        return key

    def begin_attempt(
        self, key: str, *, environment: EnvironmentRecord, rendered_input_sha256: str
    ) -> str:
        validate_sha256(rendered_input_sha256)
        state = self._load(key)
        refusal = {
            "complete": "observation is complete and is never regenerated",
            "terminal_failure": "observation ended in a terminal failure",
            "running": "an attempt is in progress; recover interrupted attempts first",
            "retries_exhausted": f"retry limit of {self.retry_policy.max_attempts} reached",
        }.get(state.status)
        if refusal:
            raise InvalidTransition(refusal)
        self._check_environment(environment)
        first = next((e for e in state.events if e.state == "running"), None)
        if first is not None and first.rendered_input_sha256 != rendered_input_sha256:
            raise InvalidTransition("a retry must use the identical rendered input")
        digest = environment_sha256(environment)
        env_path = self.run_dir / "environments" / f"{digest}.json"
        env_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            publish_bytes(env_path, serialize_project_json(environment, EnvironmentRecord), digest)
        except FileExistsError:
            pass  # identical bytes already stored
        attempt_id = new_attempt_id()
        self._append(
            key,
            state,
            state="running",
            attempt_id=attempt_id,
            attempt_number=state.attempts + 1,
            backend_kind=self.record.backend_kind,
            environment_sha256=digest,
            rendered_input_sha256=rendered_input_sha256,
            seed=state.coordinates.seed,
        )
        return attempt_id

    def receipt_of(self, state: ObservationState) -> ExecutionReceipt | None:
        """The verified receipt of a completed observation, or None if it has none."""
        last = state.events[-1]
        if last.state != "complete" or last.execution_receipt_sha256 is None:
            return None
        assert last.attempt_id is not None
        return read_project_json(
            self._receipt_path(state.key, last.attempt_id),
            last.execution_receipt_sha256,
            ExecutionReceipt,
        )

    def publish_input(self, record: GeneratorInputRecord) -> ArtifactRef:
        """Store a generator input with this run. Only an input of this run's configuration."""
        if record.rendered.resolved_config_sha256 != self.record.resolved_config_sha256:
            raise RunMismatch("generator input was rendered under another configuration")
        return publish_input_record(self.run_dir, record)

    def _running(self, state: ObservationState, attempt_id: str) -> AttemptEvent:
        if state.status != "running" or state.events[-1].attempt_id != attempt_id:
            raise InvalidTransition("that attempt is not the running attempt of this observation")
        return state.events[-1]

    def _raw_ref(self, key: str, attempt_id: str, rendered_sha: str, raw_sha: str) -> ArtifactRef:
        artifact_id = derived_artifact_id(
            artifact_kind="generator_raw_output",
            inputs=[ArtifactInput("rendered_input", key, rendered_sha)],
            transformation_id="generator_raw_output_v1",
            transformation_settings_sha256=self.record.resolved_config_sha256,
            output_sha256=raw_sha,
            output_format="text",
            serialization_version="1",
        )
        return ArtifactRef(
            artifact_id=artifact_id,
            logical_ref=f"runs/{self.run_id}/observations/{key}/raw/{attempt_id}.txt",
            sha256=raw_sha,
        )

    def complete(
        self,
        key: str,
        attempt_id: str,
        raw_output: str,
        *,
        execution: ExecutionFacts | None = None,
        input_record_sha256: str | None = None,
    ) -> None:
        """Store the raw text, then record the completion with fields parsed from it.

        Execution facts, when given, become a receipt that is checked against the stored input
        record and the configuration before anything is written.
        """
        if (execution is None) != (input_record_sha256 is None):
            raise ValueError("execution facts and the input record hash are given together")
        state = self._load(key)
        if state.status == "complete":
            raise DuplicateCompletion(f"observation {key} is already complete")
        running = self._running(state, attempt_id)
        assert running.rendered_input_sha256 is not None
        content = raw_output.encode("utf-8")
        ref = self._raw_ref(key, attempt_id, running.rendered_input_sha256, sha256_bytes(content))
        receipt_sha: str | None = None
        if execution is not None and input_record_sha256 is not None:
            assert running.environment_sha256 is not None
            receipt = ExecutionReceipt(
                run_id=self.run_id,
                observation_key=key,
                attempt_id=attempt_id,
                environment_sha256=running.environment_sha256,
                input_record_sha256=input_record_sha256,
                rendered_input_sha256=running.rendered_input_sha256,
                raw_output_sha256=ref.sha256,
                output_token_ids_sha256=token_ids_sha256(execution.output_token_ids),
                facts=execution,
            )
            self._check_receipt(key, receipt, running)
            receipt_content = serialize_project_json(receipt, ExecutionReceipt)
            receipt_sha = sha256_bytes(receipt_content)
            receipt_target = self._receipt_path(key, attempt_id)
            receipt_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                publish_bytes(receipt_target, receipt_content, receipt_sha)
            except FileExistsError:
                if receipt_target.read_bytes() != receipt_content:
                    raise PublicationConflict(
                        "a different receipt is already stored for this attempt"
                    ) from None
        target = self.root / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            publish_bytes(target, content, ref.sha256)
        except FileExistsError:
            if sha256_bytes(target.read_bytes()) != ref.sha256:
                raise PublicationConflict(
                    "different raw output already stored for this attempt"
                ) from None
        parsed = parse_output(raw_output, self.config.settings.parser_version)
        self._append(
            key,
            state,
            state="complete",
            attempt_id=attempt_id,
            attempt_number=running.attempt_number,
            raw_output=ref,
            parse_status=parsed.parse_status,
            answer=parsed.answer,
            abstained=parsed.abstained,
            parse_reason=parsed.reason,
            execution_receipt_sha256=receipt_sha,
        )

    def fail(self, key: str, attempt_id: str, *, kind: str, detail: str, retryable: bool) -> None:
        state = self._load(key)
        running = self._running(state, attempt_id)
        if retryable and kind not in self.retry_policy.retryable_failures:
            raise InvalidTransition(f"{kind!r} is not a retryable failure kind in the retry policy")
        self._append(
            key,
            state,
            state="retryable_failure" if retryable else "terminal_failure",
            attempt_id=attempt_id,
            attempt_number=running.attempt_number,
            failure_kind=kind,
            failure_detail=detail,
        )

    def recover_interrupted(self, *, previous_worker_stopped: bool) -> list[str]:
        """Record attempts left running by a dead worker as interrupted.

        The caller must know no worker is still writing to this run; a live worker's
        attempt would otherwise be closed under it.
        """
        if previous_worker_stopped is not True:
            raise UnsafeResume("recovery requires confirmation that no worker is still running")
        recovered = []
        for state in self.verify():
            if state.status == "running":
                last = state.events[-1]
                self._append(
                    state.key,
                    state,
                    state="interrupted",
                    attempt_id=last.attempt_id,
                    attempt_number=last.attempt_number,
                )
                recovered.append(state.key)
        return recovered


def official_completions(store: RunStore) -> list[ObservationState]:
    """Complete observations of a run, only if the run is an official development run."""
    if not store.record.official:
        raise NonOfficialBackend(
            f"a {store.record.backend_kind} run with a {store.record.purpose} configuration "
            "is not an official development run"
        )
    return [state for state in store.verify() if state.status == "complete"]


def check_no_duplicate_completions(root: Path) -> None:
    """The same observation key must not be completed in two runs.

    Reads each log's final event only; chain verification belongs to RunStore.open.
    """
    owner: dict[str, str] = {}
    for events_dir in sorted((root / "runs").glob("*/observations/*/events")):
        files = sorted(p for p in events_dir.glob("0*.json"))
        if not files:
            continue
        last = AttemptEvent.model_validate(json.loads(files[-1].read_text(encoding="utf-8")))
        if last.state != "complete":
            continue
        if last.observation_key in owner and owner[last.observation_key] != last.run_id:
            raise DuplicateCompletion(
                f"{last.observation_key} is complete in runs {owner[last.observation_key]} "
                f"and {last.run_id}"
            )
        owner[last.observation_key] = last.run_id
