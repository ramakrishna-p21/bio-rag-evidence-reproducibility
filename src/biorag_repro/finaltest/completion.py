"""Completion record of one rehearsal batch: the immutable end of its observation history.

Records only; publishing is in `finish.py`. A completion states that every observation the
batch planned has been decided and sealed, and says how each one ended. It is not a statement
that the batch succeeded: an observation that failed terminally or ran out of retries stays a
recorded failure with its attempts, and the outcome says so (`complete_with_recorded_missing`).

Two things about authorization are kept apart. The *citation chain* of the batch shows which
reports permitted it to run. The `PublicationAuthorization` of a completion shows what permitted
this record to be published: the report of the session that ran the batch, still held and still
bound to the batch (`same_session`), or a fresh report that was compared with the first one and
found continuous (`fresh_session`). A historical citation alone is neither. Hash dependencies
are acyclic: the record cites the registry event *before* its own event, and the event cites the
record's digest.
"""

from collections.abc import Collection, Sequence
from typing import Final, Literal, Self

from pydantic import field_validator, model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    BatchManifest,
    Nonblank,
    Sha256,
    citation_sha256,
)
from biorag_repro.finaltest.layout import FrozenRecord, RootKind, serialize
from biorag_repro.finaltest.runrecord import FinalTestRunRecord, run_record_sha256
from biorag_repro.generation.attempts import ObservationState
from biorag_repro.generation.environment import check_utc

Outcome = Literal["complete_all_observed", "complete_with_recorded_missing"]
DECIDED: Final = ("complete", "terminal_failure", "retries_exhausted")


class NotCompletable(ValueError):
    """The observations are not all decided. `code` names the first reason."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class PublicationAuthorization(FrozenRecord):
    mode: Literal["same_session", "fresh_session"]
    report_sha256: Sha256
    report_binding_sha256: Sha256
    policy_sha256: Sha256
    evaluator_code_sha256: Sha256
    scientific_identity_sha256: Sha256
    # the batch's latest citation (same session) or the continuity assessment (fresh session)
    basis_sha256: Sha256


class ObservationEnd(FrozenRecord):
    key: Nonblank
    status: Literal["complete", "terminal_failure", "retries_exhausted"]
    attempts: int
    first_event_sha256: Sha256
    final_event_sha256: Sha256  # the event the observation's seal pins
    # how the last attempt ended; an interrupted attempt records no failure kind and none is
    # invented for it
    final_state: Literal["complete", "terminal_failure", "retryable_failure", "interrupted"]
    failure_kind: Nonblank | None
    failure_detail: Nonblank | None

    @model_validator(mode="after")
    def failure_matches_status(self) -> Self:
        allowed = {
            "complete": ("complete",),
            "terminal_failure": ("terminal_failure",),
            "retries_exhausted": ("retryable_failure", "interrupted"),
        }[self.status]
        if self.final_state not in allowed:
            raise ValueError(f"a {self.status} observation cannot end in {self.final_state}")
        if (self.failure_kind is None) != (self.failure_detail is None):
            raise ValueError("a failure carries both its kind and its detail")
        if (self.final_state in ("complete", "interrupted")) != (self.failure_kind is None):
            raise ValueError("a failure kind belongs to a failed last attempt only")
        return self


class BatchCompletion(FrozenRecord):
    schema_version: Literal["1"] = "1"
    root_id: Sha256
    root_kind: RootKind
    campaign_plan_sha256: Sha256
    batch_id: Sha256
    run_id: Nonblank
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256
    manifest_sha256: Sha256
    key_set_sha256: Sha256
    ownership_sha256: Sha256
    inputs_set_sha256: Sha256
    run_record_sha256: Sha256
    citation_count: int
    first_citation_sha256: Sha256
    last_citation_sha256: Sha256
    authorizing_event_sha256: (
        Sha256  # file hash of the latest authorizing event (initial or resume)
    )
    registry_head_sequence: int  # the event before the completion event
    registry_head_event_sha256: Sha256
    authorization: PublicationAuthorization
    outcome: Outcome
    complete_count: int
    missing_count: int
    observations: list[ObservationEnd]
    created_utc: str  # informational

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        keys = [o.key for o in self.observations]
        if not keys or keys != sorted(set(keys)):
            raise ValueError("observations are listed once each, in key order")
        complete = sum(o.status == "complete" for o in self.observations)
        if (self.complete_count, self.missing_count) != (complete, len(keys) - complete):
            raise ValueError("the counts do not follow from the observations")
        wanted = (
            "complete_all_observed" if complete == len(keys) else "complete_with_recorded_missing"
        )
        if self.outcome != wanted:
            raise ValueError("the outcome does not follow from the observations")
        if self.citation_count < 1 or self.registry_head_sequence < 1:
            raise ValueError("a batch has at least one citation and a registry head")
        return self


def completion_sha256(record: BatchCompletion) -> str:
    return sha256_bytes(serialize(record))


def _ending(state: ObservationState) -> ObservationEnd:
    last = state.events[-1]
    return ObservationEnd(
        key=state.key,
        status=state.status,  # type: ignore[arg-type]  # checked against DECIDED by the caller
        attempts=state.attempts,
        first_event_sha256=state.first_event_sha256,
        final_event_sha256=state.last_event_sha256,
        final_state=last.state,  # type: ignore[arg-type]  # the model rejects the other states
        failure_kind=last.failure_kind,
        failure_detail=last.failure_detail,
    )


def check_decided(states: Sequence[ObservationState], expected: Collection[str]) -> None:
    """Raise `NotCompletable` unless every expected observation is reported once and decided.

    Eligibility is read from the observation states themselves, not from any classification: a
    key must be reported exactly once, belong to the plan, and be complete, a recorded terminal
    failure or retries exhausted. A planned, running or retryable observation, or one that is
    absent, is work that remains and no completion can be built over it.
    """
    seen = [s.key for s in states]
    if len(seen) != len(set(seen)):
        raise NotCompletable("duplicate_observation")
    if foreign := sorted(set(seen) - set(expected)):
        raise NotCompletable("foreign_observation", foreign[0])
    if absent := sorted(set(expected) - set(seen)):
        raise NotCompletable("observation_absent", absent[0])
    for state in sorted(states, key=lambda s: s.key):
        if state.status not in DECIDED:
            raise NotCompletable("observation_open", f"{state.key} is {state.status}")


def build_completion(
    *,
    manifest: BatchManifest,
    run: FinalTestRunRecord,
    citations: Sequence[AuthorizationCitation],
    authorizing_event_sha256: str,
    head: tuple[int, str],
    authorization: PublicationAuthorization,
    manifest_sha256: str,
    states: Sequence[ObservationState],
    expected: Collection[str],
    created_utc: str,
) -> BatchCompletion:
    """The completion of a batch whose planned observations are all decided, or `NotCompletable`."""
    check_decided(states, expected)
    endings = [_ending(s) for s in sorted(states, key=lambda s: s.key)]
    complete = sum(e.status == "complete" for e in endings)
    return BatchCompletion(
        root_id=run.root_id,
        root_kind=run.root_kind,
        campaign_plan_sha256=run.campaign_plan_sha256,
        batch_id=run.batch_id,
        run_id=run.run_id,
        matrix_arm=run.matrix_arm,
        resolved_config_sha256=run.resolved_config_sha256,
        manifest_sha256=manifest_sha256,
        key_set_sha256=manifest.key_set_sha256,
        ownership_sha256=manifest.ownership,
        inputs_set_sha256=run.inputs_set_sha256,
        run_record_sha256=run_record_sha256(run),
        citation_count=len(citations),
        first_citation_sha256=citation_sha256(citations[0]),
        last_citation_sha256=citation_sha256(citations[-1]),
        authorizing_event_sha256=authorizing_event_sha256,
        registry_head_sequence=head[0],
        registry_head_event_sha256=head[1],
        authorization=authorization,
        outcome=(
            "complete_all_observed"
            if complete == len(endings)
            else "complete_with_recorded_missing"
        ),
        complete_count=complete,
        missing_count=len(endings) - complete,
        observations=endings,
        created_utc=created_utc,
    )
