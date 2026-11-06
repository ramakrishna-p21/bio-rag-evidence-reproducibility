"""Execution of an observation roster through the attempt lifecycle.

The runner plans the roster deterministically, asks an injected backend for text, and
leaves every record to `RunStore`. It never invents an attempt or an output: an
observation that has not reached a final status produces no result, and an operational
failure is handed on as missing, not as an incorrect answer. It does not choose prompts,
decoding settings or seeds; those come from the configuration and the seed registry.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from biorag_repro.benchmark.models import Record
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.attempts import (
    Coordinates,
    LifecycleError,
    NonOfficialBackend,
    ObservationState,
    RunMismatch,
    RunStore,
    UnsafeResume,
    official_completions,
)
from biorag_repro.generation.environment import BackendIdentity, EnvironmentRecord
from biorag_repro.generation.prompt import RenderedInput
from biorag_repro.generation.seeds import (
    GenerationSeedRegistry,
    check_seeds_for_configuration,
    repeat_seed,
)

_FINAL_FAILURES = ("terminal_failure", "retries_exhausted")


class RosterMismatch(LifecycleError):
    """The roster contradicts the configuration or the records already stored."""


class BackendFailure(Exception):
    """An operational failure reported by a backend. Whether it is retried is the policy's call."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class GeneratorBackend(Protocol):
    identity: BackendIdentity

    def generate(
        self, rendered: RenderedInput, *, seed: int | None, max_new_tokens: int
    ) -> str: ...


class RosterEntry(Record):
    """One authorized observation with the exact input its generator call receives."""

    item_id: str
    paper_id: str
    representation_or_condition: str
    repeat_index: int
    rendered: RenderedInput


@dataclass(frozen=True)
class RunSummary:
    complete: int
    terminal_failures: int
    retries_exhausted: int
    attempts: int  # attempts started by this call


def _check_roster(store: RunStore, roster: list[RosterEntry]) -> None:
    settings = store.config.settings
    seen: set[tuple[str, str, int]] = set()
    for entry in roster:
        name = (entry.item_id, entry.representation_or_condition, entry.repeat_index)
        if name in seen:
            raise RosterMismatch(f"duplicate roster entry {name}")
        seen.add(name)
        rendered = entry.rendered
        if rendered.resolved_config_sha256 != store.record.resolved_config_sha256:
            raise RosterMismatch(f"input for {name} was rendered under another configuration")
        if rendered.prompt_sha256 != settings.prompt_sha256:
            raise RosterMismatch(f"input for {name} used a different prompt than the configuration")
        if (rendered.context is None) != (settings.matrix_arm == "closed_book"):
            raise RosterMismatch(
                f"context of {name} does not match matrix arm {settings.matrix_arm}"
            )


def run_roster(
    store: RunStore,
    roster: list[RosterEntry],
    backend: GeneratorBackend,
    *,
    capture_environment: Callable[[], EnvironmentRecord],
    seeds: GenerationSeedRegistry | None = None,
) -> RunSummary:
    """Run every undecided observation of the roster; skip decided ones untouched."""
    settings = store.config.settings
    if backend.identity.kind != store.record.backend_kind:
        raise RunMismatch(
            f"backend is {backend.identity.kind}, the run was created on {store.record.backend_kind}"
        )
    if settings.decoding.do_sample and seeds is None:
        raise ValueError("sampling decoding needs the seed registry named by the configuration")
    if seeds is not None:
        check_seeds_for_configuration(settings, seeds)
    _check_roster(store, roster)

    ordered = sorted(
        roster, key=lambda e: (e.item_id, e.representation_or_condition, e.repeat_index)
    )
    planned: list[tuple[RosterEntry, Coordinates]] = []
    for entry in ordered:
        seed = (
            repeat_seed(seeds, entry.repeat_index)
            if settings.decoding.do_sample and seeds
            else None
        )
        planned.append(
            (
                entry,
                Coordinates(
                    item_id=entry.item_id,
                    paper_id=entry.paper_id,
                    matrix_arm=settings.matrix_arm,
                    representation_or_condition=entry.representation_or_condition,
                    repeat_index=entry.repeat_index,
                    seed=seed,
                ),
            )
        )
    roster_keys = {store.key_for(coordinates) for _, coordinates in planned}
    stray = set(store.observation_keys()) - roster_keys
    if stray:
        raise RosterMismatch(f"{len(stray)} stored observations are not in the roster")

    keys = [store.plan(coordinates) for _, coordinates in planned]
    if any(store.observation(key).status == "running" for key in keys):
        raise UnsafeResume("an attempt is still marked running; recover interrupted attempts first")

    attempts = 0
    policy = store.retry_policy
    for (entry, _), key in zip(planned, keys, strict=True):
        while True:
            state = store.observation(key)
            if state.status not in ("planned", "retryable"):
                break
            attempt_id = store.begin_attempt(
                key,
                environment=capture_environment(),
                rendered_input_sha256=entry.rendered.rendered_sha256,
            )
            attempts += 1
            try:
                raw = backend.generate(
                    entry.rendered,
                    seed=state.coordinates.seed,
                    max_new_tokens=settings.budgets.max_new_tokens,
                )
            except BackendFailure as failure:
                store.fail(
                    key,
                    attempt_id,
                    kind=failure.kind,
                    detail=failure.detail.strip() or failure.kind,
                    retryable=failure.kind in policy.retryable_failures,
                )
                continue
            except Exception as error:  # noqa: BLE001  # a bug or surprise is recorded, never retried
                store.fail(
                    key,
                    attempt_id,
                    kind="unclassified_error",
                    detail=f"{type(error).__name__}: {error}",
                    retryable=False,
                )
                continue
            if not isinstance(raw, str):
                store.fail(
                    key,
                    attempt_id,
                    kind="invalid_backend_output",
                    detail=f"backend returned {type(raw).__name__}, not text",
                    retryable=False,
                )
                continue
            store.complete(key, attempt_id, raw)

    states = [store.observation(key) for key in keys]
    return RunSummary(
        complete=sum(s.status == "complete" for s in states),
        terminal_failures=sum(s.status == "terminal_failure" for s in states),
        retries_exhausted=sum(s.status == "retries_exhausted" for s in states),
        attempts=attempts,
    )


# --- handoff to Stage C ---------------------------------------------------------------------------


def _result(store: RunStore, state: ObservationState) -> GenerationResult:
    c = state.coordinates
    common: dict[str, Any] = {
        "phase": store.record.phase,
        "benchmark_version": store.record.benchmark_version,
        "item_id": c.item_id,
        "paper_id": c.paper_id,
        "matrix_arm": c.matrix_arm,
        "representation_or_condition": c.representation_or_condition,
        "repeat_index": c.repeat_index,
        "resolved_config_sha256": store.record.resolved_config_sha256,
        "observation_key": state.key,
        "run_id": store.run_id,
    }
    last = state.events[-1]
    assert last.attempt_id is not None
    if state.status == "complete":
        assert last.raw_output is not None and last.parse_status is not None
        raw = (store.root / last.raw_output.logical_ref).read_text(encoding="utf-8")
        return GenerationResult(
            **common,
            attempt_id=last.attempt_id,
            status="completed",
            output=last.raw_output,
            raw_output=raw,
            parse_status=last.parse_status,
            answer=last.answer,
            abstained=last.abstained,
        )
    return GenerationResult(
        **common,
        attempt_id=last.attempt_id,
        status="operational_failure",
        failure=f"{last.failure_kind}: {last.failure_detail} (attempts: {state.attempts})",
    )


def _decided_results(store: RunStore) -> list[GenerationResult]:
    decided = [
        state
        for state in store.verify()
        if state.status == "complete" or state.status in _FINAL_FAILURES
    ]
    decided.sort(
        key=lambda s: (
            s.coordinates.item_id,
            s.coordinates.representation_or_condition,
            s.coordinates.repeat_index,
        )
    )
    return [_result(store, state) for state in decided]


def official_results(store: RunStore) -> list[GenerationResult]:
    """Decided observations of an official development run. Undecided ones yield nothing."""
    official_completions(store)  # refuses a run that is not an official development run
    return _decided_results(store)


@dataclass(frozen=True)
class NonOfficialResults:
    """Results of a mock or smoke run, labelled so they are not mistaken for observations."""

    backend_kind: str
    results: list[GenerationResult]


def nonofficial_results(store: RunStore) -> NonOfficialResults:
    if store.record.official:
        raise NonOfficialBackend("this is an official run; use official_results")
    return NonOfficialResults(store.record.backend_kind, _decided_results(store))
