"""The store of one rehearsed final-test run: `RunStore` behind a session gate.

`RunStore` is reused unchanged for the attempt state machine, retry rule, receipts, completions
and seals. This subclass only adds, in front of every method that writes, a check supplied by
the batch start (`finaltest.execution`) and removes the ways to create, open or recover a run:

* `create`, `open`, `recover_interrupted`: refused. A run is created only by
  `create_final_test_store`, and a stored run cannot be reopened in C2B (resume needs a fresh
  authorization and is a later pass).
* `plan`, `publish_input`, `begin_attempt`: the gate runs first, with the observation key and,
  where there is one, the rendered input digest.
* `complete`, `fail`: the attempt must have been started by this store object, and the gate
  runs first.
* `_append` and `_seal`, which every writer above reaches, run the gate too, so a direct call
  to them is also checked.

The gate establishes local consistency only: the session still holds its lock, the registry and
the batch records still hash as they did when the session recorded them, and the input belongs
to the bound roster. It does not show that approvals are still valid, that the hardware is the
approved hardware, or that the registry head is the latest authorized one. A store can still
be constructed by code that skips this module; `finaltest.audit` is what classifies such a run
as unbound. `RunRecord`, `RunPhase` and every development and qualification path are unchanged.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest.layout import serialize
from biorag_repro.finaltest.runrecord import (
    FinalTestRunRecord,
    run_record_projection,
    run_record_sha256,
)
from biorag_repro.generation.attempts import (
    LifecycleError,
    ObservationState,
    PublicationConflict,
    RunStore,
)
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import EnvironmentRecord, utc_now
from biorag_repro.generation.inputs import GeneratorInputRecord

# action, observation key (if the action concerns one), rendered input digest (if it has one)
Gate = Callable[[str, str | None, str | None], None]


class GateRefused(LifecycleError):
    """A guarded operation was refused before anything was written."""


class StoreOperationRefused(LifecycleError):
    """The operation does not exist for a final-test store in this version."""


def _refused(*_args: Any, **_kwargs: Any) -> Any:
    raise StoreOperationRefused("a final-test store is created once and cannot be reopened")


class FinalTestStore(RunStore):
    create = _refused
    open = _refused
    recover_interrupted = _refused

    def __init__(
        self,
        root: Path,
        record: FinalTestRunRecord,
        config: ResolvedConfiguration[GeneratorConfiguration],
        gate: Gate,
        now: Callable[[], str],
    ) -> None:
        super().__init__(
            root, run_record_projection(record), run_record_sha256(record), config, now
        )
        self.final_record = record
        self._gate = gate
        self._begun: dict[str, str] = {}  # attempt id -> key, for attempts started here

    # --- gated writers ---------------------------------------------------------------

    def plan(self, coordinates: Any) -> str:
        self._gate("plan", self.key_for(coordinates), None)
        return super().plan(coordinates)

    def publish_input(self, record: GeneratorInputRecord) -> Any:
        self._gate("publish_input", record.observation_key, record.rendered.rendered_sha256)
        return super().publish_input(record)

    def begin_attempt(
        self, key: str, *, environment: EnvironmentRecord, rendered_input_sha256: str
    ) -> str:
        self._gate("begin_attempt", key, rendered_input_sha256)
        attempt_id = super().begin_attempt(
            key, environment=environment, rendered_input_sha256=rendered_input_sha256
        )
        self._begun[attempt_id] = key
        return attempt_id

    def _own_attempt(self, key: str, attempt_id: str) -> None:
        if self._begun.get(attempt_id) != key:
            raise GateRefused("the attempt was not started by this store in this session")

    def complete(self, key: str, attempt_id: str, raw_output: str, **kwargs: Any) -> None:
        self._own_attempt(key, attempt_id)
        self._gate("complete", key, None)
        super().complete(key, attempt_id, raw_output, **kwargs)

    def fail(self, key: str, attempt_id: str, *, kind: str, detail: str, retryable: bool) -> None:
        self._own_attempt(key, attempt_id)
        self._gate("fail", key, None)
        super().fail(key, attempt_id, kind=kind, detail=detail, retryable=retryable)

    def _append(self, key: str, prior: ObservationState | None, **fields: Any) -> Any:
        self._gate("append", key, None)
        return super()._append(key, prior, **fields)

    def _seal(self, state: ObservationState) -> None:
        self._gate("seal", state.key, None)
        super()._seal(state)


def create_final_test_store(
    root: Path,
    record: FinalTestRunRecord,
    config: ResolvedConfiguration[GeneratorConfiguration],
    gate: Gate,
    *,
    now: Callable[[], str] = utc_now,
) -> FinalTestStore:
    """Write `runs/<run_id>/` exactly as decided (the record bytes are already cited elsewhere)."""
    if record.root_kind != "rehearsal":
        raise StoreOperationRefused("only rehearsal runs can be created in this version")
    if (record.resolved_config_sha256, record.raw_config_sha256) != (
        config.configuration_sha256,
        config.raw_toml_sha256,
    ):
        raise StoreOperationRefused("the configuration is not the one the run record names")
    run_dir = root / "runs" / record.run_id
    run_dir.mkdir(parents=True)
    publish_bytes(
        run_dir / "config.canonical.json", config.canonical_json, config.configuration_sha256
    )
    publish_bytes(run_dir / "config.toml", config.raw_toml, config.raw_toml_sha256)
    content = serialize(record)
    try:
        publish_bytes(run_dir / "run.json", content, sha256_bytes(content))
    except FileExistsError as error:
        raise PublicationConflict("a run record already exists here") from error
    return FinalTestStore(root, record, config, gate, now)
