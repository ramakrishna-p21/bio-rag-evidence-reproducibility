"""A read-only view of a stored run.

`RunStore.open` seals final events that lack a seal, so it cannot be used to inspect a run
without changing it. This view reuses the store's own validation (`observation`,
`observation_keys`, seal comparison), none of which writes, and replaces every method that
writes, and both constructors, with a refusal. It changes nothing in `RunStore`.
"""

from pathlib import Path
from typing import Any

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.generation.attempts import RunRecord, RunStore
from biorag_repro.generation.config import GeneratorConfiguration


class ReadOnlyViolation(RuntimeError):
    """A method that writes was reached through a read-only view."""


def _refuse(*_args: Any, **_kwargs: Any) -> Any:
    raise ReadOnlyViolation("a read-only view cannot write")


class ReadOnlyRunView(RunStore):
    create = _refuse
    open = _refuse
    plan = _refuse
    begin_attempt = _refuse
    complete = _refuse
    fail = _refuse
    recover_interrupted = _refuse
    publish_input = _refuse
    _append = _refuse
    _seal = _refuse


def read_only_view(
    root: Path,
    record: RunRecord,
    run_sha256: str,
    config: ResolvedConfiguration[GeneratorConfiguration],
) -> ReadOnlyRunView:
    """The caller supplies the already-verified run record and its hash; nothing is read here."""
    return ReadOnlyRunView(root, record, run_sha256, config, _refuse)
