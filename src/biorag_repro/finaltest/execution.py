"""A started batch and the checks that guard its execution.

`StartedBatch` is what a successful rehearsal start returns: the published records' identities,
the exact generator inputs held in memory, the open session and the guarded store. It is a
description of a session in progress, not a credential. Once the session is closed or lost
every guarded operation refuses, and nothing here can reopen a run.

`SessionGate` is the check `FinalTestStore` calls before each operation that writes. It
re-derives, from the files on disk, what the start published: the registry events up to the
head the session recorded, the manifest, the input binding, the rosters, the citation and the
run record. Together with the live session lock this shows local consistency since the start.
It does not show that any approval is still valid, that the host or GPU is the approved one,
that the head is the latest authorized head, or that registry, witnesses and records were not
rolled back together.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest.inputs import RosterIdentityRow
from biorag_repro.finaltest.registry import registry_head
from biorag_repro.finaltest.session import Session, SessionLost
from biorag_repro.generation.environment import EnvironmentRecord
from biorag_repro.generation.finaltest_store import FinalTestStore, GateRefused
from biorag_repro.generation.runner import GeneratorBackend, RosterEntry, RunSummary, run_roster
from biorag_repro.generation.seeds import GenerationSeedRegistry
from biorag_repro.readiness.files import FileProblem, read_regular


class SessionGate:
    """Callable with the `Gate` signature of `generation.finaltest_store`."""

    def __init__(
        self,
        session: Session,
        files: Mapping[str, str],
        head: tuple[int, str],
        rows: Mapping[str, RosterIdentityRow],
    ) -> None:
        self.session = session
        self.files = dict(files)  # root-relative path -> sha256 recorded at the end of the start
        self.head = head
        self.rows = dict(rows)

    def __call__(self, action: str, key: str | None, rendered_input_sha256: str | None) -> None:
        try:
            self.session.assert_held()
        except SessionLost as error:
            raise GateRefused(f"{action}: {error}") from error
        root = self.session.root
        if registry_head(root) != self.head:
            raise GateRefused(f"{action}: the registry head is not the one the session recorded")
        for relative, expected in self.files.items():
            try:
                actual = sha256_bytes(read_regular(root, relative))
            except FileProblem as error:
                raise GateRefused(f"{action}: {relative} cannot be read: {error}") from error
            if actual != expected:
                raise GateRefused(f"{action}: {relative} differs from what the start published")
        if key is None:
            return
        row = self.rows.get(key)
        if row is None:
            raise GateRefused(f"{action}: {key} is not an observation of this batch")
        if rendered_input_sha256 is not None and rendered_input_sha256 != row.rendered_input_sha256:
            raise GateRefused(f"{action}: the input of {key} differs from the bound roster")
        if action == "begin_attempt":
            self.session.begin_worker()  # after every check, before the attempt is written


@dataclass(frozen=True)
class StartedBatch:
    session: Session
    store: FinalTestStore
    batch_id: str
    run_id: str
    binding_sha256: str
    citation_sha256: str
    run_record_sha256: str
    report_sha256: str
    head: tuple[int, str]
    rows: Mapping[str, RosterIdentityRow]
    entries: tuple[RosterEntry, ...]
    seeds: GenerationSeedRegistry | None
    root: Path


def run_batch(
    started: StartedBatch,
    backend: GeneratorBackend,
    *,
    capture_environment: Callable[[], EnvironmentRecord],
) -> RunSummary:
    """Run the batch's roster through the guarded store. The existing runner does the work."""
    started.session.assert_held()
    return run_roster(
        started.store,
        list(started.entries),
        backend,
        capture_environment=capture_environment,
        seeds=started.seeds,
    )
