"""Initialize a rehearsal root, and verify a root's complete inventory against an approval.

`verify_root` answers one question: is the storage found at the approved location exactly
the root that approval describes, with a complete and consistent history? It never says that
anything may run. Authority comes only from the `StorageAuthorization` the caller passes in,
which production code reads from the artifact that policy slot G11 approves; the files found
in the root are checked against it, never believed on their own. An empty or missing
directory, a root of another kind or identity, or a root at another location establishes
nothing and raises `AuthorityNotEstablished`.

Everything else is classified into the history codes fixed by `readiness.models`, using
`readiness.history.classify_history`: the first match of incomplete, conflicting, unauthorized
or unverified, otherwise `no_prior_final_test_execution`. P10-A has no batches, so any entry
outside the genesis files is unauthorized.

Limits. Entries are listed and read through no-follow descriptors, which guards against
mistakes and accidental substitution on a local disk, not against a process that can write the
directory during a check. `registry_lock` is `flock`: it excludes other holders on one host
and says nothing about network or shared file systems; supporting concurrent writers on the
execution host's file system needs an explicit, separately approved arrangement.

Initialization is for rehearsal roots only. The `final_test` contract is: the same three
publications in the same order, authorized by an approved G11 artifact and performed by a
separately reviewed act that does not exist yet.
"""

import fcntl
import os
import re
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    BatchManifest,
    CitationChainError,
    citation_sha256,
    observed_state_sha256,
    verify_citation_chain,
)
from biorag_repro.finaltest.continuity import ContinuityAssessment
from biorag_repro.finaltest.inputs import BatchInputBinding, ConditionRoster, roster_sha256
from biorag_repro.finaltest.layout import (
    EvidenceCitation,
    HeadCheckpoint,
    HeadWitness,
    RegistryEvent,
    RootAnchor,
    RootRecord,
    StorageAuthorization,
)
from biorag_repro.finaltest.runrecord import FinalTestRunRecord
from biorag_repro.generation.environment import utc_now
from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular
from biorag_repro.readiness.history import (
    INCOMPLETE,
    HistoryEntry,
    HistoryInventory,
    classify_history,
)

NO_PRIOR: Final = "no_prior_final_test_execution"
KNOWN_ARTIFACTS: Final = "root_and_registry"
_TEMP = re.compile(r"^\.(ROOT\.json|\d{6}\.json)\.[^/]+\.tmp$")
_MANIFEST_TEMP = re.compile(r"^\.manifest\.json\.[^/]+\.tmp$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_BATCH_DIR = re.compile(r"^batches/[0-9a-f]{64}$")
_RUN_DIR = re.compile(r"^runs/[0-9a-f]{32}$")
_ANY_TEMP = re.compile(r"^\.[^/]+\.tmp$")
_CITATION_FILE = re.compile(r"^\d{6}(\.continuity)?\.json$")
# A registry beyond genesis is pinned by nothing outside the data root the registry shares, so
# no verdict about it is positive. Genesis alone is pinned by the out-of-tree anchor.
HEAD_NOT_ESTABLISHED_TEXT: Final = (
    "incomplete: the latest registry head is not independently established"
)

Problem = tuple[str, str]  # (kind, detail); kind: incomplete, conflicting, unverified, unexpected


class AuthorityNotEstablished(Exception):
    """The storage found is not the root the approval describes (or none exists)."""

    def __init__(self, cause: str, detail: str = ""):
        super().__init__(f"{cause}: {detail}" if detail else cause)
        self.cause = cause


class InitializationRefused(Exception):
    """Initialization was not performed, or did not end in a verified root."""


class RegistryBusy(Exception):
    """Another holder has the registry lock."""


@dataclass(frozen=True)
class BatchRecord:
    state: str  # reserved, materialized, closed_incomplete
    campaign_plan_sha256: str
    ownership_sha256: str
    manifest_sha256: str
    # set once the batch is authorized: its run, the latest authorizing event (initial or resume)
    run_id: str | None = None
    authorization_event_sha256: str | None = None
    authorization_event_sequence: int | None = None
    resumes: int = 0


@dataclass(frozen=True)
class RootVerdict:
    code: str  # a history code, never an authorization
    root_kind: str
    root_id: str
    registry_head_sequence: int | None
    anchor_sha256: str | None
    inventory: dict[str, Any]  # canonical-JSON snapshot of the complete inventory
    problems: tuple[str, ...]
    batches: dict[str, BatchRecord] = field(default_factory=dict)
    # batch_id -> (what is recoverable, the exact problem text it accounts for)
    reconcile: dict[str, tuple[str, str]] = field(default_factory=dict)
    unwitnessed_head: bool = False
    head_authority: str = "genesis_anchored"  # or not_established
    checkpoint_continuity: bool | None = None  # None: no checkpoint supplied


# ---- enumeration ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Entry:
    path: str
    kind: str  # file, dir, link, other


def _enters(path: str, deep: bool) -> bool:
    if path in (layout.REGISTRY_DIR, layout.LOCKS_DIR, layout.BATCHES_DIR) or _BATCH_DIR.match(
        path
    ):
        return True
    if not deep:  # final-test roots: nothing beyond the P10-B layout is looked into
        return False
    if path == layout.RUNS_DIR:  # the run directories are listed, never entered
        return True
    parts = path.split("/")
    if len(parts) == 3 and parts[0] == layout.BATCHES_DIR:
        return parts[2] in (layout.INPUTS_DIR, layout.CITATIONS_DIR)
    return len(parts) == 4 and parts[2:] == [layout.INPUTS_DIR, layout.ROSTERS_DIR]


def _walk(directory_fd: int, prefix: str, deep: bool) -> list[_Entry]:
    found: list[_Entry] = []
    with os.scandir(directory_fd) as listing:
        items = sorted(listing, key=lambda item: item.name)
    for item in items:
        path = prefix + item.name
        if item.is_symlink():
            kind = "link"
        elif item.is_dir(follow_symlinks=False):
            kind = "dir"
        elif item.is_file(follow_symlinks=False):
            kind = "file"
        else:
            kind = "other"
        found.append(_Entry(path, kind))
        if kind == "dir" and _enters(path, deep):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            inner = os.open(item.name, flags, dir_fd=directory_fd)
            try:
                found += _walk(inner, path + "/", deep)
            finally:
                os.close(inner)
    return found


def _scan(root: Path, deep: bool = False) -> list[_Entry]:
    """Every entry of the declared scope. Only the registry and lock directories are entered;
    any other directory is one unexpected entry. `deep` (rehearsal roots only) also enters the
    input and citation directories of a batch and lists the run directories."""
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        return _walk(descriptor, "", deep)
    finally:
        os.close(descriptor)


def _role(entry: _Entry, deep: bool = False) -> str:
    """How an entry is read: root, event, lock, dir, temp, wrongkind or unexpected."""
    path = entry.path
    if entry.kind in ("link", "other"):
        return "unexpected"
    expected = {
        layout.ROOT_FILE: ("file", "root"),
        layout.REGISTRY_DIR: ("dir", "dir"),
        layout.LOCKS_DIR: ("dir", "dir"),
        layout.LOCK_FILE: ("file", "lock"),
        layout.BATCHES_DIR: ("dir", "dir"),
    }
    if path in expected:
        kind, role = expected[path]
        return role if entry.kind == kind else "wrongkind"
    directory, _, name = path.rpartition("/")
    if directory in ("", layout.REGISTRY_DIR) and entry.kind == "file" and _TEMP.match(name):
        return "temp"
    if directory == layout.REGISTRY_DIR and layout.EVENT_NAME.match(name):
        return "event" if entry.kind == "file" else "wrongkind"
    if directory == layout.BATCHES_DIR and _HEX64.match(name):
        return "batchdir" if entry.kind == "dir" else "wrongkind"
    if _BATCH_DIR.match(directory):
        if name == layout.MANIFEST_NAME:
            return "manifest" if entry.kind == "file" else "wrongkind"
        if entry.kind == "file" and _MANIFEST_TEMP.match(name):
            return "batchtemp"
    return _rehearsal_role(entry, deep) if deep else "unexpected"


def _rehearsal_role(entry: _Entry, deep: bool) -> str:
    """Roles that exist in rehearsal roots only; anywhere else these paths are unexpected."""
    path = entry.path
    name = path.rpartition("/")[2]
    parts = path.split("/")
    expected_dirs = (layout.RUNS_DIR,)
    if path in expected_dirs or (
        len(parts) == 3
        and parts[0] == layout.BATCHES_DIR
        and parts[2] in (layout.INPUTS_DIR, layout.CITATIONS_DIR)
    ):
        return "dir" if entry.kind == "dir" else "wrongkind"
    if path == layout.SESSION_LOCK_FILE:
        return "lock" if entry.kind == "file" else "wrongkind"
    if path == layout.HOLDER_FILE:
        return "holder" if entry.kind == "file" else "wrongkind"
    if _RUN_DIR.match(path):
        return "rundir" if entry.kind == "dir" else "wrongkind"
    if len(parts) == 4 and parts[0] == layout.BATCHES_DIR and parts[2] == layout.INPUTS_DIR:
        if name == layout.BINDING_NAME:
            return "binding" if entry.kind == "file" else "wrongkind"
        if name == layout.ROSTERS_DIR:
            return "dir" if entry.kind == "dir" else "wrongkind"
        if entry.kind == "file" and _ANY_TEMP.match(name):
            return "batchtemp"
    if len(parts) == 5 and parts[2:4] == [layout.INPUTS_DIR, layout.ROSTERS_DIR]:
        if entry.kind == "file" and re.fullmatch(r"[0-9a-f]{64}\.json", name):
            return "roster" if entry.kind == "file" else "wrongkind"
        if entry.kind == "file" and _ANY_TEMP.match(name):
            return "batchtemp"
    if len(parts) == 4 and parts[0] == layout.BATCHES_DIR and parts[2] == layout.CITATIONS_DIR:
        if entry.kind == "file" and _CITATION_FILE.match(name):
            return "citation"
        if entry.kind == "file" and _ANY_TEMP.match(name):
            return "batchtemp"
    return "unexpected"


# ---- records -------------------------------------------------------------------------------


def _parse(model: type[BaseModel], data: bytes, what: str, problems: list[Problem]) -> Any:
    """The record, only if the bytes are strict JSON in exactly the project's serialization."""
    try:
        record = model.model_validate(loads_strict(data), strict=True)
        if layout.serialize(record) != data:
            raise ValueError("not in the canonical project serialization")
    except ValueError as error:  # includes FileProblem and pydantic validation errors
        problems.append(("unverified", f"{what} is not a valid record: {str(error)[:200]}"))
        return None
    return record


@dataclass
class _Artifacts:
    """Rehearsal-only files found below the batches and the run directories found."""

    binding: dict[str, bytes | None] = field(default_factory=dict)
    rosters: dict[str, dict[str, bytes | None]] = field(default_factory=dict)
    citations: dict[str, dict[str, bytes | None]] = field(default_factory=dict)
    runs: dict[str, bytes | None] = field(default_factory=dict)  # run id -> run.json bytes
    runs_dir: bool = False

    def add(self, role: str, path: str, data: bytes | None) -> None:
        parts = path.split("/")
        batch, name = parts[1], parts[-1]
        if role == "binding":
            self.binding[batch] = data
        elif role == "roster":
            self.rosters.setdefault(batch, {})[name[:-5]] = data
        else:
            self.citations.setdefault(batch, {})[name] = data


@dataclass
class RegistryCheck:
    problems: list[Problem]
    events: dict[int, RegistryEvent | None]


def check_registry(root_id: str, files: dict[int, bytes | None]) -> RegistryCheck:
    """Check a registry given the bytes of each event file by number (None: unreadable)."""
    problems: list[Problem] = []
    events: dict[int, RegistryEvent | None] = {}
    if not files:
        problems.append(("incomplete", "the registry has no events"))
        return RegistryCheck(problems, events)
    for missing in sorted(set(range(1, max(files) + 1)) - set(files)):
        problems.append(("incomplete", f"registry event {missing} is missing"))
    for number in sorted(files):
        data = files[number]
        events[number] = None
        if data is None:
            continue
        event = _parse(RegistryEvent, data, f"registry event {number}", problems)
        if event is None:
            continue
        events[number] = event
        if event.root_id != root_id:
            problems.append(("conflicting", f"registry event {number} belongs to another root"))
        elif event.sequence != number:
            problems.append(("conflicting", f"registry event {number} claims {event.sequence}"))
        elif number > 1 and event.event_type == "root_initialized":
            problems.append(("conflicting", f"event {number} repeats the genesis event type"))
        elif number == 1 and event.event_type != "root_initialized":
            problems.append(("conflicting", "the first event is not the genesis event"))
        previous = files.get(number - 1)
        if previous is not None and event.previous_event_sha256 != sha256_bytes(previous):
            problems.append(("unverified", f"registry event {number} breaks the hash chain"))
    return RegistryCheck(problems, events)


def _read(root: Path, path: str, problems: list[Problem]) -> bytes | None:
    try:
        return read_regular(root, path)
    except FileProblem as error:
        problems.append(("unverified", f"{path} cannot be read: {error}"))
        return None


def _check_location(root: Path, authorization: StorageAuthorization) -> None:
    if str(root) != authorization.authorized_root or os.path.realpath(root) != str(root):
        raise AuthorityNotEstablished("root_location", "not the approved, link-free location")


def _check_root_record(record: RootRecord, authorization: StorageAuthorization) -> None:
    if record.root_id != layout.root_id_for(authorization):
        raise AuthorityNotEstablished("root_identity")
    if record.root_kind != authorization.root_kind:
        raise AuthorityNotEstablished("root_kind")
    approved = (
        layout.authorization_sha256(authorization),
        authorization.corpus_content_sha256,
        authorization.benchmark_release_sha256,
        authorization.matrix_id,
        authorization.layout_version,
    )
    recorded = (
        record.authorization_sha256,
        record.corpus_content_sha256,
        record.benchmark_release_sha256,
        record.matrix_id,
        record.layout_version,
    )
    if recorded != approved:
        raise AuthorityNotEstablished("scope")


def _check_anchor(anchor: RootAnchor, authorization: StorageAuthorization) -> None:
    identity = (
        layout.root_id_for(authorization),
        authorization.root_kind,
        layout.authorization_sha256(authorization),
    )
    if (anchor.root_id, anchor.root_kind, anchor.authorization_sha256) != identity:
        raise AuthorityNotEstablished("anchor_identity")


def verify_root(
    root: Path,
    authorization: StorageAuthorization,
    anchor_dir: Path,
    witness_dir: Path | None = None,
    checkpoint: HeadCheckpoint | None = None,
) -> RootVerdict:
    """Verify the complete inventory of `root` against the approved `authorization`.

    What each part of the verdict establishes. Internal chain validity: every event parses, the
    hashes link and the sequence has no gap. Consistency with retained witnesses (`witness_dir`):
    the registry holds the events the witnesses recorded; this catches truncation or edits of
    one of two stores, not a rollback of both together, because both live under the same data
    root. Continuity with a checkpoint: the registry still contains the event a caller
    pinned; the checkpoint is the caller's claim and is not evidence of authority. Nothing here
    proves the latest authorized head, so a registry beyond genesis never yields a positive
    history class and `head_authority` stays `not_established`.
    """
    _check_location(root, authorization)
    rehearsal = authorization.root_kind == "rehearsal"
    if not os.path.lexists(root):
        raise AuthorityNotEstablished("not_initialized", "no storage at the approved location")
    try:
        entries = _scan(root, rehearsal)
    except OSError as error:
        raise AuthorityNotEstablished("root_location", str(error)) from error
    if not entries:
        raise AuthorityNotEstablished("not_initialized", "the approved location is empty")

    expected_id = layout.root_id_for(authorization)
    problems: list[Problem] = []
    roles = {entry.path: _role(entry, rehearsal) for entry in entries}
    hashes: dict[str, str] = {}
    registry_bytes: dict[int, bytes | None] = {}
    root_bytes: bytes | None = None
    manifest_bytes: dict[str, bytes | None] = {}
    batch_dirs: set[str] = set()
    artifacts = _Artifacts()

    for entry in entries:
        role = roles[entry.path]
        if role == "unexpected":
            problems.append(("unexpected", f"unexpected {entry.kind} {entry.path}"))
        elif role == "wrongkind":
            problems.append(("unverified", f"{entry.path} is a {entry.kind}, not what is required"))
        elif role in ("temp", "batchtemp"):
            problems.append(("incomplete", f"unfinished publication {entry.path}"))
        elif role == "batchdir":
            batch_dirs.add(entry.path.rpartition("/")[2])
        elif role == "manifest":
            batch = entry.path.split("/")[1]
            manifest_bytes[batch] = data = _read(root, entry.path, problems)
            if data is not None:
                hashes[entry.path] = sha256_bytes(data)
        elif role in ("binding", "roster", "citation"):
            data = _read(root, entry.path, problems)
            if data is not None:
                hashes[entry.path] = sha256_bytes(data)
            artifacts.add(role, entry.path, data)
        elif role == "dir" and entry.path == layout.RUNS_DIR:
            artifacts.runs_dir = True
        elif role == "rundir":
            artifacts.runs[entry.path.rpartition("/")[2]] = _read(
                root, entry.path + "/run.json", problems
            )
        elif role == "lock":
            data = _read(root, entry.path, problems)
            if data:
                problems.append(("unexpected", "the lock file has content"))
            elif data is not None:
                hashes[entry.path] = sha256_bytes(data)
        elif role == "holder":
            pass  # the last session's identity: volatile, read by the next session, not evidence
        elif role == "root":
            root_bytes = _read(root, entry.path, problems)
        elif role == "event":
            number = int(entry.path.rpartition("/")[2][:6])
            registry_bytes[number] = data = _read(root, entry.path, problems)
            if data is not None:
                hashes[entry.path] = sha256_bytes(data)

    for required in (layout.ROOT_FILE, layout.REGISTRY_DIR):
        if required not in roles:
            problems.append(("incomplete", f"{required} is missing"))

    root_record = None
    if root_bytes is not None:
        hashes[layout.ROOT_FILE] = sha256_bytes(root_bytes)
        root_record = _parse(RootRecord, root_bytes, layout.ROOT_FILE, problems)
    if root_record is not None:
        _check_root_record(root_record, authorization)

    registry = check_registry(expected_id, registry_bytes)
    problems += registry.problems
    genesis = registry.events.get(1)

    anchor_name = expected_id + layout.ANCHOR_SUFFIX
    anchor_bytes = None
    anchor = None
    if not os.path.lexists(anchor_dir / anchor_name):
        problems.append(("incomplete", "the root anchor is missing"))
    else:
        try:
            anchor_bytes = read_regular(anchor_dir, anchor_name)
        except FileProblem as error:
            problems.append(("unverified", f"the root anchor cannot be read: {error}"))
        if anchor_bytes is not None:
            anchor = _parse(RootAnchor, anchor_bytes, "the root anchor", problems)
    if anchor is not None:
        _check_anchor(anchor, authorization)
        if root_record is not None and anchor.root_file_sha256 != hashes[layout.ROOT_FILE]:
            problems.append(("conflicting", "the anchor pins a different ROOT.json"))
        if genesis is not None and anchor.genesis_event_sha256 != hashes.get(
            "registry/000001.json"
        ):
            problems.append(("conflicting", "the anchor pins a different genesis event"))
    if genesis is not None and root_record is not None:
        cited = [(c.role, c.sha256) for c in genesis.evidence]
        wanted = [
            ("root_file", hashes[layout.ROOT_FILE]),
            ("storage_authorization", layout.authorization_sha256(authorization)),
        ]
        if cited != wanted:
            problems.append(("conflicting", "the genesis event cites other root records"))

    head = max(registry_bytes) if registry_bytes else None
    batches, reconcile = _check_batches(
        registry.events,
        hashes,
        manifest_bytes,
        batch_dirs,
        authorization,
        expected_id,
        problems,
        artifacts if rehearsal else None,
    )
    if layout.BATCHES_DIR in roles and not batches:
        problems.append(("unexpected", "batches directory without any reservation"))
    unwitnessed = False
    if witness_dir is not None and registry_bytes:
        unwitnessed = _check_witnesses(expected_id, registry_bytes, witness_dir, problems)
    continuity = None
    if checkpoint is not None:
        continuity = _check_checkpoint(checkpoint, expected_id, registry_bytes, problems)
    code = _classify(problems)
    beyond_genesis = head is not None and head > 1
    if code == NO_PRIOR and beyond_genesis:
        problems.append(("incomplete", HEAD_NOT_ESTABLISHED_TEXT.partition(": ")[2]))
        code = INCOMPLETE
    snapshot: dict[str, Any] = {
        "layout_version": layout.LAYOUT_VERSION,
        "root_kind": authorization.root_kind,
        "root_id": expected_id,
        "authorization_sha256": layout.authorization_sha256(authorization),
        "anchor_sha256": None if anchor_bytes is None else sha256_bytes(anchor_bytes),
        "registry_head": None
        if head is None or registry_bytes[head] is None
        else {
            "sequence": head,
            "event_file_sha256": hashes[f"{layout.REGISTRY_DIR}/{head:06d}.json"],
        },
        "entries": [
            {"path": e.path, "kind": e.kind, "sha256": hashes.get(e.path)} for e in entries
        ],
        "problems": [f"{kind}: {detail}" for kind, detail in problems],
        "classification": code,
    }
    return RootVerdict(
        code=code,
        root_kind=authorization.root_kind,
        root_id=expected_id,
        registry_head_sequence=head,
        anchor_sha256=snapshot["anchor_sha256"],
        inventory=snapshot,
        problems=tuple(snapshot["problems"]),
        batches=batches,
        reconcile=reconcile,
        unwitnessed_head=unwitnessed,
        head_authority="not_established" if beyond_genesis else "genesis_anchored",
        checkpoint_continuity=continuity,
    )


def _check_checkpoint(
    checkpoint: HeadCheckpoint,
    root_id: str,
    registry_bytes: dict[int, bytes | None],
    problems: list[Problem],
) -> bool:
    data = registry_bytes.get(checkpoint.sequence)
    if checkpoint.root_id != root_id:
        problems.append(("conflicting", "the checkpoint belongs to another root"))
    elif data is None or sha256_bytes(data) != checkpoint.event_file_sha256:
        problems.append(
            ("conflicting", "the registry does not contain the checkpointed event (rollback?)")
        )
    else:
        return True
    return False


def _witness_files(root_id: str, witness_dir: Path) -> dict[int, str]:
    pattern = re.compile(rf"^{root_id}\.(\d{{6}}){re.escape(layout.WITNESS_SUFFIX)}$")
    found = {}
    if witness_dir.is_dir():
        for name in os.listdir(witness_dir):
            if match := pattern.match(name):
                found[int(match.group(1))] = name
    return found


def _check_witnesses(
    root_id: str,
    registry_bytes: dict[int, bytes | None],
    witness_dir: Path,
    problems: list[Problem],
) -> bool:
    """Compare the registry with the retained witnesses; True if only the head lacks one.

    A genesis-only root needs no witness (the anchor pins it) unless witnesses already exist.
    """
    head = max(registry_bytes)
    names = _witness_files(root_id, witness_dir)
    if head == 1 and not names:
        return False
    if names and max(names) > head:
        problems.append(
            ("conflicting", "witnesses exist beyond the registry head (rollback or truncation?)")
        )
    previous: bytes | None = None
    gaps = []
    for number in range(1, head + 1):
        if number not in names:
            gaps.append(number)
            previous = None
            continue
        try:
            data = read_regular(witness_dir, names[number])
        except FileProblem as error:
            problems.append(("unverified", f"witness {number} cannot be read: {error}"))
            previous = None
            continue
        probe: list[Problem] = []
        witness = _parse(HeadWitness, data, f"witness {number}", probe)
        problems += probe
        if witness is None:
            previous = None
            continue
        event = registry_bytes.get(number)
        if witness.root_id != root_id or witness.sequence != number:
            problems.append(("conflicting", f"witness {number} belongs elsewhere"))
        elif event is not None and witness.event_file_sha256 != sha256_bytes(event):
            problems.append(("conflicting", f"witness {number} records a different event"))
        elif (
            number > 1
            and (number - 1) not in gaps
            and previous is not None
            and (witness.previous_witness_sha256 != sha256_bytes(previous))
        ):
            problems.append(("conflicting", f"witness {number} breaks the witness chain"))
        previous = data
    if gaps == [head] and head > 1:
        problems.append(("incomplete", f"registry event {head} has no witness"))
        return True
    for number in gaps:
        problems.append(("incomplete", f"witness {number} is missing"))
    return False


def _check_batches(
    events: dict[int, RegistryEvent | None],
    hashes: dict[str, str],
    manifests: dict[str, bytes | None],
    batch_dirs: set[str],
    authorization: StorageAuthorization,
    root_id: str,
    problems: list[Problem],
    artifacts: _Artifacts | None = None,
) -> tuple[dict[str, BatchRecord], dict[str, tuple[str, str]]]:
    """Fold the batch events into per-batch states and check them against the files.

    `artifacts` is given for rehearsal roots only; for a final-test root the two rehearsal
    event types are unauthorized, exactly like the types nothing may publish.
    """
    rehearsal = artifacts is not None
    claimed_runs: set[str] = set()
    state: dict[str, dict[str, Any]] = {}
    plans: set[str] = set()
    for number in sorted(events):
        event = events[number]
        if event is None or number == 1 or event.event_type == "root_initialized":
            continue
        kind = event.event_type
        if kind in layout.FUTURE_EVENTS or (kind in layout.REHEARSAL_EVENTS and not rehearsal):
            problems.append(("unexpected", f"event {number} has the unauthorized type {kind}"))
            continue
        cited = {c.role: c.sha256 for c in event.evidence}
        if kind in layout.REHEARSAL_EVENTS:
            wanted_roles = layout.REHEARSAL_ROLES[kind]
            if [c.role for c in event.evidence] != wanted_roles:
                problems.append(("conflicting", f"event {number} does not cite {wanted_roles}"))
                continue
        batch = cited["batch_id"]
        file_sha = hashes.get(f"{layout.REGISTRY_DIR}/{number:06d}.json")
        short = batch[:12]
        if kind == "batch_reserved":
            if batch in state:
                problems.append(("conflicting", f"batch {short} is reserved twice"))
                continue
            plans.add(cited["campaign_plan"])
            if len(plans) > 1:
                problems.append(("conflicting", "reservations cite more than one campaign plan"))
            for other, claimed in state.items():
                if claimed["ownership"] == cited["ownership"]:
                    problems.append(
                        ("conflicting", f"batches {other[:12]} and {short} claim the same keys")
                    )
            state[batch] = {
                "state": "reserved",
                "plan": cited["campaign_plan"],
                "ownership": cited["ownership"],
                "manifest": cited["batch_manifest"],
                "reservation": file_sha,
                "published": False,
            }
            continue
        info = state.get(batch)
        if info is None:
            problems.append(("conflicting", f"event {number} concerns an unreserved batch {short}"))
            continue
        if kind in layout.REHEARSAL_EVENTS and artifacts is not None:
            _fold_rehearsal(
                kind,
                number,
                batch,
                info,
                cited,
                file_sha,
                artifacts,
                root_id,
                claimed_runs,
                hashes,
                problems,
            )
            continue
        if info["state"] != "reserved":
            problems.append(
                ("conflicting", f"event {number} follows batch {short} {info['state']}")
            )
        elif cited["reservation_event"] != info["reservation"]:
            problems.append(("conflicting", f"event {number} cites another reservation"))
        elif kind == "batch_materialized":
            if cited["batch_manifest"] != info["manifest"]:
                problems.append(("conflicting", f"event {number} cites another manifest"))
            else:
                info["state"] = "materialized"
                info["materialization"] = file_sha
        elif cited["observed_state"] not in (
            observed_state_sha256(None, False),
            observed_state_sha256(info["manifest"], False),
        ):
            problems.append(("conflicting", f"closure {number} records an impossible state"))
        else:
            info["state"] = "closed_incomplete"

    reconcile: dict[str, tuple[str, str]] = {}
    for batch in sorted(batch_dirs - set(state)):
        problems.append(("unexpected", f"directory of the unreserved batch {batch[:12]}"))
    for batch in sorted(set(manifests) - set(state)):
        problems.append(("unexpected", f"manifest of the unreserved batch {batch[:12]}"))
    for batch, info in sorted(state.items()):
        short = batch[:12]
        data = manifests.get(batch)
        clean = False
        if data is not None:
            probe: list[Problem] = []
            manifest = _parse(BatchManifest, data, f"manifest of batch {short}", probe)
            problems += probe
            if manifest is not None:
                clean = _manifest_consistent(
                    manifest, data, batch, info, authorization, root_id, short, problems
                )
        info["published"] = clean
        if info["state"] == "reserved":
            if data is None and batch not in manifests:
                text = f"batch {short} is reserved but its manifest is not published"
                reconcile[batch] = ("reserved_unpublished", f"incomplete: {text}")
                problems.append(("incomplete", text))
            elif clean:
                text = f"batch {short} is published but not materialized"
                reconcile[batch] = ("reserved_published", f"incomplete: {text}")
                problems.append(("incomplete", text))
        elif info["state"] in _PAST_MATERIALIZED and batch not in manifests:
            problems.append(("incomplete", f"materialized batch {short} has no manifest"))
    if artifacts is not None:
        _check_unclaimed(state, artifacts, claimed_runs, problems)
    records = {
        batch: BatchRecord(
            info["state"],
            info["plan"],
            info["ownership"],
            info["manifest"],
            run_id=info.get("run_id"),
            authorization_event_sha256=info.get("auth_event"),
            authorization_event_sequence=info.get("auth_sequence"),
            resumes=info.get("resumes", 0),
        )
        for batch, info in state.items()
    }
    return records, reconcile


_PAST_MATERIALIZED = ("materialized", "inputs_bound", "authorized")


def _canonical(model: type[BaseModel], data: bytes | None) -> Any:
    """The record if `data` is exactly its canonical serialization, else None."""
    if data is None:
        return None
    try:
        record = model.model_validate(loads_strict(data), strict=True)
    except ValueError:
        return None
    return record if layout.serialize(record) == data else None


def _fold_rehearsal(
    kind: str,
    number: int,
    batch: str,
    info: dict[str, Any],
    cited: dict[str, str],
    file_sha: str | None,
    artifacts: _Artifacts,
    root_id: str,
    claimed_runs: set[str],
    hashes: dict[str, str],
    problems: list[Problem],
) -> None:
    short = batch[:12]
    if kind == "batch_resumed":
        if info["state"] != "authorized":
            problems.append(
                ("conflicting", f"event {number} follows batch {short} {info['state']}")
            )
        elif cited["previous_authorization_event"] != info["auth_event"]:
            problems.append(("conflicting", f"event {number} cites another authorizing event"))
        elif _check_resume(number, batch, cited, info, artifacts, hashes, problems):
            info.update(auth_event=file_sha, auth_sequence=number, resumes=info["resumes"] + 1)
        return
    if kind == "batch_inputs_bound":
        if info["state"] != "materialized":
            problems.append(
                ("conflicting", f"event {number} follows batch {short} {info['state']}")
            )
        elif cited["materialization_event"] != info.get("materialization"):
            problems.append(("conflicting", f"event {number} cites another materialization"))
        elif _check_inputs(batch, cited["inputs_binding"], info, artifacts, root_id, problems):
            info.update(
                state="inputs_bound", binding=cited["inputs_binding"], inputs_event=file_sha
            )
        return
    if info["state"] != "inputs_bound":
        problems.append(("conflicting", f"event {number} follows batch {short} {info['state']}"))
    elif cited["inputs_bound_event"] != info.get("inputs_event"):
        problems.append(("conflicting", f"event {number} cites another inputs event"))
    elif _check_authorization(batch, cited, info, artifacts, claimed_runs, problems):
        info.update(state="authorized", auth_event=file_sha, auth_sequence=number, resumes=0)


def _check_inputs(
    batch: str,
    binding_sha: str,
    info: dict[str, Any],
    artifacts: _Artifacts,
    root_id: str,
    problems: list[Problem],
) -> bool:
    short = batch[:12]
    data = artifacts.binding.get(batch)
    if data is None:
        problems.append(("incomplete", f"batch {short} has bound inputs but no binding file"))
        return False
    binding = _canonical(BatchInputBinding, data)
    if sha256_bytes(data) != binding_sha or binding is None:
        problems.append(("conflicting", f"binding of batch {short} differs from its event"))
        return False
    if (binding.batch_id, binding.root_id, binding.manifest_sha256) != (
        batch,
        root_id,
        info["manifest"],
    ) or binding.campaign_plan_sha256 != info["plan"]:
        problems.append(("conflicting", f"binding of batch {short} names another batch or plan"))
        return False
    files = artifacts.rosters.get(batch, {})
    good = True
    for ref in binding.rosters:
        roster = _canonical(ConditionRoster, files.get(ref.roster_sha256))
        if ref.roster_sha256 not in files or files[ref.roster_sha256] is None:
            problems.append(
                ("incomplete", f"roster {ref.roster_sha256[:12]} of {short} is missing")
            )
            good = False
        elif roster is None or roster_sha256(roster) != ref.roster_sha256:
            problems.append(
                ("conflicting", f"roster {ref.roster_sha256[:12]} of {short} is invalid")
            )
            good = False
        elif (roster.batch_id, roster.condition, len(roster.rows)) != (
            batch,
            ref.condition,
            ref.row_count,
        ):
            problems.append(("conflicting", f"roster {ref.roster_sha256[:12]} disagrees"))
            good = False
    for extra in sorted(set(files) - {r.roster_sha256 for r in binding.rosters}):
        problems.append(("unexpected", f"roster {extra[:12]} is not in the binding of {short}"))
        good = False
    return good


def _check_authorization(
    batch: str,
    cited: dict[str, str],
    info: dict[str, Any],
    artifacts: _Artifacts,
    claimed_runs: set[str],
    problems: list[Problem],
) -> bool:
    short = batch[:12]
    files = artifacts.citations.get(batch, {})
    data = files.get(layout.CITATION_NAME)
    if data is None:
        problems.append(("incomplete", f"authorized batch {short} has no citation file"))
        return False
    citation = _canonical(AuthorizationCitation, data)
    if citation is None or sha256_bytes(data) != cited["authorization_citation"]:
        problems.append(("conflicting", f"citation of batch {short} differs from its event"))
        return False
    if (
        citation.batch_id != batch
        or citation.inputs_set_sha256 != info["binding"]
        or citation.readiness_report_sha256 != cited["readiness_report"]
        or citation_sha256(citation) != cited["authorization_citation"]
    ):
        problems.append(("conflicting", f"citation of batch {short} is about other evidence"))
        return False
    matches = [
        run_id
        for run_id, run_bytes in artifacts.runs.items()
        if run_bytes is not None and sha256_bytes(run_bytes) == cited["run_record"]
    ]
    if not matches:
        problems.append(("incomplete", f"authorized batch {short} has no run record"))
        return False
    run_id = matches[0]
    record = _canonical(FinalTestRunRecord, artifacts.runs[run_id])
    if record is None or (
        record.run_id,
        record.batch_id,
        record.inputs_set_sha256,
        record.authorization_citation_sha256,
        record.root_kind,
    ) != (run_id, batch, info["binding"], cited["authorization_citation"], "rehearsal"):
        problems.append(("conflicting", f"run record of batch {short} is about other evidence"))
        return False
    claimed_runs.add(run_id)
    info.update(
        run_id=run_id,
        run_record=cited["run_record"],
        chain=[citation],
        assessments={},
        last_citation=cited["authorization_citation"],
        accepted={layout.CITATION_NAME},
    )
    return True


def _check_resume(
    number: int,
    batch: str,
    cited: dict[str, str],
    info: dict[str, Any],
    artifacts: _Artifacts,
    hashes: dict[str, str],
    problems: list[Problem],
) -> bool:
    """One `batch_resumed` event against its citation, continuity assessment and the chain."""
    short = batch[:12]
    ordinal = info["resumes"] + 2
    files = artifacts.citations.get(batch, {})
    citation_data = files.get(layout.citation_name(ordinal))
    assessment_data = files.get(layout.continuity_name(ordinal))
    if citation_data is None or assessment_data is None:
        problems.append(("incomplete", f"resume event {number} of {short} lacks its records"))
        return False
    citation = _canonical(AuthorizationCitation, citation_data)
    assessment = _canonical(ContinuityAssessment, assessment_data)
    if (
        citation is None
        or assessment is None
        or sha256_bytes(citation_data) != cited["authorization_citation"]
        or sha256_bytes(assessment_data) != cited["continuity_assessment"]
    ):
        problems.append(("conflicting", f"records of resume event {number} differ from it"))
        return False
    head = hashes.get(f"{layout.REGISTRY_DIR}/{number - 1:06d}.json")
    if (
        citation.kind != "resume"
        or citation.batch_id != batch
        or citation.inputs_set_sha256 != info["binding"]
        or citation.readiness_report_sha256 != cited["readiness_report"]
        or citation.continuity_assessment_sha256 != cited["continuity_assessment"]
        or cited["run_record"] != info["run_record"]
        # the head the assessment was made against is exactly the event before this one
        or (citation.registry_head_sequence, citation.registry_head_event_sha256)
        != (number - 1, head)
    ):
        problems.append(("conflicting", f"resume citation of {short} is about other evidence"))
        return False
    assessments = info["assessments"] | {cited["continuity_assessment"]: assessment}
    try:
        verify_citation_chain([*info["chain"], citation], assessments)
    except CitationChainError as error:
        problems.append(("conflicting", f"citation chain of {short} is invalid: {error}"))
        return False
    info["chain"].append(citation)
    info["assessments"] = assessments
    info["last_citation"] = cited["authorization_citation"]
    info["accepted"] |= {layout.citation_name(ordinal), layout.continuity_name(ordinal)}
    return True


def _check_unclaimed(
    state: dict[str, dict[str, Any]],
    artifacts: _Artifacts,
    claimed_runs: set[str],
    problems: list[Problem],
) -> None:
    if artifacts.runs_dir and not any(i["state"] == "authorized" for i in state.values()):
        problems.append(("unexpected", "runs directory without an authorized batch"))
    for run_id in sorted(set(artifacts.runs) - claimed_runs):
        problems.append(("unexpected", f"run directory {run_id[:12]} is not an authorized run"))
    for batch, files in sorted(artifacts.citations.items()):
        accepted = state.get(batch, {}).get("accepted", set())
        if state.get(batch, {}).get("state") == "authorized":
            for extra in sorted(set(files) - accepted):
                problems.append(
                    ("unexpected", f"citation {extra} of batch {batch[:12]} is not authorized")
                )
    for batch in sorted(set(artifacts.binding) | set(artifacts.rosters) | set(artifacts.citations)):
        known = state.get(batch, {}).get("state")
        if known is None:
            problems.append(("unexpected", f"inputs of the unreserved batch {batch[:12]}"))
        elif batch in artifacts.binding and known in (
            "reserved",
            "materialized",
            "closed_incomplete",
        ):
            problems.append(("unexpected", f"binding of batch {batch[:12]} has no inputs event"))
        elif batch in artifacts.citations and known != "authorized":
            problems.append(("unexpected", f"citation of batch {batch[:12]} has no event"))


def _manifest_consistent(
    manifest: BatchManifest,
    data: bytes,
    batch: str,
    info: dict[str, Any],
    authorization: StorageAuthorization,
    root_id: str,
    short: str,
    problems: list[Problem],
) -> bool:
    if sha256_bytes(data) != info["manifest"]:
        problems.append(("conflicting", f"manifest of batch {short} differs from its reservation"))
    elif (manifest.batch_id, manifest.root_id) != (batch, root_id):
        problems.append(("conflicting", f"manifest of batch {short} names another batch or root"))
    elif (manifest.campaign_plan_sha256, manifest.ownership) != (info["plan"], info["ownership"]):
        problems.append(("conflicting", f"manifest of batch {short} differs from its claim"))
    elif (manifest.benchmark_release_sha256, manifest.corpus_content_sha256) != (
        authorization.benchmark_release_sha256,
        authorization.corpus_content_sha256,
    ):
        problems.append(("conflicting", f"manifest of batch {short} is for another benchmark"))
    else:
        return True
    return False


def _classify(problems: list[Problem]) -> str:
    entries = [
        HistoryEntry(f"unexpected:{detail}", "unexpected", False)
        for kind, detail in problems
        if kind == "unexpected"
    ] + [
        HistoryEntry(KNOWN_ARTIFACTS, "registry", False)
        for kind, _ in problems
        if kind == "unverified"
    ]
    return classify_history(
        HistoryInventory(
            complete=not any(kind == "incomplete" for kind, _ in problems),
            entries=tuple(entries),
            authorized_batch=None,
            approved_batches=frozenset({KNOWN_ARTIFACTS}),
            contradictions=any(kind == "conflicting" for kind, _ in problems),
        )
    )


# ---- lock ----------------------------------------------------------------------------------


_HELD_HERE: set[Path] = set()  # roots whose registry lock this process holds right now


def registry_lock_held_here(root: Path) -> bool:
    return root in _HELD_HERE


@dataclass(frozen=True)
class HeldLock:
    """Proof, by construction only, that the caller is inside `registry_lock` of `root`."""

    root: Path


@contextmanager
def registry_lock(root: Path) -> Iterator[HeldLock]:
    """Exclusive non-blocking `flock` on the registry lock file.

    Valid on a local file system on one host. It is not evidence that the lock works on a
    shared or network file system. The yielded token is a structural reminder for the
    functions that must run under the lock; it is not a security credential.
    """
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(root / layout.LOCK_FILE, flags, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RegistryBusy(f"another holder has the registry lock of {root}") from None
        _HELD_HERE.add(root)
        try:
            yield HeldLock(root)
        finally:
            _HELD_HERE.discard(root)
    finally:
        os.close(descriptor)


# ---- initialization ------------------------------------------------------------------------


def _publish(target: Path, data: bytes) -> None:
    publish_bytes(target, data, sha256_bytes(data))
    target.chmod(0o444)


def initialize_root(
    root: Path,
    authorization: StorageAuthorization,
    anchor_dir: Path,
    *,
    now: Callable[[], str] = utc_now,
) -> RootVerdict:
    """Create a rehearsal root and return its verified state; read back a verified one.

    Publishes ROOT.json, the genesis event and the anchor, in that order. They are separate
    files, so an interruption leaves a partial root that `verify_root` reports as incomplete;
    it is never completed, cleaned up or overwritten here.
    """
    if authorization.root_kind != "rehearsal":
        raise InitializationRefused(
            "only a rehearsal root can be initialized in this pass; a final_test root needs an "
            "approved G11 artifact and a separately reviewed act"
        )
    return _initialize(root, authorization, anchor_dir, now)


def _initialize(
    root: Path,
    authorization: StorageAuthorization,
    anchor_dir: Path,
    now: Callable[[], str],
) -> RootVerdict:
    """The initialization steps, for any kind. Not exported: tests build synthetic `final_test`
    roots with it, production code has no caller."""
    if str(root) != authorization.authorized_root:
        raise InitializationRefused("the root is not at the approved location")
    anchors = Path(os.path.abspath(anchor_dir))
    if anchors == root or root in anchors.parents or anchors in root.parents:
        raise InitializationRefused("the anchor directory must lie outside the root")
    if os.path.lexists(root):
        if root.is_symlink() or not root.is_dir():
            raise InitializationRefused("the approved location is not a plain directory")
        if any(root.iterdir()):
            try:
                existing = verify_root(root, authorization, anchors)
            except AuthorityNotEstablished as error:
                raise InitializationRefused(
                    f"existing storage is not this root: {error}"
                ) from error
            if existing.code == NO_PRIOR:
                return existing
            raise InitializationRefused(
                f"partial or inconsistent storage ({existing.code}); evidence is preserved"
            )
    else:
        root.mkdir()
    (root / layout.REGISTRY_DIR).mkdir()
    (root / layout.LOCKS_DIR).mkdir()

    root_id = layout.root_id_for(authorization)
    authorization_sha = layout.authorization_sha256(authorization)
    stamp = now()
    root_bytes = layout.serialize(
        RootRecord(
            layout_version=layout.LAYOUT_VERSION,
            root_kind=authorization.root_kind,
            root_id=root_id,
            authorization_sha256=authorization_sha,
            corpus_content_sha256=authorization.corpus_content_sha256,
            benchmark_release_sha256=authorization.benchmark_release_sha256,
            matrix_id=authorization.matrix_id,
            created_utc=stamp,
        )
    )
    event_bytes = layout.serialize(
        layout.make_event(
            root_id=root_id,
            sequence=1,
            event_type="root_initialized",
            previous_event_sha256=None,
            evidence=[
                EvidenceCitation(role="root_file", sha256=sha256_bytes(root_bytes)),
                EvidenceCitation(role="storage_authorization", sha256=authorization_sha),
            ],
            created_utc=stamp,
        )
    )
    anchor_bytes = layout.serialize(
        RootAnchor(
            layout_version=layout.LAYOUT_VERSION,
            root_kind=authorization.root_kind,
            root_id=root_id,
            authorization_sha256=authorization_sha,
            root_file_sha256=sha256_bytes(root_bytes),
            genesis_event_sha256=sha256_bytes(event_bytes),
            created_utc=stamp,
        )
    )
    with registry_lock(root):
        _publish(root / layout.ROOT_FILE, root_bytes)
        _publish(root / layout.REGISTRY_DIR / "000001.json", event_bytes)
        anchors.mkdir(parents=True, exist_ok=True)
        _publish(anchors / (root_id + layout.ANCHOR_SUFFIX), anchor_bytes)
    verdict = verify_root(root, authorization, anchors)
    if verdict.code != NO_PRIOR:
        raise InitializationRefused(f"the new root does not verify: {verdict.code}")
    return verdict


__all__ = [
    "INCOMPLETE",
    "NO_PRIOR",
    "AuthorityNotEstablished",
    "InitializationRefused",
    "RegistryBusy",
    "RootVerdict",
    "check_registry",
    "initialize_root",
    "registry_lock",
    "verify_root",
]
