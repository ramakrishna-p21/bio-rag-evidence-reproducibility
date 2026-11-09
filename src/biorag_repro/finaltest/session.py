"""One continuous, exclusive session over a rehearsal root.

A session spans a batch start and the execution that follows it, so an authorization made at
the start cannot be reused after the process that made it has gone: when the session object
is closed, or the process ends, the lock is released and every guarded operation of that
session refuses. A new session needs a new authorization.

Lock order, always: session lock first, then registry lock. The session lock keeps other
sessions out for its whole lifetime; the registry lock (`inventory.registry_lock`) is taken
only around registry verification and mutation. Opening a session while this process holds
the registry lock is refused, and `Session.registry()` is the sanctioned way to take it inside.

Worker record. Holding the lock proves that no cooperating process holds it now. It does not
prove that the previous session's backend worker has stopped writing, so recovery of an attempt
left running needs more. When the first attempt of a session begins, `begin_worker` writes
`locks/session.holder` (pid, process start time, host, state "open"); a clean `close` rewrites
it as "closed". The next session reads the previous record before it replaces it and
`prove_previous_worker_stopped` accepts only: a record closed by its owner, or an open record of
this host whose process is gone or is a different process (pid reuse is caught by the start
time in /proc). A missing, unreadable, foreign-host or still-live record is not proof and
recovery is refused. Every write of a final-test store also passes the session gate, which
refuses once the session is closed or lost, so a stopped session cannot write either.

What this is. `flock` on a local file system of one host. Ownership is checked by evidence,
not by the lock file's existence: the descriptor must still refer to the file at its path,
and an independent descriptor must be unable to take the lock. Advisory locks bind only
cooperating processes; on network or shared file systems they may not work at all, and no
claim about such storage is made here.
"""

import fcntl
import json
import os
import socket
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path

from biorag_repro.finaltest import layout
from biorag_repro.finaltest.inventory import HeldLock, registry_lock, registry_lock_held_here


class SessionBusy(Exception):
    """Another session holds the root."""


class SessionLost(Exception):
    """This session no longer holds its lock, or has been closed."""


class LockOrderViolation(Exception):
    """The registry lock was held before the session lock was requested."""


class WorkerNotStopped(Exception):
    """The previous worker cannot be shown to have stopped; recovery is refused."""


@dataclass(frozen=True)
class WorkerStopped:
    """Evidence that the previous worker cannot write. Made only by the session that checked."""

    reason: str


def _process_start(pid: int) -> str | None:
    """Start time of a process in clock ticks since boot, or None if there is no such process."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as handle:
            stat = handle.read().decode("ascii", "replace")
    except FileNotFoundError:
        return None
    # the command name may contain spaces and parentheses; the fields after it are fixed
    return stat.rpartition(")")[2].split()[19]


def _holder_record(state: str) -> bytes:
    start = _process_start(os.getpid())
    record = {"pid": os.getpid(), "start": start, "host": socket.gethostname(), "state": state}
    return json.dumps(record, sort_keys=True).encode()


@dataclass
class Session:
    root: Path
    _descriptor: int | None
    head: tuple[int, str] | None = None  # (sequence, event file sha256) this session expects
    previous_holder: bytes | None = None  # holder record found when this session opened
    _worker: bool = False

    def begin_worker(self) -> None:
        """Record this process as the worker of the session (first attempt only)."""
        if self._worker:
            return
        self._write_holder("open")
        self._worker = True

    def _write_holder(self, state: str) -> None:
        path = self.root / layout.HOLDER_FILE
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW | os.O_CLOEXEC
        descriptor = os.open(path, flags, 0o644)
        try:
            os.write(descriptor, _holder_record(state))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def prove_previous_worker_stopped(self) -> WorkerStopped:
        """Evidence that the worker of the previous session cannot write, or `WorkerNotStopped`."""
        self.assert_held()
        data = self.previous_holder
        if data is None:
            raise WorkerNotStopped("no record of the previous worker")
        try:
            record = json.loads(data)
            state, host, pid, start = (
                record["state"],
                record["host"],
                record["pid"],
                record["start"],
            )
            if state not in ("open", "closed") or not isinstance(pid, int):
                raise ValueError(state)
        except (ValueError, KeyError, TypeError):
            raise WorkerNotStopped("the record of the previous worker is unreadable") from None
        if state == "closed":
            return WorkerStopped("the previous session was closed by its owner")
        if host != socket.gethostname():
            raise WorkerNotStopped("the previous worker ran on another host")
        if not os.path.isdir("/proc/self"):
            raise WorkerNotStopped("process identity cannot be read on this system")
        current = _process_start(pid)
        if current is not None and current == start:
            raise WorkerNotStopped(f"the previous worker (pid {pid}) is still running")
        return WorkerStopped("the previous worker process is gone")

    @property
    def active(self) -> bool:
        return self._descriptor is not None

    def assert_held(self) -> None:
        """Raise `SessionLost` unless this session verifiably holds the session lock."""
        if self._descriptor is None:
            raise SessionLost("the session is closed")
        path = self.root / layout.SESSION_LOCK_FILE
        try:
            same = os.fstat(self._descriptor).st_ino == os.stat(path, follow_symlinks=False).st_ino
        except OSError as error:
            raise SessionLost(f"the session lock file is gone: {error}") from error
        if not same:
            raise SessionLost("the session lock file was replaced")
        probe = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            try:
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return  # held, by this descriptor
            raise SessionLost("the session lock is not held")
        finally:
            os.close(probe)

    def registry(self) -> AbstractContextManager[HeldLock]:
        """The registry lock, taken inside this session (the only permitted order)."""
        self.assert_held()
        return registry_lock(self.root)

    def close(self) -> None:
        if self._descriptor is not None:
            if self._worker:
                self._write_holder("closed")
            os.close(self._descriptor)
            self._descriptor = None


@contextmanager
def open_session(root: Path) -> Iterator[Session]:
    """Take the session lock of `root` without waiting, or raise `SessionBusy`."""
    if registry_lock_held_here(root):
        raise LockOrderViolation("the registry lock is held; the session lock comes first")
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(root / layout.SESSION_LOCK_FILE, flags, 0o644)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise SessionBusy(f"another session holds {root}") from None
    session = Session(root, descriptor)
    try:
        session.previous_holder = (root / layout.HOLDER_FILE).read_bytes()
    except FileNotFoundError:
        pass
    try:
        yield session
    finally:
        session.close()
