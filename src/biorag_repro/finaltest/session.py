"""One continuous, exclusive session over a rehearsal root.

A session spans a batch start and the execution that follows it, so an authorization made at
the start cannot be reused after the process that made it has gone: when the session object
is closed, or the process ends, the lock is released and every guarded operation of that
session refuses. A new session needs a new authorization.

Lock order, always: session lock first, then registry lock. The session lock keeps other
sessions out for its whole lifetime; the registry lock (`inventory.registry_lock`) is taken
only around registry verification and mutation. Opening a session while this process holds
the registry lock is refused, and `Session.registry()` is the sanctioned way to take it inside.

What this is. `flock` on a local file system of one host. Ownership is checked by evidence,
not by the lock file's existence: the descriptor must still refer to the file at its path,
and an independent descriptor must be unable to take the lock. Advisory locks bind only
cooperating processes; on network or shared file systems they may not work at all, and no
claim about such storage is made here.
"""

import fcntl
import os
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


@dataclass
class Session:
    root: Path
    _descriptor: int | None
    head: tuple[int, str] | None = None  # (sequence, event file sha256) this session expects

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
        yield session
    finally:
        session.close()
