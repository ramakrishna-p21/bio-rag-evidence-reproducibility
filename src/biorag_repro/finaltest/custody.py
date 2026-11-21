"""Recovery proof contract for a separately administered append-only custodian.

The implementation here is an in-memory test custodian. Production must supply a
separately administered durable implementation and verify its authoritative head.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file


class CustodyRefused(ValueError):
    """The publication cannot be authenticated against independent custody."""


@dataclass(frozen=True)
class Commitment:
    batch_id: str
    manifest_sha256: str
    predecessor_head: str
    path: str
    content_sha256: str
    head: str


class Custodian(Protocol):
    def authoritative_head(self) -> str: ...

    def entries(self) -> tuple[Commitment, ...]: ...

    def commit(
        self,
        batch_id: str,
        manifest_sha256: str,
        predecessor_head: str,
        path: str,
        content_sha256: str,
    ) -> Commitment: ...


class SyntheticCustodian:
    """A deterministic test double; its state is not production evidence."""

    def __init__(self, genesis_head: str):
        self._genesis = genesis_head
        self._head = genesis_head
        self._entries: list[Commitment] = []

    def authoritative_head(self) -> str:
        return self._head

    def entries(self) -> tuple[Commitment, ...]:
        return tuple(self._entries)

    def commit(
        self,
        batch_id: str,
        manifest_sha256: str,
        predecessor_head: str,
        path: str,
        content_sha256: str,
    ) -> Commitment:
        if predecessor_head != self._head:
            raise CustodyRefused("stale or divergent custody head")
        if any(entry.path == path and entry.batch_id == batch_id for entry in self._entries):
            raise CustodyRefused("publication identity was already committed")
        body = {
            "batch_id": batch_id,
            "manifest_sha256": manifest_sha256,
            "predecessor_head": predecessor_head,
            "path": path,
            "content_sha256": content_sha256,
        }
        head = sha256_bytes(canonical_json_bytes(body))
        entry = Commitment(**body, head=head)
        self._entries.append(entry)
        self._head = head
        return entry


def verify_custody(
    custodian: Custodian,
    root: Path,
    batch_id: str,
    manifest_sha256: str,
    expected_head: str,
    genesis_head: str,
    anchors: Path | None = None,
) -> dict[str, str]:
    """Classify committed files; any uncommitted or altered byte refuses recovery."""
    if custodian.authoritative_head() != expected_head:
        raise CustodyRefused("authoritative head differs; stale, rolled back or divergent history")
    previous = genesis_head
    committed: dict[str, str] = {}
    states: dict[str, str] = {}
    for entry in custodian.entries():
        if entry.predecessor_head != previous:
            raise CustodyRefused("custody chain is broken")
        body = {
            "batch_id": entry.batch_id,
            "manifest_sha256": entry.manifest_sha256,
            "predecessor_head": entry.predecessor_head,
            "path": entry.path,
            "content_sha256": entry.content_sha256,
        }
        if sha256_bytes(canonical_json_bytes(body)) != entry.head:
            raise CustodyRefused("custody commitment was altered")
        previous = entry.head
        if entry.batch_id != batch_id or entry.manifest_sha256 != manifest_sha256:
            raise CustodyRefused("commitment belongs to another batch or manifest")
        path = Path(entry.path)
        if path.is_absolute() or not path.parts or ".." in path.parts or entry.path in committed:
            raise CustodyRefused("ambiguous publication path")
        committed[entry.path] = entry.content_sha256
        if entry.path.startswith("external/"):
            if anchors is None or len(path.parts) != 2:
                raise CustodyRefused("external commitment has no verified custody location")
            target = anchors / path.name
        else:
            target = root / path
        if target.exists():
            if (
                not target.is_file()
                or target.is_symlink()
                or sha256_file(target) != entry.content_sha256
            ):
                raise CustodyRefused("published bytes differ from the prior commitment")
            states[entry.path] = "published_committed"
        else:
            states[entry.path] = "committed_absent"
    if previous != expected_head:
        raise CustodyRefused("custody head cannot be reconstructed")
    for path in root.rglob("*") if root.exists() else ():
        if path.is_symlink():
            raise CustodyRefused("publication tree contains a link")
        if (
            path.is_file()
            and path.name != "batch.lock"
            and path.relative_to(root).as_posix() not in committed
        ):
            raise CustodyRefused("uncommitted published file")
    return states
