"""Researcher-acknowledged Drive snapshots for the approved final-test campaign."""

import re
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.finaltest.drive_publication import publish_drive_batch, verify_drive_batch
from biorag_repro.readiness.files import loads_strict


class AmendedCustodyRefused(ValueError):
    """The retained head or published copy does not authenticate the campaign."""


def _publish_drive_record(
    target: Path,
    content: bytes,
    expected_sha256: str,
    *,
    reuse_identical: bool = False,
) -> str:
    """Exclusive Drive file creation with exact post-write verification.

    Unlike local hard-link publication, this does not guarantee atomic
    visibility or remote durability on the mounted Drive filesystem.
    """
    if sha256_bytes(content) != expected_sha256:
        raise AmendedCustodyRefused("publication content hash mismatch")

    def verify_existing() -> None:
        if (
            target.is_symlink()
            or not target.is_file()
            or sha256_file(target) != expected_sha256
            or target.read_bytes() != content
        ):
            raise AmendedCustodyRefused("Drive record is missing, incomplete, or conflicting")

    if target.exists() or target.is_symlink():
        if not reuse_identical:
            raise FileExistsError(target)
        verify_existing()
        return expected_sha256

    try:
        with target.open("xb") as stream:
            stream.write(content)
            stream.flush()
    except FileExistsError:
        if not reuse_identical:
            raise

    verify_existing()
    return expected_sha256


def _outside(path: Path, *roots: Path) -> None:
    resolved = path.resolve()
    if any(resolved == root.resolve() or root.resolve() in resolved.parents for root in roots):
        raise AmendedCustodyRefused(
            "researcher acknowledgment must be outside runtime and Drive roots"
        )


def verify_acknowledgment(request: Path, acknowledgment: Path, root: Path, drive: Path) -> dict:
    """Require a researcher-supplied exact acknowledgment, retained off runtime."""
    _outside(acknowledgment, root, drive)
    expected = loads_strict(request.read_bytes())
    received = loads_strict(acknowledgment.read_bytes())
    if not isinstance(received, dict) or received != {
        "schema_version": "researcher_head_ack_v1",
        "researcher": "Rama Krishna Pudota",
        "request_sha256": sha256_file(request),
        "campaign_id": expected["campaign_id"],
        "batch_id": expected["batch_id"],
        "new_head_sha256": expected["new_head_sha256"],
    }:
        raise AmendedCustodyRefused("researcher acknowledgment does not match pending request")
    return expected


def retain_acknowledgment(request: Path, acknowledgment: Path, root: Path, drive: Path) -> Path:
    """Copy exact researcher-supplied evidence into the append-only Drive history."""
    expected = verify_acknowledgment(request, acknowledgment, root, drive)
    target = drive / "acknowledgments" / f"{expected['new_head_sha256']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    data = acknowledgment.read_bytes()
    _publish_drive_record(target, data, sha256_bytes(data), reuse_identical=True)
    return target


def publish_request(
    campaign_id: str, batch: Path, drive: Path, previous_head: str, new_head: str
) -> Path:
    """Write one immutable request after exact Drive copy/readback."""
    if not re.fullmatch(r"[a-zA-Z0-9_]{1,80}", campaign_id):
        raise AmendedCustodyRefused("invalid campaign identity")
    snapshot = drive / "snapshots" / new_head
    published = publish_drive_batch(batch, snapshot)
    files = verify_drive_batch(published)
    manifest = published / "drive_manifest.json"
    record = {
        "schema_version": "amended_campaign_head_request_v1",
        "campaign_id": campaign_id,
        "batch_id": batch.name,
        "previous_head_sha256": previous_head,
        "new_head_sha256": new_head,
        "manifest_sha256": sha256_file(manifest),
        "files": files,
    }
    target = drive / "requests" / f"{new_head}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    _publish_drive_record(
        target,
        canonical_json_bytes(record),
        sha256_bytes(canonical_json_bytes(record)),
        reuse_identical=True,
    )
    return target


def verify_chain(
    campaign_id: str,
    root: Path,
    drive: Path,
    records: list[dict],
    genesis_head: str,
    retained_head: str,
    acknowledgment: Path | None,
    pending_snapshot_head: str | None = None,
) -> None:
    """Authenticate every snapshot and require the latest off-runtime acknowledgment."""
    previous = genesis_head
    seen = set()
    for record in records:
        head = record["anchor_sha256"]
        request = drive / "requests" / f"{head}.json"
        if head in seen or not request.is_file():
            raise AmendedCustodyRefused("missing or duplicate campaign head request")
        seen.add(head)
        value = loads_strict(request.read_bytes())
        snapshot = drive / "snapshots" / head / record["batch_id"]
        files = verify_drive_batch(snapshot)
        if (
            value.get("campaign_id") != campaign_id
            or value.get("batch_id") != record["batch_id"]
            or value.get("previous_head_sha256") != previous
            or value.get("new_head_sha256") != head
            or value.get("manifest_sha256") != sha256_file(snapshot / "drive_manifest.json")
            or value.get("files") != files
            or files != record["files"]
        ):
            raise AmendedCustodyRefused("campaign request or Drive snapshot diverges")
        previous = head
    for folder in ("requests", "snapshots"):
        path = drive / folder
        actual = (
            set()
            if not path.exists()
            else {p.stem if folder == "requests" else p.name for p in path.iterdir()}
        )
        allowed = seen | ({pending_snapshot_head} if pending_snapshot_head else set())
        if actual != allowed and actual != seen:
            raise AmendedCustodyRefused("Drive history has missing or unanchored entries")
    if previous != retained_head:
        raise AmendedCustodyRefused("retained head is stale or history was rolled back")
    if records:
        if acknowledgment is None:
            raise AmendedCustodyRefused("latest researcher acknowledgment is missing")
        request = drive / "requests" / f"{retained_head}.json"
        retain_acknowledgment(request, acknowledgment, root, drive)
        for head in seen:
            stored = drive / "acknowledgments" / f"{head}.json"
            if not stored.is_file():
                raise AmendedCustodyRefused("earlier researcher acknowledgment is missing")
            verified = loads_strict(stored.read_bytes())
            prior_request = drive / "requests" / f"{head}.json"
            prior = loads_strict(prior_request.read_bytes())
            if verified != {
                "schema_version": "researcher_head_ack_v1",
                "researcher": "Rama Krishna Pudota",
                "request_sha256": sha256_file(prior_request),
                "campaign_id": prior["campaign_id"],
                "batch_id": prior["batch_id"],
                "new_head_sha256": head,
            }:
                raise AmendedCustodyRefused("stored researcher acknowledgment diverges")
        actual = {p.stem for p in (drive / "acknowledgments").iterdir()}
        if actual != seen:
            raise AmendedCustodyRefused("Drive history has unanchored acknowledgments")
