"""Externally anchored, append-only final-test batch history.

The caller must supply the latest anchor SHA-256 from independent retention. Anchor
files live outside the approved data root and are published write-once. A data-root
writer cannot make a rolled-back or substituted batch match the retained head.
"""

import re
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest.layout import (
    ANCHOR_SUFFIX,
    ROOT_FILE,
    RegistryEvent,
    RootAnchor,
    RootRecord,
    StorageAuthorization,
    authorization_sha256,
    root_id_for,
)
from biorag_repro.readiness.files import FileProblem, loads_strict, read_regular


class HistoryRefused(ValueError):
    """The retained head does not authenticate the complete execution history."""


def _outside(root: Path, anchors: Path) -> None:
    root, anchors = root.resolve(), anchors.resolve()
    if root == anchors or root in anchors.parents or anchors in root.parents:
        raise HistoryRefused("anchors must be retained outside the execution root")


def _genesis(root: Path, authorization: StorageAuthorization, anchors: Path) -> str:
    if str(root) != authorization.authorized_root or authorization.root_kind != "final_test":
        raise HistoryRefused("root differs from approved final-test storage")
    _outside(root, anchors)
    root_id = root_id_for(authorization)
    anchor_bytes = read_regular(anchors, root_id + ANCHOR_SUFFIX)
    anchor = RootAnchor.model_validate(loads_strict(anchor_bytes))
    root_record = RootRecord.model_validate(loads_strict(read_regular(root, ROOT_FILE)))
    genesis = RegistryEvent.model_validate(loads_strict(read_regular(root, "registry/000001.json")))
    if (
        anchor.root_id != root_id
        or anchor.authorization_sha256 != authorization_sha256(authorization)
        or root_record.root_id != root_id
        or root_record.authorization_sha256 != authorization_sha256(authorization)
        or genesis.root_id != root_id
        or genesis.event_type != "root_initialized"
        or anchor.root_file_sha256 != sha256_bytes(read_regular(root, ROOT_FILE))
        or anchor.genesis_event_sha256 != sha256_bytes(read_regular(root, "registry/000001.json"))
    ):
        raise HistoryRefused("genesis differs from its independent root anchor")
    if {p.name for p in root.iterdir()} - {ROOT_FILE, "registry", "locks", "batches"}:
        raise HistoryRefused("unexpected entry in execution root")
    if {p.name for p in (root / "registry").iterdir()} != {"000001.json"}:
        raise HistoryRefused("unexpected registry history")
    if {p.name for p in (root / "locks").iterdir()} != {"registry.lock"}:
        raise HistoryRefused("unexpected lock entry")
    if read_regular(root, "locks/registry.lock"):
        raise HistoryRefused("registry lock contains data")
    batches = root / "batches"
    if batches.exists() and (batches.is_symlink() or not batches.is_dir()):
        raise HistoryRefused("batch directory is not a plain directory")
    return sha256_bytes(anchor_bytes)


def _files(batch: Path) -> dict[str, str]:
    if batch.is_symlink() or not batch.is_dir():
        raise HistoryRefused("batch is missing or is not a plain directory")
    result = {}
    for path in sorted(batch.rglob("*")):
        if path.is_symlink():
            raise HistoryRefused("batch contains a link")
        if path.is_file():
            result[path.relative_to(batch).as_posix()] = sha256_file(path)
        elif not path.is_dir():
            raise HistoryRefused("batch contains an unsupported entry")
    return result


def _verify_internal_seals(files: dict[str, str], batch: Path) -> None:
    """The latest write-once seal must account for every published batch file."""
    seals = sorted(name for name in files if re.fullmatch(r"anchors/\d{6}\.json", name))
    if seals != [f"anchors/{number:06d}.json" for number in range(1, len(seals) + 1)]:
        raise HistoryRefused("internal seal sequence is missing or has a gap")
    if not seals or "batch.json" not in files:
        raise HistoryRefused("batch has no identity or internal seal")
    try:
        identity = loads_strict((batch / "batch.json").read_bytes())
    except FileProblem as error:
        raise HistoryRefused("batch identity is unreadable") from error
    if not isinstance(identity, dict) or identity.get("run_id") != batch.name:
        raise HistoryRefused("batch identity differs from its directory")
    previous = None
    for name in seals:
        try:
            record = loads_strict((batch / name).read_bytes())
        except FileProblem as error:
            raise HistoryRefused("internal seal is unreadable") from error
        listed = record.get("files") if isinstance(record, dict) else None
        if (
            not isinstance(listed, dict)
            or record.get("previous_anchor_sha256") != previous
            or any(
                not isinstance(key, str) or not isinstance(value, str) or files.get(key) != value
                for key, value in listed.items()
            )
            or "batch.json" not in listed
        ):
            raise HistoryRefused("internal seal does not verify")
        previous = files[name]
    unsealed = set(files) - set(listed) - set(seals) - {"batch.lock"}
    if unsealed:
        raise HistoryRefused("batch contains files missing from its latest internal seal")


def _verify_history(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    expected_head_sha256: str,
    pending_batch_id: str | None = None,
) -> list[dict]:
    """Verify genesis, every batch and the exact independently retained latest head."""
    previous = _genesis(root, authorization, anchors)
    root_id = root_id_for(authorization)
    pattern = re.compile(rf"^{root_id}\.(\d{{6}})\.json$")
    names = sorted(p.name for p in anchors.iterdir() if pattern.fullmatch(p.name))
    if names != [f"{root_id}.{n:06d}.json" for n in range(1, len(names) + 1)]:
        raise HistoryRefused("external anchor sequence has a gap or duplicate")
    seen = set()
    latest_files = {}
    records = []
    for number, name in enumerate(names, 1):
        raw = read_regular(anchors, name)
        record = loads_strict(raw)
        if not isinstance(record, dict):
            raise HistoryRefused("external anchor is not a record")
        batch_id = record.get("batch_id")
        if (
            record.get("schema_version") != "final_test_batch_anchor_v2"
            or record.get("root_id") != root_id
            or record.get("sequence") != number
            or record.get("previous_anchor_sha256") != previous
            or not isinstance(batch_id, str)
            or not re.fullmatch(r"run_[0-9a-f]{32}", batch_id)
        ):
            raise HistoryRefused("external batch anchor breaks continuity")
        batch = root / "batches" / batch_id
        actual_files = _files(batch)
        sealed_files = record.get("files")
        if (
            not isinstance(sealed_files, dict)
            or any(actual_files.get(name) != digest for name, digest in sealed_files.items())
            or (
                batch_id in latest_files
                and any(
                    sealed_files.get(name) != digest
                    for name, digest in latest_files[batch_id].items()
                )
            )
        ):
            raise HistoryRefused("batch differs from its independent anchor")
        _verify_internal_seals(
            sealed_files if batch_id == pending_batch_id else actual_files, batch
        )
        seen.add(batch_id)
        latest_files[batch_id] = sealed_files
        records.append(record)
        previous = sha256_bytes(raw)
    batch_dir = root / "batches"
    actual = set() if not batch_dir.exists() else {p.name for p in batch_dir.iterdir()}
    allowed = seen | ({pending_batch_id} if pending_batch_id else set())
    if actual != allowed:
        raise HistoryRefused("execution root has missing or unanchored batches")
    for batch_id in seen - ({pending_batch_id} if pending_batch_id else set()):
        if _files(root / "batches" / batch_id) != latest_files[batch_id]:
            raise HistoryRefused("batch contains unanchored changes")
    if previous != expected_head_sha256:
        raise HistoryRefused("latest head differs from independently retained digest")
    return records


def verify_history(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    expected_head_sha256: str,
) -> list[dict]:
    """Public strict readback: no unanchored batch may exist."""
    return _verify_history(root, authorization, anchors, expected_head_sha256)


def anchor_batch(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    batch_id: str,
    expected_head_sha256: str,
    *,
    custodian=None,
    manifest_sha256: str | None = None,
    custody_genesis_head: str | None = None,
    before_publish=None,
) -> str:
    """Publish one complete batch after verifying its predecessor and all existing batches."""
    records = _verify_history(root, authorization, anchors, expected_head_sha256, batch_id)
    if not re.fullmatch(r"run_[0-9a-f]{32}", batch_id):
        raise HistoryRefused("invalid batch identity")
    batch = root / "batches" / batch_id
    files = _files(batch)
    _verify_internal_seals(files, batch)
    if custodian is not None:
        from biorag_repro.finaltest.custody import verify_custody

        if manifest_sha256 is None or custody_genesis_head is None:
            raise HistoryRefused("recovery anchor lacks a committed manifest or custody genesis")
        states = verify_custody(
            custodian,
            batch,
            batch_id,
            manifest_sha256,
            custodian.authoritative_head(),
            custody_genesis_head,
            anchors,
        )
        if any(
            state == "committed_absent" and not path.startswith("external/")
            for path, state in states.items()
        ):
            raise HistoryRefused("scientific publication is committed but absent")
    if records and records[-1]["batch_id"] == batch_id and records[-1]["files"] == files:
        raise HistoryRefused("duplicate batch anchor without new published files")
    record = {
        "schema_version": "final_test_batch_anchor_v2",
        "root_id": root_id_for(authorization),
        "sequence": len(records) + 1,
        "previous_anchor_sha256": expected_head_sha256,
        "batch_id": batch_id,
        "files": files,
    }
    data = canonical_json_bytes(record)
    target = anchors / f"{root_id_for(authorization)}.{len(records) + 1:06d}.json"
    if custodian is not None:
        relative = f"external/{target.name}"
        existing = [entry for entry in custodian.entries() if entry.path == relative]
        if existing:
            if len(existing) != 1 or existing[0].content_sha256 != sha256_bytes(data):
                raise HistoryRefused("external anchor differs from its prior commitment")
            if target.exists():
                if not target.is_file() or sha256_file(target) != sha256_bytes(data):
                    raise HistoryRefused("committed external anchor is altered")
                return sha256_bytes(data)
        if target.exists():
            raise HistoryRefused("external anchor exists without independent commitment")
        if not existing:
            custodian.commit(
                batch_id,
                manifest_sha256,
                custodian.authoritative_head(),
                relative,
                sha256_bytes(data),
            )
    if before_publish is not None:
        before_publish(sha256_bytes(data))
    publish_bytes(target, data, sha256_bytes(data))
    return sha256_bytes(data)


def preview_batch_head(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    batch_id: str,
    expected_head_sha256: str,
) -> str:
    """Derive the next anchor digest from a sealed pending batch without publishing it."""
    records = _verify_history(root, authorization, anchors, expected_head_sha256, batch_id)
    batch = root / "batches" / batch_id
    files = _files(batch)
    _verify_internal_seals(files, batch)
    if records and records[-1]["batch_id"] == batch_id and records[-1]["files"] == files:
        raise HistoryRefused("no pending sealed batch changes")
    record = {
        "schema_version": "final_test_batch_anchor_v2",
        "root_id": root_id_for(authorization),
        "sequence": len(records) + 1,
        "previous_anchor_sha256": expected_head_sha256,
        "batch_id": batch_id,
        "files": files,
    }
    return sha256_bytes(canonical_json_bytes(record))
