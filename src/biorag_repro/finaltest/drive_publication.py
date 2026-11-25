"""Verified batch copy to a mounted Drive folder.

This provides recovery from interrupted copies. It does not establish independent
custody or an authoritative head for G13.
"""

import shutil
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_file
from biorag_repro.readiness.files import loads_strict


class DrivePublicationError(ValueError):
    """A copied batch is incomplete or differs from its manifest."""


def _files(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise DrivePublicationError("batch directory is unavailable")
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DrivePublicationError("batch contains a symbolic link")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if name == "drive_manifest.json":
                continue
            result[name] = sha256_file(path)
        elif not path.is_dir():
            raise DrivePublicationError("batch contains an unsupported entry")
    return result


def verify_drive_batch(destination: Path) -> dict[str, str]:
    """Read every published file and require exact manifest coverage and hashes."""
    manifest = destination / "drive_manifest.json"
    if manifest.is_symlink() or not manifest.is_file():
        raise DrivePublicationError("batch has no publication manifest")
    record = loads_strict(manifest.read_bytes())
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != "drive_batch_copy_v1"
        or record.get("batch_id") != destination.name
        or not isinstance(record.get("files"), dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in record["files"].items())
    ):
        raise DrivePublicationError("invalid publication manifest")
    files = _files(destination)
    if files != record["files"]:
        raise DrivePublicationError("published files differ from manifest")
    return files


def publish_drive_batch(source: Path, drive_root: Path) -> Path:
    """Copy a sealed batch, resume identical partial copies, publish manifest last."""
    files = _files(source)
    if not files or "batch.json" not in files:
        raise DrivePublicationError("source lacks batch identity")
    destination = drive_root / source.name
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination / "drive_manifest.json"
    if manifest.exists():
        if verify_drive_batch(destination) != files:
            raise DrivePublicationError("published batch differs from source")
        return destination
    existing = _files(destination)
    if any(files.get(name) != digest for name, digest in existing.items()):
        raise DrivePublicationError("partial copy differs from source")
    for name, digest in files.items():
        if name in existing:
            continue
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with (source / name).open("rb") as input_file, target.open("xb") as output_file:
            shutil.copyfileobj(input_file, output_file)
        if sha256_file(target) != digest:
            raise DrivePublicationError("copied file differs from source")
    if _files(destination) != files:
        raise DrivePublicationError("copied batch is incomplete")
    record = {"schema_version": "drive_batch_copy_v1", "batch_id": source.name, "files": files}
    with manifest.open("xb") as output_file:
        output_file.write(canonical_json_bytes(record))
    verify_drive_batch(destination)
    return destination
