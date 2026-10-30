"""Validated project JSON and immutable local artifact publication."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from biorag_repro.core.hashing import (
    HashMismatchError,
    sha256_bytes,
    validate_sha256,
    verify_file_sha256,
)
from biorag_repro.core.manifest import BuildManifest


def serialize_manifest(manifest: BuildManifest) -> bytes:
    """Encode a validated manifest as deterministic, human-readable project JSON."""
    validated = BuildManifest.model_validate(
        manifest.model_dump(mode="python", warnings="error"), strict=True
    )
    data = validated.model_dump(mode="python")
    return (
        json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def read_manifest(path: Path, expected_sha256: str) -> BuildManifest:
    """Check exact stored bytes and validate a persisted build manifest."""
    validate_sha256(expected_sha256)
    content = path.read_bytes()
    actual_sha256 = sha256_bytes(content)
    if actual_sha256 != expected_sha256:
        raise HashMismatchError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    data = json.loads(content.decode("utf-8"), parse_constant=_reject_nonfinite)
    return BuildManifest.model_validate(data, strict=True)


def publish_bytes(
    target: Path,
    content: bytes,
    expected_sha256: str,
    *,
    validate_temp: Callable[[Path], object] | None = None,
) -> str:
    """Publish complete, verified bytes without replacing an existing target.

    The target directory must support same-directory hard links. File data is
    fsynced; persistence of the directory entry across a system crash is not claimed.
    """
    validate_sha256(expected_sha256)
    descriptor, name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if validate_temp is not None:
            validate_temp(temporary)
        verify_file_sha256(temporary, expected_sha256)
        os.link(temporary, target)
        return expected_sha256
    finally:
        temporary.unlink(missing_ok=True)


def verify_manifest_outputs(manifest: BuildManifest, output_paths: Mapping[str, Path]) -> None:
    """Verify caller-resolved final output files before manifest publication."""
    for output in manifest.outputs:
        ref = output.artifact
        if ref.logical_ref not in output_paths:
            raise ValueError(f"no published path supplied for {ref.logical_ref}")
        path = output_paths[ref.logical_ref]
        if not path.exists():
            raise FileNotFoundError(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"published output must be a regular file: {path}")
        verify_file_sha256(path, ref.sha256)


def publish_manifest(
    manifest: BuildManifest, target: Path, output_paths: Mapping[str, Path]
) -> str:
    """Publish a build manifest after checking all listed retained outputs."""
    content = serialize_manifest(manifest)
    verify_manifest_outputs(manifest, output_paths)
    expected_sha256 = sha256_bytes(content)
    return publish_bytes(
        target,
        content,
        expected_sha256,
        validate_temp=lambda path: read_manifest(path, expected_sha256),
    )
