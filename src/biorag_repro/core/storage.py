"""Validated project JSON and immutable local artifact publication."""

import json
import os
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from biorag_repro.core.hashing import (
    read_verified_bytes,
    sha256_bytes,
    validate_sha256,
    verify_file_sha256,
)
from biorag_repro.core.manifest import BuildManifest

ModelT = TypeVar("ModelT", bound=BaseModel)


def serialize_project_json(value: ModelT, model_type: type[ModelT]) -> bytes:
    """Store a revalidated persistent model in the project's exact JSON format."""
    validated = model_type.model_validate(
        value.model_dump(mode="python", warnings="error"), strict=True
    )
    return (
        json.dumps(
            validated.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def serialize_manifest(manifest: BuildManifest) -> bytes:
    """Encode a validated manifest as deterministic, human-readable project JSON."""
    return serialize_project_json(manifest, BuildManifest)


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"nonfinite JSON value: {value}")


def read_project_json(path: Path, expected_sha256: str, model_type: type[ModelT]) -> ModelT:
    """Check exact stored bytes and validate a persisted project model."""
    content = read_verified_bytes(path, expected_sha256)
    data = json.loads(content.decode("utf-8"), parse_constant=_reject_nonfinite)
    return model_type.model_validate(data, strict=True)


def read_manifest(path: Path, expected_sha256: str) -> BuildManifest:
    """Check exact stored bytes and validate a persisted build manifest."""
    return read_project_json(path, expected_sha256, BuildManifest)


def publish_bytes(
    target: Path,
    content: bytes,
    expected_sha256: str,
    *,
    validate_temp: Callable[[Path], object] | None = None,
    reuse_identical: bool = False,
) -> str:
    """Publish complete, verified bytes without replacing an existing target.

    The target directory must support same-directory hard links. File data is
    fsynced; persistence of the directory entry across a system crash is not claimed.
    """
    validate_sha256(expected_sha256)

    def verify_existing() -> str:
        if target.is_symlink() or not target.is_file():
            raise ValueError("existing publication must be a regular file")
        if read_verified_bytes(target, expected_sha256) != content:
            raise ValueError("existing publication differs from expected bytes")
        if validate_temp is not None:
            validate_temp(target)
        return expected_sha256

    if reuse_identical and target.exists():
        return verify_existing()
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
        try:
            os.link(temporary, target)
        except FileExistsError:
            if not reuse_identical:
                raise
            return verify_existing()
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
    manifest: BuildManifest,
    target: Path,
    output_paths: Mapping[str, Path],
    *,
    reuse_identical: bool = False,
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
        reuse_identical=reuse_identical,
    )
