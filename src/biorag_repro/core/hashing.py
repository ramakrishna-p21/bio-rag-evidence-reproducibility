"""Canonical semantic serialization and exact-byte SHA-256 hashing."""

import re
from hashlib import sha256
from pathlib import Path
from typing import Any

import rfc8785
from pydantic import JsonValue, TypeAdapter

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class HashMismatchError(ValueError):
    """Stored file bytes differ from their declared SHA-256 digest."""


def validate_sha256(value: str) -> None:
    """Reject digests outside the project's lowercase SHA-256 representation."""
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise ValueError("expected lowercase 64-character SHA-256 hexadecimal digest")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize an RFC 8785 JSON value for a specified semantic identity."""
    return rfc8785.dumps(_JSON_VALUE.validate_python(value, strict=True))


def sha256_bytes(data: bytes) -> str:
    """Return the lowercase SHA-256 digest of exact bytes."""
    return sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash the exact stored bytes without loading the file into memory."""
    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str) -> None:
    """Validate a file's exact bytes without changing the file."""
    validate_sha256(expected_sha256)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise HashMismatchError(
            f"SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
