"""Canonical semantic serialization and exact-byte SHA-256 hashing."""

from hashlib import sha256
from pathlib import Path
from typing import Any

import rfc8785
from pydantic import JsonValue, TypeAdapter

_JSON_VALUE: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)


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
