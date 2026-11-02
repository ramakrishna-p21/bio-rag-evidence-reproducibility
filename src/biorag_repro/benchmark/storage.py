"""Deterministic benchmark JSONL serialization and integrity-checked reads."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256

ModelT = TypeVar("ModelT", bound=BaseModel)


def encode_jsonl(records: Sequence[BaseModel]) -> bytes:
    """Write one sorted, compact UTF-8 JSON object per line with final newline."""
    if not records:
        return b""
    lines = []
    for record in records:
        valid = type(record).model_validate(record.model_dump(mode="python"), strict=True)
        lines.append(
            json.dumps(
                valid.model_dump(mode="json"),
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def decode_jsonl(content: bytes, model: type[ModelT]) -> list[ModelT]:
    text = content.decode("utf-8")
    if text and (not text.endswith("\n") or "\n\n" in text):
        raise ValueError("JSONL requires exactly one object per terminated line")
    records = [model.model_validate(json.loads(line), strict=True) for line in text.splitlines()]
    if encode_jsonl(records) != content:
        raise ValueError("JSONL bytes are not canonical project form")
    return records


def read_jsonl(path: Path, expected_sha256: str, model: type[ModelT]) -> list[ModelT]:
    verify_file_sha256(path, expected_sha256)
    return decode_jsonl(path.read_bytes(), model)


def jsonl_sha256(records: Sequence[BaseModel]) -> str:
    return sha256_bytes(encode_jsonl(records))
