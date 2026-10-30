"""Typed scientific configuration resolution and identity."""

import tomllib
from dataclasses import dataclass
from typing import Generic, TypeVar

from pydantic import BaseModel

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

ConfigT = TypeVar("ConfigT", bound=BaseModel)


@dataclass(frozen=True)
class ResolvedConfiguration(Generic[ConfigT]):
    settings: ConfigT
    canonical_json: bytes
    configuration_sha256: str
    raw_toml: bytes
    raw_toml_sha256: str


def resolve_config(raw_toml: bytes, model_type: type[ConfigT]) -> ResolvedConfiguration[ConfigT]:
    """Resolve declared defaults and hash effective settings, not TOML formatting."""
    if model_type.model_config.get("validate_default") is not True:
        raise ValueError("configuration model requires validate_default=True")

    parsed = tomllib.loads(raw_toml.decode("utf-8"))
    settings = model_type.model_validate(parsed, strict=True, extra="forbid")
    effective = settings.model_dump(mode="python", warnings="error")

    version = effective.get("schema_version")
    if not isinstance(version, str) or not version:
        raise ValueError("configuration model requires a nonempty schema_version")

    canonical = canonical_json_bytes(effective)
    return ResolvedConfiguration(
        settings=settings,
        canonical_json=canonical,
        configuration_sha256=sha256_bytes(canonical),
        raw_toml=raw_toml,
        raw_toml_sha256=sha256_bytes(raw_toml),
    )
