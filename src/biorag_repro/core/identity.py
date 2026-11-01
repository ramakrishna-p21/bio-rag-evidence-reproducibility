"""Stable derived and planned-observation identities."""

from collections.abc import Sequence
from dataclasses import dataclass
from uuid import uuid4

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, validate_sha256


@dataclass(frozen=True)
class ArtifactInput:
    role: str
    artifact_id: str
    sha256: str


def derived_artifact_id(
    *,
    artifact_kind: str,
    inputs: Sequence[ArtifactInput],
    transformation_id: str,
    transformation_settings_sha256: str | None,
    output_sha256: str,
    output_format: str,
    serialization_version: str,
    identity_version: str = "1",
) -> str:
    """Identify exact output bytes and ordered lineage using transformation-specific settings."""
    for name, value in (
        ("identity_version", identity_version),
        ("artifact_kind", artifact_kind),
        ("transformation_id", transformation_id),
        ("output_format", output_format),
        ("serialization_version", serialization_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    if not inputs:
        raise ValueError("inputs must contain at least one artifact")
    for item in inputs:
        if not isinstance(item, ArtifactInput):
            raise TypeError("inputs must contain ArtifactInput values")
        if not isinstance(item.role, str) or not item.role:
            raise ValueError("input role must be a nonempty string")
        if not isinstance(item.artifact_id, str) or not item.artifact_id:
            raise ValueError("input artifact_id must be a nonempty string")
        validate_sha256(item.sha256)
    if transformation_settings_sha256 is not None:
        validate_sha256(transformation_settings_sha256)
    validate_sha256(output_sha256)

    payload = {
        "identity_version": identity_version,
        "artifact_kind": artifact_kind,
        "inputs": [
            {"role": item.role, "artifact_id": item.artifact_id, "sha256": item.sha256}
            for item in inputs
        ],
        "transformation_id": transformation_id,
        "transformation_settings_sha256": transformation_settings_sha256,
        "output_sha256": output_sha256,
        "output_format": output_format,
        "serialization_version": serialization_version,
    }
    return "drv_" + sha256_bytes(canonical_json_bytes(payload))


def raw_source_artifact_id(*, paper_id: str, representation: str, sha256: str) -> str:
    """Identify accepted source bytes within one paper and manifestation."""
    if not isinstance(paper_id, str) or not paper_id:
        raise ValueError("paper_id must be a nonempty string")
    if representation not in {"JATS", "PDF"}:
        raise ValueError("representation must be JATS or PDF")
    validate_sha256(sha256)
    payload = {
        "identity_version": "1",
        "paper_id": paper_id,
        "representation": representation,
        "sha256": sha256,
    }
    return "raw_" + sha256_bytes(canonical_json_bytes(payload))


def observation_key(
    *,
    phase: str,
    benchmark_version: str,
    item_id: str,
    matrix_arm: str,
    representation_or_condition: str,
    repeat_index: int,
    resolved_config_sha256: str,
    identity_version: str = "1",
) -> str:
    """Identify planned scientific coordinates independently of execution attempts."""
    for name, value in (
        ("phase", phase),
        ("benchmark_version", benchmark_version),
        ("item_id", item_id),
        ("matrix_arm", matrix_arm),
        ("representation_or_condition", representation_or_condition),
        ("identity_version", identity_version),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a nonempty string")
    if type(repeat_index) is not int or repeat_index < 0:
        raise ValueError("repeat_index must be a nonnegative integer")
    validate_sha256(resolved_config_sha256)

    payload = {
        "identity_version": identity_version,
        "phase": phase,
        "benchmark_version": benchmark_version,
        "item_id": item_id,
        "matrix_arm": matrix_arm,
        "representation_or_condition": representation_or_condition,
        "repeat_index": repeat_index,
        "resolved_config_sha256": resolved_config_sha256,
    }
    return "obs_" + sha256_bytes(canonical_json_bytes(payload))


def new_run_id() -> str:
    """Return an opaque execution-batch locator."""
    return uuid4().hex


def new_attempt_id() -> str:
    """Return an opaque execution-attempt locator."""
    return uuid4().hex
