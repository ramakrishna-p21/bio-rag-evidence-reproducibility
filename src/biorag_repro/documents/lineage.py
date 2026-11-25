"""Build-manifest lineage and settings checks shared by the ingestion and authorization code."""

from pathlib import Path

from biorag_repro.core.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    verify_file_sha256,
)
from biorag_repro.core.manifest import ArtifactRef, BuildManifest
from biorag_repro.core.storage import (
    read_manifest,
)


def check_manifest(
    ref: ArtifactRef,
    data_root: Path,
    expected_output: ArtifactRef,
    expected_inputs: list[ArtifactRef],
) -> BuildManifest:
    if ref.artifact_id != "build_" + ref.sha256:
        raise ValueError("build manifest artifact identity mismatch")
    manifest = read_manifest(data_root / ref.logical_ref, ref.sha256)
    if manifest.status != "complete" or len(manifest.outputs) != 1:
        raise ValueError("complete build manifest is required")
    if manifest.outputs[0].artifact != expected_output:
        raise ValueError("build output lineage mismatch")
    if [item.artifact for item in manifest.inputs] != expected_inputs:
        raise ValueError("build input lineage mismatch")
    verify_file_sha256(data_root / expected_output.logical_ref, expected_output.sha256)
    return manifest


def runtime_settings(configuration: str, environment: str, assets: str) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "configuration_sha256": configuration,
                "environment_sha256": environment,
                "asset_inventory_sha256": assets,
            }
        )
    )


def check_manifest_contract(
    ref: ArtifactRef,
    root: Path,
    output: ArtifactRef,
    inputs: list[ArtifactRef],
    roles: list[str],
    kind: str,
    transform: str,
    settings: str,
    version: str,
) -> BuildManifest:
    manifest = check_manifest(ref, root, output, inputs)
    item = manifest.outputs[0]
    if (
        [x.role for x in manifest.inputs] != roles
        or manifest.transformation_id != transform
        or manifest.transformation_settings_sha256 != settings
        or (
            item.artifact_kind,
            item.output_format,
            item.serialization_version,
            item.identity_version,
        )
        != (kind, "json", version, "1")
    ):
        raise ValueError("manifest transformation contract mismatch (roles, metadata, settings)")
    return manifest
