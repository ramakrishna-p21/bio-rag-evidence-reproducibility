"""Immutable canonical JATS publication through core artifact primitives."""

from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.documents.models import CanonicalDocument


def publish_canonical_jats(
    doc: CanonicalDocument, data_root: Path, source_content: bytes
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish validated canonical bytes, then their verified complete build manifest."""
    if doc.representation != "JATS" or doc.status != "complete" or doc.native_artifact is not None:
        raise ValueError("expected complete JATS canonical document")
    from biorag_repro.documents.phase4 import validate_jats_coverage

    if sha256_bytes(source_content) != doc.source.sha256:
        raise ValueError("JATS source hash mismatch")
    validate_jats_coverage(doc, source_content)
    content = serialize_project_json(doc, CanonicalDocument)
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind="canonical_jats",
        inputs=[ArtifactInput("raw_jats", doc.source.artifact_id, doc.source.sha256)],
        transformation_id=doc.transformation_id,
        transformation_settings_sha256=doc.configuration_sha256,
        output_sha256=digest,
        output_format="json",
        serialization_version="3",
    )
    logical_ref = f"corpus/derived/canonical/JATS/{artifact_id}.json"
    ref = ArtifactRef(artifact_id=artifact_id, logical_ref=logical_ref, sha256=digest)
    target = data_root / logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        content,
        digest,
        validate_temp=lambda path: read_project_json(path, digest, CanonicalDocument),
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=[BuildInput(role="raw_jats", artifact=doc.source)],
        transformation_id=doc.transformation_id,
        transformation_settings_sha256=doc.configuration_sha256,
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="canonical_jats",
                output_format="json",
                serialization_version="3",
                identity_version="1",
            )
        ],
        status="complete",
    )
    manifest_ref = f"corpus/derived/manifests/{artifact_id}.build.json"
    manifest_target = data_root / manifest_ref
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    manifest_hash = publish_manifest(manifest, manifest_target, {logical_ref: target})
    return ref, ArtifactRef(
        artifact_id="build_" + manifest_hash, logical_ref=manifest_ref, sha256=manifest_hash
    )
