"""Publication of checked input-selection records; no scientific transforms run here."""

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
from biorag_repro.retrieval.models import (
    Bm25Index,
    CanonicalInput,
    Chunk,
    ChunkSet,
    ContextRecord,
    DenseIndex,
    RankedRetrieval,
)
from biorag_repro.retrieval.validation import CanonicalAuthority, load_canonical_input


def _inputs(value: CanonicalInput) -> list[BuildInput]:
    return [
        BuildInput(role=role, artifact=ref)
        for role, ref in (
            ("canonical", value.canonical),
            ("phase4_validation", value.validation),
            ("canonical_build_manifest", value.build_manifest),
        )
    ]


def input_ref(value: CanonicalInput) -> ArtifactRef:
    """Use existing ordered-lineage identity and exact stored-byte serialization."""
    content = serialize_project_json(value, CanonicalInput)
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind="canonical_input",
        inputs=[
            ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in _inputs(value)
        ],
        transformation_id="canonical_input_v1",
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/phase5_inputs/{artifact_id}.json",
        sha256=digest,
    )


def record_ref(
    value: Chunk | ChunkSet | Bm25Index | DenseIndex | RankedRetrieval | ContextRecord,
) -> ArtifactRef:
    """Identify interface records; this does not certify rendering or computation."""
    content = serialize_project_json(value, type(value))
    dependencies: list[tuple[str, ArtifactRef]]
    if isinstance(value, (Chunk, ChunkSet)):
        kind = "chunk" if isinstance(value, Chunk) else "chunk_set"
        dependencies = [(item.role, item.artifact) for item in _inputs(value.input)]
        if isinstance(value, ChunkSet):
            dependencies.append(("chunks_jsonl", value.chunks))
    elif isinstance(value, (Bm25Index, DenseIndex)):
        kind = "bm25_index" if isinstance(value, Bm25Index) else "dense_index"
        dependencies = [("chunk_set", item) for item in value.chunk_sets]
        if isinstance(value, DenseIndex):
            dependencies.append(("vectors", value.vectors))
    elif isinstance(value, RankedRetrieval):
        kind, dependencies = "ranked_retrieval", [("index", value.index)]
    else:
        kind, dependencies = "context", [("retrieval", value.retrieval)]
    dependencies.append(("configuration", value.configuration.artifact))
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[ArtifactInput(role, ref.artifact_id, ref.sha256) for role, ref in dependencies],
        transformation_id=f"{kind}_record_v1",
        transformation_settings_sha256=value.configuration.resolved_sha256,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/phase5/{kind}/{artifact_id}.json",
        sha256=digest,
    )


def publish_input(
    value: CanonicalInput,
    authority: CanonicalAuthority,
    output_root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish input validation only, with the complete manifest exposed last."""
    value = CanonicalInput.model_validate(value.model_dump())
    load_canonical_input(value, authority)
    ref = input_ref(value)
    target = output_root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, CanonicalInput),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, CanonicalInput),
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=_inputs(value),
        transformation_id="canonical_input_v1",
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="canonical_input",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    logical_ref = f"corpus/derived/phase5_inputs/{ref.artifact_id}.build.json"
    digest = publish_manifest(manifest, output_root / logical_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)
