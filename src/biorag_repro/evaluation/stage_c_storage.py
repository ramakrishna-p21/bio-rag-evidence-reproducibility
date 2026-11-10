"""Immutable publication and scientific read-back of Stage C scores."""

from pathlib import Path

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.evaluation.stage_c import (
    dependencies,
    score_ref,
    transformation_id,
    validate_score,
)
from biorag_repro.evaluation.stage_c_models import GenerationResult, StageCScore


def publish_score(
    value: StageCScore, item: BenchmarkItem, result: GenerationResult, root: Path
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_score(value, item, result)
    ref = score_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, StageCScore),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, StageCScore),
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=[BuildInput(role=r, artifact=a) for r, a in dependencies(value)],
        transformation_id=transformation_id(value),
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="stage_c_score",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    logical_ref = f"evaluation/stage_c/build/{ref.artifact_id}.build.json"
    (root / logical_ref).parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(manifest, root / logical_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


def read_score(
    ref: ArtifactRef, root: Path, item: BenchmarkItem, result: GenerationResult
) -> StageCScore:
    value = read_project_json(root / ref.logical_ref, ref.sha256, StageCScore)
    if score_ref(value) != ref:
        raise ValueError("stored Stage C score does not match its reference identity")
    validate_score(value, item, result)
    return value
