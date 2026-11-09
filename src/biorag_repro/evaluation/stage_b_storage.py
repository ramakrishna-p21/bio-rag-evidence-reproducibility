"""Immutable publication and scientific read-back of Stage B records."""

from pathlib import Path
from typing import TypeVar

from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.evaluation.stage_b import (
    Inputs,
    dependencies,
    stage_b_ref,
    validate_candidate,
    validate_judgment,
)
from biorag_repro.evaluation.stage_b_models import StageBCandidate, StageBJudgment

RecordT = TypeVar("RecordT", StageBCandidate, StageBJudgment)


def _publish(value: RecordT, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    model = type(value)
    ref = stage_b_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, model),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, model),
    )
    kind = "stage_b_candidate" if model is StageBCandidate else "stage_b_judgment"
    manifest = BuildManifest(
        schema_version="1",
        inputs=[BuildInput(role=r, artifact=a) for r, a in dependencies(value)],
        transformation_id=f"{kind}_v1",
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    logical_ref = f"evaluation/stage_b/build/{ref.artifact_id}.build.json"
    (root / logical_ref).parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(manifest, root / logical_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


def publish_candidate(
    candidate: StageBCandidate, inputs: Inputs, root: Path
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_candidate(candidate, inputs)
    return _publish(candidate, root)


def publish_judgment(
    judgment: StageBJudgment, inputs: Inputs, root: Path
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_judgment(judgment, inputs)
    return _publish(judgment, root)


def _read(ref: ArtifactRef, root: Path, model: type[RecordT]) -> RecordT:
    value = read_project_json(root / ref.logical_ref, ref.sha256, model)
    if stage_b_ref(value) != ref:
        raise ValueError("stored Stage B record does not match its reference identity")
    return value


def read_candidate(ref: ArtifactRef, root: Path, inputs: Inputs) -> StageBCandidate:
    value = _read(ref, root, StageBCandidate)
    validate_candidate(value, inputs)
    return value


def read_judgment(ref: ArtifactRef, root: Path, inputs: Inputs) -> StageBJudgment:
    value = _read(ref, root, StageBJudgment)
    validate_judgment(value, inputs)
    return value
