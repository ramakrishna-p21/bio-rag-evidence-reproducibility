"""Immutable publication and scientific read-back of Stage A records."""

from pathlib import Path
from typing import TypeVar

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import StageACandidate, StageAJudgment
from biorag_repro.evaluation.stage_a import (
    stage_a_dependencies,
    stage_a_ref,
    validate_candidate,
    validate_judgment,
)

RecordT = TypeVar("RecordT", StageACandidate, StageAJudgment)


def _publish(value: RecordT, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    model = type(value)
    ref = stage_a_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, model),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, model),
    )
    kind = "stage_a_candidate" if model is StageACandidate else "stage_a_judgment"
    manifest = BuildManifest(
        schema_version="1",
        inputs=[BuildInput(role=r, artifact=a) for r, a in stage_a_dependencies(value)],
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
    logical_ref = f"evaluation/stage_a/build/{ref.artifact_id}.build.json"
    (root / logical_ref).parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(manifest, root / logical_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


def publish_candidate(
    candidate: StageACandidate,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish only a candidate the locator reproduces from this item and document."""
    validate_candidate(candidate, item, document, canonical)
    return _publish(candidate, root)


def publish_judgment(
    judgment: StageAJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish a judgment only after scientific validation; never creates reviewer fields."""
    validate_judgment(judgment, item, document, canonical)
    return _publish(judgment, root)


def _read(ref: ArtifactRef, root: Path, model: type[RecordT]) -> RecordT:
    value = read_project_json(root / ref.logical_ref, ref.sha256, model)
    if stage_a_ref(value) != ref:
        raise ValueError("stored Stage A record does not match its reference identity")
    return value


def read_candidate(
    ref: ArtifactRef,
    root: Path,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> StageACandidate:
    value = _read(ref, root, StageACandidate)
    validate_candidate(value, item, document, canonical)
    return value


def read_judgment(
    ref: ArtifactRef,
    root: Path,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> StageAJudgment:
    value = _read(ref, root, StageAJudgment)
    validate_judgment(value, item, document, canonical)
    return value
