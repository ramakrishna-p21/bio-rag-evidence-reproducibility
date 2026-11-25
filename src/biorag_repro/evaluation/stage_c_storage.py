"""Immutable publication and scientific read-back of Stage C scores."""

from pathlib import Path

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.stage_c import (
    dependencies,
    score_ref,
    transformation_id,
    validate_score,
)
from biorag_repro.evaluation.stage_c_models import GenerationResult, StageCScore
from biorag_repro.evaluation.stage_records import publish_record, read_record


def publish_score(
    value: StageCScore, item: BenchmarkItem, result: GenerationResult, root: Path
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_score(value, item, result)
    return publish_record(
        value,
        score_ref(value),
        "stage_c_score",
        "stage_c",
        dependencies(value),
        transformation_id(value),
        root,
    )


def read_score(
    ref: ArtifactRef, root: Path, item: BenchmarkItem, result: GenerationResult
) -> StageCScore:
    value = read_record(ref, root, StageCScore, score_ref, "Stage C score")
    validate_score(value, item, result)
    return value
