"""Immutable publication and scientific read-back of Stage B records."""

from pathlib import Path
from typing import TypeVar

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.stage_b import (
    Inputs,
    dependencies,
    stage_b_ref,
    validate_candidate,
    validate_judgment,
)
from biorag_repro.evaluation.stage_b_models import StageBCandidate, StageBJudgment
from biorag_repro.evaluation.stage_records import publish_record, read_record

RecordT = TypeVar("RecordT", StageBCandidate, StageBJudgment)


def _publish(value: RecordT, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    kind = "stage_b_candidate" if isinstance(value, StageBCandidate) else "stage_b_judgment"
    return publish_record(
        value, stage_b_ref(value), kind, "stage_b", dependencies(value), f"{kind}_v1", root
    )


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
    return read_record(ref, root, model, stage_b_ref, "Stage B record")


def read_candidate(ref: ArtifactRef, root: Path, inputs: Inputs) -> StageBCandidate:
    value = _read(ref, root, StageBCandidate)
    validate_candidate(value, inputs)
    return value


def read_judgment(ref: ArtifactRef, root: Path, inputs: Inputs) -> StageBJudgment:
    value = _read(ref, root, StageBJudgment)
    validate_judgment(value, inputs)
    return value
