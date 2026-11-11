"""Immutable publication and scientific read-back of Stage A records."""

from pathlib import Path
from typing import TypeVar

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import StageACandidate, StageAJudgment
from biorag_repro.evaluation.stage_a import (
    stage_a_dependencies,
    stage_a_ref,
    validate_candidate,
    validate_judgment,
)
from biorag_repro.evaluation.stage_records import publish_record, read_record

RecordT = TypeVar("RecordT", StageACandidate, StageAJudgment)


def _publish(value: RecordT, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    kind = "stage_a_candidate" if isinstance(value, StageACandidate) else "stage_a_judgment"
    return publish_record(
        value, stage_a_ref(value), kind, "stage_a", stage_a_dependencies(value), f"{kind}_v1", root
    )


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
    return read_record(ref, root, model, stage_a_ref, "Stage A record")


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
