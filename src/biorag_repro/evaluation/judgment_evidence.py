"""Evidence for the analysis table from published Stage A and Stage B judgment inventories.

An inventory row is only an index. Every judgment is read back through its authoritative
reader, which recomputes its validity from the benchmark item, the canonical document and
(for Stage B) the context records; the row's label, status, reviewer and timestamp must then
agree with the stored record. Nothing is invented for a missing judgment:

* no Stage A row            -> no entry (the caller sees `None`, a planned observation
                               without evidence);
* Stage A row, no Stage B   -> Stage A evidence only; Stage B stays unassessed. A
                               candidate that is not an item-level judgment (for example a
                               recorded-placement non-delivery) is never promoted;
* Stage A unresolved/absent/corrupted -> a Stage B record may only be the matching
                               non-evaluable / not-reachable record (the Stage B validator
                               decides); a delivery claim is refused.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import StringConstraints

from biorag_repro.benchmark.models import BenchmarkItem, Record
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest, read_project_json
from biorag_repro.documents.chunking import read_chunking
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation import stage_b_storage
from biorag_repro.evaluation.analysis_table import Evidence
from biorag_repro.evaluation.models import StageAJudgment
from biorag_repro.evaluation.stage_b import Inputs
from biorag_repro.evaluation.stage_b_models import StageBJudgment
from biorag_repro.evaluation.storage import read_judgment as read_stage_a
from biorag_repro.generation.roster import ContextSource
from biorag_repro.retrieval import bm25
from biorag_repro.retrieval import context as phase5
from biorag_repro.retrieval.models import ChunkSet
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import load_canonical_input

Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
RowT = TypeVar("RowT", "StageARow", "StageBRow")
Key = tuple[str, str]  # (item_id, representation)

DocumentFor = Callable[[BenchmarkItem, str], tuple[CanonicalDocument, ArtifactRef]]
InputsFor = Callable[[BenchmarkItem, str, StageAJudgment], Inputs]


class EvidenceError(Exception):
    """A published judgment or its inventory row cannot be accepted."""


class StageARow(Record):
    item_id: Nonblank
    representation: Nonblank
    judgment: ArtifactRef
    manifest: ArtifactRef
    label: Nonblank
    reviewer: str | None = None  # a deterministic judgment carries no reviewer
    at_utc: str | None = None


class StageBRow(Record):
    item_id: Nonblank
    representation: Nonblank
    judgment: ArtifactRef
    manifest: ArtifactRef
    status: Nonblank
    basis: Literal["deterministic", "reviewer"]
    context_provenance: Literal["assembled_only"]  # a published judgment never claims delivery
    reviewer: str | None = None
    at_utc: str | None = None


def _rows(path: Path, model: type[RowT]) -> list[RowT]:
    try:
        data = json.loads(path.read_bytes())
        if not isinstance(data, list):
            raise TypeError("an inventory is a JSON list")
        return [model.model_validate(row) for row in data]
    except (OSError, TypeError, ValueError) as error:
        raise EvidenceError(f"inventory {path} is not usable: {error}") from error


def _check_manifest(root: Path, manifest: ArtifactRef, judgment: ArtifactRef) -> None:
    try:
        built = read_manifest(root / manifest.logical_ref, manifest.sha256)
    except (OSError, ValueError) as error:
        raise EvidenceError(f"build manifest of {judgment.artifact_id}: {error}") from error
    if [o.artifact for o in built.outputs] != [judgment] or built.status != "complete":
        raise EvidenceError(f"build manifest does not publish exactly {judgment.artifact_id}")


def read_evidence(
    *,
    stage_a_inventory: Path,
    stage_b_inventory: Path | None,
    root: Path,
    items: Sequence[BenchmarkItem],
    document_for: DocumentFor,
    inputs_for: InputsFor,
) -> Mapping[Key, Evidence]:
    """Validated evidence per (item_id, representation); absent keys have no judgment.

    `items` must be the authorized development items: a row for any other item is refused.
    """
    by_id = {item.item_id: item for item in items}
    a_rows = _rows(stage_a_inventory, StageARow)
    b_rows = [] if stage_b_inventory is None else _rows(stage_b_inventory, StageBRow)

    stage_a: dict[Key, tuple[StageAJudgment, CanonicalDocument, ArtifactRef]] = {}
    for row in a_rows:
        key = (row.item_id, row.representation)
        if key in stage_a:
            raise EvidenceError(f"duplicate Stage A row for {key}")
        item = by_id.get(row.item_id)
        if item is None:
            raise EvidenceError(f"Stage A row for {row.item_id}, not an authorized item")
        document, canonical = document_for(item, row.representation)
        _check_manifest(root, row.manifest, row.judgment)
        try:
            judgment = read_stage_a(row.judgment, root, item, document, canonical)
        except (OSError, ValueError) as error:
            raise EvidenceError(f"Stage A judgment for {key}: {error}") from error
        if (row.label, row.reviewer, row.at_utc) != (
            judgment.label,
            judgment.reviewer,
            judgment.at_utc,
        ):
            raise EvidenceError(f"Stage A row for {key} disagrees with the stored judgment")
        stage_a[key] = (judgment, document, canonical)

    stage_b: dict[Key, tuple[StageBJudgment, Inputs]] = {}
    for b_row in b_rows:
        key = (b_row.item_id, b_row.representation)
        if key in stage_b:
            raise EvidenceError(f"duplicate Stage B row for {key}")
        if key not in stage_a:
            raise EvidenceError(f"Stage B row for {key} has no Stage A judgment")
        a = stage_a[key][0]
        _check_manifest(root, b_row.manifest, b_row.judgment)
        try:
            b_inputs = inputs_for(by_id[b_row.item_id], b_row.representation, a)
            b_judgment = stage_b_storage.read_judgment(b_row.judgment, root, b_inputs)
        except (OSError, ValueError) as error:
            raise EvidenceError(f"Stage B judgment for {key}: {error}") from error
        if (b_row.status, b_row.basis, b_row.reviewer, b_row.at_utc) != (
            b_judgment.status,
            b_judgment.basis,
            b_judgment.reviewer,
            b_judgment.at_utc,
        ):
            raise EvidenceError(f"Stage B row for {key} disagrees with the stored judgment")
        stage_b[key] = (b_judgment, b_inputs)

    evidence: dict[Key, Evidence] = {}
    for key, (a, document, canonical) in stage_a.items():
        b_found = stage_b.get(key)
        b, inputs = b_found if b_found else (None, None)
        evidence[key] = Evidence(document, canonical, a, b, inputs)
    return evidence


class SourceInputs:
    """`document_for` and `inputs_for` built from the context sources of a roster.

    Every record is re-read through the authoritative context readers; the sources only say
    where to look. Designated-paper sources are supported: Stage B inputs name one chunk set.
    """

    def __init__(self, sources: Sequence[ContextSource], context_tokenizer_root: Path) -> None:
        self.tokenizer_root = context_tokenizer_root
        self._by_key: dict[Key, ContextSource] = {}
        for source in sources:
            selection = source.selection
            if selection.query.scope != "designated_paper" or len(selection.sources) != 1:
                raise EvidenceError("only single-chunk-set designated-paper sources are supported")
            chunks = selection.sources[0]
            stored = read_project_json(
                chunks.chunk_root / chunks.chunk_set.logical_ref, chunks.chunk_set.sha256, ChunkSet
            )
            key = (selection.query.query_id, stored.input.representation)
            if key in self._by_key:
                raise EvidenceError(f"duplicate context source for {key}")
            self._by_key[key] = source

    def _source(self, item: BenchmarkItem, representation: str) -> ContextSource:
        try:
            return self._by_key[(item.item_id, representation)]
        except KeyError:
            raise EvidenceError(f"no context source for {(item.item_id, representation)}") from None

    def document_for(
        self, item: BenchmarkItem, representation: str
    ) -> tuple[CanonicalDocument, ArtifactRef]:
        chunks = self._source(item, representation).selection.sources[0]
        stored = read_project_json(
            chunks.chunk_root / chunks.chunk_set.logical_ref, chunks.chunk_set.sha256, ChunkSet
        )
        try:
            return load_canonical_input(stored.input, chunks.authority), stored.input.canonical
        except (OSError, ValueError) as error:
            raise EvidenceError(f"canonical document of {item.item_id}: {error}") from error

    def inputs_for(
        self, item: BenchmarkItem, representation: str, stage_a: StageAJudgment
    ) -> Inputs:
        source = self._source(item, representation)
        selection = source.selection
        chunks = selection.sources[0]
        try:
            ranked = bm25.read_retrieval(
                selection.retrieval,
                selection.retrieval_manifest,
                selection.index,
                selection.index_manifest,
                selection.sources,
                selection.query,
                selection.retrieval_configuration,
                selection.retrieval_configuration_root,
                selection.retrieval_root,
            )
            ctx = phase5.read(
                source.context,
                source.manifest,
                selection,
                source.configuration,
                source.configuration_root,
                self.tokenizer_root,
                source.root,
            )
            read = read_chunking(
                chunks.chunk_set,
                chunks.manifest,
                chunks.authority,
                chunks.configuration,
                chunks.configuration_root,
                chunks.chunk_root,
                tokenizer_root=chunks.tokenizer_root,
                fixture_counter=chunks.fixture_counter,
            )
            document, canonical = self.document_for(item, representation)
        except (OSError, ValueError) as error:
            raise EvidenceError(f"Stage B inputs of {item.item_id}: {error}") from error
        return Inputs(
            item,
            document,
            stage_a,
            ctx,
            ranked,
            read.chunk_set,
            {record_ref(c).artifact_id: c for c in read.chunks},
            canonical,
            source.context,
        )
