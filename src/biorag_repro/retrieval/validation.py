"""Scientific input and contributor checks against separately selected evidence."""

from dataclasses import dataclass
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.authorization import ScientificAuthority, validate_scientific_pair
from biorag_repro.documents.models import CanonicalDocument, CanonicalNode, SourceMapping
from biorag_repro.documents.phase4 import Phase4GateIndex, Phase4Validation
from biorag_repro.retrieval.models import (
    Bm25Index,
    CanonicalInput,
    Chunk,
    ChunkSet,
    ConfigurationRef,
    ContextRecord,
    DenseIndex,
    Phase5Configuration,
    RankedRetrieval,
    SourceOccurrence,
)


@dataclass(frozen=True)
class CanonicalAuthority:
    """Caller-selected evidence, not a deserializable approval or execution proof.

    The caller must select reviewed evidence outside the submitted input record.
    Existing Phase 4 validation supplies the frozen release/config/native checks.
    """

    corpus_root: Path
    artifact_root: Path
    scientific: ScientificAuthority
    evidence: Phase4GateIndex
    reproduction_revision: str


def load_canonical_input(value: CanonicalInput, authority: CanonicalAuthority) -> CanonicalDocument:
    """Check the selected frozen pair, then resolve this exact canonical input."""
    if not isinstance(authority, CanonicalAuthority):
        raise TypeError("canonical authority must be independently selected by the caller")
    value = CanonicalInput.model_validate(value.model_dump())
    evidence = Phase4GateIndex.model_validate(authority.evidence.model_dump())
    if (
        value.corpus_version != evidence.corpus_version
        or value.corpus_release_sha256 != authority.scientific.release_sha256
        or value.paper_id != evidence.paper_id
    ):
        raise ValueError("input differs from independently selected corpus/paper selection")
    validate_scientific_pair(
        evidence,
        authority.artifact_root,
        authority.corpus_root,
        authority.scientific,
        authority.reproduction_revision,
    )
    selected_validation = (
        evidence.jats_validation if value.representation == "JATS" else evidence.pdf_validation
    )
    report = read_project_json(
        authority.artifact_root / selected_validation.logical_ref,
        selected_validation.sha256,
        Phase4Validation,
    )
    if (
        value.validation != selected_validation
        or value.canonical != report.canonical_artifact
        or value.build_manifest != report.build_manifest
        or value.representation != report.representation
    ):
        raise ValueError("canonical/manifest/validation selection differs from reviewed evidence")
    return read_project_json(
        authority.artifact_root / value.canonical.logical_ref,
        value.canonical.sha256,
        CanonicalDocument,
    )


def occurrence_id(
    canonical: ArtifactRef | dict[str, object],
    mapping: SourceMapping | dict[str, object],
    field: str,
) -> str:
    """Identify a source field, excluding presentation ordinals and precision claims."""
    ref = ArtifactRef.model_validate(canonical)
    source = SourceMapping.model_validate(mapping)
    locator = {
        "source_artifact_id": source.source_artifact_id,
        "kind": source.kind,
        "path": source.path,
        "native_artifact": (
            {
                "artifact_id": source.native_artifact.artifact_id,
                "sha256": source.native_artifact.sha256,
            }
            if source.native_artifact
            else None
        ),
        "native_ref": source.native_ref,
        "field": "text"
        if source.kind == "docling" and field in {"segment", "table_cell"}
        else field,
    }
    return derived_artifact_id(
        artifact_kind="source_occurrence",
        inputs=[ArtifactInput("canonical", ref.artifact_id, ref.sha256)],
        transformation_id="source_occurrence_v1",
        transformation_settings_sha256=None,
        output_sha256=sha256_bytes(canonical_json_bytes(locator)),
        output_format="json",
        serialization_version="1",
    )


def validate_occurrences(
    occurrences: list[SourceOccurrence],
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> None:
    """Validate submitted ownership without discovering or rendering occurrences."""
    document = CanonicalDocument.model_validate(document.model_dump())
    nodes = {node.node_id: node for node in document.nodes}
    cells = {
        cell.node_id: cell for node in document.nodes if node.table for cell in node.table.cells
    }
    seen: set[str] = set()
    for submitted in occurrences:
        item = SourceOccurrence.model_validate(submitted.model_dump())
        if item.occurrence_id in seen:
            raise ValueError("duplicate source occurrence")
        seen.add(item.occurrence_id)
        if item.canonical != canonical or item.owner_node_id not in nodes:
            raise ValueError("occurrence canonical or owner differs from selected input")
        node = nodes[item.owner_node_id]
        if item.occurrence_id != occurrence_id(canonical, item.mapping, item.field):
            raise ValueError("occurrence identity differs from source location")
        mapping, text = node.mapping, None
        if item.field == "segment":
            if item.segment_index is None or item.segment_index >= len(node.segments):
                raise ValueError("occurrence segment is outside canonical owner")
            segment = node.segments[item.segment_index]
            mapping, text = segment.mapping, segment.text
        elif item.field == "table_cell":
            if document.representation == "JATS":
                raise ValueError("JATS table text uses segment occurrences, not an aggregate")
            if node.node_id not in cells:
                raise ValueError("occurrence owner is not a structured table cell")
            cell = cells[node.node_id]
            mapping, text = cell.mapping, cell.text
        elif item.field == "native_orig":
            text = node.native_orig
        elif item.field == "native_marker":
            text = node.native_marker
        elif (
            node.segments
            or node.native_orig
            or node.native_marker
            or (node.node_id in cells and cells[node.node_id].text)
        ):
            raise ValueError("nontextual occurrence hides canonical textual content")
        if (
            item.mapping != mapping
            or item.text != text
            or item.content_state != node.content_state
            or item.interpretation_state != node.interpretation_state
            or item.native_order_state != node.native_order_state
            or item.orig_qualification != node.orig_qualification
        ):
            raise ValueError(
                "occurrence content, mapping, or uncertainty differs from canonical owner"
            )


def _node_contains(node: CanonicalNode, occurrence: SourceOccurrence) -> bool:
    if occurrence.field == "segment":
        # Segment ordinals are local to each containing node, not source identity.
        return any(
            part.text == occurrence.text
            and part.mapping.model_copy(update={"segment_ordinal": None})
            == occurrence.mapping.model_copy(update={"segment_ordinal": None})
            for part in node.segments
        )
    if occurrence.field == "table_cell" and node.table:
        return any(cell.node_id == occurrence.owner_node_id for cell in node.table.cells)
    return node.node_id == occurrence.owner_node_id


def validate_chunk_contributors(
    chunk: Chunk,
    document: CanonicalDocument,
    occurrences: list[SourceOccurrence],
) -> None:
    """Check contributor lineage only; rendering and real token counts are later gates."""
    chunk = Chunk.model_validate(chunk.model_dump())
    if (chunk.input.paper_id, chunk.input.representation) != (
        document.paper_id,
        document.representation,
    ):
        raise ValueError("chunk paper or representation differs from canonical input")
    validate_occurrences(occurrences, document, chunk.input.canonical)
    by_id = {item.occurrence_id: item for item in occurrences}
    nodes = {node.node_id: node for node in document.nodes}
    primary_ids = [item.occurrence_id for item in chunk.contributors if item.usage == "content"]
    if len(primary_ids) != len(set(primary_ids)):
        raise ValueError("one source occurrence has duplicate primary contributions")
    for contributor in chunk.contributors:
        occurrence = by_id.get(contributor.occurrence_id)
        node = nodes.get(contributor.node_id)
        if occurrence is None or node is None or not _node_contains(node, occurrence):
            raise ValueError("contributor does not reference this canonical occurrence")


def load_configuration(ref: ConfigurationRef, root: Path) -> Phase5Configuration:
    settings = read_project_json(
        root / ref.artifact.logical_ref,
        ref.artifact.sha256,
        Phase5Configuration,
    )
    if sha256_bytes(canonical_json_bytes(settings.model_dump())) != ref.resolved_sha256:
        raise ValueError("resolved configuration identity differs from stored settings")
    return settings


def validate_chunk_set_links(
    value: ChunkSet,
    chunks: list[Chunk],
    document: CanonicalDocument,
    settings: Phase5Configuration,
) -> None:
    """Check accounting and configuration limits, without rendering or tokenizing."""
    from biorag_repro.retrieval.storage import record_ref

    value = ChunkSet.model_validate(value.model_dump())
    settings = Phase5Configuration.model_validate(settings.model_dump())
    if value.configuration.resolved_sha256 != sha256_bytes(
        canonical_json_bytes(settings.model_dump())
    ):
        raise ValueError("chunk set configuration identity differs from supplied settings")
    if value.chunk_ids != [record_ref(chunk).artifact_id for chunk in chunks]:
        raise ValueError("chunk set order or identities differ from submitted chunks")
    validate_occurrences(value.occurrences, document, value.input.canonical)
    used: dict[str, list[str]] = {item.occurrence_id: [] for item in value.occurrences}
    for chunk, chunk_id in zip(chunks, value.chunk_ids, strict=True):
        if chunk.input != value.input or chunk.configuration != value.configuration:
            raise ValueError("chunk canonical/configuration references differ from chunk set")
        if chunk.construction_tokens > settings.chunk_tokens:
            raise ValueError("chunk declared count exceeds construction budget")
        validate_chunk_contributors(chunk, document, value.occurrences)
        rendered = {
            item.occurrence_id
            for item in chunk.contributors
            if item.usage != "containing_reference"
        }
        for item_id in rendered:
            used[item_id].append(chunk_id)
    for item in value.accounting:
        if item.chunk_ids != used[item.occurrence_id]:
            raise ValueError("occurrence accounting differs from chunk contributors")


def validate_index_links(index: Bm25Index | DenseIndex, chunk_sets: list[ChunkSet]) -> None:
    from biorag_repro.retrieval.storage import record_ref

    index = type(index).model_validate(index.model_dump())
    if isinstance(index, Bm25Index):
        chunk_sets = sorted(chunk_sets, key=lambda item: record_ref(item).artifact_id)
    if index.chunk_sets != [record_ref(item) for item in chunk_sets]:
        raise ValueError("index chunk-set references differ from selected inputs")
    expected_ids = [chunk_id for item in chunk_sets for chunk_id in item.chunk_ids]
    if isinstance(index, Bm25Index):
        expected_ids.sort()
    if index.ordered_chunk_ids != expected_ids:
        raise ValueError("index chunk order differs from selected chunk sets")
    if any(item.configuration != index.configuration for item in chunk_sets):
        raise ValueError("index configuration differs from selected chunk sets")
    if len({(item.input.representation) for item in chunk_sets}) != 1:
        raise ValueError("index mixes source representations")


def validate_retrieval_links(
    value: RankedRetrieval,
    index: Bm25Index | DenseIndex,
    chunks: list[Chunk],
) -> None:
    from biorag_repro.retrieval.storage import record_ref

    value = RankedRetrieval.model_validate(value.model_dump())
    expected_retriever = "bm25" if isinstance(index, Bm25Index) else "bge"
    if (
        value.index != record_ref(index)
        or value.configuration != index.configuration
        or value.retriever != expected_retriever
    ):
        raise ValueError("ranked retrieval index/configuration/retriever mismatch")
    by_id = {record_ref(item).artifact_id: item for item in chunks}
    for hit in value.hits:
        chunk = by_id.get(hit.chunk.artifact_id)
        if (
            chunk is None
            or hit.chunk.artifact_id not in index.ordered_chunk_ids
            or hit.chunk != record_ref(chunk)
            or hit.paper_id != chunk.input.paper_id
            or value.representation != chunk.input.representation
        ):
            raise ValueError("ranked hit differs from indexed chunk lineage")


def validate_context_links(value: ContextRecord, ranked: RankedRetrieval) -> None:
    from biorag_repro.retrieval.storage import record_ref

    value = ContextRecord.model_validate(value.model_dump())
    if (
        value.retrieval != record_ref(ranked)
        or value.configuration != ranked.configuration
        or value.representation != ranked.representation
    ):
        raise ValueError("context retrieval/configuration/representation mismatch")
    if [(item.rank, item.chunk_id) for item in value.decisions] != [
        (item.rank, item.chunk.artifact_id) for item in ranked.hits
    ]:
        raise ValueError("context candidate decisions differ from ranked inputs")
