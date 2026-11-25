"""Explicit development-only readback of retained Stage B generator contexts."""

import hashlib
import json
from pathlib import Path

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.stage_a_retrospective import read_judgment as read_stage_a
from biorag_repro.evaluation.stage_b import Inputs, validate_judgment
from biorag_repro.evaluation.stage_b_models import StageBJudgment
from biorag_repro.retrieval.models import Bm25Index, Chunk, ChunkSet, ContextRecord, RankedRetrieval
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import _node_contains, validate_occurrences

VERSION = "historical_supplied_context_readback_v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read(path: Path, digest: str) -> dict:
    raw = path.read_bytes()
    if _sha(raw) != digest:
        raise ValueError(f"retained artifact hash differs: {path}")
    return json.loads(raw)


def _artifact(root: Path, value: dict, model: type):
    ref = ArtifactRef.model_validate(value)
    record = model.model_validate(_read(root / ref.logical_ref, ref.sha256))
    if record_ref(record) != ref:
        raise ValueError("retained artifact identity differs")
    return record


def check_delivery(recorded: str, sealed: str, spans: list[dict]) -> bool:
    if recorded != sealed:
        raise ValueError("supplied context changed")
    for span in spans:
        if recorded[span["start"] : span["end"]] != span["text"]:
            raise ValueError("evidence span is absent from supplied context")
    return True


def _links(chunks: list[Chunk], chunk_set: ChunkSet, document: CanonicalDocument) -> list[dict]:
    validate_occurrences(chunk_set.occurrences, document, chunk_set.input.canonical)
    by_occurrence = {o.occurrence_id: o for o in chunk_set.occurrences}
    by_node = {n.node_id: n for n in document.nodes}
    failures = []
    for chunk in chunks:
        for contributor in chunk.contributors:
            occurrence = by_occurrence.get(contributor.occurrence_id)
            node = by_node.get(contributor.node_id)
            if occurrence is None or node is None or not _node_contains(node, occurrence):
                failures.append(
                    {
                        "chunk_id": record_ref(chunk).artifact_id,
                        "occurrence_id": contributor.occurrence_id,
                        "declared_node_id": contributor.node_id,
                        "owner_node_id": occurrence.owner_node_id if occurrence else None,
                        "canonical_sha256": chunk.input.canonical.sha256,
                        "source_artifact_id": document.source.artifact_id,
                        "source_sha256": document.source.sha256,
                    }
                )
    return failures


def read_case(
    *,
    packet: dict,
    draft: dict,
    accounting: dict | None,
    known_failures: list[dict] | None,
    observations: dict[str, dict],
    data_root: Path,
    corpus_root: Path,
    canonical_root: Path,
    foreign_canonical_root: Path,
    selected_canonical_roots: dict[tuple[str, str], Path],
    frozen_sources: dict[tuple[str, str], tuple[str, str]],
    source_selection: dict,
    item: BenchmarkItem,
    phase: str,
    return_inputs: bool = False,
) -> dict | tuple[dict, Inputs]:
    """Read one pre-existing development case; never derive a new selected context."""
    number = packet["number"]
    if phase != "historical_development" or number not in range(1, 284):
        raise ValueError("historical reader requires an authorized development case")
    record = draft["record"]
    keys = draft["observation_keys"]
    if packet["number"] != draft["case_number"] or not keys or len(keys) != len(set(keys)):
        raise ValueError("historical case or observation accounting conflicts")
    if packet["candidate_ref"] != record["candidate"] or packet["candidate"] != _read(
        data_root / record["candidate"]["logical_ref"], record["candidate"]["sha256"]
    ):
        raise ValueError("original candidate differs from reviewed record")
    if (record["item_id"], record["paper_id"], record["representation"]) != (
        item.item_id,
        item.paper_id,
        source_selection["representation"],
    ):
        raise ValueError("item or selected source identity differs")
    if source_selection["paper_id"] != item.paper_id:
        raise ValueError("unknown selected source")
    if record["status"] != "delivered" or packet["review_status"] != "pending_researcher_judgment":
        raise ValueError("historical route requires the original reviewed YES case")
    context = _artifact(data_root, record["context"], ContextRecord)
    ranked = _artifact(data_root, record["retrieval"], RankedRetrieval)
    chunk_set = _artifact(data_root, record["chunk_set"], ChunkSet)
    if context.exact_context != packet["exact_supplied_context"]:
        raise ValueError("packet and retained context bytes differ")
    if (
        context.retrieval != ArtifactRef.model_validate(record["retrieval"])
        or chunk_set.input.canonical.sha256 != source_selection["canonical"]["sha256"]
    ):
        raise ValueError("selected context source differs")
    canonical_ref = ArtifactRef.model_validate(source_selection["canonical"])
    canonical = CanonicalDocument.model_validate(
        _read(canonical_root / canonical_ref.logical_ref, canonical_ref.sha256)
    )
    source_ref = source_selection["source"]
    if (source_ref["artifact_id"], source_ref["sha256"]) != frozen_sources.get(
        (item.paper_id, context.representation)
    ):
        raise ValueError("designated source differs from frozen corpus manifest")
    if canonical.source.model_dump() != source_ref or chunk_set.input.canonical != canonical_ref:
        raise ValueError("canonical or frozen source identity differs")
    _verify_raw(corpus_root / "raw" / source_ref["logical_ref"], source_ref["sha256"])
    raw_chunks = (data_root / chunk_set.chunks.logical_ref).read_bytes()
    if _sha(raw_chunks) != chunk_set.chunks.sha256:
        raise ValueError("historical chunks bytes differ")
    chunks = [Chunk.model_validate(json.loads(line)) for line in raw_chunks.splitlines()]
    if [record_ref(chunk).artifact_id for chunk in chunks] != chunk_set.chunk_ids:
        raise ValueError("historical chunk identities differ")
    by_id = {record_ref(chunk).artifact_id: chunk for chunk in chunks}
    foreign_sets = {}
    foreign_chunks = {}
    if any(hit.chunk.artifact_id not in by_id for hit in ranked.hits):
        index = _artifact(data_root, ranked.index.model_dump(), Bm25Index)
        sets_by_paper = {}
        for ref in index.chunk_sets:
            other_set = _artifact(data_root, ref.model_dump(), ChunkSet)
            sets_by_paper[other_set.input.paper_id] = other_set
        for hit in ranked.hits:
            if hit.chunk.artifact_id in by_id:
                continue
            other_set = sets_by_paper.get(hit.paper_id)
            if other_set is None or hit.chunk.artifact_id not in other_set.chunk_ids:
                raise ValueError("foreign ranked chunk has no original chunk set")
            if hit.paper_id not in foreign_chunks:
                foreign_bytes = (data_root / other_set.chunks.logical_ref).read_bytes()
                if _sha(foreign_bytes) != other_set.chunks.sha256:
                    raise ValueError("foreign historical chunk bytes differ")
                all_foreign = [
                    Chunk.model_validate(json.loads(line)) for line in foreign_bytes.splitlines()
                ]
                if [record_ref(c).artifact_id for c in all_foreign] != other_set.chunk_ids:
                    raise ValueError("foreign historical chunk-set identities differ")
                foreign_chunks[hit.paper_id] = dict(
                    zip(other_set.chunk_ids, all_foreign, strict=True)
                )
            if hit.chunk.artifact_id in context.included_chunk_ids:
                foreign_sets[hit.paper_id] = other_set
            chunk = foreign_chunks[hit.paper_id][hit.chunk.artifact_id]
            if record_ref(chunk) != hit.chunk or chunk.input != other_set.input:
                raise ValueError("foreign ranked chunk identity differs")
            by_id[hit.chunk.artifact_id] = chunk
    if [decision.chunk_id for decision in context.decisions] != [
        hit.chunk.artifact_id for hit in ranked.hits
    ]:
        raise ValueError("ranked selected-chunk identities differ")
    if context.included_chunk_ids != [
        d.chunk_id for d in context.decisions if d.decision == "included"
    ]:
        raise ValueError("included-chunk accounting differs")
    selected = [by_id[cid] for cid in context.included_chunk_ids]
    for hit in ranked.hits:
        if (
            hit.chunk.artifact_id not in by_id
            or record_ref(by_id[hit.chunk.artifact_id]) != hit.chunk
        ):
            raise ValueError("substituted ranked chunk identity")
    designated_selected = [chunk for chunk in selected if chunk.input.paper_id == item.paper_id]
    selected_failures = _links(designated_selected, chunk_set, canonical)
    full_failures = _links(chunks, chunk_set, canonical)
    missing_foreign = []
    missing_selected_foreign = []
    for paper_id, other_set in foreign_sets.items():
        foreign_canonical_ref = other_set.input.canonical
        foreign_path = foreign_canonical_root / foreign_canonical_ref.logical_ref
        if not foreign_path.is_file():
            selected_root = selected_canonical_roots.get(
                (paper_id, context.representation), canonical_root
            )
            foreign_path = selected_root / foreign_canonical_ref.logical_ref
        if not foreign_path.is_file():
            missing_foreign.append(
                {"paper_id": paper_id, "canonical": foreign_canonical_ref.model_dump()}
            )
            if any(c.input.paper_id == paper_id for c in selected):
                missing_selected_foreign.append(paper_id)
            continue
        foreign_document = CanonicalDocument.model_validate(
            _read(foreign_path, foreign_canonical_ref.sha256)
        )
        if (
            foreign_document.paper_id != paper_id
            or foreign_document.representation != context.representation
        ):
            raise ValueError("foreign canonical source differs")
        if (
            foreign_document.source.artifact_id,
            foreign_document.source.sha256,
        ) != frozen_sources.get((paper_id, context.representation)):
            raise ValueError("foreign source differs from frozen corpus manifest")
        _verify_raw(
            corpus_root / "raw" / foreign_document.source.logical_ref,
            foreign_document.source.sha256,
        )
        other_selected = [c for c in selected if c.input.paper_id == paper_id]
        selected_failures.extend(_links(other_selected, other_set, foreign_document))
        full_failures.extend(
            _links(list(foreign_chunks[paper_id].values()), other_set, foreign_document)
        )
    approved = accounting is not None
    actual_links = {
        (v["chunk_id"], v["occurrence_id"], v["declared_node_id"]) for v in selected_failures
    }
    expected_links = {
        (v["chunk_id"], v["occurrence_id"], v["contributor_node_id"]) for v in known_failures or []
    }
    if approved and (
        accounting["case_number"] != number
        or accounting["observation_keys"] != keys
        or accounting["delivery_decision"] != "YES"
        or accounting["contributor_node_provenance_certification"] != "FAILED"
    ):
        raise ValueError("approved exception accounting conflicts")
    if approved and (not expected_links or not expected_links <= actual_links):
        raise ValueError("approved exception accounting is missing or differs")
    if not approved and known_failures:
        raise ValueError("exception link accounting lacks approval")
    if actual_links - expected_links:
        disposition = "RESEARCHER_DECISION_REQUIRED"
    else:
        disposition = "APPROVED" if approved else "NOT_NEEDED"
    judgment = StageBJudgment.model_validate(
        {**record, "reviewer": "Rama Krishna Pudota", "at_utc": "2026-11-20T04:54:26.985015Z"}
    )
    spans = [s.model_dump() for delivered in judgment.delivered for s in delivered.spans]
    for key in keys:
        sealed = observations.get(key)
        if sealed is None or sealed["context"] != record["context"]:
            raise ValueError(f"missing or conflicting sealed observation {key}")
        check_delivery(context.exact_context, sealed["exact_context"], spans)
    stage_a = read_stage_a(
        ArtifactRef.model_validate(record["stage_a"]), data_root, item, canonical, canonical_ref
    )
    inputs = Inputs(
        item,
        canonical,
        stage_a,
        context,
        ranked,
        chunk_set,
        by_id,
        canonical_ref,
        ArtifactRef.model_validate(record["context"]),
        data_root,
    )
    validate_judgment(judgment, inputs)
    result = {
        "case_number": number,
        "observation_keys": keys,
        "item_id": item.item_id,
        "paper_id": item.paper_id,
        "representation": record["representation"],
        "source_sha256": source_ref["sha256"],
        "canonical_sha256": canonical_ref.sha256,
        "context_sha256": record["context"]["sha256"],
        "source_authenticity": "NOT_ESTABLISHED" if missing_selected_foreign else "PASS",
        "designated_source_authenticity": "PASS",
        "exact_context_delivery": "PASS",
        "evidence_sufficiency": "PASS",
        "selected_context_provenance": "FAILED"
        if selected_failures
        else "NOT_ESTABLISHED"
        if missing_selected_foreign
        else "PASS",
        "foreign_selected_canonical_status": "NOT_ESTABLISHED"
        if missing_selected_foreign
        else "PASS"
        if foreign_sets
        else "NOT_APPLICABLE",
        "complete_set_provenance": "FAILED"
        if full_failures
        else "NOT_ESTABLISHED"
        if missing_foreign
        else "PASS",
        "missing_foreign_canonical": missing_foreign,
        "selected_failures": selected_failures,
        "complete_set_failure_count": len(full_failures),
        "researcher_disposition": disposition,
        "ordinary_official_readback": "NOT_RUN",
        "original_status": record["status"],
    }
    return (result, inputs) if return_inputs else result


def _verify_raw(path: Path, digest: str) -> None:
    if _sha(path.read_bytes()) != digest:
        raise ValueError(f"frozen source hash differs: {path}")
