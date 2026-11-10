"""Stage B: did the exact supplied context contain one complete, faithful sufficient set?

Delivery is decided from Stage A's located occurrences and the verified chunk
lineage, then confirmed against the text of the supplied chunk. A retrieved hit,
or a source occurrence ID inside an included chunk, is not delivery by itself.
Nothing here reads generator output or benchmark gold.
"""

from dataclasses import dataclass

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import Placement, StageAJudgment
from biorag_repro.evaluation.stage_a import _blocking, _spans, item_digest, stage_a_ref
from biorag_repro.evaluation.stage_a import validate_judgment as validate_stage_a
from biorag_repro.evaluation.stage_b_models import (
    ComponentSpan,
    Delivered,
    Determination,
    Diagnosis,
    Reason,
    StageBCandidate,
    StageBIdentity,
    StageBJudgment,
    Status,
    composition_of,
)
from biorag_repro.retrieval.context import POLICY, _same_evidence
from biorag_repro.retrieval.models import Chunk, ChunkSet, ContextRecord, RankedRetrieval
from biorag_repro.retrieval.storage import record_ref

OPEN, CLOSE = "[evidence]\n", "\n[/evidence]"
# Later entries got further through the pipeline before the evidence was lost.
LOSS_ORDER: list[Reason] = [
    "excluded_in_chunking",
    "not_ranked",
    "lost_in_packing",
    "partial_components",
]


@dataclass(frozen=True)
class Inputs:
    """Records the caller obtained through the authoritative context readers."""

    item: BenchmarkItem
    document: CanonicalDocument
    stage_a: StageAJudgment
    context: ContextRecord
    ranked: RankedRetrieval
    chunk_set: ChunkSet
    chunks: dict[str, Chunk]
    canonical: ArtifactRef
    context_ref: ArtifactRef


@dataclass
class Supplied:
    """The verified final context: included chunks in order, with their offsets."""

    included: dict[str, tuple[int, Chunk, int]]  # chunk id -> (rank, chunk, offset in context)
    designated: dict[str, Chunk]  # every chunk of the item's own canonical artifact
    decisions: dict[str, tuple[str, int | None]]  # chunk id -> (decision, retained rank)
    ranks: dict[int, str]
    designated_included: int
    other_included: int


def _verify(i: Inputs) -> Supplied:
    item, ctx, ranked = i.item, i.context, i.ranked
    validate_stage_a(i.stage_a, item, i.document, i.canonical)
    if (
        i.chunk_set.input.canonical != i.canonical
        or i.chunk_set.input.paper_id != item.paper_id
        or len({i.document.representation, ctx.representation, ranked.representation}) != 1
        or i.chunk_set.input.representation != i.document.representation
    ):
        raise ValueError("item, canonical artifact, chunk set and retrieval name different inputs")
    if i.context_ref != record_ref(ctx) or ctx.retrieval != record_ref(ranked):
        raise ValueError("context is not the claimed retained artifact of this retrieval")
    if ctx.packing_policy not in (None, POLICY):
        raise ValueError("unapproved context packing policy")
    if ranked.scope == "designated_paper" and ranked.designated_paper_id != item.paper_id:
        raise ValueError("retrieval was scoped to another paper")
    if [(h.rank, h.chunk.artifact_id) for h in ranked.hits] != [
        (d.rank, d.chunk_id) for d in ctx.decisions
    ]:
        raise ValueError("context decisions do not follow the ranked retrieval")

    def authentic(chunk_id: str) -> Chunk:
        chunk = i.chunks.get(chunk_id)
        if chunk is None or record_ref(chunk).artifact_id != chunk_id:
            raise ValueError("supplied chunk does not match its recorded identity")
        if chunk.input.representation != i.document.representation:
            raise ValueError("chunk representation differs from the item's arm")
        return chunk

    designated = {cid: authentic(cid) for cid in i.chunk_set.chunk_ids}
    if any(
        c.input.canonical != i.canonical or c.input.paper_id != item.paper_id
        for c in designated.values()
    ):
        raise ValueError("chunk set contains a chunk of another canonical artifact")
    by_rank = {d.rank: d for d in ctx.decisions}
    included: dict[str, tuple[int, Chunk, int]] = {}
    decisions: dict[str, tuple[str, int | None]] = {}
    text, designated_n, other_n = "", 0, 0
    for hit in ranked.hits:
        chunk = authentic(hit.chunk.artifact_id)
        if record_ref(chunk) != hit.chunk or chunk.input.paper_id != hit.paper_id:
            raise ValueError("ranked chunk differs from the supplied chunk")
        if chunk.input.paper_id == item.paper_id and hit.chunk.artifact_id not in designated:
            raise ValueError("designated-paper chunk is not part of the item's chunk set")
        decision = by_rank[hit.rank]
        retained: int | None = None
        if decision.decision == "included":
            wrapped = OPEN + chunk.rendered_text + CLOSE
            text += ("\n\n" if text else "") + wrapped
            included[decision.chunk_id] = (hit.rank, chunk, len(text) - len(wrapped) + len(OPEN))
            retained = hit.rank
            if chunk.input.paper_id == item.paper_id:
                designated_n += 1
            else:
                other_n += 1
        elif decision.decision == "duplicate":
            retained = decision.duplicate_of_rank
            kept = authentic(by_rank[retained].chunk_id) if retained else None
            if kept is None or _same_evidence(kept) != _same_evidence(chunk):
                raise ValueError("duplicate decision does not follow identical evidence")
        decisions[decision.chunk_id] = (decision.decision, retained)
    if text != ctx.exact_context:
        raise ValueError("exact context is not the rendering of the included chunks")
    return Supplied(
        included,
        designated,
        decisions,
        {d.rank: d.chunk_id for d in ctx.decisions},
        designated_n,
        other_n,
    )


def _content(chunk: Chunk) -> set[str]:
    return {c.occurrence_id for c in chunk.contributors if c.usage != "containing_reference"}


def _delivering(v: Supplied, chunk_id: str) -> tuple[int, str] | None:
    """Rank and ID of the included chunk that actually supplies this chunk's evidence."""
    decision, retained = v.decisions.get(chunk_id, ("", None))
    if decision in {"included", "duplicate"} and retained is not None:
        return retained, v.ranks[retained]
    return None


def _review(set_index: int, placement: Placement, detail: str) -> Diagnosis:
    return Diagnosis(
        set_index=set_index,
        container_node_id=placement.container_node_id,
        outcome="review",
        detail=detail,
    )


def _lost(set_index: int, placement: Placement, reason: Reason, detail: str) -> Diagnosis:
    return Diagnosis(
        set_index=set_index,
        container_node_id=placement.container_node_id,
        outcome="lost",
        reason=reason,
        detail=detail,
    )


def _confirm(
    i: Inputs, v: Supplied, set_index: int, placement: Placement, rank: int, chunk_id: str
) -> tuple[Diagnosis, Delivered | None]:
    """Check that the chunk supplying the evidence carries it faithfully and unambiguously."""
    _, chunk, offset = v.included[chunk_id]
    components = placement.components
    if any(not c.verbatim for c in components):
        return _review(
            set_index, placement, "altered evidence is delivered; faithfulness needs review"
        ), None
    known = {o.occurrence_id: o for o in i.chunk_set.occurrences}
    ids = [o for c in components for o in c.occurrence_ids]
    if any(o not in known for o in ids):
        raise ValueError("component occurrence is not in the chunk set accounting")
    problems = sorted({p for o in ids for p in _blocking(known[o])})
    if problems:
        return _review(set_index, placement, "qualified mapping: " + ", ".join(problems)), None
    spans = []
    for c in components:
        found = _spans(chunk.rendered_text, c.text)
        if not found:
            return _review(
                set_index, placement, f"{c.role} text is not verbatim in the supplied chunk"
            ), None
        start, end = found[0]
        spans.append(
            ComponentSpan(role=c.role, text=c.text, start=offset + start, end=offset + end)
        )
    answers = [c.text for c in components if c.role == "answer"]
    rest = [c.text for c in components if c.role != "answer"]
    for other_id, (_, other, _) in v.included.items():
        if other_id == chunk_id:
            continue
        has_all = all(_spans(other.rendered_text, t) for t in rest)
        if has_all and not any(_spans(other.rendered_text, t) for t in answers):
            return (
                _review(
                    set_index,
                    placement,
                    "conflicting value: the context restates this fact differently",
                ),
                None,
            )
    return (
        Diagnosis(
            set_index=set_index,
            container_node_id=placement.container_node_id,
            outcome="delivered",
            detail=f"complete in the included chunk at rank {rank}",
        ),
        Delivered(
            set_index=set_index,
            container_node_id=placement.container_node_id,
            chunk_id=chunk_id,
            rank=rank,
            spans=spans,
        ),
    )


def _placement(
    i: Inputs, v: Supplied, set_index: int, placement: Placement
) -> tuple[Diagnosis, Delivered | None]:
    holders = [
        {cid for cid, ch in v.designated.items() if set(c.occurrence_ids) & _content(ch)}
        for c in placement.components
    ]
    whole = set.intersection(*holders)
    if whole:
        supplied = sorted(d for cid in whole if (d := _delivering(v, cid)) is not None)
        if supplied:
            rank, chunk_id = supplied[0]
            return _confirm(i, v, set_index, placement, rank, chunk_id)
        reason: Reason = "lost_in_packing" if whole & v.decisions.keys() else "not_ranked"
        return _lost(
            set_index,
            placement,
            reason,
            "the chunk holding the whole set did not reach the context",
        ), None
    if not all(holders):
        return _lost(
            set_index, placement, "excluded_in_chunking", "a component is in no chunk"
        ), None
    parts = [any(_delivering(v, cid) for cid in h) for h in holders]
    if all(parts):
        return _review(set_index, placement, "components are in different included chunks"), None
    if any(parts):
        missing = [c.text for c, ok in zip(placement.components, parts, strict=True) if not ok]
        if any(_spans(i.context.exact_context, t) for t in missing):
            return _review(
                set_index, placement, "missing component text occurs elsewhere in the context"
            ), None
        return _lost(
            set_index, placement, "partial_components", "only some components were supplied"
        ), None
    held = set().union(*holders)
    reason = "lost_in_packing" if held & v.decisions.keys() else "not_ranked"
    return _lost(
        set_index, placement, reason, "no chunk with these components reached the context"
    ), None


def _identity(i: Inputs, v: Supplied) -> StageBIdentity:
    return StageBIdentity(
        benchmark_version=i.item.benchmark_version,
        corpus_version=i.item.corpus_version,
        item_id=i.item.item_id,
        item_sha256=item_digest(i.item),
        paper_id=i.item.paper_id,
        representation=i.document.representation,
        stage_a=stage_a_ref(i.stage_a),
        context=i.context_ref,
        retrieval=i.context.retrieval,
        chunk_set=record_ref(i.chunk_set),
        configuration=i.context.configuration,
        composition=composition_of(v.designated_included, v.other_included),
        designated_chunks_included=v.designated_included,
        other_chunks_included=v.other_included,
    )


def _candidate(
    base: StageBIdentity,
    determination: Determination,
    diagnoses: list[Diagnosis],
    delivered: list[Delivered] | None = None,
    reason: Reason | None = None,
) -> StageBCandidate:
    return StageBCandidate.model_validate(
        {
            **base.model_dump(),
            "determination": determination,
            "reason": reason,
            "delivered": [d.model_dump() for d in delivered or []],
            "diagnoses": [d.model_dump() for d in diagnoses],
        }
    )


def evaluate(i: Inputs) -> StageBCandidate:
    v = _verify(i)
    base = _identity(i, v)
    placements = [(s.set_index, p) for s in i.stage_a.sets for p in s.placements]
    label = i.stage_a.label
    if label in {"scientifically_corrupted", "absent"}:
        return _candidate(base, "not_reachable_stage_a", [])
    if label == "unresolved":
        return _candidate(base, "non_evaluable_stage_a", [])
    if base.composition in {"empty", "other_only"}:
        reason: Reason = "empty_context" if base.composition == "empty" else "other_paper_only"
        diagnoses = [_lost(s, p, reason, f"context is {base.composition}") for s, p in placements]
        return _candidate(base, "automatic_not_delivered", diagnoses, reason=reason)
    results = [_placement(i, v, s, p) for s, p in placements]
    diagnoses = [d for d, _ in results]
    delivered = [x for _, x in results if x is not None]
    if delivered:
        return _candidate(base, "automatic_delivered", diagnoses, delivered)
    if any(d.outcome == "review" for d in diagnoses):
        return _candidate(base, "needs_review", diagnoses)
    best = max((d.reason for d in diagnoses if d.reason is not None), key=LOSS_ORDER.index)
    return _candidate(base, "automatic_not_delivered", diagnoses, reason=best)


def dependencies(value: StageBCandidate | StageBJudgment) -> list[tuple[str, ArtifactRef]]:
    items = [
        ("stage_a", value.stage_a),
        ("context", value.context),
        ("retrieval", value.retrieval),
        ("chunk_set", value.chunk_set),
        (
            "benchmark_item",
            ArtifactRef(
                artifact_id=value.item_id,
                logical_ref=f"benchmark/{value.benchmark_version}/items/{value.item_id}",
                sha256=value.item_sha256,
            ),
        ),
    ]
    if isinstance(value, StageBJudgment):
        items.append(("stage_b_candidate", value.candidate))
    return items


def stage_b_ref(value: StageBCandidate | StageBJudgment) -> ArtifactRef:
    """Identify a record by its bytes and the exact Stage A, context and chunk lineage."""
    kind = "stage_b_candidate" if isinstance(value, StageBCandidate) else "stage_b_judgment"
    digest = sha256_bytes(serialize_project_json(value, type(value)))
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[
            ArtifactInput(role, ref.artifact_id, ref.sha256) for role, ref in dependencies(value)
        ],
        transformation_id=f"{kind}_v1",
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"evaluation/stage_b/{kind}/{artifact_id}.json",
        sha256=digest,
    )


_STATUS: dict[str, Status] = {
    "automatic_delivered": "delivered",
    "automatic_not_delivered": "not_delivered",
    "not_reachable_stage_a": "not_reachable_stage_a",
    "non_evaluable_stage_a": "non_evaluable",
}


def deterministic_judgment(candidate: StageBCandidate) -> StageBJudgment:
    """The only reviewer-free judgment: a restatement of an automatic determination."""
    if candidate.determination == "needs_review":
        raise ValueError("a review case has no deterministic judgment")
    identity = {k: getattr(candidate, k) for k in StageBIdentity.model_fields}
    return StageBJudgment(
        **identity,
        status=_STATUS[candidate.determination],
        basis="deterministic",
        candidate=stage_b_ref(candidate),
        reason=candidate.reason,
        delivered=candidate.delivered,
    )


def validate_candidate(candidate: StageBCandidate, i: Inputs) -> None:
    if evaluate(i) != candidate:
        raise ValueError("candidate differs from the evaluator output for these inputs")


def validate_judgment(judgment: StageBJudgment, i: Inputs) -> None:
    """Scientific read-back: lineage, determination and any reviewer citation."""
    candidate = evaluate(i)
    identity = {k: getattr(candidate, k) for k in StageBIdentity.model_fields}
    if {k: getattr(judgment, k) for k in identity} != identity:
        raise ValueError("judgment identity differs from the supplied inputs")
    if judgment.candidate != stage_b_ref(candidate):
        raise ValueError("judgment cites a candidate that is not the evaluator output")
    if judgment.basis == "deterministic":
        if judgment != deterministic_judgment(candidate):
            raise ValueError("deterministic judgment is not reproduced by the evaluator")
        return
    if candidate.determination != "needs_review":
        raise ValueError("a reviewer may decide only cases the evaluator routed to review")
    if judgment.reason == "empty_context" or judgment.reason == "other_paper_only":
        raise ValueError("composition-based reasons are not reviewable")
    supplied = {(s.set_index, p.container_node_id) for s in i.stage_a.sets for p in s.placements}
    v = _verify(i)
    for cited in judgment.delivered:
        entry = v.included.get(cited.chunk_id)
        if entry is None or entry[1].input.paper_id != i.item.paper_id or entry[0] != cited.rank:
            raise ValueError(
                "cited evidence is not a designated-paper chunk of the supplied context"
            )
        if (cited.set_index, cited.container_node_id) not in supplied:
            raise ValueError("cited evidence names no located Stage A placement")
        if any(i.context.exact_context[s.start : s.end] != s.text for s in cited.spans):
            raise ValueError("cited span is not in the exact supplied context")
