"""Stage B: did the exact supplied context contain one complete, faithful sufficient set?

Delivery is decided from Stage A's located occurrences and the verified chunk
lineage, then confirmed against the text of the supplied chunk. A retrieved hit,
or a source occurrence ID inside an included chunk, is not delivery by itself.
Nothing here reads generator output or benchmark gold.
"""

from dataclasses import dataclass

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.manifest import ArtifactRef
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
    Rules,
    Span,
    StageBCandidate,
    StageBIdentity,
    StageBJudgment,
    Status,
    composition_of,
)
from biorag_repro.evaluation.stage_records import stage_record_ref
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
# Rules v2: these say only that the recorded placements were lost. Another passage of the
# supplied context may still carry the set, so they need a whole-context review.
LINEAGE_LOSSES = set(LOSS_ORDER)


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


def kind_of(value: StageBCandidate | StageBJudgment) -> str:
    return "stage_b_candidate" if isinstance(value, StageBCandidate) else "stage_b_judgment"


def transformation_id(value: StageBCandidate | StageBJudgment) -> str:
    rules = "v1" if isinstance(value, StageBCandidate) else value.rules
    return f"{kind_of(value)}_{rules}"


def stage_b_ref(value: StageBCandidate | StageBJudgment) -> ArtifactRef:
    """Identify a record by its bytes and the exact Stage A, context and chunk lineage."""
    return stage_record_ref(
        value, kind_of(value), "stage_b", dependencies(value), transformation_id(value)
    )


_STATUS: dict[str, Status] = {
    "automatic_delivered": "delivered",
    "automatic_not_delivered": "not_delivered",
    "not_reachable_stage_a": "not_reachable_stage_a",
    "non_evaluable_stage_a": "non_evaluable",
}


def _unambiguous_prose(i: Inputs, v: Supplied, delivered: Delivered) -> bool:
    """Rules v2: an automatic positive needs prose whose every component occurs once in the
    supplying chunk, on one line. Tables, captions and repeated wording go to review."""
    node = next((n for n in i.document.nodes if n.node_id == delivered.container_node_id), None)
    if node is None or node.table is not None or node.kind not in {"paragraph", "list_item"}:
        return False
    _, chunk, offset = v.included[delivered.chunk_id]
    text = chunk.rendered_text
    lines = set()
    for span in delivered.spans:
        if span.role in {"row", "column"} or len(_spans(text, span.text)) != 1:
            return False
        lines.add(text.count("\n", 0, span.start - offset))
        lines.add(text.count("\n", 0, span.end - offset))
    return len(lines) == 1


def _checked(candidate: StageBCandidate, i: Inputs) -> Supplied:
    if evaluate(i) != candidate:
        raise ValueError("candidate differs from the evaluator output for these inputs")
    return _verify(i)


def _v2_delivered(candidate: StageBCandidate, i: Inputs) -> list[Delivered]:
    v = _checked(candidate, i)
    return [d for d in candidate.delivered if _unambiguous_prose(i, v, d)]


def needs_review_v2(candidate: StageBCandidate, i: Inputs) -> bool:
    """Whether rules v2 leave this context to a whole-context reviewer."""
    if candidate.determination == "needs_review":
        _checked(candidate, i)
        return True
    if candidate.determination == "automatic_not_delivered":
        _checked(candidate, i)
        return candidate.reason in LINEAGE_LOSSES
    if candidate.determination == "automatic_delivered":
        return not _v2_delivered(candidate, i)
    _checked(candidate, i)
    return False


def deterministic_judgment(
    candidate: StageBCandidate, rules: Rules = "v1", inputs: Inputs | None = None
) -> StageBJudgment:
    """The only reviewer-free judgment: a restatement of an automatic determination.

    Rules v2 keep only the Stage A states, composition losses and unambiguous verbatim prose;
    v2 automatic delivery is rechecked against the inputs the candidate was evaluated from.
    """
    if candidate.determination == "needs_review":
        raise ValueError("a review case has no deterministic judgment")
    delivered = candidate.delivered
    if rules == "v2":
        if candidate.reason in LINEAGE_LOSSES:
            raise ValueError("a lineage-only loss needs whole-context review under rules v2")
        if candidate.determination == "automatic_delivered":
            if inputs is None:
                raise ValueError("v2 automatic delivery is checked against its inputs")
            delivered = _v2_delivered(candidate, inputs)
            if not delivered:
                raise ValueError("delivered evidence is not unambiguous prose: needs review")
    identity = {k: getattr(candidate, k) for k in StageBIdentity.model_fields}
    return StageBJudgment(
        **identity,
        status=_STATUS[candidate.determination],
        basis="deterministic",
        candidate=stage_b_ref(candidate),
        reason=candidate.reason,
        delivered=delivered,
        rules=rules,
    )


def validate_candidate(candidate: StageBCandidate, i: Inputs) -> None:
    if evaluate(i) != candidate:
        raise ValueError("candidate differs from the evaluator output for these inputs")


def cite(i: Inputs, chunk_id: str, quote: str, occurrence: int | None = None) -> Span:
    """Exact span of a quote inside one included designated-paper chunk of the context."""
    v = _verify(i)
    entry = v.included.get(chunk_id)
    if entry is None or entry[1].input.paper_id != i.item.paper_id:
        raise ValueError("cited chunk is not a designated-paper chunk of the supplied context")
    _, chunk, offset = entry
    text, hits = chunk.rendered_text, []
    at = text.find(quote) if quote else -1
    while at >= 0:
        hits.append(at)
        at = text.find(quote, at + 1)
    if not hits:
        raise ValueError("quote is not in the cited chunk")
    if occurrence is None:
        if len(hits) > 1:
            raise ValueError(f"quote occurs {len(hits)} times in the chunk; give its occurrence")
        occurrence = 1
    if not 1 <= occurrence <= len(hits):
        raise ValueError("occurrence out of range")
    start = offset + hits[occurrence - 1]
    return Span(chunk_id=chunk_id, start=start, end=start + len(quote), text=quote)


def _check_span(i: Inputs, v: Supplied, span: Span) -> None:
    entry = v.included.get(span.chunk_id)
    if entry is None or entry[1].input.paper_id != i.item.paper_id:
        raise ValueError("cited chunk is not a designated-paper chunk of the supplied context")
    _, chunk, offset = entry
    if not (offset <= span.start and span.end <= offset + len(chunk.rendered_text)):
        raise ValueError("cited span lies outside its chunk")
    if i.context.exact_context[span.start : span.end] != span.text:
        raise ValueError("cited span is not in the exact supplied context")


def _validate_review(judgment: StageBJudgment, i: Inputs, candidate: StageBCandidate) -> None:
    """Rules v2 whole-context review: every citation is exact supplied text; a positive cites
    every component of one set (from any number of chunks); a negative names a missing or
    unfaithful component of every set. Whether the text means it is the reviewer's judgment."""
    if not needs_review_v2(candidate, i):
        raise ValueError("a reviewer may decide only cases rules v2 routes to review")
    v = _verify(i)
    sets = [len(s.components) for s in i.item.evidence_sets]
    named = [(c.set_index, c.component_index) for c in judgment.citations] + [
        (m.set_index, m.component_index) for m in judgment.missing_components
    ]
    for set_index, component_index in named:
        if set_index >= len(sets) or component_index >= sets[set_index]:
            raise ValueError("citation names a set or component the item does not have")
    for span in [*judgment.citations, *judgment.conflict_citations]:
        _check_span(i, v, span)
    cited = {(c.set_index, c.component_index) for c in judgment.citations}
    complete = [s for s, n in enumerate(sets) if all((s, k) in cited for k in range(n))]
    if judgment.status == "delivered":
        if len({c.set_index for c in judgment.citations}) != 1 or not complete:
            raise ValueError("delivered cites every component of one evidence set")
    elif judgment.status == "not_delivered":
        if {m.set_index for m in judgment.missing_components} != set(range(len(sets))):
            raise ValueError("not_delivered names a missing or unfaithful component of every set")
        if complete and judgment.reason not in {"unfaithful_in_context", "ambiguous_referent"}:
            raise ValueError(
                "a fully cited set is lost only as unfaithful_in_context or ambiguous_referent"
            )


def validate_judgment(judgment: StageBJudgment, i: Inputs) -> None:
    """Scientific read-back: lineage, determination and any reviewer citation."""
    candidate = evaluate(i)
    identity = {k: getattr(candidate, k) for k in StageBIdentity.model_fields}
    if {k: getattr(judgment, k) for k in identity} != identity:
        raise ValueError("judgment identity differs from the supplied inputs")
    if judgment.candidate != stage_b_ref(candidate):
        raise ValueError("judgment cites a candidate that is not the evaluator output")
    if judgment.basis == "deterministic":
        if judgment != deterministic_judgment(candidate, judgment.rules, i):
            raise ValueError("deterministic judgment is not reproduced by the evaluator")
        return
    if judgment.rules == "v2":
        _validate_review(judgment, i, candidate)
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
