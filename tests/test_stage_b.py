"""Stage B: was a complete, faithful sufficient evidence set in the exact supplied context?"""

import inspect
from dataclasses import dataclass, field

import pytest
from pydantic import ValidationError
from test_chunking import build
from test_retrieval_models import configuration, ref
from test_stage_a import (
    CANONICAL,
    PROSE_COMPONENTS,
    judgment,
    locus,
    make_item,
    occurrence_ids,
    with_text,
)

from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation import stage_b
from biorag_repro.evaluation.models import StageAJudgment
from biorag_repro.evaluation.stage_a import automatic_judgment, locate, stage_a_ref
from biorag_repro.evaluation.stage_b_models import Delivered, StageBIdentity, StageBJudgment
from biorag_repro.evaluation.stage_b_storage import (
    publish_candidate,
    publish_judgment,
    read_candidate,
    read_judgment,
)
from biorag_repro.retrieval import context
from biorag_repro.retrieval.models import Chunk, ChunkSet, ContextRecord, RankedHit, RankedRetrieval
from biorag_repro.retrieval.storage import record_ref

P_A = "Under condition A, treatment effect was −1.2 × 10⁻³ in the cohort."
P_A2 = "Under condition A, treatment effect was −3.4 × 10⁻³ in the replication."
P_B = "Under condition B, vaccine titre was 4.5 in the cohort."
P_C = "Unrelated methods text about sequencing depth."
B_COMPONENTS = [
    ("subject", "vaccine"),
    ("relation", "titre"),
    ("answer", "4.5"),
    ("condition", "condition B"),
]


def prose_xml(*paragraphs):
    body = "".join(f"<p>{p}</p>" for p in paragraphs)
    return f"<article><body><sec><title>Results</title>{body}</sec></body></article>"


def table_xml(second_value="0.55"):
    return (
        "<article><body><table-wrap><caption><p>Classifier performance</p></caption><table>"
        "<thead><tr><th>Gene</th><th>AUROC</th></tr></thead>"
        "<tbody><tr><th>TP53</th><td>0.91</td></tr>"
        f"<tr><th>KRAS</th><td>{second_value}</td></tr></tbody></table>"
        "<table-wrap-foot><fn><p>adjusted for age</p></fn></table-wrap-foot>"
        "</table-wrap></body></article>"
    )


TABLE_COMPONENTS = [
    ("subject", "Classifier performance"),
    ("relation", "AUROC"),
    ("answer", "0.91"),
    ("row", "TP53"),
    ("column", "AUROC"),
    ("qualifier", "adjusted for age"),
]


@dataclass
class World:
    document: CanonicalDocument
    item: object
    stage_a: StageAJudgment
    chunk_set: ChunkSet
    chunks: dict[str, Chunk]
    by_text: dict = field(default_factory=dict)

    def chunk(self, needle, *, whole=False):
        for c in self.chunks.values():
            if (c.rendered_text == needle) if whole else (needle in c.rendered_text):
                return c
        raise KeyError(needle)


def prose_world(*paragraphs, components=(PROSE_COMPONENTS,), per_paragraph=11):
    document, _, _, _, result = build(
        prose_xml(*paragraphs), counter=lambda s: len(s.split()) * per_paragraph
    )
    return finish(document, result, make_item(document, *components))


def table_world(second_value="0.55"):
    document, _, _, _, result = build(table_xml(second_value), counter=lambda s: len(s.split()) * 3)
    return finish(document, result, make_item(document, TABLE_COMPONENTS))


def finish(document, result, item, stage_a=None):
    if stage_a is None:
        candidate = locate(item, document, CANONICAL)
        stage_a = automatic_judgment(candidate, stage_a_ref(candidate))
    chunks = {record_ref(c).artifact_id: c for c in result.chunks}
    return World(document, item, stage_a, result.chunk_set, chunks)


@dataclass
class Case:
    world: World
    ranked: RankedRetrieval
    ctx: ContextRecord
    chunks: dict[str, Chunk]

    def inputs(self, **override):
        args = {
            "item": self.world.item,
            "document": self.world.document,
            "stage_a": self.world.stage_a,
            "context": self.ctx,
            "ranked": self.ranked,
            "chunk_set": self.world.chunk_set,
            "chunks": self.chunks,
            "canonical": CANONICAL,
            "context_ref": record_ref(self.ctx),
        }
        args.update(override)
        return stage_b.Inputs(**args)

    def evaluate(self, **override):
        return stage_b.evaluate(self.inputs(**override))


def deliver(world, picks, count=lambda s: 1, scope="designated_paper", extra=None):
    decisions, text, tokens, stop = context.pack(picks, count)
    hits = [
        RankedHit(rank=i, chunk=record_ref(c), paper_id=c.input.paper_id, score=100.0 - i)
        for i, c in enumerate(picks, 1)
    ]
    ranked = RankedRetrieval(
        configuration=configuration(),
        index=ref("index"),
        retriever="bm25",
        query_id="q_001",
        question="What was reported?",
        representation=world.document.representation,
        scope=scope,
        designated_paper_id=world.item.paper_id if scope == "designated_paper" else None,
        hits=hits,
    )
    ctx = ContextRecord(
        configuration=configuration(),
        retrieval=record_ref(ranked),
        packing_policy=context.POLICY,
        representation=world.document.representation,
        decisions=decisions,
        included_chunk_ids=[d.chunk_id for d in decisions if d.decision == "included"],
        exact_context=text,
        generator_tokens=tokens,
        stop_reason=stop,
    )
    chunks = dict(world.chunks)
    chunks.update({record_ref(c).artifact_id: c for c in picks})
    chunks.update(extra or {})
    return Case(world, ranked, ctx, chunks)


def other_paper(chunk, paper_id="P002"):
    return chunk.model_copy(update={"input": chunk.input.model_copy(update={"paper_id": paper_id})})


def strip_from_chunk(world, chunk, text, dropped_ids):
    """A chunk set in which one chunk no longer carries some text (Phase 5 split variant)."""
    new = chunk.model_copy(
        update={
            "rendered_text": chunk.rendered_text.replace(text, ""),
            "contributors": [c for c in chunk.contributors if c.occurrence_id not in dropped_ids],
        }
    )
    old_id, new_id = record_ref(chunk).artifact_id, record_ref(new).artifact_id
    accounting = [
        a.model_copy(
            update={
                "chunk_ids": [
                    new_id if i == old_id else i
                    for i in a.chunk_ids
                    if not (i == old_id and a.occurrence_id in dropped_ids)
                ]
            }
        )
        for a in world.chunk_set.accounting
    ]
    chunk_set = world.chunk_set.model_copy(
        update={
            "chunk_ids": [new_id if i == old_id else i for i in world.chunk_set.chunk_ids],
            "accounting": accounting,
        }
    )
    chunks = {i: c for i, c in world.chunks.items() if i != old_id}
    chunks[new_id] = new
    return World(world.document, world.item, world.stage_a, chunk_set, chunks), new


FOOTNOTE = "[footnote]\nadjusted for age\n[/footnote]\n"


# ---- delivery requires the complete set in the exact final context ------------------------


def test_complete_sufficient_evidence_is_delivered_with_inspectable_spans():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    result = case.evaluate()
    assert result.determination == "automatic_delivered"
    assert result.composition == "designated_only"
    (delivered,) = result.delivered
    assert delivered.rank == 1 and delivered.set_index == 0
    for span in delivered.spans:
        assert case.ctx.exact_context[span.start : span.end] == span.text
    assert {s.role for s in delivered.spans} == {r for r, _ in PROSE_COMPONENTS}


def test_one_complete_alternative_set_is_enough():
    w = prose_world(P_A, P_B, P_C, components=(PROSE_COMPONENTS, B_COMPONENTS))
    assert [s.set_index for s in w.stage_a.sets] == [0, 1]
    result = deliver(w, [w.chunk(P_B)]).evaluate()
    assert result.determination == "automatic_delivered"
    assert [d.set_index for d in result.delivered] == [1]
    lost = [d for d in result.diagnoses if d.outcome == "lost"]
    assert [(d.set_index, d.reason) for d in lost] == [(0, "not_ranked")]


def test_table_value_under_another_row_is_not_delivery_of_the_designated_cell():
    w = table_world(second_value="0.91")  # TP53 and KRAS both report 0.91
    case = deliver(w, [w.chunk("KRAS")])
    assert "0.91" in case.ctx.exact_context
    result = case.evaluate()
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "not_ranked"


def test_table_row_chunk_delivers_value_with_row_column_and_qualifier():
    w = table_world()
    result = deliver(w, [w.chunk("TP53")]).evaluate()
    assert result.determination == "automatic_delivered"
    assert {s.role for s in result.delivered[0].spans} >= {"answer", "row", "column", "qualifier"}


def test_value_present_but_necessary_qualifier_missing_is_not_delivery():
    w = table_world()
    row = w.chunk("TP53")
    qualifier_ids = set(occurrence_ids(w.document, "adjusted for age"))
    variant, stripped = strip_from_chunk(w, row, FOOTNOTE, qualifier_ids)
    assert "adjusted for age" not in stripped.rendered_text
    result = deliver(variant, [stripped]).evaluate()
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "partial_components"


def test_required_evidence_split_across_chunks_delivers_only_part():
    w = table_world()
    qualifier_ids = set(occurrence_ids(w.document, "adjusted for age"))
    variant, stripped = strip_from_chunk(w, w.chunk("TP53"), FOOTNOTE, qualifier_ids)
    footnote_only = variant.chunk("adjusted for age", whole=True)
    both = deliver(variant, [stripped, footnote_only]).evaluate()
    # Every part is present, but not inside one scientifically associated chunk: a reviewer decides.
    assert both.determination == "needs_review"
    only_note = deliver(variant, [footnote_only]).evaluate()
    assert only_note.determination == "automatic_not_delivered"
    assert only_note.reason in {"partial_components", "not_ranked"}


def test_lineage_without_verbatim_component_text_is_reviewed_not_delivered():
    w = prose_world(P_A, P_B, P_C)
    variant, restated = strip_from_chunk(w, w.chunk(P_A), "condition A", set())
    assert "condition A" not in restated.rendered_text
    result = deliver(variant, [restated]).evaluate()
    assert result.determination == "needs_review"
    assert any("verbatim" in d.detail for d in result.diagnoses)


# ---- loss between retrieval and the generator, and in chunk construction -------------------


def test_correct_chunk_ranked_but_excluded_from_final_context():
    w = prose_world(P_A, P_B, P_C)
    wanted = w.chunk(P_A)
    case = deliver(w, [w.chunk(P_B), wanted], count=lambda s: 3000 if "treatment" in s else 10)
    assert any(h.chunk == record_ref(wanted) for h in case.ranked.hits)
    result = case.evaluate()
    assert [d.decision for d in case.ctx.decisions] == ["included", "nonfitting"]
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "lost_in_packing"


def test_evidence_after_the_first_nonfitting_chunk_is_never_processed():
    w = prose_world(P_A, P_B, P_C)
    picks = [w.chunk(P_C), w.chunk(P_B), w.chunk(P_A)]
    case = deliver(w, picks, count=lambda s: 3000 if "vaccine" in s else 10)
    assert [d.decision for d in case.ctx.decisions] == ["included", "nonfitting", "not_processed"]
    result = case.evaluate()
    assert result.reason == "lost_in_packing"


@pytest.mark.parametrize("copy_first", [False, True])
def test_duplicate_removal_keeps_genuinely_equivalent_evidence(copy_first):
    w = prose_world(P_A, P_B, P_C)
    original = w.chunk(P_A)
    twin = original.model_copy(
        update={
            "contributors": [
                c.model_copy(update={"node_id": "another-containing-node"})
                for c in original.contributors
            ]
        }
    )
    twin_id, original_id = record_ref(twin).artifact_id, record_ref(original).artifact_id
    chunk_set = w.chunk_set.model_copy(
        update={
            "chunk_ids": [*w.chunk_set.chunk_ids, twin_id],
            "accounting": [
                a.model_copy(update={"chunk_ids": [*a.chunk_ids, twin_id]})
                if original_id in a.chunk_ids
                else a
                for a in w.chunk_set.accounting
            ],
        }
    )
    variant = World(w.document, w.item, w.stage_a, chunk_set, {**w.chunks, twin_id: twin})
    picks = [twin, original] if copy_first else [original, twin]
    case = deliver(variant, picks)
    assert [d.decision for d in case.ctx.decisions] == ["included", "duplicate"]
    result = case.evaluate()
    assert result.determination == "automatic_delivered"
    assert result.delivered[0].rank == 1


def test_evidence_excluded_during_chunk_construction():
    document, _, _, _, result = build(
        prose_xml(P_A, P_B),
        counter=lambda s: 1000 if "treatment" in s else len(s.split()),
    )
    w = finish(document, result, make_item(document, PROSE_COMPONENTS))
    assert {a.status for a in w.chunk_set.accounting} >= {"oversized"}
    case = deliver(w, [w.chunk(P_B)])
    outcome = case.evaluate()
    assert outcome.determination == "automatic_not_delivered"
    assert outcome.reason == "excluded_in_chunking"


# ---- context composition (D4) is separate from the loss reason ------------------------------


def test_other_paper_only_context_is_wrong_paper_selection():
    w = prose_world(P_A, P_B, P_C)
    foreign = other_paper(w.chunk(P_A))
    case = deliver(w, [foreign], scope="full_corpus", extra={})
    result = case.evaluate()
    assert result.composition == "other_only"
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "other_paper_only"
    assert result.other_chunks_included == 1 and result.designated_chunks_included == 0


def test_empty_context_is_not_wrong_paper_retrieval():
    w = prose_world(P_A, P_B, P_C)
    result = deliver(w, []).evaluate()
    assert result.composition == "empty"
    assert result.reason == "empty_context"
    assert result.determination == "automatic_not_delivered"


def test_mixed_context_reports_designated_chunks_present_but_insufficient():
    w = prose_world(P_A, P_B, P_C)
    foreign = other_paper(w.chunk(P_A))
    case = deliver(w, [foreign, w.chunk(P_C)], scope="full_corpus")
    result = case.evaluate()
    assert result.composition == "mixed"
    assert result.designated_chunks_included == 1 and result.other_chunks_included == 1
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "not_ranked"


def test_designated_evidence_still_delivers_in_a_mixed_context():
    w = prose_world(P_A, P_B, P_C)
    foreign = other_paper(w.chunk(P_B))
    result = deliver(w, [foreign, w.chunk(P_A)], scope="full_corpus").evaluate()
    assert result.composition == "mixed"
    assert result.determination == "automatic_delivered"


# ---- Stage A dependency ----------------------------------------------------------------------


def altered_world(label, **fields):
    """Source text carries the exponent as 10⁻² so the benchmark evidence is not there."""
    document, _, _, _, result = build(
        prose_xml(P_A.replace("10⁻³", "10⁻²"), P_B), counter=lambda s: len(s.split()) * 9
    )
    item = make_item(document, PROSE_COMPONENTS)
    stage_a = judgment(item, document, label, **fields)
    return finish(document, result, item, stage_a)


@pytest.mark.parametrize(
    "label,fields",
    [("scientifically_corrupted", {"alteration": "exponent −3 became −2"}), ("absent", {})],
)
def test_stage_a_loss_is_never_a_retrieval_miss(label, fields):
    w = altered_world(label, **fields)
    case = deliver(w, [w.chunk(P_B)])
    result = case.evaluate()
    assert result.determination == "not_reachable_stage_a"
    assert result.reason is None and not result.delivered


def test_stage_a_unresolved_keeps_delivery_non_evaluable():
    w = altered_world("unresolved")
    result = deliver(w, [w.chunk(P_B)]).evaluate()
    assert result.determination == "non_evaluable_stage_a"


# ---- ambiguous or nonverbatim evidence is reviewed, never auto-classified --------------------


def altered_but_sufficient_world():
    document, _, _, _, result = build(prose_xml(P_A, P_B), counter=lambda s: len(s.split()) * 9)
    items = with_text(PROSE_COMPONENTS, "answer", "−1.2×10⁻³")
    item = make_item(document, items)
    sets = [locus(document, items, observed={"answer": "−1.2 × 10⁻³"})]
    stage_a = judgment(
        item, document, "altered_but_sufficient", sets=sets, alteration="spacing around ×"
    )
    return finish(document, result, item, stage_a)


def test_altered_but_sufficient_evidence_requires_scientific_review():
    w = altered_but_sufficient_world()
    result = deliver(w, [w.chunk(P_A)]).evaluate()
    assert result.determination == "needs_review"
    assert not result.delivered
    assert any(d.outcome == "review" for d in result.diagnoses)


def test_reviewer_attested_equivalence_is_delivered_only_after_review():
    document, _, _, _, result = build(prose_xml(P_A, P_B), counter=lambda s: len(s.split()) * 9)
    item = make_item(document, PROSE_COMPONENTS)
    sets = [locus(document, PROSE_COMPONENTS)]
    subject = sets[0].placements[0].components[0]
    subject.equivalence_note = "descriptor paraphrases the sentence subject"
    stage_a = judgment(item, document, "preserved", sets=sets)
    w = finish(document, result, item, stage_a)
    outcome = deliver(w, [w.chunk(P_A)]).evaluate()
    assert outcome.determination == "needs_review"
    assert not outcome.delivered


def test_altered_evidence_that_never_reached_the_context_is_still_not_delivered():
    w = altered_but_sufficient_world()
    result = deliver(w, [w.chunk(P_B)]).evaluate()
    assert result.determination == "automatic_not_delivered"
    assert result.reason == "not_ranked"


def test_qualified_mapping_on_delivered_evidence_requires_review():
    w = prose_world(P_A, P_B, P_C)
    ids = set(occurrence_ids(w.document, "treatment effect"))
    w.chunk_set = w.chunk_set.model_copy(
        update={
            "occurrences": [
                o.model_copy(update={"interpretation_state": "ambiguous"})
                if o.occurrence_id in ids
                else o
                for o in w.chunk_set.occurrences
            ]
        }
    )
    result = deliver(w, [w.chunk(P_A)]).evaluate()
    assert result.determination == "needs_review"
    assert any("qualified" in d.detail for d in result.diagnoses)


def test_repeated_conflicting_value_in_context_requires_review():
    w = prose_world(P_A, P_A2, P_C)
    case = deliver(w, [w.chunk(P_A), w.chunk(P_A2)])
    result = case.evaluate()
    assert result.determination == "needs_review"
    assert any("conflict" in d.detail for d in result.diagnoses)
    assert deliver(w, [w.chunk(P_A)]).evaluate().determination == "automatic_delivered"


# ---- provenance: counterfeit or mismatched inputs are rejected, not scored -------------------


def test_chunk_not_matching_its_ranked_identity_is_rejected():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    key = record_ref(w.chunk(P_A)).artifact_id
    forged = w.chunk(P_A).model_copy(update={"rendered_text": P_A.replace("−1.2", "−9.9")})
    with pytest.raises(ValueError):
        case.evaluate(chunks={**case.chunks, key: forged})


def test_chunk_from_another_paper_cannot_pose_as_designated_evidence():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    key = record_ref(w.chunk(P_A)).artifact_id
    with pytest.raises(ValueError):
        case.evaluate(chunks={**case.chunks, key: other_paper(w.chunk(P_A))})


@pytest.mark.parametrize("field", ["item_id", "paper_id", "item_sha256", "representation"])
def test_stage_a_judgment_for_another_identity_is_rejected(field):
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    value = {
        "item_id": "q_999",
        "paper_id": "P002",
        "item_sha256": "f" * 64,
        "representation": "PDF_DOCLING",
    }
    forged = w.stage_a.model_copy(update={field: value[field]})
    with pytest.raises(ValueError):
        case.evaluate(stage_a=forged)


def test_context_differing_from_claimed_retained_artifact_is_rejected():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    tampered = case.ctx.model_copy(update={"exact_context": case.ctx.exact_context + " "})
    with pytest.raises(ValueError):
        case.evaluate(context=tampered)  # claimed ref no longer matches the bytes
    with pytest.raises(ValueError):
        case.evaluate(context=tampered, context_ref=record_ref(tampered))  # rehashed counterfeit


def test_context_not_built_from_the_ranked_retrieval_is_rejected():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    other = deliver(w, [w.chunk(P_B)])
    with pytest.raises(ValueError):
        case.evaluate(ranked=other.ranked)


def test_chunk_set_of_another_canonical_artifact_is_rejected():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    foreign = w.chunk_set.model_copy(
        update={"input": w.chunk_set.input.model_copy(update={"canonical": ref("elsewhere")})}
    )
    with pytest.raises(ValueError):
        case.evaluate(chunk_set=foreign)


def test_retrieval_hit_without_delivery_is_not_delivery():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_B), w.chunk(P_A)], count=lambda s: 3000 if "treatment" in s else 1)
    assert record_ref(w.chunk(P_A)) in [h.chunk for h in case.ranked.hits]
    assert record_ref(w.chunk(P_A)).artifact_id not in case.ctx.included_chunk_ids
    assert case.evaluate().determination == "automatic_not_delivered"


# ---- independence from the answer ------------------------------------------------------------


def test_stage_b_has_no_access_to_generated_answers_or_gold():
    parameters = set(inspect.signature(stage_b.Inputs).parameters)
    assert not {p for p in parameters if "answer" in p or "output" in p or "gold" in p}
    source = inspect.getsource(stage_b)
    assert ".gold" not in source and "generation" not in source.lower()


# ---- judgments, review path, publication ------------------------------------------------------


def reviewer_judgment(candidate, status, **fields):
    values = {
        **{k: getattr(candidate, k) for k in StageBIdentity.model_fields},
        "status": status,
        "basis": "reviewer",
        "candidate": stage_b.stage_b_ref(candidate),
        "reason": None,
        "delivered": [],
        "rationale": "inspected the supplied context against the benchmark evidence",
        "reviewer": "Rama Krishna Pudota",
        "at_utc": "2026-11-05T12:00:00Z",
    }
    values.update(fields)
    return StageBJudgment.model_validate(values)


def review_case():
    w = altered_but_sufficient_world()
    case = deliver(w, [w.chunk(P_A)])
    return case, case.evaluate()


def test_deterministic_judgment_follows_the_candidate_and_publishes(tmp_path):
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    candidate = case.evaluate()
    judged = stage_b.deterministic_judgment(candidate)
    assert judged.status == "delivered" and judged.basis == "deterministic"
    ref_c, _ = publish_candidate(candidate, case.inputs(), tmp_path)
    ref_j, _ = publish_judgment(judged, case.inputs(), tmp_path)
    assert read_candidate(ref_c, tmp_path, case.inputs()) == candidate
    assert read_judgment(ref_j, tmp_path, case.inputs()) == judged
    with pytest.raises(FileExistsError):  # published records are immutable
        publish_candidate(candidate, case.inputs(), tmp_path)
    path = tmp_path / ref_j.logical_ref
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        read_judgment(ref_j, tmp_path, case.inputs())


def test_deterministic_judgment_is_unavailable_for_review_cases():
    _, candidate = review_case()
    with pytest.raises(ValueError):
        stage_b.deterministic_judgment(candidate)


def test_reviewer_decides_a_review_case_with_cited_evidence(tmp_path):
    case, candidate = review_case()
    chunk_id = case.ctx.included_chunk_ids[0]
    container = case.world.stage_a.sets[0].placements[0].container_node_id
    cited = Delivered(set_index=0, container_node_id=container, chunk_id=chunk_id, rank=1, spans=[])
    good = reviewer_judgment(candidate, "delivered", delivered=[cited])
    stage_b.validate_judgment(good, case.inputs())
    publish_judgment(good, case.inputs(), tmp_path)
    no = reviewer_judgment(candidate, "not_delivered", reason="partial_components")
    stage_b.validate_judgment(no, case.inputs())


def test_reviewer_cannot_cite_evidence_that_was_not_supplied():
    case, candidate = review_case()
    uncited = reviewer_judgment(
        candidate,
        "delivered",
        delivered=[
            Delivered(
                set_index=0,
                container_node_id="node",
                chunk_id="chunk-never-supplied",
                rank=1,
                spans=[],
            )
        ],
    )
    with pytest.raises(ValueError):
        stage_b.validate_judgment(uncited, case.inputs())


def test_reviewer_cannot_override_automatic_or_stage_a_outcomes():
    w = prose_world(P_A, P_B, P_C)
    automatic = deliver(w, [w.chunk(P_B)]).evaluate()
    assert automatic.determination == "automatic_not_delivered"
    with pytest.raises(ValueError):
        stage_b.validate_judgment(
            reviewer_judgment(automatic, "not_delivered", reason="not_ranked"),
            deliver(w, [w.chunk(P_B)]).inputs(),
        )
    with pytest.raises(ValidationError):
        reviewer_judgment(automatic, "not_reachable_stage_a", reason=None)


def test_reviewer_fields_are_structurally_enforced():
    _, candidate = review_case()
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "not_delivered", reason="not_ranked", reviewer=None)
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "not_delivered", reason="not_ranked", rationale=" ")
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "not_delivered", reason="not_ranked", at_utc="2026-11-05")
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "not_delivered")


def test_publication_validates_before_writing(tmp_path):
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    candidate = case.evaluate()
    forged = candidate.model_copy(
        update={"determination": "automatic_not_delivered", "reason": "not_ranked", "delivered": []}
    )
    with pytest.raises(ValueError):
        publish_candidate(forged, case.inputs(), tmp_path)
    assert not list(tmp_path.rglob("*.json"))
