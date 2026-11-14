"""Stage B rules v2: narrow automatic decisions plus whole-context review of the supplied text."""

import inspect
import json
from dataclasses import replace

import pytest
from pydantic import ValidationError
from test_analysis_export import observation, tbl, world
from test_stage_a import CANONICAL
from test_stage_b import (
    P_A,
    P_A2,
    P_B,
    P_C,
    Delivered,
    altered_but_sufficient_world,
    deliver,
    other_paper,
    prose_world,
    review_case,
    reviewer_judgment,
    table_world,
)

from biorag_repro.evaluation import stage_b
from biorag_repro.evaluation.analysis_storage import _settings_sha256, publish_table, read_table
from biorag_repro.evaluation.analysis_table import (
    Evidence,
    build_table,
    rules_of,
    scoring_reference,
)
from biorag_repro.evaluation.attribution import attribute
from biorag_repro.evaluation.stage_b_models import Citation, ComponentRef, Span
from biorag_repro.evaluation.stage_b_storage import publish_judgment, read_judgment

# ---- v1 is unchanged -------------------------------------------------------------------------


def test_v1_judgment_identities_are_byte_identical():
    w = prose_world(P_A, P_B, P_C)
    candidate = deliver(w, [w.chunk(P_A)]).evaluate()
    deterministic = stage_b.deterministic_judgment(candidate)
    assert stage_b.stage_b_ref(deterministic).artifact_id == (
        "drv_9e0bb7f2c25e8ae91c744b2532c574d0d33437b48d1ad02a8b23329a41b1a9e1"
    )
    assert stage_b.stage_b_ref(candidate).artifact_id == (
        "drv_af5e953509d449fa95ef26523705f8eb35bda1ae2e2c79ca00afeecd4e3004e8"
    )
    case, review = review_case()
    container = case.world.stage_a.sets[0].placements[0].container_node_id
    cited = Delivered(
        set_index=0,
        container_node_id=container,
        chunk_id=case.ctx.included_chunk_ids[0],
        rank=1,
        spans=[],
    )
    reviewed = reviewer_judgment(review, "delivered", delivered=[cited])
    assert stage_b.stage_b_ref(reviewed).artifact_id == (
        "drv_554042a1407a8e44c8285edeb82555f45152592f50a487b5cac3a74f51ecff00"
    )
    assert "rules" not in reviewed.model_dump(mode="json")


def test_v1_analysis_settings_are_unchanged():
    v1 = "ded32b91b3919e129e9a483af8ae2ba1a6013e3f904c961b8fe243ede9bda566"
    v2 = "7180caba3f9e4786c002be180be7b2932c4f2d541009946a97bf743a56369abc"
    assert _settings_sha256("v1") == v1 and _settings_sha256("v2") == v2
    assert _settings_sha256("v1", "v1") == v1
    assert _settings_sha256("v1", "v2") != v1
    assert rules_of(scoring_reference("v2", "v2")) == ("v2", "v2")
    assert rules_of(scoring_reference()) == ("v1", "v1")


def test_v1_refuses_v2_statuses_reasons_and_fields():
    _, candidate = review_case()
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "unresolved", unresolved_reason="cannot tell")
    with pytest.raises(ValidationError):
        reviewer_judgment(candidate, "not_delivered", reason="ambiguous_referent")
    with pytest.raises(ValidationError):
        reviewer_judgment(
            candidate,
            "not_delivered",
            reason="partial_components",
            missing_components=[ComponentRef(set_index=0, component_index=2)],
        )


# ---- v2 automatic decisions are narrower ------------------------------------------------------


def v2(candidate, inputs=None):
    return stage_b.deterministic_judgment(candidate, "v2", inputs)


def test_v2_keeps_unambiguous_verbatim_prose_delivery():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    judged = v2(case.evaluate(), case.inputs())
    assert judged.status == "delivered" and judged.rules == "v2"
    stage_b.validate_judgment(judged, case.inputs())
    assert stage_b.stage_b_ref(judged) != stage_b.stage_b_ref(
        stage_b.deterministic_judgment(case.evaluate())
    )


def test_v2_automatic_delivery_is_checked_against_its_inputs():
    w = prose_world(P_A, P_B, P_C)
    candidate = deliver(w, [w.chunk(P_A)]).evaluate()
    with pytest.raises(ValueError):
        v2(candidate)
    other = deliver(w, [w.chunk(P_B)])
    with pytest.raises(ValueError):
        v2(candidate, other.inputs())


def test_v2_sends_table_associations_to_review():
    w = table_world()
    case = deliver(w, [w.chunk("TP53")])
    candidate = case.evaluate()
    assert candidate.determination == "automatic_delivered"  # v1 behaviour is kept
    with pytest.raises(ValueError, match="review"):
        v2(candidate, case.inputs())
    assert stage_b.needs_review_v2(candidate, case.inputs())


def test_v2_sends_ambiguous_prose_to_review():
    # Two statements in one chunk repeat subject, relation and condition with different values.
    w = prose_world(P_A, P_A2, P_C, per_paragraph=1)
    chunk = w.chunk(P_A)
    assert P_A2 in chunk.rendered_text
    case = deliver(w, [chunk])
    candidate = case.evaluate()
    assert candidate.determination == "automatic_delivered"
    with pytest.raises(ValueError, match="review"):
        v2(candidate, case.inputs())


def test_v2_refuses_lineage_only_negatives():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_B)])
    candidate = case.evaluate()
    assert (candidate.determination, candidate.reason) == ("automatic_not_delivered", "not_ranked")
    stage_b.deterministic_judgment(candidate)  # v1 unchanged
    with pytest.raises(ValueError, match="whole-context review"):
        v2(candidate, case.inputs())
    assert stage_b.needs_review_v2(candidate, case.inputs())


def test_v2_keeps_composition_and_stage_a_outcomes_deterministic():
    w = prose_world(P_A, P_B, P_C)
    empty = deliver(w, [])
    assert v2(empty.evaluate(), empty.inputs()).reason == "empty_context"
    foreign = deliver(w, [other_paper(w.chunk(P_A))], scope="full_corpus", extra={})
    judged = v2(foreign.evaluate(), foreign.inputs())
    assert judged.reason == "other_paper_only"
    stage_b.validate_judgment(judged, foreign.inputs())
    assert not stage_b.needs_review_v2(foreign.evaluate(), foreign.inputs())


# ---- v2 whole-context review ------------------------------------------------------------------


def two_chunk_review():
    w = altered_but_sufficient_world()
    case = deliver(w, [w.chunk(P_A), w.chunk(P_B)])
    first, second = case.ctx.included_chunk_ids
    assert first != second
    return case, case.evaluate(), first, second


def cite(case, chunk_id, quote, set_index=0, component_index=0, occurrence=None):
    span = stage_b.cite(case.inputs(), chunk_id, quote, occurrence)
    return Citation(**span.model_dump(), set_index=set_index, component_index=component_index)


def review(candidate, status, **fields):
    return reviewer_judgment(candidate, status, rules="v2", **fields)


def full_set(case, first, second):
    return [
        cite(case, first, "treatment", component_index=0),
        cite(case, first, "effect", component_index=1),
        cite(case, first, "−1.2 × 10⁻³", component_index=2),
        cite(case, second, "condition", component_index=3),
    ]


def test_distributed_evidence_across_chunks_is_delivered(tmp_path):
    case, candidate, first, second = two_chunk_review()
    assert candidate.determination == "needs_review"
    judged = review(candidate, "delivered", citations=full_set(case, first, second))
    stage_b.validate_judgment(judged, case.inputs())
    ref, _ = publish_judgment(judged, case.inputs(), tmp_path)
    assert read_judgment(ref, tmp_path, case.inputs()) == judged
    assert ref.logical_ref.startswith("evaluation/stage_b/stage_b_judgment/")


def test_delivered_needs_every_component_of_one_set():
    case, candidate, first, second = two_chunk_review()
    partial = review(candidate, "delivered", citations=full_set(case, first, second)[:3])
    with pytest.raises(ValueError, match="every component"):
        stage_b.validate_judgment(partial, case.inputs())
    unknown = [*full_set(case, first, second), cite(case, first, "treatment", 0, 9)]
    with pytest.raises(ValueError):
        stage_b.validate_judgment(review(candidate, "delivered", citations=unknown), case.inputs())
    with pytest.raises(ValidationError):
        review(candidate, "delivered")


@pytest.mark.parametrize(
    "change",
    [
        {"text": "treatmenT"},  # text differs from the context
        {"chunk_id": "chunk-never-supplied"},
        {"start": 0, "end": 9},  # inside the [evidence] wrapper, outside the chunk
    ],
)
def test_citation_must_be_exact_text_of_a_supplied_chunk(change):
    case, candidate, first, second = two_chunk_review()
    citations = full_set(case, first, second)
    bad = citations[0].model_copy(update=change)
    if "start" in change:
        bad = bad.model_copy(update={"text": case.ctx.exact_context[0:9]})
    judged = review(candidate, "delivered", citations=[bad, *citations[1:]])
    with pytest.raises(ValueError):
        stage_b.validate_judgment(judged, case.inputs())


def test_cite_needs_an_occurrence_when_a_quote_repeats():
    case, _, first, _ = two_chunk_review()
    text = case.chunks[first].rendered_text
    repeated = "e"
    assert text.count(repeated) > 1
    with pytest.raises(ValueError, match="occurrence"):
        stage_b.cite(case.inputs(), first, repeated)
    second = stage_b.cite(case.inputs(), first, repeated, occurrence=2)
    assert case.ctx.exact_context[second.start : second.end] == repeated
    with pytest.raises(ValueError):
        stage_b.cite(case.inputs(), first, "not in the chunk")


def test_negative_needs_no_complete_evidence_but_names_missing_components():
    case, candidate, first, _ = two_chunk_review()
    missing = [ComponentRef(set_index=0, component_index=3)]
    judged = review(
        candidate,
        "not_delivered",
        reason="partial_components",
        missing_components=missing,
        citations=[cite(case, first, "treatment")],
    )
    stage_b.validate_judgment(judged, case.inputs())
    bare = review(
        candidate, "not_delivered", reason="absent_in_context", missing_components=missing
    )
    stage_b.validate_judgment(bare, case.inputs())
    with pytest.raises(ValidationError):
        review(candidate, "not_delivered", reason="partial_components")


def test_negative_must_name_a_missing_component_of_every_set():
    w = prose_world(P_A, P_B, P_C, components=(P_A_SET, B_SET))
    case = deliver(w, [w.chunk(P_B)])
    candidate = case.evaluate()
    one_set = review(
        candidate,
        "not_delivered",
        reason="partial_components",
        missing_components=[ComponentRef(set_index=0, component_index=2)],
    )
    with pytest.raises(ValueError, match="every set"):
        stage_b.validate_judgment(one_set, case.inputs())


P_A_SET = [
    ("subject", "treatment"),
    ("relation", "effect"),
    ("answer", "−1.2 × 10⁻³"),
    ("condition", "condition A"),
]
B_SET = [("subject", "vaccine"), ("relation", "titre"), ("answer", "4.5"), ("condition", "X")]


def test_fully_cited_negative_must_be_unfaithful_or_ambiguous():
    case, candidate, first, second = two_chunk_review()
    missing = [ComponentRef(set_index=0, component_index=3)]
    cited = full_set(case, first, second)
    partial = review(
        candidate,
        "not_delivered",
        reason="partial_components",
        missing_components=missing,
        citations=cited,
    )
    with pytest.raises(ValueError, match="unfaithful_in_context or ambiguous_referent"):
        stage_b.validate_judgment(partial, case.inputs())
    ambiguous = partial.model_copy(update={"reason": "ambiguous_referent"})
    stage_b.validate_judgment(ambiguous, case.inputs())


def test_unresolved_needs_a_reason_and_is_non_evaluable():
    case, candidate, _, _ = two_chunk_review()
    judged = review(candidate, "unresolved", unresolved_reason="two defensible readings")
    stage_b.validate_judgment(judged, case.inputs())
    with pytest.raises(ValidationError):
        review(candidate, "unresolved")
    with pytest.raises(ValidationError):
        review(candidate, "unresolved", unresolved_reason=" ")
    out = attribute(
        closed_book=False,
        stage_a="preserved",
        stage_b="unresolved",
        stage_c="correct",
        execution="completed",
    )
    assert (out.evidence_delivered, out.evidence_reason, out.earliest_failure) == (
        "non_evaluable",
        "stage_b_unresolved",
        "unresolved",
    )
    assert out.correct_without_evidence == "NA"


def test_conflicting_context_flag_needs_cited_contradiction():
    case, candidate, first, second = two_chunk_review()
    citations = full_set(case, first, second)
    conflict = stage_b.cite(case.inputs(), second, "4.5")
    flagged = review(
        candidate,
        "delivered",
        citations=citations,
        flags=["conflicting_context"],
        conflict_citations=[conflict],
    )
    stage_b.validate_judgment(flagged, case.inputs())
    with pytest.raises(ValidationError):
        review(candidate, "delivered", citations=citations, flags=["conflicting_context"])
    with pytest.raises(ValidationError):
        review(candidate, "delivered", citations=citations, conflict_citations=[conflict])
    with pytest.raises(ValidationError):
        review(
            candidate,
            "not_delivered",
            reason="partial_components",
            missing_components=[ComponentRef(set_index=0, component_index=3)],
            flags=["conflicting_context"],
            conflict_citations=[conflict],
        )


def test_review_cites_spans_not_stage_a_placements():
    case, candidate, first, second = two_chunk_review()
    container = case.world.stage_a.sets[0].placements[0].container_node_id
    placement = Delivered(
        set_index=0, container_node_id=container, chunk_id=first, rank=1, spans=[]
    )
    with pytest.raises(ValidationError):
        review(
            candidate, "delivered", citations=full_set(case, first, second), delivered=[placement]
        )


def test_reviewer_cannot_override_v2_automatic_outcomes():
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_A)])
    automatic = case.evaluate()
    judged = review(automatic, "unresolved", unresolved_reason="prefer to look")
    with pytest.raises(ValueError, match="routes to review"):
        stage_b.validate_judgment(judged, case.inputs())
    empty = deliver(w, [])
    negative = review(
        empty.evaluate(),
        "not_delivered",
        reason="absent_in_context",
        missing_components=[ComponentRef(set_index=0, component_index=0)],
    )
    with pytest.raises(ValueError):
        stage_b.validate_judgment(negative, empty.inputs())


def test_lineage_negative_can_be_reviewed_as_delivered_from_other_text():
    # The recorded placement was not ranked, but the supplied text may still carry the set.
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_B)])
    chunk = case.ctx.included_chunk_ids[0]
    citations = [
        cite(case, chunk, "vaccine", component_index=0),
        cite(case, chunk, "titre", component_index=1),
        cite(case, chunk, "4.5", component_index=2),
        cite(case, chunk, "condition B", component_index=3),
    ]
    judged = review(case.evaluate(), "delivered", citations=citations)
    stage_b.validate_judgment(judged, case.inputs())  # structure only; meaning is the reviewer's


def test_v2_review_has_no_access_to_generated_answers_or_stage_c():
    for function in (stage_b.validate_judgment, stage_b.cite, stage_b.needs_review_v2):
        names = set(inspect.signature(function).parameters)
        assert not {n for n in names if "answer" in n or "output" in n or "score" in n}
    source = inspect.getsource(stage_b)
    assert "stage_c" not in source and "generation" not in source.lower()


def test_span_records_are_structurally_checked():
    with pytest.raises(ValidationError):
        Span(chunk_id="c", start=5, end=5, text="")
    with pytest.raises(ValidationError):
        Span(chunk_id="c", start=0, end=3, text="ab")


# ---- analysis table ---------------------------------------------------------------------------


def with_judgment(w, case, judged, **plan):
    o = observation(w, case=case, **plan)
    return replace(o, evidence=Evidence(w.document, CANONICAL, w.stage_a, judged, case.inputs()))


def test_table_names_stage_b_rules_and_keeps_unresolved_non_evaluable(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_B)])
    judged = review(case.evaluate(), "unresolved", unresolved_reason="two defensible readings")
    table = tbl([with_judgment(w, case, judged)])
    assert table.stage_b_rules == "v2"
    (row,) = table.rows
    assert (
        row.stage_b_status,
        row.evidence_delivered,
        row.evidence_reason,
        row.earliest_failure,
    ) == ("unresolved", "non_evaluable", "stage_b_unresolved", "unresolved")
    _, inventory, build = publish_table(table, tmp_path)
    assert read_table(build, tmp_path) == table.rows
    reference = json.loads((tmp_path / inventory.logical_ref).read_text())["scoring_reference"]
    assert reference["stage_b"] == "stage_b_judgment_v2"


def test_table_rejects_mixed_stage_b_rules():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    first = observation(w, case=case)  # v1 deterministic judgment
    v2_judged = stage_b.deterministic_judgment(case.evaluate(), "v2", case.inputs())
    second = with_judgment(w, case, v2_judged, condition="PDF_DOCLING")
    with pytest.raises(ValueError, match="mixed Stage B"):
        build_table([first, second], tbl([first]).roster)
