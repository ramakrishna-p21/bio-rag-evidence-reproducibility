"""Reviewer-attested answer-surface wording: separate from genuine source-to-canonical alteration."""

import pytest
from pydantic import ValidationError
from test_documents_jats import parse
from test_stage_a import (
    CANONICAL,
    TABLE,
    TABLE_COMPONENTS,
    judgment,
    make_item,
    node_of,
    occurrence_ids,
)
from test_stage_b import prose_xml

from biorag_repro.evaluation.models import (
    AnswerSurface,
    ComponentLocus,
    Placement,
    SetLocus,
)
from biorag_repro.evaluation.stage_a import automatic_judgment, locate, validate_judgment
from biorag_repro.evaluation.surface import surface_problem

SENTENCE = (
    "High CCDC69 expression gave significantly better overall survival in TNBC patients "
    "and the analysis included 693 preterm infants with 25 of 26 evaluations passing."
)
COMPONENTS = [
    ("subject", "TNBC patients"),
    ("relation", "overall survival"),
    ("answer", "better survival with high expression"),
    ("comparison", "High CCDC69 expression"),
]
SURFACE = "significantly better overall survival"
NOTE = "same direction of association; the high-expression condition is the comparison component"


@pytest.fixture
def world():
    document = parse(prose_xml(SENTENCE).encode())
    return document, make_item(document, COMPONENTS)


def components(document, *, answer_text="better survival with high expression", **surface):
    def plain(role, text):
        return ComponentLocus(role=role, text=text, occurrence_ids=occurrence_ids(document, text))

    surface = {"canonical_text": SURFACE, "note": NOTE} | surface
    answer = ComponentLocus(
        role="answer",
        text=answer_text,
        occurrence_ids=occurrence_ids(document, surface["canonical_text"]),
        answer_surface=AnswerSurface(**surface),
    )
    return [plain("subject", "TNBC patients"), plain("relation", "overall survival"), answer] + [
        plain("comparison", "High CCDC69 expression")
    ]


def verdict(item, document, parts, label="preserved", **fields):
    placement = Placement(container_node_id=node_of(document, "p"), components=parts)
    result = judgment(
        item, document, label, sets=[SetLocus(set_index=0, placements=[placement])], **fields
    )
    validate_judgment(result, item, document, CANONICAL)
    return result


# ---- accepted -----------------------------------------------------------------------------


def test_preserved_with_attested_answer_surface_and_condition_carried_elsewhere(world):
    document, item = world
    assert verdict(item, document, components(document)).label == "preserved"


def test_extra_canonical_number_needs_the_note_to_state_its_role():
    item = AnswerSurface(canonical_text="25 of 26 evaluations", note="26 is the denominator")
    assert surface_problem("25 evaluations", item, []) is None
    missing = AnswerSurface(canonical_text="25 of 26 evaluations", note="same count")
    assert "extra" in surface_problem("25 evaluations", missing, [])


@pytest.mark.parametrize(
    ("answer", "canonical"),
    [
        ("125000", "125,000"),
        ("693 infants", "693 preterm infants"),
        ("positive", "positively correlated"),
        ("−1.2", "−1.2 in total"),  # U+2212 sign kept
        ("12%", "12% of cases"),
    ],
)
def test_conservative_checks_pass_faithful_wording(answer, canonical):
    assert surface_problem(answer, AnswerSurface(canonical_text=canonical, note="n"), []) is None


# ---- rejected by the conservative checks --------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "canonical", "fragment"),
    [
        ("125000", "12,500", "number"),  # magnitude
        ("1.2", "0.12", "number"),
        ("1.2", "1.20", "number"),  # no rounding or trailing-zero equivalence
        ("−1.2", "1.2", "number"),  # dropped sign
        ("1.2", "−1.2", "number"),  # added sign
        ("5 and 7", "7 and 5", "number"),  # order
        ("p < 0.05", "p = 0.05", "comparator"),
        ("p < 0.05", "p ≤ 0.05", "comparator"),
        ("more than 5", "fewer than 5", "comparator"),
        ("12%", "12 days", "unit"),
        ("12 mg", "12", "unit"),
        ("12", "12 months", "unit"),
        ("positive", "negatively correlated", "direction"),
        ("higher", "lower", "direction"),
        ("improved", "improved not", "negation"),
        ("no effect", "effect", "negation"),
        ("better survival", "better survival and lower risk", "direction"),
    ],
)
def test_conservative_checks_reject_changed_scientific_content(answer, canonical, fragment):
    problem = surface_problem(answer, AnswerSurface(canonical_text=canonical, note="n"), [])
    assert problem is not None and fragment in problem


def test_answer_qualifier_missing_from_the_surface_must_be_carried_by_another_component():
    surface = AnswerSurface(canonical_text="significantly better overall survival", note="n")
    answer = "better survival with high expression"
    assert "high" in surface_problem(answer, surface, ["TNBC patients", "overall survival"])
    assert surface_problem(answer, surface, ["high CCDC69 expression"]) is None


# ---- structure, label rules, identity -----------------------------------------------------


@pytest.mark.parametrize(
    "fields",
    [
        {"role": "subject"},  # answer role only
        {"role": "answer", "observed_text": "x y"},
        {"role": "answer", "equivalence_note": "n"},
        {"role": "answer", "form": "mean_sd"},
    ],
)
def test_answer_surface_is_exclusive_and_answer_only(fields):
    with pytest.raises(ValidationError):
        ComponentLocus(
            text="t",
            occurrence_ids=["o"],
            answer_surface=AnswerSurface(canonical_text="u", note="n"),
            **fields,
        )


@pytest.mark.parametrize(
    "fields", [{"canonical_text": " ", "note": "n"}, {"canonical_text": "u", "note": " "}]
)
def test_answer_surface_needs_text_and_note(fields):
    with pytest.raises(ValidationError):
        AnswerSurface(**fields)


def test_surface_must_differ_from_the_benchmark_answer():
    with pytest.raises(ValidationError):
        ComponentLocus(
            role="answer",
            text="u",
            occurrence_ids=["o"],
            answer_surface=AnswerSurface(canonical_text="u", note="n"),
        )


def test_observed_text_keeps_its_meaning_and_still_blocks_preserved(world):
    document, item = world
    parts = components(document)
    parts[2] = ComponentLocus(
        role="answer",
        text="better survival with high expression",
        occurrence_ids=occurrence_ids(document, SURFACE),
        observed_text=SURFACE,
    )
    with pytest.raises(ValueError, match="verbatim"):
        verdict(item, document, parts, "preserved")
    assert verdict(item, document, parts, "altered_but_sufficient", alteration="x").label


def test_surface_alone_does_not_justify_altered_but_sufficient(world):
    document, item = world
    with pytest.raises(ValueError, match="observed text"):
        verdict(item, document, components(document), "altered_but_sufficient", alteration="x")


def test_surface_text_must_be_carried_by_the_named_occurrences(world):
    document, item = world
    parts = components(document)
    parts[2] = parts[2].model_copy(
        update={
            "answer_surface": AnswerSurface(
                canonical_text="significantly better overall survival rates", note=NOTE
            )
        }
    )
    with pytest.raises(ValueError, match="do not carry"):
        verdict(item, document, parts)


def test_check_failure_blocks_the_judgment(world):
    document, item = world
    parts = components(document, canonical_text="693 preterm infants")
    with pytest.raises(ValueError):
        verdict(item, document, parts)


def test_condition_component_is_still_required(world):
    document, item = world
    parts = components(document)[:3]
    with pytest.raises(ValueError, match="exactly the item's evidence components"):
        verdict(item, document, parts)


def test_answer_surface_is_prose_only():
    document = parse(TABLE.encode())
    item = make_item(document, TABLE_COMPONENTS)
    answer = ComponentLocus(
        role="answer",
        text="0.91",
        occurrence_ids=occurrence_ids(document, "0.91"),
        answer_surface=AnswerSurface(canonical_text="0.91 ", note="n"),
    )
    plain = [
        ComponentLocus(role=r, text=t, occurrence_ids=occurrence_ids(document, t))
        for r, t in TABLE_COMPONENTS
        if r != "answer"
    ]
    placement = Placement(container_node_id=node_of(document, "table"), components=[*plain, answer])
    result = judgment(
        item, document, "preserved", sets=[SetLocus(set_index=0, placements=[placement])]
    )
    with pytest.raises(ValueError):
        validate_judgment(result, item, document, CANONICAL)


def test_unused_field_leaves_existing_records_byte_identical():
    plain = ComponentLocus(role="answer", text="1", occurrence_ids=["o"])
    assert "answer_surface" not in plain.model_dump(mode="json")
    used = ComponentLocus(
        role="answer",
        text="1",
        occurrence_ids=["o"],
        answer_surface=AnswerSurface(canonical_text="1.0", note="n"),
    )
    assert used.model_dump(mode="json")["answer_surface"] == {"canonical_text": "1.0", "note": "n"}
    assert not used.verbatim


def test_automatic_path_cannot_carry_an_answer_surface(world):
    document, item = world
    candidate = locate(item, document, CANONICAL)
    with pytest.raises(ValueError):
        automatic_judgment(candidate, CANONICAL)
