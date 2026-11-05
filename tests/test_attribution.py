"""Earliest-failure attribution and the two co-primary outcomes (independent truth table)."""

import pytest

from biorag_repro.evaluation.attribution import attribute

A_OK = ["preserved", "altered_but_sufficient"]
A_FAIL = ["scientifically_corrupted", "absent"]

# (stage_a, stage_b, stage_c, execution) -> (delivered, reason, correct, earliest, cwe)
# Written from EXPERIMENT_PLAN sections 7 and 9, not generated from the implementation.
ROWS = [
    # Stage A usable, Stage B delivered: answer decides
    ("preserved", "delivered", "correct", "completed", "yes", None, "yes", "none", "FALSE"),
    ("preserved", "delivered", "incorrect", "completed", "yes", None, "no", "stage_c", "FALSE"),
    (
        "altered_but_sufficient",
        "delivered",
        "non_evaluable",
        "completed",
        "yes",
        None,
        "non_evaluable",
        "unresolved",
        "NA",
    ),
    (
        "preserved",
        "delivered",
        "missing",
        "operational_failure",
        "yes",
        None,
        "missing",
        "incomplete",
        "NA",
    ),
    (
        "preserved",
        "delivered",
        None,
        "planned_not_executed",
        "yes",
        None,
        "not_executed",
        "incomplete",
        "NA",
    ),
    # Stage A usable, Stage B lost: stage_b whatever the answer did; correct stays visible
    ("preserved", "not_delivered", "correct", "completed", "no", None, "yes", "stage_b", "TRUE"),
    ("preserved", "not_delivered", "incorrect", "completed", "no", None, "no", "stage_b", "FALSE"),
    (
        "preserved",
        "not_delivered",
        "non_evaluable",
        "completed",
        "no",
        None,
        "non_evaluable",
        "stage_b",
        "NA",
    ),
    (
        "preserved",
        "not_delivered",
        "missing",
        "operational_failure",
        "no",
        None,
        "missing",
        "stage_b",
        "NA",
    ),
    # Stage A lost: Stage B not reachable (no B judgment is needed or allowed to say otherwise)
    *[
        (a, "not_reachable_stage_a", "correct", "completed", "no", None, "yes", "stage_a", "TRUE")
        for a in A_FAIL
    ],
    *[(a, None, "correct", "completed", "no", None, "yes", "stage_a", "TRUE") for a in A_FAIL],
    *[(a, None, "incorrect", "completed", "no", None, "no", "stage_a", "FALSE") for a in A_FAIL],
    *[
        (a, None, None, "planned_not_executed", "no", None, "not_executed", "stage_a", "NA")
        for a in A_FAIL
    ],
    # Stage A unresolved: delivery non-evaluable, correctness still scored on its own
    (
        "unresolved",
        "non_evaluable",
        "correct",
        "completed",
        "non_evaluable",
        "stage_a_unresolved",
        "yes",
        "unresolved",
        "NA",
    ),
    (
        "unresolved",
        "non_evaluable",
        "incorrect",
        "completed",
        "non_evaluable",
        "stage_a_unresolved",
        "no",
        "unresolved",
        "NA",
    ),
    (
        "unresolved",
        None,
        "missing",
        "operational_failure",
        "non_evaluable",
        "stage_a_unresolved",
        "missing",
        "unresolved",
        "NA",
    ),
    # Not yet assessed is not unresolved and not a failure
    (
        None,
        None,
        "correct",
        "completed",
        "non_evaluable",
        "stage_a_not_assessed",
        "yes",
        "incomplete",
        "NA",
    ),
    (
        "preserved",
        None,
        "incorrect",
        "completed",
        "non_evaluable",
        "stage_b_not_assessed",
        "no",
        "incomplete",
        "NA",
    ),
]


@pytest.mark.parametrize(
    "a,b,c,execution,delivered,reason,correct,earliest,cwe",
    ROWS,
)
def test_truth_table(a, b, c, execution, delivered, reason, correct, earliest, cwe):
    out = attribute(closed_book=False, stage_a=a, stage_b=b, stage_c=c, execution=execution)
    assert (
        out.evidence_delivered,
        out.evidence_reason,
        out.answer_correct,
        out.earliest_failure,
        out.correct_without_evidence,
    ) == (delivered, reason, correct, earliest, cwe)


@pytest.mark.parametrize(
    "c,execution,correct",
    [
        ("correct", "completed", "yes"),
        ("incorrect", "completed", "no"),
        ("missing", "operational_failure", "missing"),
        ("non_evaluable", "completed", "non_evaluable"),
        (None, "planned_not_executed", "not_executed"),
    ],
)
def test_closed_book_has_no_evidence_stages(c, execution, correct):
    out = attribute(closed_book=True, stage_a=None, stage_b=None, stage_c=c, execution=execution)
    assert out.evidence_delivered == "not_applicable"
    assert out.evidence_reason is None
    assert out.earliest_failure == "not_applicable"
    assert out.correct_without_evidence == "NA"
    assert out.answer_correct == correct


@pytest.mark.parametrize(
    "a,b",
    [
        ("preserved", "not_reachable_stage_a"),
        ("preserved", "non_evaluable"),
        ("altered_but_sufficient", "non_evaluable"),
        ("absent", "delivered"),
        ("scientifically_corrupted", "not_delivered"),
        ("scientifically_corrupted", "non_evaluable"),
        ("unresolved", "delivered"),
        ("unresolved", "not_delivered"),
        (None, "delivered"),
    ],
)
def test_stage_b_must_agree_with_stage_a(a, b):
    with pytest.raises(ValueError):
        attribute(closed_book=False, stage_a=a, stage_b=b, stage_c="correct", execution="completed")


@pytest.mark.parametrize("a,b", [("preserved", None), ("absent", "not_reachable_stage_a")])
def test_closed_book_refuses_stage_a_or_b(a, b):
    with pytest.raises(ValueError):
        attribute(closed_book=True, stage_a=a, stage_b=b, stage_c="correct", execution="completed")


@pytest.mark.parametrize(
    "c,execution",
    [
        ("correct", "planned_not_executed"),
        ("missing", "completed"),
        ("correct", "operational_failure"),
        ("incorrect", "operational_failure"),
        (None, "completed"),
        (None, "operational_failure"),
    ],
)
def test_score_must_agree_with_execution(c, execution):
    with pytest.raises(ValueError):
        attribute(closed_book=True, stage_a=None, stage_b=None, stage_c=c, execution=execution)


def test_unusable_gold_on_an_operational_failure_stays_non_evaluable():
    out = attribute(
        closed_book=True,
        stage_a=None,
        stage_b=None,
        stage_c="non_evaluable",
        execution="operational_failure",
    )
    assert out.answer_correct == "non_evaluable"


def test_correct_answer_never_repairs_a_stage_failure_and_wrong_answer_never_causes_one():
    # every Stage A/B result paired with every scoreable Stage C result
    for a, b, expected_e in [
        ("absent", None, "no"),
        ("preserved", "not_delivered", "no"),
        ("preserved", "delivered", "yes"),
    ]:
        for c in ("correct", "incorrect"):
            out = attribute(
                closed_book=False, stage_a=a, stage_b=b, stage_c=c, execution="completed"
            )
            assert out.evidence_delivered == expected_e
            assert out.answer_correct == ("yes" if c == "correct" else "no")
