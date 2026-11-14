"""Combine Stage A, B and C results into the two co-primary outcomes and the earliest failure.

The inputs are statuses of already validated records. Delivery (Stage A then B) and
answer correctness (Stage C) are derived independently: a correct answer never repairs
a lost evidence stage, and a wrong answer never implies one.
"""

from dataclasses import dataclass
from typing import Literal

Execution = Literal["planned_not_executed", "operational_failure", "completed"]
Delivered = Literal["yes", "no", "non_evaluable", "not_applicable"]
Correct = Literal["yes", "no", "non_evaluable", "missing", "not_executed"]
Earliest = Literal[
    "none", "stage_a", "stage_b", "stage_c", "unresolved", "not_applicable", "incomplete"
]
EvidenceReason = Literal[
    "stage_a_not_assessed", "stage_a_unresolved", "stage_b_not_assessed", "stage_b_unresolved"
]
Tri = Literal["TRUE", "FALSE", "NA"]

A_FAILED = {"scientifically_corrupted", "absent"}
A_USABLE = {"preserved", "altered_but_sufficient"}
_ANSWER: dict[str, Correct] = {
    "correct": "yes",
    "incorrect": "no",
    "non_evaluable": "non_evaluable",
    "missing": "missing",
}


@dataclass(frozen=True)
class Outcomes:
    evidence_delivered: Delivered
    evidence_reason: EvidenceReason | None
    answer_correct: Correct
    earliest_failure: Earliest
    correct_without_evidence: Tri


def _delivery(
    stage_a: str | None, stage_b: str | None
) -> tuple[Delivered, EvidenceReason | None, Earliest | None]:
    """Evidence delivery and, where evidence alone decides it, the earliest failure."""
    if stage_a is None:
        if stage_b is not None:
            raise ValueError("a Stage B record needs the Stage A judgment it depends on")
        return "non_evaluable", "stage_a_not_assessed", None
    if stage_a in A_FAILED:
        if stage_b not in (None, "not_reachable_stage_a"):
            raise ValueError("Stage B must be not reachable after a Stage A loss")
        return "no", None, "stage_a"
    if stage_a == "unresolved":
        if stage_b not in (None, "non_evaluable"):
            raise ValueError("Stage B must be non-evaluable while Stage A is unresolved")
        return "non_evaluable", "stage_a_unresolved", "unresolved"
    if stage_a not in A_USABLE:
        raise ValueError(f"unknown Stage A label {stage_a!r}")
    if stage_b is None:
        return "non_evaluable", "stage_b_not_assessed", None
    if stage_b == "delivered":
        return "yes", None, None
    if stage_b == "not_delivered":
        return "no", None, "stage_b"
    if stage_b == "unresolved":  # a reviewed context that could not be decided
        return "non_evaluable", "stage_b_unresolved", "unresolved"
    raise ValueError(f"Stage B status {stage_b!r} contradicts a usable Stage A label")


def _answer(stage_c: str | None, execution: Execution) -> Correct:
    if execution == "planned_not_executed":
        if stage_c is not None:
            raise ValueError("an unexecuted observation has no answer score")
        return "not_executed"
    if stage_c is None:
        raise ValueError("an attempted observation needs its Stage C score")
    if stage_c == "missing" and execution != "operational_failure":
        raise ValueError("only an operational failure is missing")
    if execution == "operational_failure" and stage_c in {"correct", "incorrect"}:
        raise ValueError("an operational failure has no scoreable answer")
    return _ANSWER[stage_c]


def attribute(
    *,
    closed_book: bool,
    stage_a: str | None,
    stage_b: str | None,
    stage_c: str | None,
    execution: Execution,
) -> Outcomes:
    answer = _answer(stage_c, execution)
    if closed_book:
        if stage_a is not None or stage_b is not None:
            raise ValueError("Stage A and B do not apply to the closed-book baseline")
        return Outcomes("not_applicable", None, answer, "not_applicable", "NA")

    delivered, reason, earliest = _delivery(stage_a, stage_b)
    if earliest is None:
        if delivered == "non_evaluable":  # not yet assessed
            earliest = "incomplete"
        elif answer == "yes":
            earliest = "none"
        elif answer == "no":
            earliest = "stage_c"
        elif answer == "non_evaluable":
            earliest = "unresolved"
        else:
            earliest = "incomplete"
    scored = delivered in {"yes", "no"} and answer in {"yes", "no"}
    flag: Tri = "NA"
    if scored:
        flag = "TRUE" if delivered == "no" and answer == "yes" else "FALSE"
    return Outcomes(delivered, reason, answer, earliest, flag)
