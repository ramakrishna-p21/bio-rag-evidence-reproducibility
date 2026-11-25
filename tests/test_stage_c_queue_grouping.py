"""Stage C review queue: one entry per item, rules, automatic reason and normalized answer.

`interpreted` is empty for several adjudicable reasons, so the normalized answer itself must
separate entries; otherwise one reviewer decision would silently cover different answers.
"""

import pytest
from test_stage_c import CONCLUSION, SET_GOLD, entity, make_item, make_result, quantity, rekey

from biorag_repro.evaluation import final_test, stage_c
from biorag_repro.evaluation.stage_c_models import ADJUDICABLE

AT = "2026-11-25T00:00:00Z"
ENTITY = entity("TP53", ["p53"])
# (gold, automatic reason, two distinct answers that both get that reason)
CASES = [
    (quantity("1000"), "uninterpretable_numeric_text", ["5-10", "6-12"]),
    (quantity("5", unit="mg"), "unit_missing", ["5", "7"]),
    (quantity("0.91"), "unit_not_in_gold", ["0.91 mg", "0.91 g"]),
    (quantity("5", unit="mg"), "unknown_unit_conversion", ["5 mol/L", "5 mol"]),
    (ENTITY, "approved_name_in_longer_text", ["TP53 is mutated", "It is not TP53."]),
    (ENTITY, "case_only_difference", ["tp53", "Tp53"]),
    (SET_GOLD, "case_only_difference", [["TP53", "KRAS", "Egfr"], ["TP53", "Kras", "EGFR"]]),
    (SET_GOLD, "set_segmentation_ambiguous", ["TP53, KRAS and EGFR", "TP53 and KRAS, EGFR"]),
    (
        CONCLUSION,
        "wording_not_approved",
        ["Drug X decreased survival in KRAS-mutant tumours.", "Drug X increased survival."],
    ),
]


def scored(gold, answers, **coords):
    item = make_item(gold)
    results = [rekey(make_result(item, answer=a), **coords) for a in answers]
    return item, results, [stage_c.score(item, r, "v2") for r in results]


def queue(item, results, scores, seed=3):
    answers = {r.observation_key: r.answer for r in results}
    return final_test.stage_c_queue(scores, {item.item_id: item}, seed, answers)


@pytest.mark.parametrize("gold,reason,answers", CASES)
def test_distinct_answers_with_one_reason_get_separate_entries(gold, reason, answers):
    item = make_item(gold)
    results = [rekey(make_result(item, answer=a), repeat_index=n) for n, a in enumerate(answers, 1)]
    scores = [stage_c.score(item, r, "v2") for r in results]
    assert [s.reason for s in scores] == [reason, reason]
    entries, mapping = queue(item, results, scores)
    assert len(entries) == 2 and all(len(v) == 1 for v in mapping.values())
    # one decision per entry: the two answers can be decided differently
    decisions = {
        entries[0]["handle"]: {"status": "correct", "rationale": "C1", "at_utc": AT},
        entries[1]["handle"]: {"status": "incorrect", "rationale": "C1", "at_utc": AT},
    }
    out = final_test.apply_decisions(scores, mapping, decisions, "R")
    assert sorted(s.status for s in out) == ["correct", "incorrect"]


def test_every_adjudicable_reason_is_exercised():
    covered = {reason for _, reason, _ in CASES}
    assert covered == ADJUDICABLE - {"gold_unusable"}


def test_negated_and_plain_answers_never_share_a_decision():
    item = make_item(ENTITY)
    plain = rekey(make_result(item, answer="TP53 is mutated"), repeat_index=1)
    negated = rekey(make_result(item, answer="TP53 is not mutated"), repeat_index=2)
    scores = [stage_c.score(item, r, "v2") for r in (plain, negated)]
    entries, mapping = queue(item, [plain, negated], scores)
    assert len(entries) == 2
    decided = final_test.apply_decisions(
        scores,
        mapping,
        {
            entries[0]["handle"]: {"status": "correct", "rationale": "C1", "at_utc": AT},
            entries[1]["handle"]: {"status": "incorrect", "rationale": "C1", "at_utc": AT},
        },
        "R",
    )
    assert {s.status for s in decided} == {"correct", "incorrect"}


def test_identical_normalized_answers_still_share_one_entry_across_arms_and_repeats():
    item = make_item(ENTITY)
    answers = ["TP53 is mutated", "TP53  is\nmutated", "TP53 is mutated"]
    results = [
        rekey(make_result(item, answer=a), repeat_index=n, matrix_arm=arm)
        for n, (a, arm) in enumerate(zip(answers, ("a", "b", "c"), strict=True), 1)
    ]
    scores = [stage_c.score(item, r, "v2") for r in results]
    entries, mapping = queue(item, results, scores)
    assert len(entries) == 1 and len(mapping["M0001"]) == 3
    assert len(entries[0]["answers"]) == 2  # the reviewer still sees both raw spellings


def test_categorical_final_full_stop_is_the_only_extra_normalization():
    item = make_item(CONCLUSION)
    texts = [
        "Drug X decreased survival.",
        "Drug X decreased survival",
        "drug X decreased survival.",
    ]
    results = [rekey(make_result(item, answer=t), repeat_index=n) for n, t in enumerate(texts, 1)]
    scores = [stage_c.score(item, r, "v2") for r in results]
    _, mapping = queue(item, results, scores)
    sizes = sorted(len(v) for v in mapping.values())
    assert sizes == [1, 2]  # final period dropped for categorical conclusions; case kept


def test_other_items_and_rules_never_merge():
    a, b = make_item(ENTITY), make_item(ENTITY, item_id="q_002")
    ra = rekey(make_result(a, answer="TP53 is mutated"))
    rb = rekey(make_result(b, answer="TP53 is mutated"))
    scores = [stage_c.score(a, ra, "v2"), stage_c.score(b, rb, "v2")]
    answers = {r.observation_key: r.answer for r in (ra, rb)}
    entries, _ = final_test.stage_c_queue(scores, {a.item_id: a, b.item_id: b}, 3, answers)
    assert len(entries) == 2


def test_scoring_behaviour_is_unchanged_by_the_grouping_key():
    item = make_item(ENTITY)
    result = make_result(item, answer="It is not TP53.")
    before = stage_c.score(item, result, "v2")
    queue(item, [result], [before])
    assert stage_c.score(item, result, "v2") == before
