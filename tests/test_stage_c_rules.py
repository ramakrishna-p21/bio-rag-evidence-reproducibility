"""Stage C scoring rules: v1 stays exactly as published, v2 adds two numeric forms.

The oracle rows are hand-derived from the approved policy, not computed by the scorer.
v2 recognizes only (1) strictly grouped English thousands separators and (2) a plain-text
exponent whose sign directly follows `10` after a multiplication sign. Everything else
keeps its v1 outcome.
"""

import json
from dataclasses import replace

import pytest
from pydantic import ValidationError
from test_analysis_export import observation, tbl, world
from test_stage_b import P_A, deliver
from test_stage_c import (
    entity,
    failed_result,
    make_item,
    make_result,
    member,
    quantity,
)

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import read_manifest, serialize_project_json
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.analysis_storage import publish_table, read_table
from biorag_repro.evaluation.analysis_table import SCORING_REFERENCE, build_table
from biorag_repro.evaluation.answer_scoring import score_answer
from biorag_repro.evaluation.stage_c_models import StageCScore
from biorag_repro.evaluation.stage_c_storage import publish_score, read_score

NE_NUMBER = ("non_evaluable", "uninterpretable_numeric_text")
CORRECT = ("correct", None)
WRONG_VALUE = ("incorrect", "value_mismatch")

# --- v1 identities captured from the code before v2 existed ---------------------------------
# The schema blob below is archival: the schema was intentionally regenerated afterwards.
V1_SCHEMA_BLOB_BEFORE_V2 = "e87457e1edfff0b08d509e2d5d929d3c0804706b"
V1_GOLDEN = {
    "correct": (
        "30b550ce342e184f4a7bb878b2db4d4179bfd94d784c1cac65afe3b221e4e265",
        "drv_cc7cc2c86906ef413fbf35456b964801c916f992983e2440541b1ed936f58c9b",
    ),
    "non_evaluable": (
        "ba6ddc1798efc8cee7527dc75892fd3d0089ad8823b406313c4e3b585fa98143",
        "drv_74912a7aa3afcc3d610862ff0bcf4d953c797641cc60d3b3f78b7e2dbe997bf9",
    ),
    "thousands": (
        "6d53d30afe9ebb2e2974f7e965cd8387429b9454ef4cff9918e392764b25dfab",
        "drv_f779fafd3390f781af3fc1f9d2c9aa36ce607cd8066ceaa9b29fe591dfa602d0",
    ),
    "missing": (
        "c9d23f8cc4ee3f533792c2e8cd5168bd4c80fc9ab450e0f9c469b3c45f8c92a7",
        "drv_fbd6fc691a7c856aaece632a3e48e7fb62a4a7c7871b17e5c7fa2a3f1524fc6e",
    ),
    "adjudicated": (
        "fed1efb5a91207f9eef07a53f98b1e89e7c6e72565072402b96bbc0a1448c8ae",
        "drv_28a1d73956f06ee4314d3b7c312838ea2e7f471f528f54a0d74fc0acd102904c",
    ),
}


def v1_cases():
    quantity_item = make_item(quantity("0.05"))
    entity_item = make_item(entity("Brca1"))
    grouped_item = make_item(quantity("750076"))
    return {
        "correct": (quantity_item, make_result(quantity_item)),
        "non_evaluable": (entity_item, make_result(entity_item, answer="BRCA1")),
        "thousands": (grouped_item, make_result(grouped_item, answer="750,076")),
        "missing": (quantity_item, failed_result(quantity_item)),
    }


def digest(score):
    return sha256_bytes(serialize_project_json(score, StageCScore)), stage_c.score_ref(
        score
    ).artifact_id


@pytest.mark.parametrize("name", ["correct", "non_evaluable", "thousands", "missing"])
def test_v1_serialization_and_identity_are_unchanged(name):
    item, result = v1_cases()[name]
    score = stage_c.score(item, result)
    assert score.scoring_rules == "v1"
    assert b"scoring_rules" not in serialize_project_json(score, StageCScore)
    assert digest(score) == V1_GOLDEN[name]


def test_v1_adjudication_chain_is_unchanged():
    item, result = v1_cases()["non_evaluable"]
    judged = stage_c.adjudicate(
        stage_c.score(item, result),
        status="correct",
        reviewer="Reviewer One",
        at_utc="2026-11-05T12:00:00Z",
        rationale="same referent",
    )
    assert digest(judged) == V1_GOLDEN["adjudicated"]
    stage_c.validate_score(judged, item, result)


# --- numeric oracle ------------------------------------------------------------------------
# (gold value, comparator, extra gold fields, answer, v1 outcome, v2 outcome)
ORACLE = [
    # thousands separators: recognized by v2 only
    ("750076", "eq", {}, "750,076", NE_NUMBER, CORRECT),
    ("1234567", "eq", {}, "1,234,567", NE_NUMBER, CORRECT),
    ("-12000", "eq", {}, "−12,000", NE_NUMBER, CORRECT),  # Unicode minus
    ("-12000", "eq", {}, "-12,000", NE_NUMBER, CORRECT),
    ("1234.5", "eq", {}, "1,234.5", NE_NUMBER, CORRECT),
    ("1234", "eq", {}, "1,234", NE_NUMBER, CORRECT),
    ("999999", "eq", {}, "999,999", NE_NUMBER, CORRECT),
    ("1000", "lt", {}, "<1,000", NE_NUMBER, CORRECT),
    ("1000", "lt", {}, "1,000", NE_NUMBER, ("incorrect", "comparator_mismatch")),
    ("1234000", "eq", {}, "1,234 × 10+3", NE_NUMBER, CORRECT),
    (
        "1250",
        "eq",
        {"reported_precision": "10", "rounding_interval": ["1245", "1255"]},
        "1,252",
        NE_NUMBER,
        CORRECT,
    ),  # interval is declared by the gold, not by the normalizer
    (
        "1250",
        "eq",
        {"reported_precision": "10", "rounding_interval": ["1245", "1255"]},
        "1,260",
        NE_NUMBER,
        WRONG_VALUE,
    ),
    ("1", "eq", {"unit": "mg"}, "1,000 μg", NE_NUMBER, CORRECT),  # registered conversion
    ("1", "eq", {"unit": "mg"}, "1,000 mL", NE_NUMBER, ("incorrect", "incompatible_units")),
    # a different number is still different
    ("1234", "eq", {}, "1,235", NE_NUMBER, WRONG_VALUE),
    ("1234", "eq", {}, "1,234,000", NE_NUMBER, WRONG_VALUE),
    # malformed or ambiguous grouping: unchanged
    ("15", "eq", {}, "1,5", NE_NUMBER, NE_NUMBER),
    ("1.5", "eq", {}, "1,5", NE_NUMBER, NE_NUMBER),
    ("123", "eq", {}, "0,123", NE_NUMBER, NE_NUMBER),
    ("12345", "eq", {}, "1,2345", NE_NUMBER, NE_NUMBER),
    ("1234567", "eq", {}, "12,345,67", NE_NUMBER, NE_NUMBER),
    ("123456", "eq", {}, "1,234,56", NE_NUMBER, NE_NUMBER),
    ("1234", "eq", {}, "1,234,5", NE_NUMBER, NE_NUMBER),
    ("750076", "eq", {}, "750 076", NE_NUMBER, NE_NUMBER),
    ("750076", "eq", {}, ",750,076", NE_NUMBER, NE_NUMBER),
    ("750076", "eq", {}, "750,076,", NE_NUMBER, NE_NUMBER),
    ("1234.5", "eq", {}, "1,234.5.6", NE_NUMBER, NE_NUMBER),
    ("1234.56789", "eq", {}, "1,234.567,89", NE_NUMBER, NE_NUMBER),
    ("1234", "eq", {}, "1,234.", NE_NUMBER, NE_NUMBER),
    ("0.1234", "eq", {}, "0.1,234", NE_NUMBER, NE_NUMBER),
    ("123400", "eq", {}, "1,234e2", NE_NUMBER, NE_NUMBER),  # grouping is not mixed with e-notation
    ("1234", "eq", {}, "1,234 and 5,678", NE_NUMBER, NE_NUMBER),
    # a decimal point is never a thousands separator
    ("750076", "eq", {}, "750.076", WRONG_VALUE, WRONG_VALUE),
    ("750.076", "eq", {}, "750.076", CORRECT, CORRECT),
    ("750.076", "eq", {}, "750,076", NE_NUMBER, WRONG_VALUE),  # grouped 750076, not 750.076
    # plain-text exponent with an explicit adjacent sign
    ("0.0002725", "eq", {}, "2.725 × 10−4", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725×10-4", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725 x 10−4", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725 X 10-4", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725 · 10-4", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725 * 10-4", NE_NUMBER, CORRECT),
    ("300", "eq", {}, "3 x 10+2", NE_NUMBER, CORRECT),
    ("300", "eq", {}, "3 × 10+2", NE_NUMBER, CORRECT),
    ("0.05", "lt", {}, "<5 × 10−2", NE_NUMBER, CORRECT),
    ("0.0002725", "eq", {}, "2.725 × 10−5", NE_NUMBER, WRONG_VALUE),  # magnitude differs
    ("0.0002725", "eq", {}, "2.725 × 10+4", NE_NUMBER, WRONG_VALUE),  # sign differs
    ("-0.0002725", "eq", {}, "2.725 × 10−4", NE_NUMBER, WRONG_VALUE),  # sign of the value
    # forms that must not become exponents
    ("23.25", "eq", {}, "2.725 × 10 − 4", NE_NUMBER, NE_NUMBER),  # arithmetic, not an exponent
    ("0.0002725", "eq", {}, "2.725 × 10 − 4", NE_NUMBER, NE_NUMBER),
    ("0.0002725", "eq", {}, "2.725 × 10 -4", NE_NUMBER, NE_NUMBER),
    ("27250", "eq", {}, "2.725 × 104", NE_NUMBER, NE_NUMBER),  # flattened superscript, no sign
    ("283.4", "eq", {}, "2.725 × 104", NE_NUMBER, NE_NUMBER),
    ("0.0002725", "eq", {}, "2.725 × 10 4", NE_NUMBER, NE_NUMBER),
    ("0.0002725", "eq", {}, "2.725 × 10-", NE_NUMBER, NE_NUMBER),
    ("0.0002725", "eq", {}, "2.725 × 10–4", NE_NUMBER, NE_NUMBER),  # en dash
    ("0.0002725", "eq", {}, "2.725 × 10—4", NE_NUMBER, NE_NUMBER),  # em dash
    ("10", "eq", {}, "5–10", NE_NUMBER, NE_NUMBER),  # a range, never a negative number
    ("-10", "eq", {}, "5–10", NE_NUMBER, NE_NUMBER),
    ("0", "eq", {}, "1 × 10-999", NE_NUMBER, NE_NUMBER),  # outside the supported exponent range
    # forms supported before v2 stay supported
    ("0.0002725", "eq", {}, "2.725 × 10^-4", CORRECT, CORRECT),
    ("0.0002725", "eq", {}, "2.725 × 10⁻⁴", CORRECT, CORRECT),
    ("0.0002725", "eq", {}, "2.725 × 10^−4", CORRECT, CORRECT),
    ("0.0002725", "eq", {}, "2.725e-4", CORRECT, CORRECT),
    ("750076", "eq", {}, "750076", CORRECT, CORRECT),
    (
        "23",
        "eq",
        {},
        "23 genes",
        ("non_evaluable", "unit_not_in_gold"),
        ("non_evaluable", "unit_not_in_gold"),
    ),  # no inferred counted noun
    (
        "45",
        "eq",
        {"unit": "%"},
        "0.45",
        ("non_evaluable", "unit_missing"),
        ("non_evaluable", "unit_missing"),
    ),  # no percent/proportion conversion
]


def outcome(gold_fields, comparator, extra, answer, rules):
    gold = make_item(quantity(gold_fields, comparator, **extra)).gold
    verdict = score_answer(gold, answer, rules)
    return verdict.status, verdict.reason


@pytest.mark.parametrize("value, comparator, extra, answer, v1, v2", ORACLE)
def test_numeric_oracle(value, comparator, extra, answer, v1, v2):
    assert outcome(value, comparator, extra, answer, "v1") == v1
    assert outcome(value, comparator, extra, answer, "v2") == v2


def test_default_rules_are_v1():
    gold = make_item(quantity("750076")).gold
    assert score_answer(gold, "750,076").reason == "uninterpretable_numeric_text"
    assert score_answer(gold, "750,076", "v2").status == "correct"


def test_unknown_rules_are_refused():
    item = make_item(quantity("1"))
    with pytest.raises(ValueError):
        score_answer(item.gold, "1", "v3")
    with pytest.raises(ValueError):
        stage_c.score(item, make_result(item, answer="1"), "v3")
    with pytest.raises(ValidationError):
        StageCScore.model_validate(
            {
                **stage_c.score(item, make_result(item, answer="1")).model_dump(),
                "scoring_rules": "v3",
            }
        )


@pytest.mark.parametrize(
    "gold, answer",
    [
        (entity("TP53", ["p53"]), "p53"),
        (entity("TP53", ["p53"]), "tp53"),
        (entity("TP53", ["p53"]), "not TP53"),
        (entity("TP53", ["p53"]), "KRAS"),
        ({"answer_class": "bounded_entity_set", "members": [member("A"), member("B")]}, "A; B"),
        ({"answer_class": "bounded_entity_set", "members": [member("A"), member("B")]}, "A, C"),
        (
            {
                "answer_class": "categorical_conclusion",
                "subject": "s",
                "relation": "r",
                "direction": "up",
                "equivalent_wording": ["higher in tumors"],
            },
            "higher in tumors.",
        ),
        (
            {
                "answer_class": "categorical_conclusion",
                "subject": "s",
                "relation": "r",
                "direction": "up",
            },
            "increased",
        ),
    ],
)
def test_non_quantity_scoring_is_identical_in_v2(gold, answer):
    item = make_item(gold)
    assert score_answer(item.gold, answer, "v1") == score_answer(item.gold, answer, "v2")


# --- version identity ----------------------------------------------------------------------


def v2_pair():
    item = make_item(quantity("750076"))
    result = make_result(item, answer="750,076")
    return item, result, stage_c.score(item, result, "v2")


def test_v2_record_names_its_rules_and_identity():
    item, result, v2 = v2_pair()
    v1 = stage_c.score(item, result)
    assert (v1.status, v1.reason) == NE_NUMBER and v1.scoring_rules == "v1"
    assert (v2.status, v2.reason, v2.scoring_rules) == ("correct", None, "v2")
    assert json.loads(serialize_project_json(v2, StageCScore))["scoring_rules"] == "v2"
    assert stage_c.score_ref(v2).artifact_id != stage_c.score_ref(v1).artifact_id
    stage_c.validate_score(v2, item, result)
    stage_c.validate_score(v1, item, result)


def test_validation_recomputes_under_the_recorded_rules_only():
    item, result, v2 = v2_pair()
    with pytest.raises(ValueError):
        stage_c.validate_score(v2.model_copy(update={"scoring_rules": "v1"}), item, result)
    v1 = stage_c.score(item, result)
    with pytest.raises(ValueError):
        stage_c.validate_score(v1.model_copy(update={"scoring_rules": "v2"}), item, result)


def test_identical_verdicts_still_carry_distinct_identities(tmp_path):
    item = make_item(quantity("0.05"))
    result = make_result(item)
    v1, v2 = stage_c.score(item, result), stage_c.score(item, result, "v2")
    assert (v1.status, v2.status) == ("correct", "correct")
    ref1, ref2 = stage_c.score_ref(v1), stage_c.score_ref(v2)
    assert ref1.artifact_id != ref2.artifact_id and ref1.sha256 != ref2.sha256
    ref, _ = publish_score(v2, item, result, tmp_path)
    assert read_score(ref, tmp_path, item, result) == v2
    # the v2 bytes cannot be read under the v1 identity, nor the reverse
    (tmp_path / ref1.logical_ref).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / ref1.logical_ref).write_bytes((tmp_path / ref.logical_ref).read_bytes())
    with pytest.raises(ValueError):
        read_score(ref1, tmp_path, item, result)


@pytest.mark.parametrize("rules", ["v1", "v2"])
def test_published_score_and_manifest_carry_the_rules(tmp_path, rules):
    item, result = v1_cases()["correct"]
    score = stage_c.score(item, result, rules)
    ref, build = publish_score(score, item, result, tmp_path)
    assert read_score(ref, tmp_path, item, result) == score
    manifest = read_manifest(tmp_path / build.logical_ref, build.sha256)
    assert manifest.transformation_id == f"stage_c_score_{rules}"


def test_adjudication_inherits_and_checks_the_automatic_rules():
    item = make_item(entity("Brca1"))
    result = make_result(item, answer="BRCA1")
    args = {
        "status": "correct",
        "reviewer": "Reviewer One",
        "at_utc": "2026-11-05T12:00:00Z",
        "rationale": "same referent",
    }
    for rules in ("v1", "v2"):
        automatic = stage_c.score(item, result, rules)
        judged = stage_c.adjudicate(automatic, **args)
        assert judged.scoring_rules == rules
        assert judged.automatic == stage_c.score_ref(automatic)
        stage_c.validate_score(judged, item, result)
    other = "v1" if judged.scoring_rules == "v2" else "v2"
    with pytest.raises(ValueError):  # the cited automatic score was built under other rules
        stage_c.validate_score(judged.model_copy(update={"scoring_rules": other}), item, result)


def test_missing_and_malformed_outcomes_do_not_depend_on_rules():
    item = make_item(quantity("750076"))
    for result, expected in (
        (failed_result(item), ("missing", "operational_failure")),
        (
            make_result(item, answer=None, parse_status="unparseable", abstained=None),
            ("incorrect", "unparseable"),
        ),
        (make_result(item, answer=None, abstained=True), ("incorrect", "abstained")),
    ):
        for rules in ("v1", "v2"):
            s = stage_c.score(item, result, rules)
            assert (s.status, s.reason) == expected


# --- analysis table ------------------------------------------------------------------------


def scored(w, rules, **plan):
    o = observation(w, **plan)
    return replace(o, score=stage_c.score(w.item, o.result, rules))


def test_table_names_the_scoring_rules_it_was_built_with(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    v1 = tbl([observation(w, case=case)])
    assert v1.stage_c_rules == "v1"
    _, inv1, build1 = publish_table(v1, tmp_path / "v1")
    v2_obs = scored(w, "v2", case=case)
    v2 = tbl([v2_obs])
    assert v2.stage_c_rules == "v2"
    _, inv2, build2 = publish_table(v2, tmp_path / "v2")
    read1 = read_table(build1, tmp_path / "v1")
    read2 = read_table(build2, tmp_path / "v2")
    assert (read1, read2) == (v1.rows, v2.rows)

    def reference(root, ref):
        return json.loads((root / ref.logical_ref).read_text())["scoring_reference"]

    assert reference(tmp_path / "v1", inv1) == SCORING_REFERENCE
    assert reference(tmp_path / "v2", inv2) == SCORING_REFERENCE | {"stage_c": "stage_c_score_v2"}
    settings = [
        read_manifest(r / b.logical_ref, b.sha256).transformation_settings_sha256
        for r, b in ((tmp_path / "v1", build1), (tmp_path / "v2", build2))
    ]
    assert settings[0] != settings[1]


def test_table_rejects_mixed_scoring_rules():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    first = scored(w, "v1", case=case)
    second = scored(w, "v2", case=case, condition="PDF_DOCLING")
    with pytest.raises(ValueError, match="mixed"):
        build_table([first, second], tbl([first]).roster)


def test_table_readback_rejects_a_scoring_reference_that_disagrees_with_the_build(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    _, inventory_ref, build_ref = publish_table(tbl([scored(w, "v2", case=case)]), tmp_path)
    path = tmp_path / inventory_ref.logical_ref
    data = json.loads(path.read_text())
    data["scoring_reference"]["stage_c"] = "stage_c_score_v1"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        read_table(build_ref, tmp_path)
