"""Stage C: scientific answer scoring against adjudicated gold.

The oracle tables below are hand-authored from the frozen scoring rules
(BENCHMARK_SPEC section 6) and decisions D2/D3, before the scorer existed.
"""

import inspect

import pytest
from pydantic import ValidationError

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.answer_scoring import normalize_text, score_answer
from biorag_repro.evaluation.stage_c_models import GenerationResult, StageCScore
from biorag_repro.evaluation.stage_c_storage import publish_score, read_score

CONFIG = "c" * 64


def quantity(value, comparator="eq", **extra):
    return {"answer_class": "quantitative_value", "value": value, "comparator": comparator, **extra}


def entity(canonical, aliases=(), answer_class="biological_entity"):
    return {"answer_class": answer_class, "canonical": canonical, "aliases": list(aliases)}


def member(canonical, aliases=()):
    return {"answer_class": "biological_entity", "canonical": canonical, "aliases": list(aliases)}


def make_item(gold, **changes) -> BenchmarkItem:
    data = {
        "schema_version": "1",
        "benchmark_version": "benchmark_v1",
        "corpus_version": "corpus_v1",
        "item_id": "q_001",
        "paper_id": "P001",
        "question": "What was reported?",
        "answer_class": gold["answer_class"],
        "gold": gold,
        "evidence_sets": [
            {
                "components": [
                    {"role": "subject", "text": "s"},
                    {"role": "relation", "text": "r"},
                    {"role": "answer", "text": "a"},
                ]
            }
        ],
        "jats_refs": [
            {
                "artifact_id": "raw_jats",
                "source_sha256": "a" * 64,
                "paper_id": "P001",
                "xml_path": "/article/body/p[1]",
                "source_text": "source",
            }
        ],
        "pdf_refs": [
            {
                "artifact_id": "raw_pdf",
                "source_sha256": "b" * 64,
                "paper_id": "P001",
                "page": 1,
                "source_text": "source",
            }
        ],
        "jats_status": "present_equivalent",
        "pdf_status": "present_equivalent",
        "split": "development",
        "candidate_status": "accepted",
        "review_status": "reviewed",
        "audit_status": "not_selected",
        "decision_ids": ["d1"],
    }
    data.update(changes)
    return BenchmarkItem.model_validate(data)


def verdict(gold, answer):
    item = make_item(gold)
    v = score_answer(item.gold, answer)
    return v.status, v.reason


# --- D2 normalization --------------------------------------------------------


def test_normalization_is_only_unicode_composition_and_whitespace():
    assert normalize_text("  T  cell\n") == "T cell"
    assert normalize_text("é") == "é"  # NFC composition
    assert normalize_text("BRCA1") == "BRCA1"  # no case folding
    assert normalize_text("TP53-") == "TP53-"  # punctuation untouched
    assert normalize_text("T cell") != normalize_text("Tcell")


# --- D3 quantitative oracle ---------------------------------------------------

QUANT = [
    # (gold value, comparator, extra gold, answer, expected status, expected reason)
    ("0.000012", "eq", {}, "1.2e-5", "correct", None),
    ("0.000012", "eq", {}, "1.2E-5", "correct", None),
    ("0.000012", "eq", {}, "1.2 × 10⁻⁵", "correct", None),
    ("0.000012", "eq", {}, "1.2 x 10^-5", "correct", None),
    ("0.000012", "eq", {}, "1.2 × 10^−5", "correct", None),  # Unicode minus exponent
    ("-3.5", "eq", {}, "−3.5", "correct", None),  # Unicode minus
    ("-3.5", "eq", {}, "-3.50", "correct", None),
    ("-3.5", "eq", {}, "3.5", "incorrect", "value_mismatch"),  # sign lost
    ("3.5", "eq", {}, "-3.5", "incorrect", "value_mismatch"),
    ("0.05", "eq", {}, "0.5", "incorrect", "value_mismatch"),
    ("0.05", "eq", {}, "0.05", "correct", None),
    ("0.000012", "eq", {}, "1.2e-4", "incorrect", "value_mismatch"),  # magnitude
    ("0.001", "lt", {}, "p<0.001", "correct", None),
    ("0.001", "lt", {}, "p < 0.001", "correct", None),
    ("0.001", "lt", {}, "< 0.001", "correct", None),
    ("0.001", "lt", {}, "p=0.001", "incorrect", "comparator_mismatch"),
    ("0.001", "lt", {}, "0.001", "incorrect", "comparator_mismatch"),  # bare = equality
    ("0.001", "eq", {}, "p<0.001", "incorrect", "comparator_mismatch"),
    ("0.001", "lt", {}, "p>0.001", "incorrect", "comparator_mismatch"),
    ("0.001", "lt", {}, "p<=0.001", "incorrect", "comparator_mismatch"),
    ("0.001", "le", {}, "p≤0.001", "correct", None),
    ("0.001", "ge", {}, "p>=0.001", "correct", None),
    ("0.05", "lt", {}, "p<0.01", "incorrect", "value_mismatch"),  # boundary differs
    # units
    ("5", "eq", {"unit": "mg"}, "5 mg", "correct", None),
    ("5", "eq", {"unit": "mg"}, "5000 μg", "correct", None),
    ("5", "eq", {"unit": "mg"}, "5000 µg", "correct", None),  # micro sign
    ("5", "eq", {"unit": "mg"}, "0.005 g", "correct", None),
    ("5", "eq", {"unit": "mg"}, "5 g", "incorrect", "value_mismatch"),
    ("90", "eq", {"unit": "s"}, "1.5 min", "correct", None),
    ("1", "eq", {"unit": "min"}, "1 s", "incorrect", "value_mismatch"),  # inexact, exact arith
    ("5", "eq", {"unit": "mg"}, "5 mL", "incorrect", "incompatible_units"),
    ("5", "eq", {"unit": "mg"}, "5 mol/L", "non_evaluable", "unknown_unit_conversion"),
    ("5", "eq", {"unit": "mg"}, "5", "non_evaluable", "unit_missing"),
    ("0.91", "eq", {}, "0.91 mg", "non_evaluable", "unit_not_in_gold"),
    ("5", "eq", {"unit": "kDa"}, "5 kDa", "correct", None),  # identical unit needs no registry
    ("5", "eq", {"unit": "kDa"}, "5000 Da", "non_evaluable", "unknown_unit_conversion"),
    # rounding: exact reported value unless the item authorizes an interval
    ("0.0482", "eq", {}, "0.05", "incorrect", "value_mismatch"),
    ("0.0482", "eq", {}, "0.0482", "correct", None),
    (
        "0.05",
        "eq",
        {"reported_precision": "0.01", "rounding_interval": ["0.045", "0.055"]},
        "0.0482",
        "correct",
        None,
    ),
    (
        "0.05",
        "eq",
        {"reported_precision": "0.01", "rounding_interval": ["0.045", "0.055"]},
        "0.06",
        "incorrect",
        "value_mismatch",
    ),
    ("0.91", "eq", {"approved_equivalents": ["91"]}, "91", "correct", None),
    ("0.91", "eq", {"approved_equivalents": ["91"]}, "0.9", "incorrect", "value_mismatch"),
    # text that merely contains digits is not guessed
    ("1000", "eq", {}, "1,000", "non_evaluable", "uninterpretable_numeric_text"),
    ("5", "eq", {}, "5-10", "non_evaluable", "uninterpretable_numeric_text"),
    ("5", "eq", {}, "about five", "incorrect", "no_numeric_value"),
    ("1.2", "eq", {}, "1.2 × 10-5", "non_evaluable", "uninterpretable_numeric_text"),
    ("5", "eq", {}, "1e999999", "non_evaluable", "uninterpretable_numeric_text"),
]


@pytest.mark.parametrize("value,comparator,extra,answer,status,reason", QUANT)
def test_quantitative_oracle(value, comparator, extra, answer, status, reason):
    assert verdict(quantity(value, comparator, **extra), answer) == (status, reason)


def test_exact_arithmetic_has_no_float_error():
    assert verdict(quantity("0.3", unit="g"), "300 mg") == ("correct", None)
    assert verdict(quantity("0.1", unit="g"), "100.0000000000001 mg") == (
        "incorrect",
        "value_mismatch",
    )


# --- entities ---------------------------------------------------------------

ENTITY = [
    ("TP53", ["p53"], "TP53", "correct", None),
    ("TP53", ["p53"], "p53", "correct", None),
    ("TP53", ["p53"], "  TP53 ", "correct", None),
    ("TP53", ["p53"], "TP63", "incorrect", "entity_not_approved"),
    ("TP53", ["p53"], "TP53 ", "correct", None),
    ("TP53", ["p53"], "Tp53", "non_evaluable", "case_only_difference"),
    ("Brca1", [], "BRCA1", "non_evaluable", "case_only_difference"),
    ("rs1801133 C>T", [], "rs1801133 C>T", "correct", None),
    ("rs1801133 C>T", [], "rs1801131 C>T", "incorrect", "entity_not_approved"),
    ("BRAF V600E", [], "BRAF V600K", "incorrect", "entity_not_approved"),
    ("CD4+ T cell", [], "CD4+ T cells", "incorrect", "entity_not_approved"),  # no substring match
    ("TCGA-BRCA", [], "TCGA-BRCA cohort", "non_evaluable", "approved_name_in_longer_text"),
    ("TP53", ["p53"], "TP53.", "non_evaluable", "approved_name_in_longer_text"),
    ("TP53", ["p53"], "(p53)", "non_evaluable", "approved_name_in_longer_text"),
    ("TP53", ["p53"], "The gene is TP53", "non_evaluable", "approved_name_in_longer_text"),
    ("TP53", ["p53"], "It is not TP53.", "non_evaluable", "approved_name_in_longer_text"),
    ("TP53", [], "TP530", "incorrect", "entity_not_approved"),  # boundary, not substring
    ("TP53", [], "TP63 and TP73", "incorrect", "entity_not_approved"),
    ("TP53", [], "tp53 mutation", "incorrect", "entity_not_approved"),
    ("TP53", [], "", "incorrect", "empty_answer"),
]


@pytest.mark.parametrize("canonical,aliases,answer,status,reason", ENTITY)
def test_entity_oracle(canonical, aliases, answer, status, reason):
    gold = entity(canonical, aliases)
    assert verdict(gold, answer) == (status, reason)


def test_cohort_uses_same_rules():
    gold = entity("GTEx", ["Genotype-Tissue Expression"], "dataset_cohort")
    assert verdict(gold, "Genotype-Tissue Expression") == ("correct", None)
    assert verdict(gold, "GTEx v8") == ("non_evaluable", "approved_name_in_longer_text")


# --- bounded sets -----------------------------------------------------------

SET_GOLD = {
    "answer_class": "bounded_entity_set",
    "members": [member("TP53", ["p53"]), member("KRAS"), member("EGFR")],
}


SETS = [
    (["TP53", "KRAS", "EGFR"], "correct", None),
    (["EGFR", "TP53", "KRAS"], "correct", None),  # order
    (["p53", "KRAS", "EGFR"], "correct", None),  # alias
    (["TP53", "p53", "KRAS", "EGFR"], "correct", None),  # same referent twice
    (["TP53", "KRAS"], "incorrect", "set_incomplete"),
    (["TP53", "KRAS", "EGFR", "PIK3CA"], "incorrect", "set_extra_member"),
    (["TP53", "KRAS", "PIK3CA"], "incorrect", "set_extra_member"),  # swap, no partial credit
    ([], "incorrect", "empty_answer"),
    (["TP53", "KRAS", "Egfr"], "non_evaluable", "case_only_difference"),
    (["TP53", "KRAS", "Egfr", "PIK3CA"], "incorrect", "set_extra_member"),
    (["TP53", "Kras"], "incorrect", "set_incomplete"),  # EGFR missing under any reading
    (["TP53", "KRAS", "EGFR", "tp53"], "non_evaluable", "case_only_difference"),
    (["TP53", "KRAS", "EGFR", "Kras"], "non_evaluable", "case_only_difference"),
    ("TP53, tp53, KRAS, EGFR", "non_evaluable", "case_only_difference"),
    (["TP53", "KRAS", "EGFR", "tp53", "PIK3CA"], "incorrect", "set_extra_member"),
    (["TP53", "p53", "KRAS", "EGFR", "P53"], "non_evaluable", "case_only_difference"),
    ("TP53; KRAS; EGFR", "correct", None),
    ("TP53, KRAS, EGFR", "correct", None),
    ("TP53\nKRAS\nEGFR", "correct", None),
    ("TP53, KRAS and EGFR", "non_evaluable", "set_segmentation_ambiguous"),
    ("TP53, KRAS", "incorrect", "set_incomplete"),
]


@pytest.mark.parametrize("answer,status,reason", SETS)
def test_set_oracle(answer, status, reason):
    assert verdict(SET_GOLD, answer) == (status, reason)


def test_set_with_delimiter_inside_member_name_is_not_split_on_it():
    gold = {
        "answer_class": "bounded_entity_set",
        "members": [member("IL-2, soluble"), member("KRAS")],
    }
    assert verdict(gold, "IL-2, soluble, KRAS") == ("non_evaluable", "set_segmentation_ambiguous")
    assert verdict(gold, ["IL-2, soluble", "KRAS"]) == ("correct", None)


# --- categorical ------------------------------------------------------------

CONCLUSION = {
    "answer_class": "categorical_conclusion",
    "subject": "drug X",
    "relation": "survival",
    "direction": "increased",
    "condition": "in KRAS-mutant tumours",
    "equivalent_wording": [
        "Drug X increased survival in KRAS-mutant tumours.",
        "Survival was higher with drug X in KRAS-mutant tumours.",
    ],
}


CATEGORICAL = [
    ("Drug X increased survival in KRAS-mutant tumours.", "correct", None),
    ("Drug X increased survival in KRAS-mutant tumours", "correct", None),  # final period
    ("Drug X  increased\nsurvival in KRAS-mutant tumours.", "correct", None),  # formatting only
    ("Survival was higher with drug X in KRAS-mutant tumours.", "correct", None),  # curated
    ("drug X increased survival in KRAS-mutant tumours.", "non_evaluable", "case_only_difference"),
    # not curated: never accepted, never auto-failed
    ("Drug X decreased survival in KRAS-mutant tumours.", "non_evaluable", "wording_not_approved"),
    ("Drug X increased survival.", "non_evaluable", "wording_not_approved"),  # condition omitted
    ("Drug X increased survival in tumours.", "non_evaluable", "wording_not_approved"),
    ("", "incorrect", "empty_answer"),
]


@pytest.mark.parametrize("answer,status,reason", CATEGORICAL)
def test_categorical_oracle(answer, status, reason):
    assert verdict(CONCLUSION, answer) == (status, reason)


def test_categorical_without_curated_wording_cannot_auto_score_anything_correct():
    gold = {k: v for k, v in CONCLUSION.items() if k != "equivalent_wording"}
    assert verdict(gold, "Drug X increased survival in KRAS-mutant tumours.") == (
        "non_evaluable",
        "wording_not_approved",
    )


# --- generation result and observation scoring -------------------------------


def make_result(item=None, **changes) -> GenerationResult:
    item = item or make_item(quantity("0.05"))
    raw = '{"answer": "0.05", "abstain": false}'
    data = {
        "phase": "phase7_dev",
        "benchmark_version": item.benchmark_version,
        "item_id": item.item_id,
        "paper_id": item.paper_id,
        "matrix_arm": "jats_bm25",
        "representation_or_condition": "JATS",
        "repeat_index": 1,
        "resolved_config_sha256": CONFIG,
        "run_id": "run1",
        "attempt_id": "attempt1",
        "status": "completed",
        "output": ArtifactRef(
            artifact_id="gen_1",
            logical_ref="generation/gen_1.json",
            sha256=sha256_bytes(raw.encode()),
        ),
        "raw_output": raw,
        "parse_status": "parsed",
        "answer": "0.05",
        "abstained": False,
    }
    data["observation_key"] = observation_key(
        phase=data["phase"],
        benchmark_version=data["benchmark_version"],
        item_id=data["item_id"],
        matrix_arm=data["matrix_arm"],
        representation_or_condition=data["representation_or_condition"],
        repeat_index=data["repeat_index"],
        resolved_config_sha256=data["resolved_config_sha256"],
    )
    data.update(changes)
    return GenerationResult(**data)


def rekey(result: GenerationResult, **changes) -> GenerationResult:
    """Change coordinates and keep the observation key honest."""
    data = result.model_dump() | changes
    data["observation_key"] = observation_key(
        phase=data["phase"],
        benchmark_version=data["benchmark_version"],
        item_id=data["item_id"],
        matrix_arm=data["matrix_arm"],
        representation_or_condition=data["representation_or_condition"],
        repeat_index=data["repeat_index"],
        resolved_config_sha256=data["resolved_config_sha256"],
    )
    return GenerationResult(**data)


def failed_result(item=None, **changes):
    base = make_result(item)
    data = base.model_dump() | {
        "status": "operational_failure",
        "output": None,
        "raw_output": None,
        "parse_status": None,
        "answer": None,
        "abstained": None,
        "failure": "CUDA out of memory",
    }
    data.update(changes)
    return GenerationResult(**data)


def test_correct_incorrect_completed_outputs():
    item = make_item(quantity("0.05"))
    s = stage_c.score(item, make_result(item))
    assert (s.status, s.reason, s.basis) == ("correct", None, "deterministic")
    s = stage_c.score(item, make_result(item, answer="0.5"))
    assert (s.status, s.reason) == ("incorrect", "value_mismatch")


def test_completed_abstention_on_answerable_item_is_incorrect():
    item = make_item(quantity("0.05"))
    s = stage_c.score(item, make_result(item, abstained=True, answer=None))
    assert (s.status, s.reason) == ("incorrect", "abstained")
    # an abstention flag wins over whatever answer text accompanies it
    s = stage_c.score(item, make_result(item, abstained=True, answer="0.05"))
    assert (s.status, s.reason) == ("incorrect", "abstained")


def test_completed_unparseable_output_is_incorrect():
    item = make_item(quantity("0.05"))
    result = make_result(item, parse_status="unparseable", answer=None, abstained=None)
    s = stage_c.score(item, result)
    assert (s.status, s.reason) == ("incorrect", "unparseable")


def test_operational_failure_is_missing_not_incorrect():
    item = make_item(quantity("0.05"))
    s = stage_c.score(item, failed_result(item))
    assert (s.status, s.reason) == ("missing", "operational_failure")
    assert "CUDA" in s.detail


def test_unresolved_scoring_is_non_evaluable_with_reason():
    item = make_item(entity("Brca1"))
    s = stage_c.score(item, make_result(item, answer="BRCA1"))
    assert (s.status, s.reason) == ("non_evaluable", "case_only_difference")
    assert s.detail


def test_unusable_gold_is_not_scored_incorrect():
    item = make_item(quantity("0.05"), candidate_status="proposed", review_status="pending")
    s = stage_c.score(item, make_result(item, answer="0.5"))
    assert (s.status, s.reason) == ("non_evaluable", "gold_unusable")
    excluded = make_item(
        quantity("0.05"), candidate_status="excluded", review_status="reviewed", split=None
    )
    assert stage_c.score(excluded, make_result(excluded)).reason == "gold_unusable"


def test_correct_answer_is_correct_whatever_evidence_did():
    """Stage C takes no Stage A/B argument: correctness cannot depend on them."""
    assert list(inspect.signature(stage_c.score).parameters) == ["item", "result"]
    fields = set(StageCScore.model_fields) | set(GenerationResult.model_fields)
    assert not any("stage_a" in f or "stage_b" in f or "context" in f for f in fields)
    item = make_item(quantity("0.05"))
    right = stage_c.score(item, make_result(item))
    wrong = stage_c.score(item, make_result(item, answer="0.06"))
    assert (right.status, wrong.status) == ("correct", "incorrect")


def test_answer_matching_another_items_gold_is_not_credited():
    item = make_item(quantity("0.05"))
    other_item_gold = "0.91"
    s = stage_c.score(item, make_result(item, answer=other_item_gold))
    assert s.status == "incorrect"


@pytest.mark.parametrize(
    "change",
    [
        {"item_id": "q_002"},
        {"paper_id": "P002"},
        {"benchmark_version": "benchmark_v2"},
    ],
)
def test_result_for_another_item_paper_or_benchmark_is_refused(change):
    item = make_item(quantity("0.05"))
    with pytest.raises(ValueError, match="designated"):
        stage_c.score(item, rekey(make_result(item), **change))


def test_counterfeit_observation_key_is_refused():
    item = make_item(quantity("0.05"))
    forged = make_result(item).model_copy(update={"observation_key": "obs_" + "0" * 64})
    with pytest.raises(ValueError, match="observation"):
        stage_c.score(item, forged)
    other_condition = rekey(make_result(item), representation_or_condition="PDF_DOCLING")
    assert stage_c.score(item, other_condition).status == "correct"  # key itself is honest
    mismatched = other_condition.model_copy(
        update={"observation_key": make_result(item).observation_key}
    )
    with pytest.raises(ValueError, match="observation"):
        stage_c.score(item, mismatched)


def test_generation_result_rejects_inconsistent_shapes():
    with pytest.raises(ValidationError, match="raw output"):
        make_result(raw_output='{"answer": "tampered"}')
    with pytest.raises(ValidationError):
        make_result(parse_status="parsed", abstained=None)
    with pytest.raises(ValidationError):
        make_result(parse_status="unparseable", answer="0.05")
    with pytest.raises(ValidationError):
        failed_result(raw_output="partial")
    with pytest.raises(ValidationError):
        failed_result(failure=" ")
    with pytest.raises(ValidationError):
        make_result(repeat_index=-1)


# --- adjudication ------------------------------------------------------------


def adjudicated(item, result, status="correct", **changes):
    automatic = stage_c.score(item, result)
    args = {
        "status": status,
        "reviewer": "Reviewer One",
        "at_utc": "2026-11-05T12:00:00Z",
        "rationale": "Mouse Brca1 and human BRCA1 are the same referent for this item.",
    }
    args.update(changes)
    return automatic, stage_c.adjudicate(automatic, **args)


def test_reviewer_resolves_only_non_evaluable_scientific_cases():
    item = make_item(entity("Brca1"))
    result = make_result(item, answer="BRCA1")
    automatic, judged = adjudicated(item, result)
    assert (judged.status, judged.basis, judged.reviewer) == ("correct", "reviewer", "Reviewer One")
    assert judged.automatic == stage_c.score_ref(automatic)
    stage_c.validate_score(judged, item, result)
    # gold and answer are not edited to reach the verdict
    assert item.gold.canonical == "Brca1" and judged.interpreted == automatic.interpreted
    for bad in (
        make_result(item, answer="Brca1"),  # already correct
        make_result(item, answer="TP53"),  # already incorrect
    ):
        with pytest.raises(ValueError):
            stage_c.adjudicate(
                stage_c.score(item, bad),
                status="correct",
                reviewer="R",
                at_utc="2026-11-05T12:00:00Z",
                rationale="x",
            )
    with pytest.raises(ValueError):
        stage_c.adjudicate(
            stage_c.score(item, failed_result(item)),
            status="incorrect",
            reviewer="R",
            at_utc="2026-11-05T12:00:00Z",
            rationale="x",
        )
    unusable = make_item(entity("Brca1"), candidate_status="proposed", review_status="pending")
    with pytest.raises(ValueError):
        stage_c.adjudicate(
            stage_c.score(unusable, make_result(unusable)),
            status="correct",
            reviewer="R",
            at_utc="2026-11-05T12:00:00Z",
            rationale="x",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"reviewer": " "},
        {"rationale": ""},
        {"at_utc": "2026-11-05T12:00:00"},
        {"at_utc": "2026-11-05T12:00:00+02:00"},
        {"status": "missing"},
        {"status": "non_evaluable"},
    ],
)
def test_adjudication_requires_identity_time_decision_and_rationale(changes):
    item = make_item(entity("Brca1"))
    automatic = stage_c.score(item, make_result(item, answer="BRCA1"))
    args = {
        "status": "correct",
        "reviewer": "R",
        "at_utc": "2026-11-05T12:00:00Z",
        "rationale": "because",
    } | changes
    with pytest.raises((ValueError, ValidationError)):
        stage_c.adjudicate(automatic, **args)


def test_deterministic_record_cannot_carry_reviewer_fields():
    item = make_item(quantity("0.05"))
    s = stage_c.score(item, make_result(item))
    with pytest.raises(ValidationError):
        StageCScore(**(s.model_dump() | {"reviewer": "R"}))
    with pytest.raises(ValidationError):
        StageCScore(**(s.model_dump() | {"status": "correct", "reason": "value_mismatch"}))


# --- publication and read-back -----------------------------------------------


def test_publish_read_back_and_immutability(tmp_path):
    item = make_item(quantity("0.05"))
    result = make_result(item)
    s = stage_c.score(item, result)
    ref, _ = publish_score(s, item, result, tmp_path)
    assert ref == stage_c.score_ref(s)
    assert read_score(ref, tmp_path, item, result) == s
    with pytest.raises(FileExistsError):  # published records are immutable
        publish_score(s, item, result, tmp_path)
    path = tmp_path / ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b'"correct"', b'"incorrect"'))
    with pytest.raises(ValueError):
        read_score(ref, tmp_path, item, result)


def test_read_back_rejects_rehashed_forgery_and_wrong_inputs(tmp_path):
    item = make_item(quantity("0.05"))
    result = make_result(item, answer="0.5")
    s = stage_c.score(item, result)
    ref, _ = publish_score(s, item, result, tmp_path)
    # a forged "correct" record, honestly hashed and honestly addressed
    forged = StageCScore(**(s.model_dump() | {"status": "correct", "reason": None}))
    with pytest.raises(ValueError, match="evaluator"):
        publish_score(forged, item, result, tmp_path)
    # inputs that differ from those the record was scored from
    with pytest.raises(ValueError):
        read_score(ref, tmp_path, item, make_result(item, answer="0.05"))
    other = make_item(quantity("0.5"))
    with pytest.raises(ValueError):
        read_score(ref, tmp_path, other, result)
    with pytest.raises(ValueError):
        stage_c.validate_score(s.model_copy(update={"item_sha256": "0" * 64}), item, result)


def test_adjudicated_record_publishes_and_reads_back(tmp_path):
    item = make_item(entity("Brca1"))
    result = make_result(item, answer="BRCA1")
    automatic, judged = adjudicated(item, result, "incorrect")
    publish_score(automatic, item, result, tmp_path)
    ref, _ = publish_score(judged, item, result, tmp_path)
    assert read_score(ref, tmp_path, item, result) == judged
    assert judged.status == "incorrect"
    # reviewer record that cites an automatic record from other inputs is refused
    other_result = make_result(item, answer="Brca2")
    with pytest.raises(ValueError):
        stage_c.validate_score(judged, item, other_result)
    # reviewer record whose cited automatic record was not unresolved
    clean = stage_c.score(item, make_result(item, answer="Brca1"))
    bad = judged.model_copy(update={"automatic": stage_c.score_ref(clean)})
    with pytest.raises(ValueError):
        stage_c.validate_score(bad, item, result)


def test_identity_depends_on_every_scientific_input():
    item = make_item(quantity("0.05"))
    base = stage_c.score_ref(stage_c.score(item, make_result(item)))
    other_repeat = stage_c.score_ref(stage_c.score(item, rekey(make_result(item), repeat_index=2)))
    other_arm = stage_c.score_ref(
        stage_c.score(item, rekey(make_result(item), matrix_arm="pdf_bm25"))
    )
    assert len({base.artifact_id, other_repeat.artifact_id, other_arm.artifact_id}) == 3
    assert stage_c.score(item, make_result(item)) == stage_c.score(item, make_result(item))
    raw = serialize_project_json(stage_c.score(item, make_result(item)), StageCScore)
    assert sha256_bytes(raw) == base.sha256


def test_punctuated_entity_is_adjudicable_and_keeps_the_automatic_score():
    item = make_item(entity("TP53", ["p53"]))
    result = make_result(item, answer="TP53.")
    automatic = stage_c.score(item, result)
    assert (automatic.status, automatic.reason) == ("non_evaluable", "approved_name_in_longer_text")
    _, judged = adjudicated(item, result, rationale="Trailing full stop only.")
    assert (judged.status, judged.basis, judged.reviewer) == ("correct", "reviewer", "Reviewer One")
    assert judged.automatic == stage_c.score_ref(automatic)
    assert judged.rationale == "Trailing full stop only." and judged.at_utc
    stage_c.validate_score(judged, item, result)
    _, rejected = adjudicated(item, make_result(item, answer="It is not TP53."), status="incorrect")
    assert rejected.status == "incorrect" and rejected.reason == "adjudicated_incorrect"


def test_other_incorrect_scores_stay_closed_to_review():
    item = make_item(entity("TP53"))
    for answer in ("TP63", "tumor protein p53", "TP530"):
        with pytest.raises(ValueError):
            stage_c.adjudicate(
                stage_c.score(item, make_result(item, answer=answer)),
                status="correct",
                reviewer="R",
                at_utc="2026-11-05T12:00:00Z",
                rationale="x",
            )
    # a completed unparseable output is not turned into an answer by review
    unparsed = make_result(item, parse_status="unparseable", answer=None, abstained=None)
    with pytest.raises(ValueError):
        stage_c.adjudicate(
            stage_c.score(item, unparsed),
            status="correct",
            reviewer="R",
            at_utc="2026-11-05T12:00:00Z",
            rationale="x",
        )
