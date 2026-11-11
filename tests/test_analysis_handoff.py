# ruff: noqa: F811
"""Analysis handoff: the Python-authored record that precedes any R execution.

Synthetic rows prove the contract only; nothing here is a result. Official-origin tests use
the scripted official run of test_generation_official (simulated approvals, no model).
"""

import dataclasses
import json
from typing import get_args

import pytest
from generation_support import needs_tokenizer
from test_analysis_export import (
    CFG,
    P_A,
    P_B,
    closed_book,
    deliver,
    failed_for,
    observation,
    result_for,
    roster_of,
    world,
)
from test_generation_official import (  # noqa: F401  (fixtures)
    CONTEXTS,
    anchored,
    build_world,
    recognised,
    verify,
    view,
)
from test_generation_roster import make_scope
from test_stage_b import altered_world

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.hashing import HashMismatchError, canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_models import (
    COLUMNS,
    PRIMARY_SELECTION,
    AnalysisInventory,
    AnalysisRow,
    PlannedRoster,
    RosterItem,
)
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import build_table, primary_pairs, serialize_csv
from biorag_repro.evaluation.attribution import Correct, Delivered
from biorag_repro.generation.environment import GitState
from biorag_repro.generation.handoff import official_observations

CODE = GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0)
BOTH = ("JATS", "PDF_DOCLING")


# ---- helpers ----------------------------------------------------------------------------


def pair(w, case=None, *, jats=..., pdf=...):
    """Confirmatory repeat-1 pair. PDF rows carry no Stage A (no PDF world in this fixture)."""
    if jats is ...:
        jats = observation(w, case=case or deliver(w, [w.chunk(P_A)]))
    if pdf is ...:
        pdf = observation(
            w,
            evidence=None,
            condition="PDF_DOCLING",
            result=result_for(w.item, representation_or_condition="PDF_DOCLING"),
        )
    return [jats, pdf]


def publish(tmp_path, obs, roster=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    table = build_table(obs, roster or roster_of(obs))
    _csv, _inventory, build = publish_table(table, tmp_path)
    return table, build


def rehearsal(tmp_path, obs, roster=None):
    table, build = publish(tmp_path, obs, roster)
    return table, build, ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True)


def outcome(handoff, name):
    return next(o for o in handoff.accounting if o.outcome == name)


def entry(handoff, name, item_id):
    return next(e for e in outcome(handoff, name).eligible_items if e.item_id == item_id)


# ---- frozen analysis constants (independent oracle: EXPERIMENT_PLAN sections 6-9) ----------


def test_analysis_constants_equal_the_frozen_method():
    spec = ah.ANALYSIS_SPEC.model_dump(mode="json")
    assert spec["contrast"] == "JATS - PDF_DOCLING"
    assert spec["direction"] == "JATS_minus_PDF_DOCLING"
    assert spec["estimand"] == "item_weighted_paired_risk_difference"
    assert spec["primary_scope"] == PRIMARY_SELECTION
    assert (spec["pair_key"], spec["cluster_key"]) == ("item_id", "paper_id")
    assert spec["bootstrap_method"] == "whole_paper_percentile"
    assert (spec["bootstrap_resamples"], spec["interval_percent"]) == (10000, 95)
    assert spec["bootstrap_seed_name"] == "primary_bootstrap"
    assert (spec["model"], spec["variance"]) == ("intercept_only_paired_differences", "CR2")
    assert (spec["test"], spec["package"]) == ("satterthwaite_wald_two_sided", "clubSandwich")
    assert (spec["multiplicity"], spec["multiplicity_family"]) == (
        "holm",
        ["evidence_delivered", "answer_correct"],
    )
    assert spec["sensitivity"] == ["equal_paper_weighted", "best_worst_case_bounds"]
    assert "seed" not in " ".join(k for k in spec if k != "bootstrap_seed_name")


def test_outcome_rules_follow_the_row_vocabularies_not_an_assumed_yes_no():
    rules = {r.name: r for r in ah.ANALYSIS_SPEC.outcomes}
    assert list(rules) == ["evidence_delivered", "answer_correct"]
    # The authoritative literals, written out independently of the module under test.
    assert set(get_args(Delivered)) == {"yes", "no", "non_evaluable", "not_applicable"}
    assert set(get_args(Correct)) == {"yes", "no", "non_evaluable", "missing", "not_executed"}
    assert AnalysisRow.model_fields["evidence_delivered"].annotation is Delivered
    assert AnalysisRow.model_fields["answer_correct"].annotation is Correct
    for name, states in (("evidence_delivered", Delivered), ("answer_correct", Correct)):
        rule = rules[name]
        assert (rule.column, rule.success, rule.failure) == (name, "yes", "no")
        assert set(rule.non_evaluable) == set(get_args(states)) - {"yes", "no"}
    assert set(rules["evidence_delivered"].non_evaluable) == {"non_evaluable", "not_applicable"}
    assert set(rules["answer_correct"].non_evaluable) == {
        "non_evaluable",
        "missing",
        "not_executed",
    }


# ---- cross-language CSV contract ---------------------------------------------------------


def test_csv_contract_describes_the_published_bytes(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    contract = handoff.csv_contract
    data = (tmp_path / handoff.csv.logical_ref).read_bytes()
    header = data.decode("utf-8").split("\n", 1)[0].split(",")
    assert [c.name for c in contract.columns] == header == COLUMNS
    assert (contract.encoding, contract.line_ending, contract.delimiter) == ("utf-8", "\n", ",")
    assert (contract.quote, contract.byte_order_mark) == ('"', False)
    assert (contract.missing_token, contract.logical_tokens) == ("NA", ["TRUE", "FALSE"])
    assert contract.row_count == len(table.rows) == data.decode().count("\n") - 1
    assert data.endswith(b"\n") and b"\r" not in data
    assert handoff.csv.sha256 == sha256_bytes(data)
    expected = sha256_bytes(
        canonical_json_bytes(
            [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in contract.columns]
        )
    )
    assert contract.columns_sha256 == expected


def test_csv_column_types_and_nullability_match_the_row_model(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    by_name = {c.name: c for c in handoff.csv_contract.columns}
    assert by_name["repeat_index"].type == "integer" and not by_name["repeat_index"].nullable
    assert by_name["primary_row"].type == "logical"
    assert by_name["run_id"].nullable and by_name["stage_a_label"].nullable
    assert not by_name["item_id"].nullable
    assert by_name["evidence_delivered"].type == by_name["answer_correct"].type == "string"
    assert handoff.csv_contract.sort_key == [
        "matrix_arm",
        "item_id",
        "condition",
        "repeat_index",
        "observation_key",
    ]


def test_na_is_both_the_missing_token_and_a_value_of_one_column(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    assert handoff.csv_contract.na_value_columns == ["correct_without_evidence"]
    # The non-evaluable PDF row really carries the literal NA in a non-nullable column.
    pdf = next(r for r in table.rows if r.condition == "PDF_DOCLING")
    assert pdf.correct_without_evidence == "NA"
    assert not next(
        c for c in handoff.csv_contract.columns if c.name == "correct_without_evidence"
    ).nullable


# ---- accounting: independent oracle ------------------------------------------------------


def test_complete_cases_when_both_representations_are_evaluable(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    answer = outcome(handoff, "answer_correct")
    assert answer.complete_item_ids == [w.item.item_id]
    cell = entry(handoff, "answer_correct", w.item.item_id)
    assert (cell.jats.state, cell.pdf_docling.state) == ("yes", "yes")
    assert cell.complete_case and cell.paper_id == w.item.paper_id
    # PDF has no Stage A judgment here, so delivery is not evaluable for the pair.
    delivered = outcome(handoff, "evidence_delivered")
    assert delivered.complete_item_ids == []
    cell = entry(handoff, "evidence_delivered", w.item.item_id)
    assert (cell.jats.state, cell.pdf_docling.state) == ("yes", "non_evaluable")
    assert cell.pdf_docling.reason == "stage_a_not_assessed" and not cell.complete_case


def test_the_two_outcomes_are_accounted_separately(tmp_path):
    u = altered_world("unresolved")
    jats = observation(u, case=deliver(u, [u.chunk(P_B)]))  # delivery unresolved, answer scored
    _, _, handoff = rehearsal(tmp_path, pair(u, jats=jats))
    delivery = entry(handoff, "evidence_delivered", u.item.item_id)
    answer = entry(handoff, "answer_correct", u.item.item_id)
    assert (delivery.jats.state, delivery.jats.reason) == ("non_evaluable", "stage_a_unresolved")
    assert answer.jats.state == "yes" and answer.jats.reason is None
    assert answer.complete_case and not delivery.complete_case


@pytest.mark.parametrize(
    ("make", "state", "execution", "reason"),
    [
        (
            lambda w: failed_for(w.item, representation_or_condition="PDF_DOCLING"),
            "missing",
            "operational_failure",
            "operational_failure",
        ),
        (None, "not_executed", "planned_not_executed", None),
        (
            lambda w: result_for(w.item, answer="1 mg", representation_or_condition="PDF_DOCLING"),
            "non_evaluable",
            "completed",
            "unit_not_in_gold",
        ),
        (
            lambda w: result_for(w.item, answer="2", representation_or_condition="PDF_DOCLING"),
            "no",
            "completed",
            "value_differs_from_gold",
        ),
    ],
)
def test_every_answer_state_is_preserved_with_its_execution_status(
    tmp_path, make, state, execution, reason
):
    w = world()
    result = None if make is None else make(w)
    pdf = observation(w, evidence=None, condition="PDF_DOCLING", result=result)
    _, _, handoff = rehearsal(tmp_path, pair(w, pdf=pdf))
    cell = entry(handoff, "answer_correct", w.item.item_id).pdf_docling
    assert (cell.state, cell.execution_status) == (state, execution)
    assert cell.observation_key == pdf_key(handoff, w)
    if reason is not None:
        assert cell.reason is not None
    complete = state in {"yes", "no"}
    assert entry(handoff, "answer_correct", w.item.item_id).complete_case is complete
    counts = outcome(handoff, "answer_correct").state_counts
    assert counts[f"PDF_DOCLING|{state}"] == 1 and counts["JATS|yes"] == 1


def pdf_key(handoff, w):
    return entry(handoff, "answer_correct", w.item.item_id).pdf_docling.observation_key


def test_repeats_and_robustness_arms_never_change_primary_accounting(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    base = pair(w, case)
    _, _, plain = rehearsal(tmp_path / "plain", base)

    def pdf(repeat, result):
        return observation(
            w, evidence=None, condition="PDF_DOCLING", repeat_index=repeat, result=result
        )

    extra = [
        *base,
        observation(w, case=case, repeat_index=2, result=None),
        observation(w, case=case, repeat_index=3, result=None),
        pdf(
            2,
            failed_for(w.item, representation_or_condition="PDF_DOCLING", repeat_index=2),
        ),
        pdf(3, None),
        observation(w, evidence=None, matrix_arm="chunk_robustness", result=None),
        observation(
            w, evidence=None, condition="PDF_DOCLING", matrix_arm="chunk_robustness", result=None
        ),
        closed_book(w),
    ]
    _, _, noisy = rehearsal(tmp_path / "noisy", extra)
    assert noisy.accounting == plain.accounting


def test_a_representation_cell_missing_from_the_roster_is_not_collapsed(tmp_path):
    w = world()
    only_jats = [pair(w)[0]]
    _, _, handoff = rehearsal(tmp_path, only_jats)
    cell = entry(handoff, "answer_correct", w.item.item_id)
    assert cell.pdf_docling is None and cell.jats.state == "yes" and not cell.complete_case
    assert outcome(handoff, "answer_correct").state_counts["PDF_DOCLING|not_in_roster"] == 1


def test_non_included_roster_items_keep_identity_and_reason(tmp_path):
    w = world()
    obs = pair(w)
    extra = [
        RosterItem(item_id="q_x1", state="missing", reason="no generation record yet"),
        RosterItem(item_id="q_x2", state="unresolved", reason="Stage A under review"),
        RosterItem(item_id="q_x3", state="excluded", reason="item withdrawn from the split"),
    ]
    base = roster_of(obs)
    roster = base.model_copy(
        update={"items": sorted([*base.items, *extra], key=lambda i: i.item_id)}
    )
    _, _, handoff = rehearsal(tmp_path, obs, roster)
    assert handoff.non_included_items == extra
    for name in ("evidence_delivered", "answer_correct"):
        assert [e.item_id for e in outcome(handoff, name).eligible_items] == [w.item.item_id]


def test_eligible_set_lists_every_included_item_including_unresolved_and_missing(tmp_path):
    w = world()
    failed = observation(
        w,
        evidence=None,
        condition="PDF_DOCLING",
        result=failed_for(w.item, representation_or_condition="PDF_DOCLING"),
    )
    _, _, handoff = rehearsal(tmp_path, pair(w, pdf=failed))
    for name in ("evidence_delivered", "answer_correct"):
        listed = outcome(handoff, name)
        assert [e.item_id for e in listed.eligible_items] == [w.item.item_id]
        assert listed.complete_item_ids == []


def row_copy(row, item_id, paper_id, **update):
    """A row for another item/paper; accounting reads fields only."""
    return row.model_copy(
        update={
            "item_id": item_id,
            "paper_id": paper_id,
            "observation_key": f"{row.observation_key}-{item_id}",
            **update,
        }
    )


def test_counts_by_paper_and_independence_from_row_order(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    jats = next(r for r in table.rows if r.condition == "JATS")
    pdf = next(r for r in table.rows if r.condition == "PDF_DOCLING")
    good = {"evidence_delivered": "yes", "evidence_reason": None}
    rows = [
        jats,
        pdf.model_copy(update=good),  # complete for both outcomes
        row_copy(jats, "q_b", "paper_b"),
        row_copy(pdf, "q_b", "paper_b", **good),
        row_copy(jats, "q_c", "paper_b"),
        row_copy(pdf, "q_c", "paper_b", answer_correct="missing"),
    ]
    items = [
        RosterItem(item_id=i, state="included", paper_id=p, item_sha256="1" * 64)
        for i, p in ((w.item.item_id, w.item.paper_id), ("q_b", "paper_b"), ("q_c", "paper_b"))
    ]
    roster = handoff_roster(tmp_path, handoff).model_copy(update={"items": items})
    forward = ah.derive_accounting(rows, roster)
    assert forward == ah.derive_accounting(rows[::-1], roster)
    answer = next(o for o in forward if o.outcome == "answer_correct")
    assert answer.complete_item_ids == sorted([w.item.item_id, "q_b"])
    by_paper = {p.paper_id: (p.eligible, p.complete) for p in answer.papers}
    assert by_paper == {w.item.paper_id: (1, 1), "paper_b": (2, 1)}
    assert answer.state_counts["PDF_DOCLING|missing"] == 1


def test_an_included_item_without_a_paper_is_refused_by_a_checked_error(tmp_path):
    w = world()
    obs = pair(w)
    table, _, _ = rehearsal(tmp_path, obs)
    roster = roster_of(obs)
    unvalidated = roster.items[0].model_copy(update={"paper_id": None})  # skips model validation
    broken = roster.model_copy(update={"items": [unvalidated]})
    with pytest.raises(ah.HandoffError, match="paper"):
        ah.derive_accounting(table.rows, broken)


def handoff_roster(root, handoff):
    return PlannedRoster.model_validate_json((root / handoff.roster.logical_ref).read_bytes())


def test_accounting_agrees_with_the_published_primary_pair_definition(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    paired = set(primary_pairs(table.rows, "development"))
    for name in ("evidence_delivered", "answer_correct"):
        both = {
            e.item_id
            for e in outcome(handoff, name).eligible_items
            if e.jats is not None and e.pdf_docling is not None
        }
        assert both == paired


# ---- construction, publication and read-back (synthetic rehearsal) -----------------------


def test_handoff_pins_the_published_table_and_names_its_origin(tmp_path):
    w = world()
    table, build, handoff = rehearsal(tmp_path, pair(w))
    manifest = read_manifest(tmp_path / build.logical_ref, build.sha256)
    outputs = {o.artifact_kind: o.artifact for o in manifest.outputs}
    assert handoff.table_build == build
    assert handoff.csv == outputs["analysis_table"]
    assert handoff.inventory == outputs["analysis_inventory"]
    assert handoff.roster == outputs["planned_roster"]
    assert handoff.input_origin == "synthetic_rehearsal" and handoff.source_runs == []
    assert handoff.phase == "development" and handoff.scope == "partial_development"
    assert [(a.matrix_arm, a.resolved_config_sha256) for a in handoff.arms] == [
        ("confirmatory", CFG)
    ]
    inventory = AnalysisInventory.model_validate_json(
        (tmp_path / handoff.inventory.logical_ref).read_bytes()
    )
    assert handoff.scoring_reference == inventory.scoring_reference
    assert (handoff.benchmark_version, handoff.corpus_version) == (
        table.rows[0].benchmark_version,
        table.rows[0].corpus_version,
    )


def test_serialized_handoff_has_no_local_paths_or_unobserved_execution_values(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    text = json.dumps(handoff.model_dump(mode="json"))
    assert str(tmp_path) not in text and "/home/" not in text
    names = set(ah.AnalysisHandoff.model_fields)
    assert not names & {"r_version", "r_lock", "package_inventory", "outputs", "bootstrap_seed"}
    assert "seed" not in ah.ANALYSIS_SPEC.model_dump(mode="json").keys() - {"bootstrap_seed_name"}
    assert handoff.awaiting_approval == ["primary_bootstrap_seed"]
    assert "r_version" in handoff.execution_provenance_required
    assert "output_hashes" in handoff.execution_provenance_required
    assert "r_version" not in handoff.awaiting_approval


def test_publication_is_immutable_deterministic_and_has_a_valid_build_manifest(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    ref, build = ah.publish_handoff(handoff, tmp_path)
    bytes_before = (tmp_path / ref.logical_ref).read_bytes()
    manifest = read_manifest(tmp_path / build.logical_ref, build.sha256)
    assert manifest.status == "complete" and [o.artifact for o in manifest.outputs] == [ref]
    assert {i.role for i in manifest.inputs} == {"analysis_table_build"}
    assert sha256_bytes(bytes_before) == ref.sha256
    with pytest.raises(FileExistsError):
        ah.publish_handoff(handoff, tmp_path)
    assert (tmp_path / ref.logical_ref).read_bytes() == bytes_before


def test_the_same_inputs_publish_the_same_bytes(tmp_path):
    w = world()
    obs = pair(w)
    *_, first = rehearsal(tmp_path / "a", obs)
    *_, second = rehearsal(tmp_path / "b", obs)
    ra, _ = ah.publish_handoff(first, tmp_path / "a")
    rb, _ = ah.publish_handoff(second, tmp_path / "b")
    assert ra == rb


def test_read_back_verifies_and_a_rehearsal_has_no_scientific_authority(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    _, build = ah.publish_handoff(handoff, tmp_path)
    verified = ah.read_handoff(build, tmp_path)
    assert verified.handoff == handoff and verified.rows == table.rows
    assert verified.official is False


def test_rehearsal_and_official_inputs_cannot_be_mixed(tmp_path):
    w = world()
    _, build = publish(tmp_path, pair(w))
    with pytest.raises(ah.HandoffError, match="exactly one"):
        ah.build_handoff(build, tmp_path, code=CODE)
    with pytest.raises(ah.HandoffError, match="exactly one"):
        ah.build_handoff(build, tmp_path, code=CODE, rehearsal=True, official_runs=[object()])  # type: ignore[list-item]


def test_a_final_test_handoff_is_refused_by_the_model(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    data = handoff.model_dump(mode="json") | {"phase": "final_test"}
    with pytest.raises(ValueError):
        ah.AnalysisHandoff.model_validate(data)


# ---- fail-closed verification --------------------------------------------------------------


def stored(tmp_path, handoff):
    """Publish without the pre-publication check, to exercise read-back on a bad record."""
    return ah._store(handoff, tmp_path)[1]


def test_tampered_handoff_bytes_are_detected(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    ref, build = ah.publish_handoff(handoff, tmp_path)
    path = tmp_path / ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run"))
    with pytest.raises(HashMismatchError):
        ah.read_handoff(build, tmp_path)


def test_tampered_table_bytes_are_detected(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    _, build = ah.publish_handoff(handoff, tmp_path)
    path = tmp_path / handoff.csv.logical_ref
    path.write_bytes(path.read_bytes().replace(b"JATS", b"JATX", 1))
    with pytest.raises(ValueError):
        ah.read_handoff(build, tmp_path)


def test_tampered_build_manifest_is_detected(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    _, build = ah.publish_handoff(handoff, tmp_path)
    path = tmp_path / build.logical_ref
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(HashMismatchError):
        ah.read_handoff(build, tmp_path)


@pytest.mark.parametrize("field", ["csv", "inventory", "roster"])
def test_a_reference_that_differs_from_the_table_build_is_refused(tmp_path, field):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    other = getattr(handoff, field).model_copy(update={"sha256": "0" * 64})
    bad = handoff.model_copy(update={field: other})
    with pytest.raises(ah.HandoffError):
        ah.read_handoff(stored(tmp_path, bad), tmp_path)


def test_a_swapped_column_hash_is_refused(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    contract = handoff.csv_contract.model_copy(update={"columns_sha256": "0" * 64})
    with pytest.raises(ah.HandoffError):
        ah.read_handoff(
            stored(tmp_path, handoff.model_copy(update={"csv_contract": contract})), tmp_path
        )


def test_accounting_that_differs_from_the_rows_is_refused(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    stale = [
        o.model_copy(update={"complete_item_ids": [w.item.item_id, "q_ghost"]})
        for o in handoff.accounting
    ]
    with pytest.raises(ah.HandoffError, match="accounting"):
        ah.read_handoff(
            stored(tmp_path, handoff.model_copy(update={"accounting": stale})), tmp_path
        )


def test_wrong_or_missing_arm_configuration_is_refused(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    wrong = [handoff.arms[0].model_copy(update={"resolved_config_sha256": "e" * 64})]
    with pytest.raises(ah.HandoffError, match="arm"):
        ah.read_handoff(stored(tmp_path, handoff.model_copy(update={"arms": wrong})), tmp_path)
    with pytest.raises(ValueError):
        ah.AnalysisHandoff.model_validate(handoff.model_dump(mode="json") | {"arms": []})


def test_publication_refuses_a_handoff_that_cannot_be_verified(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    bad = handoff.model_copy(update={"benchmark_version": "benchmark_v9"})
    with pytest.raises(ah.HandoffError):
        ah.publish_handoff(bad, tmp_path)
    assert not (tmp_path / "analysis" / "handoff").exists() or not list(
        (tmp_path / "analysis" / "handoff").iterdir()
    )


def test_origin_must_agree_with_source_runs(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    data = handoff.model_dump(mode="json") | {"input_origin": "verified_official_run"}
    with pytest.raises(ValueError, match="source run"):
        ah.AnalysisHandoff.model_validate(data)


def test_a_rehearsal_record_cannot_be_read_back_as_official(tmp_path):
    w = world()
    _, _, handoff = rehearsal(tmp_path, pair(w))
    _, build = ah.publish_handoff(handoff, tmp_path)
    with pytest.raises(ah.HandoffError, match="rehearsal"):
        ah.read_handoff(build, tmp_path, require_official=True)


# ---- official origin (scripted official run, no model) --------------------------------------


def official_table(tmp_path, view, tiny, **options):
    world_ = build_world(tmp_path / "run", view, tiny, **options)
    verified = verify(world_, anchored(world_))
    items = read_development_view(*view)[1]
    scope = make_scope(*view, {"q_a", "q_b"}, conditions=("JATS", "PDF_DOCLING"))
    observations = official_observations(verified, items, lambda item, condition: None)
    table = build_table(observations, scope)
    root = tmp_path / "analysis_root"
    root.mkdir()
    _csv, _inventory, build = publish_table(table, root)
    return verified, root, build


@needs_tokenizer
def test_official_handoff_records_and_rechecks_the_source_run(tmp_path, view, tiny, recognised):
    verified, root, build = official_table(tmp_path, view, tiny)
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=[verified])
    assert handoff.input_origin == "verified_official_run"
    [run] = handoff.source_runs
    completion = verified.completed.completion
    assert run.run_id == verified.run_id and run.completion.sha256 == verified.manifest_sha256
    assert run.resolved_config_sha256 == completion.resolved_config_sha256
    assert run.matrix_arm == completion.matrix_arm
    assert [a.matrix_arm for a in handoff.arms] == [completion.matrix_arm]
    _, built = ah.publish_handoff(handoff, root, official_runs=[verified])
    assert "run_completion" in {
        i.role for i in read_manifest(root / built.logical_ref, built.sha256).inputs
    }
    result = ah.read_handoff(built, root, official_runs=[verified])
    assert result.official is True


@needs_tokenizer
def test_official_read_back_without_fresh_verified_runs_fails_closed(
    tmp_path, view, tiny, recognised
):
    verified, root, build = official_table(tmp_path, view, tiny)
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=[verified])
    _, built = ah.publish_handoff(handoff, root, official_runs=[verified])
    with pytest.raises(ah.OfficialProvenanceNotVerified):
        ah.read_handoff(built, root)
    with pytest.raises(ah.OfficialProvenanceNotVerified):
        ah.publish_handoff(handoff.model_copy(update={"code": CODE}), root)


@needs_tokenizer
def test_official_handoff_needs_the_run_the_rows_cite(tmp_path, view, tiny, recognised):
    verified, root, build = official_table(tmp_path, view, tiny)
    other = dataclasses.replace(verified, run_id="some_other_run")
    with pytest.raises(ah.HandoffError, match="run"):
        ah.build_handoff(build, root, code=CODE, official_runs=[other])


@needs_tokenizer
@pytest.mark.parametrize(
    "change",
    [
        {"resolved_config_sha256": "f" * 64},
        {"matrix_arm": "chunk_robustness"},
    ],
)
def test_official_handoff_checks_configuration_and_arm(tmp_path, view, tiny, recognised, change):
    verified, root, build = official_table(tmp_path, view, tiny)
    completion = verified.completed.completion.model_copy(update=change)
    altered = dataclasses.replace(
        verified, completed=dataclasses.replace(verified.completed, completion=completion)
    )
    with pytest.raises(ah.HandoffError):
        ah.build_handoff(build, root, code=CODE, official_runs=[altered])


@needs_tokenizer
def test_official_completed_observation_missing_from_the_table_is_refused(
    tmp_path, view, tiny, recognised
):
    verified, root, build = official_table(tmp_path, view, tiny)
    extra = verified.results[0].model_copy(update={"observation_key": "obs_not_in_table"})
    altered = dataclasses.replace(verified, results=[*verified.results, extra])
    with pytest.raises(ah.HandoffError, match="absent from the analysis table"):
        ah.build_handoff(build, root, code=CODE, official_runs=[altered])


@needs_tokenizer
def test_official_result_that_differs_from_the_table_is_refused(tmp_path, view, tiny, recognised):
    verified, root, build = official_table(tmp_path, view, tiny)
    first = verified.results[0].model_copy(update={"attempt_id": "attempt_other"})
    altered = dataclasses.replace(verified, results=[first, *verified.results[1:]])
    with pytest.raises(ah.HandoffError):
        ah.build_handoff(build, root, code=CODE, official_runs=[altered])


@needs_tokenizer
def test_official_origin_cannot_be_claimed_for_a_synthetic_table(tmp_path, view, tiny, recognised):
    verified, _, _ = official_table(tmp_path, view, tiny)
    w = world()
    _, build = publish(tmp_path / "synthetic_root", pair(w))
    with pytest.raises(ah.HandoffError):
        ah.build_handoff(build, tmp_path / "synthetic_root", code=CODE, official_runs=[verified])


def test_serialization_of_csv_is_unchanged_by_the_handoff(tmp_path):
    w = world()
    table, _, handoff = rehearsal(tmp_path, pair(w))
    assert (tmp_path / handoff.csv.logical_ref).read_bytes() == serialize_csv(table.rows)
