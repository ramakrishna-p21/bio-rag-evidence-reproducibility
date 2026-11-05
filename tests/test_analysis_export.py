"""Item-level analysis table: row derivation, provenance checks, CSV and publication."""

import csv
import io
import random

import pytest
from pydantic import ValidationError
from test_chunking import build
from test_retrieval_models import ref
from test_stage_a import CANONICAL, PROSE_COMPONENTS, judgment, locus, make_item
from test_stage_b import P_A, P_B, P_C, altered_world, deliver, prose_world, prose_xml
from test_stage_c import failed_result, make_result, rekey

from biorag_repro.core.hashing import HashMismatchError, sha256_bytes
from biorag_repro.evaluation import stage_b, stage_c
from biorag_repro.evaluation.analysis_models import COLUMNS, AnalysisRow, GeneratorInput
from biorag_repro.evaluation.analysis_storage import publish_table, read_table
from biorag_repro.evaluation.analysis_table import (
    Evidence,
    Observation,
    Table,
    build_row,
    build_table,
    parse_csv,
    serialize_csv,
)
from biorag_repro.retrieval.storage import record_ref

CFG = "b" * 64  # resolved configuration hash used by the Stage B fixtures


def world():
    return prose_world(P_A, P_B, P_C)


COORDS = {
    "phase": "development",
    "matrix_arm": "confirmatory",
    "representation_or_condition": "JATS",
    "repeat_index": 1,
    "resolved_config_sha256": CFG,
}


def result_for(item, *, answer="1", **changes):
    return rekey(make_result(item), answer=answer, **(COORDS | changes))


def failed_for(item, **changes):
    return rekey(failed_result(item), **(COORDS | changes))


def evidence_for(w, case):
    judged = stage_b.deterministic_judgment(case.evaluate())
    return Evidence(w.document, CANONICAL, w.stage_a, judged, case.inputs())


def link(result, case, **changes):
    values = {
        "observation_key": result.observation_key,
        "attempt_id": result.attempt_id,
        "context": record_ref(case.ctx),
        "exact_context_sha256": sha256_bytes(case.ctx.exact_context.encode("utf-8")),
        "input": ref("generator_input"),
        "provenance": "simulated",
    }
    return GeneratorInput(**(values | changes))


def observation(w, *, case=None, result=..., evidence=..., gin=None, **plan):
    """One planned observation; by default executed, scored and with delivered evidence."""
    fields = {
        "phase": "development",
        "matrix_arm": "confirmatory",
        "condition": "JATS",
        "repeat_index": 1,
        "resolved_config_sha256": CFG,
    } | plan
    if case is None and evidence is ...:
        case = deliver(w, [w.chunk(P_A)])
    if result is ...:
        result = result_for(
            w.item,
            phase=fields["phase"],
            matrix_arm=fields["matrix_arm"],
            representation_or_condition=fields["condition"],
            repeat_index=fields["repeat_index"],
            resolved_config_sha256=fields["resolved_config_sha256"],
        )
    if evidence is ...:
        evidence = evidence_for(w, case)
    score = None if result is None else stage_c.score(w.item, result)
    return Observation(
        item=w.item, result=result, score=score, evidence=evidence, generator_input=gin, **fields
    )


PLAN = {
    "phase": "development",
    "matrix_arm": "confirmatory",
    "condition": "JATS",
    "repeat_index": 1,
    "resolved_config_sha256": CFG,
}


def row(obs):
    return build_row(obs)[0]


def closed_book(w, **plan):
    fields = {"matrix_arm": "closed_book", "condition": "CLOSED_BOOK"} | plan
    result = result_for(
        w.item,
        matrix_arm="closed_book",
        representation_or_condition="CLOSED_BOOK",
        **{k: v for k, v in plan.items() if k == "repeat_index"},
    )
    return observation(w, result=result, evidence=None, **fields)


# ---- earliest-failure attribution through real Stage A/B/C records ------------------------


def test_delivered_and_correct_has_no_failure():
    w = world()
    r = row(observation(w))
    assert (r.evidence_delivered, r.answer_correct, r.earliest_failure) == ("yes", "yes", "none")
    assert r.stage_a_label == "preserved" and r.stage_b_status == "delivered"
    assert r.correct_without_evidence == "FALSE" and r.context_provenance == "assembled_only"
    assert r.execution_status == "completed" and r.completion_kind == "answer"


def test_incorrect_despite_sufficient_evidence_is_a_stage_c_failure():
    w = world()
    r = row(observation(w, result=result_for(w.item, answer="2")))
    assert (r.evidence_delivered, r.answer_correct, r.earliest_failure) == ("yes", "no", "stage_c")


def test_correct_answer_without_delivered_evidence_stays_visible():
    w = world()
    case = deliver(w, [w.chunk(P_B)])
    r = row(observation(w, case=case))
    assert (r.evidence_delivered, r.answer_correct) == ("no", "yes")
    assert (r.earliest_failure, r.correct_without_evidence) == ("stage_b", "TRUE")
    assert r.stage_b_status == "not_delivered"


def test_incorrect_without_delivered_evidence_is_attributed_to_stage_b_not_c():
    w = world()
    case = deliver(w, [w.chunk(P_B)])
    r = row(observation(w, case=case, result=result_for(w.item, answer="2")))
    assert (r.answer_correct, r.earliest_failure, r.correct_without_evidence) == (
        "no",
        "stage_b",
        "FALSE",
    )


@pytest.mark.parametrize(
    "label,extra", [("absent", {}), ("scientifically_corrupted", {"alteration": "x"})]
)
def test_stage_a_loss_is_not_a_retrieval_failure_and_needs_no_b_record(label, extra):
    w = altered_world(label, **extra)
    r = row(
        observation(
            w,
            result=result_for(w.item, answer="1"),
            evidence=Evidence(w.document, CANONICAL, w.stage_a),
        )
    )
    assert (r.evidence_delivered, r.earliest_failure) == ("no", "stage_a")
    assert r.stage_b_status == "not_reachable_stage_a" and r.stage_b_artifact is None
    assert r.answer_correct == "yes" and r.correct_without_evidence == "TRUE"


def test_stage_a_loss_with_its_stage_b_record_is_accepted_and_cited():
    w = altered_world("absent")
    case = deliver(w, [w.chunk(P_B)])
    r = row(observation(w, case=case))
    assert r.stage_b_status == "not_reachable_stage_a" and r.stage_b_artifact is not None
    assert r.earliest_failure == "stage_a"


def test_stage_a_unresolved_keeps_delivery_non_evaluable_and_scores_the_answer():
    w = altered_world("unresolved")
    case = deliver(w, [w.chunk(P_B)])
    r = row(observation(w, case=case, result=result_for(w.item, answer="2")))
    assert (r.evidence_delivered, r.evidence_reason) == ("non_evaluable", "stage_a_unresolved")
    assert (r.answer_correct, r.earliest_failure, r.correct_without_evidence) == (
        "no",
        "unresolved",
        "NA",
    )


def test_stage_c_unresolved_keeps_stage_b_evaluable():
    w = world()
    r = row(observation(w, result=result_for(w.item, answer="1 mg")))  # unit absent from gold
    assert r.stage_c_status == "non_evaluable" and r.stage_c_reason == "unit_not_in_gold"
    assert (r.evidence_delivered, r.answer_correct, r.earliest_failure) == (
        "yes",
        "non_evaluable",
        "unresolved",
    )


def test_evidence_not_yet_assessed_is_neither_failure_nor_unresolved():
    w = world()
    r = row(observation(w, evidence=None))
    assert (r.evidence_delivered, r.evidence_reason) == ("non_evaluable", "stage_a_not_assessed")
    assert r.earliest_failure == "incomplete" and r.stage_a_label is None


def test_stage_a_only_is_not_assessed_at_stage_b():
    w = world()
    ev = Evidence(w.document, CANONICAL, w.stage_a)
    r = row(observation(w, evidence=ev))
    assert r.evidence_reason == "stage_b_not_assessed" and r.earliest_failure == "incomplete"


# ---- planned, attempted, completed -----------------------------------------------------------


def test_completed_abstention_is_incorrect_and_not_an_operational_failure():
    w = world()
    raw = '{"abstain": true}'
    abstained = result_for(
        w.item,
        raw_output=raw,
        output=make_result(w.item).output.model_copy(update={"sha256": sha256_bytes(raw.encode())}),
        answer=None,
        abstained=True,
    )
    r = row(observation(w, result=abstained))
    assert (r.execution_status, r.completion_kind) == ("completed", "abstained")
    assert (r.answer_correct, r.stage_c_reason) == ("no", "abstained")


def test_operational_failure_with_attempt_is_missing_not_incorrect():
    w = world()
    failed = failed_for(w.item)
    r = row(observation(w, result=failed))
    assert (r.execution_status, r.answer_correct) == ("operational_failure", "missing")
    assert r.attempt_id == failed.attempt_id and r.completion_kind is None
    assert r.evidence_delivered == "yes"  # evidence outcomes do not depend on the answer


def test_planned_but_unexecuted_observation_is_not_a_failure_and_has_no_attempt():
    w = world()
    r = row(observation(w, result=None))
    assert r.execution_status == "planned_not_executed"
    assert r.answer_correct == "not_executed" and r.stage_c_status is None
    assert r.run_id is None and r.attempt_id is None and r.stage_c_artifact is None
    assert r.earliest_failure == "incomplete"


def test_a_score_cannot_exist_without_the_execution_record():
    w = world()
    complete = observation(w)
    with pytest.raises(ValueError):
        build_row(
            Observation(
                item=w.item,
                phase="development",
                matrix_arm="confirmatory",
                condition="JATS",
                repeat_index=1,
                resolved_config_sha256=CFG,
                result=None,
                score=complete.score,
                evidence=complete.evidence,
            )
        )


def test_an_executed_observation_needs_its_score():
    w = world()
    o = observation(w)
    with pytest.raises(ValueError):
        build_row(Observation(**{**o.__dict__, "score": None}))


# ---- identity: wrong or counterfeit inputs are rejected, never reconciled -------------------


@pytest.mark.parametrize(
    "change",
    [
        {"item_id": "q_999"},
        {"paper_id": "P002"},
        {"benchmark_version": "benchmark_v2"},
        {"repeat_index": 2},
        {"representation_or_condition": "PDF_DOCLING"},
        {"matrix_arm": "chunk_robustness"},
        {"phase": "qualification"},
        {"resolved_config_sha256": "d" * 64},
    ],
)
def test_result_for_other_coordinates_is_rejected(change):
    w = world()
    o = observation(w)
    with pytest.raises(ValueError):
        other = rekey(result_for(w.item), **change)
        build_row(
            Observation(**{**o.__dict__, "result": other, "score": stage_c.score(w.item, other)})
        )


def test_forged_observation_key_is_rejected():
    w = world()
    good = result_for(w.item)
    forged = good.model_copy(update={"observation_key": "obs_" + "0" * 64})
    o = observation(w)
    with pytest.raises(ValueError):
        build_row(Observation(**{**o.__dict__, "result": forged, "score": o.score}))


def test_score_of_another_attempt_or_result_is_rejected():
    w = world()
    o = observation(w)
    other = result_for(w.item, answer="2", attempt_id="attempt2")
    with pytest.raises(ValueError):
        build_row(Observation(**{**o.__dict__, "score": stage_c.score(w.item, other)}))


def test_score_not_reproduced_from_the_result_is_rejected():
    w = world()
    o = observation(w, result=result_for(w.item, answer="2"))
    forged = o.score.model_copy(update={"status": "correct", "reason": None})
    with pytest.raises(ValueError):
        build_row(Observation(**{**o.__dict__, "score": forged}))


def test_stage_b_judgment_for_other_inputs_is_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    lost_case = deliver(w, [w.chunk(P_B)])
    wrong = stage_b.deterministic_judgment(lost_case.evaluate())
    ev = Evidence(w.document, CANONICAL, w.stage_a, wrong, case.inputs())
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, evidence=ev))


def test_forged_stage_b_status_is_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_B)])
    honest = stage_b.deterministic_judgment(case.evaluate())
    forged = honest.model_copy(update={"status": "delivered"})
    ev = Evidence(w.document, CANONICAL, w.stage_a, forged, case.inputs())
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, evidence=ev))


def test_stage_a_judgment_for_another_item_is_rejected():
    w = world()
    other = judgment(w.item, w.document, "absent")  # refused: the fact is located verbatim
    with pytest.raises(ValueError):
        build_row(observation(w, evidence=Evidence(w.document, CANONICAL, other)))


def test_evidence_for_another_representation_is_rejected():
    w = world()
    with pytest.raises(ValueError):
        build_row(observation(w, condition="PDF_DOCLING"))  # JATS evidence under a PDF arm


def test_stage_b_configuration_must_be_the_observations():
    w = world()
    with pytest.raises(ValueError):
        build_row(observation(w, resolved_config_sha256="d" * 64))


@pytest.mark.parametrize("arm", ["dense_robustness", "scope_robustness"])
def test_arm_must_match_the_retrieval_that_was_run(arm):
    w = world()  # BM25 over the designated paper
    with pytest.raises(ValueError):
        build_row(observation(w, matrix_arm=arm))


# ---- generator-input provenance ------------------------------------------------------------


def test_simulated_generator_input_is_marked_and_cited():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    result = result_for(w.item)
    r = row(observation(w, case=case, result=result, gin=link(result, case)))
    assert r.context_provenance == "generator_input_simulated"
    assert r.generator_input_artifact == "generator_input"


def test_recorded_generator_input_is_distinct_from_simulation():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    result = result_for(w.item)
    r = row(observation(w, case=case, result=result, gin=link(result, case, provenance="recorded")))
    assert r.context_provenance == "generator_input_recorded"


@pytest.mark.parametrize(
    "change",
    [
        {"observation_key": "obs_" + "1" * 64},
        {"attempt_id": "other"},
        {"exact_context_sha256": "f" * 64},
        {"context": ref("another_context")},
    ],
)
def test_generator_input_must_match_the_attempt_and_exact_context(change):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    result = result_for(w.item)
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, result=result, gin=link(result, case, **change)))


def test_context_changed_after_generator_input_is_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    other = deliver(w, [w.chunk(P_A), w.chunk(P_B)])
    result = result_for(w.item)
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, result=result, gin=link(result, other)))


def test_unexecuted_or_closed_book_observation_has_no_generator_input():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    result = result_for(w.item)
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, result=None, gin=link(result, case)))
    cb = closed_book(w)
    with pytest.raises(ValueError):
        build_row(Observation(**{**cb.__dict__, "generator_input": link(result, case)}))


def test_final_test_rows_are_not_built_before_the_readiness_gate():
    w = world()
    with pytest.raises(ValueError):
        build_row(observation(w, phase="final_test"))
    document, _, _, _, _ = build(prose_xml(P_A, P_B, P_C), counter=lambda s: len(s.split()) * 11)
    item = make_item(document, PROSE_COMPONENTS, split="final_test")
    result = result_for(item)
    with pytest.raises(ValueError):
        build_row(Observation(item=item, result=result, score=stage_c.score(item, result), **PLAN))
    closed = result_for(
        item,
        phase="final_test",
        matrix_arm="closed_book",
        representation_or_condition="CLOSED_BOOK",
    )
    gate_only = {
        **PLAN,
        "phase": "final_test",
        "matrix_arm": "closed_book",
        "condition": "CLOSED_BOOK",
    }
    with pytest.raises(ValueError, match="readiness gate"):
        build_row(
            Observation(item=item, result=closed, score=stage_c.score(item, closed), **gate_only)
        )


def test_row_model_requires_the_recorded_generator_input_for_final_test_execution():
    base = row(observation(world())).model_dump()
    final = base | {"phase": "final_test", "split": "final_test", "analysis_role": "confirmatory"}
    with pytest.raises(ValueError):
        AnalysisRow.model_validate(final)  # assembled-only context
    linked = final | {
        "context_provenance": "generator_input_simulated",
        "generator_input_artifact": "generator_input",
    }
    with pytest.raises(ValueError):
        AnalysisRow.model_validate(linked)  # simulated is not real delivery
    recorded = linked | {"context_provenance": "generator_input_recorded"}
    assert AnalysisRow.model_validate(recorded).analysis_role == "confirmatory"
    mislabelled = recorded | {"analysis_role": "development"}
    with pytest.raises(ValueError):
        AnalysisRow.model_validate(mislabelled)


def test_development_rows_are_never_labelled_confirmatory():
    w = world()
    assert row(observation(w)).analysis_role == "development"
    assert row(observation(w, phase="qualification")).analysis_role == "development"


# ---- closed book, repeats -------------------------------------------------------------------


def test_closed_book_has_no_stage_a_or_b():
    w = world()
    r = row(closed_book(w))
    assert (r.condition, r.matrix_arm, r.analysis_role) == (
        "CLOSED_BOOK",
        "closed_book",
        "development",
    )
    assert (r.evidence_delivered, r.earliest_failure, r.correct_without_evidence) == (
        "not_applicable",
        "not_applicable",
        "NA",
    )
    assert r.stage_a_label is None and r.stage_b_status is None and r.context_artifact is None
    assert r.context_provenance == "not_applicable" and r.answer_correct == "yes"


def test_closed_book_refuses_evidence_and_the_arm_condition_pairing_is_fixed():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    with pytest.raises(ValueError):
        build_row(observation(w, case=case, matrix_arm="closed_book", condition="CLOSED_BOOK"))
    with pytest.raises(ValueError):
        build_row(observation(w, matrix_arm="closed_book", condition="JATS", evidence=None))
    with pytest.raises(ValueError):
        build_row(observation(w, matrix_arm="confirmatory", condition="CLOSED_BOOK", evidence=None))


def test_only_repeat_one_is_primary():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    flags = [row(observation(w, case=case, repeat_index=n)).primary_row for n in (1, 2, 3)]
    assert flags == [True, False, False]


@pytest.mark.parametrize(
    "repeat,arm", [(0, "confirmatory"), (4, "confirmatory"), (2, "chunk_robustness")]
)
def test_repeat_range_follows_the_matrix(repeat, arm):
    w = world()
    with pytest.raises(ValueError):
        build_row(observation(w, repeat_index=repeat, matrix_arm=arm))


# ---- table-level rules --------------------------------------------------------------------


def three_repeats(w, case):
    return [
        observation(
            w,
            case=case,
            repeat_index=n,
            result=result_for(w.item, repeat_index=n, attempt_id=f"a{n}"),
        )
        for n in (1, 2, 3)
    ]


def test_repeats_two_and_three_do_not_inflate_the_primary_rows():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    table = build_table(three_repeats(w, case) + [closed_book(w)])
    assert len(table.rows) == 4
    assert sum(r.primary_row for r in table.rows if r.condition == "JATS") == 1
    assert sum(r.primary_row for r in table.rows) == 2  # JATS repeat 1 and closed-book repeat 1


def test_table_order_is_deterministic_whatever_the_input_order():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    obs = three_repeats(w, case) + [closed_book(w)]
    first = serialize_csv(build_table(obs).rows)
    for seed in range(5):
        shuffled = obs[:]
        random.Random(seed).shuffle(shuffled)
        assert serialize_csv(build_table(shuffled).rows) == first


def test_duplicate_observation_is_rejected():
    w = world()
    o = observation(w)
    with pytest.raises(ValueError):
        build_table([o, o])


def test_same_coordinates_under_another_configuration_are_a_duplicate_condition():
    w = world()
    other = observation(w, resolved_config_sha256="d" * 64, evidence=None)
    with pytest.raises(ValueError):
        build_table([observation(w), other])


def test_mixed_phases_are_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    with pytest.raises(ValueError):
        build_table(
            [
                observation(w, case=case),
                observation(w, case=case, phase="qualification", repeat_index=2),
            ]
        )


def test_repeats_of_one_condition_share_one_evidence_outcome():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    first = observation(w, case=case)
    second = observation(
        w, evidence=None, repeat_index=2, result=result_for(w.item, repeat_index=2, attempt_id="a2")
    )
    with pytest.raises(ValueError):
        build_table([first, second])


def test_pairing_columns_identify_item_and_paper():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    jats = observation(w, case=case)
    pdf = observation(
        w,
        evidence=None,
        condition="PDF_DOCLING",
        result=result_for(w.item, representation_or_condition="PDF_DOCLING"),
    )
    table = build_table([jats, pdf])
    assert {(r.item_id, r.paper_id, r.item_sha256) for r in table.rows} == {
        (w.item.item_id, w.item.paper_id, jats.score.item_sha256)
    }
    assert [r.condition for r in table.rows] == ["JATS", "PDF_DOCLING"]


def test_two_co_primary_outcomes_are_evaluable_independently():
    w = world()
    lost = deliver(w, [w.chunk(P_B)])
    unresolved = row(observation(w, result=result_for(w.item, answer="1 mg")))
    evaluable_delivery_only = unresolved
    assert evaluable_delivery_only.evidence_delivered == "yes"
    assert evaluable_delivery_only.answer_correct == "non_evaluable"
    failed = failed_for(w.item)
    missing = row(observation(w, case=lost, result=failed))
    assert (missing.evidence_delivered, missing.answer_correct) == ("no", "missing")
    u = altered_world("unresolved")
    answer_only = row(observation(u, case=deliver(u, [u.chunk(P_B)])))
    assert (answer_only.evidence_delivered, answer_only.answer_correct) == ("non_evaluable", "yes")


# ---- CSV ---------------------------------------------------------------------------------


def sample_table():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    obs = [
        observation(w, case=case),
        observation(
            w, case=case, repeat_index=2, result=result_for(w.item, repeat_index=2, attempt_id="a2")
        ),
        observation(w, evidence=None, condition="PDF_DOCLING", result=None),
        closed_book(w),
    ]
    return build_table(obs)


def test_csv_layout_types_and_missing_values():
    table = sample_table()
    data = serialize_csv(table.rows)
    assert data.decode("utf-8").splitlines()[0] == ",".join(COLUMNS)
    assert not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data
    records = list(csv.DictReader(io.StringIO(data.decode("utf-8"))))
    assert len(records) == len(table.rows)
    jats = next(r for r in records if r["condition"] == "JATS" and r["repeat_index"] == "1")
    assert jats["primary_row"] == "TRUE" and jats["repeat_index"] == "1"
    planned = next(r for r in records if r["execution_status"] == "planned_not_executed")
    assert planned["run_id"] == "NA" and planned["stage_c_artifact"] == "NA"
    assert planned["answer_correct"] == "not_executed"
    cb = next(r for r in records if r["condition"] == "CLOSED_BOOK")
    assert cb["stage_a_label"] == "NA" and cb["evidence_delivered"] == "not_applicable"
    assert next(r for r in records if r["repeat_index"] == "2")["primary_row"] == "FALSE"


def test_csv_round_trip_is_exact():
    table = sample_table()
    data = serialize_csv(table.rows)
    assert parse_csv(data) == table.rows
    assert serialize_csv(parse_csv(data)) == data


def edit_csv(data, column, old, new, row_index=0):
    lines = data.decode("utf-8").splitlines()
    column_index = lines[0].split(",").index(column)
    hits = [i for i, line in enumerate(lines[1:], 1) if line.split(",")[column_index] == old]
    cells = lines[hits[row_index]].split(",")
    cells[column_index] = new
    lines[hits[row_index]] = ",".join(cells)
    return ("\n".join(lines) + "\n").encode("utf-8")


@pytest.mark.parametrize(
    "column,old,new",
    [
        ("earliest_failure", "none", "stage_c"),
        ("answer_correct", "yes", "no"),
        ("evidence_delivered", "yes", "no"),
        ("primary_row", "TRUE", "FALSE"),
        ("primary_row", "TRUE", "true"),
        ("repeat_index", "1", "01"),
        ("stage_c_status", "correct", "incorrect"),
        ("correct_without_evidence", "FALSE", "NA"),
        ("context_provenance", "assembled_only", "generator_input_recorded"),
    ],
)
def test_edited_csv_cells_fail_re_derivation_or_canonical_form(column, old, new):
    data = serialize_csv(sample_table().rows)
    with pytest.raises((ValueError, ValidationError)):
        parse_csv(edit_csv(data, column, old, new))


def test_csv_must_be_in_canonical_form():
    data = serialize_csv(sample_table().rows)
    lines = data.decode("utf-8").splitlines()
    cases = [
        data.replace(b"\n", b"\r\n"),
        b"\xef\xbb\xbf" + data,
        data.replace(COLUMNS[0].encode(), b"x_" + COLUMNS[0].encode(), 1),
        ("\n".join([lines[0] + ",extra"] + lines[1:]) + "\n").encode(),
        ("\n".join([lines[0], lines[2], lines[1]] + lines[3:]) + "\n").encode(),  # unsorted
        ("\n".join(lines + [lines[1]]) + "\n").encode(),  # duplicate row
        ("\n".join(lines[:-1]) + "\n").encode()[:-1],  # no trailing newline
        data + b"\n",
    ]
    for bad in cases:
        with pytest.raises((ValueError, ValidationError)):
            parse_csv(bad)


# ---- publication and read-back ---------------------------------------------------------------


def test_publish_and_verified_read_back(tmp_path):
    table = sample_table()
    csv_ref, inventory_ref, build_ref = publish_table(table, tmp_path)
    assert read_table(build_ref, tmp_path) == table.rows
    assert sha256_bytes((tmp_path / csv_ref.logical_ref).read_bytes()) == csv_ref.sha256
    assert inventory_ref.logical_ref != csv_ref.logical_ref


def test_publication_never_replaces_an_existing_table(tmp_path):
    table = sample_table()
    publish_table(table, tmp_path)
    with pytest.raises(FileExistsError):
        publish_table(table, tmp_path)


def test_identity_follows_the_rows_and_their_sources(tmp_path):
    table = sample_table()
    first = publish_table(table, tmp_path / "a")
    second = publish_table(Table(list(reversed(table.rows)), table.inputs), tmp_path / "b")
    assert first == second  # row order is canonicalized
    changed_rows = table.rows[:-1]
    third = publish_table(Table(changed_rows, table.inputs), tmp_path / "c")
    assert third[0].artifact_id != first[0].artifact_id


def test_tampered_csv_is_detected(tmp_path):
    csv_ref, _, build_ref = publish_table(sample_table(), tmp_path)
    path = tmp_path / csv_ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b"TRUE", b"FALSE", 1))
    with pytest.raises(HashMismatchError):
        read_table(build_ref, tmp_path)


def test_tampered_inventory_or_manifest_is_detected(tmp_path):
    _, inventory_ref, build_ref = publish_table(sample_table(), tmp_path)
    inv = tmp_path / inventory_ref.logical_ref
    original = inv.read_bytes()
    inv.write_bytes(original.replace(b'"row_count": 4', b'"row_count": 5'))
    with pytest.raises(HashMismatchError):
        read_table(build_ref, tmp_path)
    inv.write_bytes(original)
    manifest = tmp_path / build_ref.logical_ref
    manifest.write_bytes(manifest.read_bytes().replace(b"stage_c_score", b"stage_c_scoreX", 1))
    with pytest.raises(HashMismatchError):
        read_table(build_ref, tmp_path)


def test_rows_must_cite_exactly_the_published_inputs(tmp_path):
    table = sample_table()
    with pytest.raises(ValueError):
        publish_table(Table(table.rows, table.inputs[:-1]), tmp_path / "missing")
    extra = table.inputs + [
        table.inputs[0].model_copy(update={"role": "stage_c_score", "artifact": ref("not_cited")})
    ]
    with pytest.raises(ValueError):
        publish_table(Table(table.rows, extra), tmp_path / "extra")
    forged = table.rows[0].model_copy(update={"stage_c_artifact": "drv_" + "9" * 64})
    with pytest.raises(ValueError):
        publish_table(Table([forged, *table.rows[1:]], table.inputs), tmp_path / "forged")


def test_inventory_lists_included_missing_and_unresolved_observations(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    failed = failed_for(w.item, repeat_index=3, attempt_id="a3")
    obs = [
        observation(w, case=case),
        observation(
            w,
            case=case,
            repeat_index=2,
            result=result_for(w.item, answer="1 mg", repeat_index=2, attempt_id="a2"),
        ),
        observation(w, case=case, repeat_index=3, result=failed),
        observation(w, evidence=None, condition="PDF_DOCLING", result=None),
    ]
    table = build_table(obs)
    _, inventory_ref, build_ref = publish_table(table, tmp_path)
    from biorag_repro.core.storage import read_project_json
    from biorag_repro.evaluation.analysis_models import AnalysisInventory

    inv = read_project_json(
        tmp_path / inventory_ref.logical_ref, inventory_ref.sha256, AnalysisInventory
    )
    assert inv.row_count == 4
    by_state = {(e.condition, e.repeat_index): e for e in inv.not_evaluable}
    assert by_state[("JATS", 2)].answer_correct == "non_evaluable"
    assert by_state[("JATS", 3)].answer_correct == "missing"
    assert by_state[("PDF_DOCLING", 1)].answer_correct == "not_executed"
    assert ("JATS", 1) not in by_state
    assert inv.primary_repeat_index == 1
    assert [c.name for c in inv.columns] == COLUMNS
    assert {c.name: c.type for c in inv.columns}["repeat_index"] == "integer"
    assert {c.name: c.type for c in inv.columns}["primary_row"] == "logical"
    assert sum(inv.counts.values()) == 4
    assert read_table(build_ref, tmp_path) == table.rows


# ---- schemas and verification command -----------------------------------------------------


def test_analysis_schemas_are_registered_and_current():
    from biorag_repro.evaluation.analysis_models import AnalysisInventory
    from scripts.export_schemas import SCHEMA_DIR, SCHEMAS, schema_bytes

    for name, model in {
        "analysis_row.schema.json": AnalysisRow,
        "analysis_inventory.schema.json": AnalysisInventory,
        "generator_input.schema.json": GeneratorInput,
    }.items():
        assert SCHEMAS[name] is model
        assert (SCHEMA_DIR / name).read_bytes() == schema_bytes(model)


def test_verify_command_accepts_a_published_table_and_rejects_tampering(tmp_path, capsys):
    from biorag_repro.evaluation.analysis_cli import main

    csv_ref, _, build_ref = publish_table(sample_table(), tmp_path)
    args = [
        "verify",
        "--root",
        str(tmp_path),
        "--build",
        build_ref.logical_ref,
        "--sha256",
        build_ref.sha256,
    ]
    assert main(args) == 0
    assert csv_ref.sha256 in capsys.readouterr().out
    path = tmp_path / csv_ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b"TRUE", b"FALSE", 1))
    assert main(args) == 1


def test_stage_a_cited_by_the_row_must_be_the_one_stage_b_was_judged_from():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    other = judgment(w.item, w.document, "preserved", sets=[locus(w.document, PROSE_COMPONENTS)])
    assert other != w.stage_a
    ev = Evidence(w.document, CANONICAL, other, evidence_for(w, case).stage_b, case.inputs())
    with pytest.raises(ValueError, match="other item, document or Stage A"):
        build_row(observation(w, case=case, evidence=ev))


def test_read_back_checks_the_inventory_against_the_rows(tmp_path, monkeypatch):
    from biorag_repro.evaluation import analysis_storage

    _, _, build_ref = publish_table(sample_table(), tmp_path)
    honest = analysis_storage.make_inventory

    def miscounted(rows, csv_ref):
        return honest(rows, csv_ref).model_copy(update={"row_count": 99})

    monkeypatch.setattr(analysis_storage, "make_inventory", miscounted)
    with pytest.raises(ValueError, match="inventory"):
        read_table(build_ref, tmp_path)


def test_known_stage_c_limits_stay_visible_in_the_exported_reason():
    """An entity inside a sentence scores incorrect (Pass 3 limit); the table says why."""
    from test_stage_c import entity
    from test_stage_c import make_item as stage_c_item

    item = stage_c_item(entity("TP53"))
    result = result_for(
        item,
        answer="The gene is TP53",
        matrix_arm="closed_book",
        representation_or_condition="CLOSED_BOOK",
    )
    plan = {**PLAN, "matrix_arm": "closed_book", "condition": "CLOSED_BOOK"}
    r = row(Observation(item=item, result=result, score=stage_c.score(item, result), **plan))
    assert (r.answer_correct, r.stage_c_reason, r.stage_c_basis) == (
        "no",
        "entity_not_approved",
        "deterministic",
    )
    assert "entity_not_approved" in serialize_csv([r]).decode()
