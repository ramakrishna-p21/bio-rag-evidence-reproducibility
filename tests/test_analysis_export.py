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
from biorag_repro.evaluation.analysis_models import (
    COLUMNS,
    FROZEN_MATRIX,
    AnalysisRow,
    GeneratorInput,
    PlannedRoster,
    RosterArm,
    RosterItem,
)
from biorag_repro.evaluation.analysis_storage import publish_table, read_table
from biorag_repro.evaluation.analysis_table import (
    Evidence,
    Observation,
    Table,
    build_row,
    build_table,
    parse_csv,
    primary_pairs,
    serialize_csv,
)
from biorag_repro.evaluation.stage_a import final_test_scope
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
    if fields["matrix_arm"] != "closed_book":
        fields.setdefault("retrieval_config_sha256", CFG)
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


def _roster(cells, source, scope="partial_development", **changes):
    """Smallest roster covering exactly `cells` {arm: (conditions, repeats)} and `items`."""
    arms = [
        RosterArm(matrix_arm=a, conditions=list(c), repeats=list(r))
        for a, (c, r) in FROZEN_MATRIX.items()
        if a in cells
        for c, r in [cells[a]]
    ]
    fields = {
        "scope": scope,
        "phase": "development",
        "benchmark_version": source[0].benchmark_version,
        "corpus_version": source[0].corpus_version,
        "declared_by": "Rama Krishna Pudota",
        "basis": "synthetic test scope",
        "arms": arms,
        "items": sorted(
            (
                RosterItem(
                    item_id=i.item_id,
                    state="included",
                    paper_id=i.paper_id,
                    item_sha256=stage_c.item_digest(i),
                )
                for i in source
            ),
            key=lambda x: x.item_id,
        ),
    }
    return PlannedRoster(**(fields | changes))


def roster_of(obs):
    """Partial roster that declares exactly the arms, conditions and repeats observed."""
    cells = {}
    for o in obs:
        conditions, repeats = cells.setdefault(o.matrix_arm, (set(), set()))
        conditions.add(o.condition)
        repeats.add(o.repeat_index)
    order = {a: list(FROZEN_MATRIX[a][0]) for a in FROZEN_MATRIX}
    cells = {a: ([c for c in order[a] if c in cs], sorted(rs)) for a, (cs, rs) in cells.items()}
    return _roster(cells, list({o.item.item_id: o.item for o in obs}.values()))


def tbl(obs):
    return build_table(obs, roster_of(obs))


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


def test_incomplete_retrospective_provenance_keeps_independent_answer_score():
    w = world()
    r = row(observation(w, evidence=None, retrospective_provenance_incomplete=True))
    assert r.evidence_delivered == "non_evaluable"
    assert r.evidence_reason == "retrospective_provenance_incomplete"
    assert r.stage_a_label is None and r.stage_b_status is None
    assert r.answer_correct == "yes"


def test_retrospective_exception_cannot_enter_final_test_rows():
    w = world()
    with pytest.raises(ValueError, match="final-test rows"):
        row(
            observation(
                w, evidence=None, phase="final_test", retrospective_provenance_incomplete=True
            )
        )


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


def test_full_configuration_is_distinct_from_the_retrieval_configuration_it_contains():
    """D6: the observation's resolved configuration is not the Phase 5 retrieval hash."""
    w = world()
    full = "d" * 64
    r = row(observation(w, resolved_config_sha256=full))  # retrieval stays the Phase 5 hash (CFG)
    assert (r.config_sha256, r.retrieval_config_sha256) == (full, CFG)
    assert r.observation_key != row(observation(w)).observation_key  # full identity is in the key
    both = serialize_csv([r]).decode()
    assert full in both and CFG in both


def test_distinct_context_configuration_retains_ranked_retrieval_identity():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    case.ctx = case.ctx.model_copy(
        update={
            "configuration": case.ctx.configuration.model_copy(update={"resolved_sha256": "c" * 64})
        }
    )
    result = row(observation(w, case=case))
    assert result.retrieval_config_sha256 == case.ranked.configuration.resolved_sha256
    assert case.ctx.configuration.resolved_sha256 != result.retrieval_config_sha256


def test_stage_b_must_come_from_the_retrieval_configuration_the_observation_names():
    w = world()
    with pytest.raises(ValueError, match="ranked retrieval configuration"):
        build_row(observation(w, retrieval_config_sha256="e" * 64))


def test_stage_b_context_identity_must_match_the_actual_context():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    evidence = evidence_for(w, case)
    wrong = evidence.stage_b.model_copy(
        update={
            "configuration": evidence.stage_b.configuration.model_copy(
                update={"resolved_sha256": "c" * 64}
            )
        }
    )
    with pytest.raises(ValueError):
        build_row(
            observation(
                w, evidence=Evidence(w.document, CANONICAL, w.stage_a, wrong, case.inputs())
            )
        )


def test_stage_b_context_reference_must_match_its_inputs():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    evidence = evidence_for(w, case)
    with pytest.raises(ValueError):
        build_row(
            observation(
                w,
                evidence=Evidence(
                    w.document,
                    CANONICAL,
                    w.stage_a,
                    evidence.stage_b,
                    case.inputs(context_ref=ref("other_context")),
                ),
            )
        )


def test_retrieval_configuration_is_required_for_retrieval_and_absent_for_closed_book():
    w = world()
    with pytest.raises(ValueError, match="retrieval configuration"):
        build_row(observation(w, retrieval_config_sha256=None))
    with pytest.raises(ValueError, match="retrieval configuration"):
        build_row(closed_book(w, retrieval_config_sha256=CFG))
    assert row(closed_book(w)).retrieval_config_sha256 is None


def test_one_arm_runs_under_one_retrieval_configuration():
    from biorag_repro.evaluation.analysis_table import check_rows

    w = world()
    first = row(observation(w, case=deliver(w, [w.chunk(P_A)])))
    other = first.model_copy(
        update={"repeat_index": 2, "primary_row": False, "observation_key": "k2"}
        | {"retrieval_config_sha256": "e" * 64}
    )
    with pytest.raises(ValueError, match="retrieval configuration"):
        check_rows([first, other])


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


def test_final_test_rows_are_built_only_inside_the_final_test_evaluation():
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
    with pytest.raises(ValueError, match="authorized final-test evaluation"):
        build_row(
            Observation(item=item, result=closed, score=stage_c.score(item, closed), **gate_only)
        )
    with final_test_scope():
        row, _ = build_row(
            Observation(item=item, result=closed, score=stage_c.score(item, closed), **gate_only)
        )
        assert (row.phase, row.split, row.analysis_role) == (
            "final_test",
            "final_test",
            "secondary",
        )
        with pytest.raises(ValueError):  # a development observation of a final-test item
            build_row(
                Observation(item=item, result=result, score=stage_c.score(item, result), **PLAN)
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
    table = tbl(three_repeats(w, case) + [closed_book(w)])
    assert len(table.rows) == 4
    assert sum(r.primary_row for r in table.rows if r.condition == "JATS") == 1
    assert sum(r.primary_row for r in table.rows) == 2  # JATS repeat 1 and closed-book repeat 1


def test_table_order_is_deterministic_whatever_the_input_order():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    obs = three_repeats(w, case) + [closed_book(w)]
    first = serialize_csv(tbl(obs).rows)
    for seed in range(5):
        shuffled = obs[:]
        random.Random(seed).shuffle(shuffled)
        assert serialize_csv(tbl(shuffled).rows) == first


def test_duplicate_observation_is_rejected():
    w = world()
    o = observation(w)
    with pytest.raises(ValueError):
        tbl([o, o])


def test_same_coordinates_under_another_configuration_are_a_duplicate_condition():
    w = world()
    other = observation(w, resolved_config_sha256="d" * 64, evidence=None)
    with pytest.raises(ValueError):
        tbl([observation(w), other])


def test_mixed_phases_are_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    with pytest.raises(ValueError):
        tbl(
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
        tbl([first, second])


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
    table = tbl([jats, pdf])
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
        observation(w, evidence=None, condition="PDF_DOCLING", repeat_index=2, result=None),
        closed_book(w),
    ]
    return tbl(obs)


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
    second = publish_table(
        Table(list(reversed(table.rows)), table.roster, table.inputs), tmp_path / "b"
    )
    assert first == second  # row order is canonicalized
    third = publish_table(tbl([observation(world())]), tmp_path / "c")
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
    inv.write_bytes(original.replace(b'"row_count": 5', b'"row_count": 6'))
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
        publish_table(Table(table.rows, table.roster, table.inputs[:-1]), tmp_path / "missing")
    extra = table.inputs + [
        table.inputs[0].model_copy(update={"role": "stage_c_score", "artifact": ref("not_cited")})
    ]
    with pytest.raises(ValueError):
        publish_table(Table(table.rows, table.roster, extra), tmp_path / "extra")
    forged = table.rows[0].model_copy(update={"stage_c_artifact": "drv_" + "9" * 64})
    with pytest.raises(ValueError):
        publish_table(
            Table([forged, *table.rows[1:]], table.roster, table.inputs), tmp_path / "forged"
        )


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
        observation(w, evidence=None, condition="PDF_DOCLING", repeat_index=2, result=None),
        observation(w, evidence=None, condition="PDF_DOCLING", repeat_index=3, result=None),
    ]
    table = tbl(obs)
    _, inventory_ref, build_ref = publish_table(table, tmp_path)
    from biorag_repro.core.storage import read_project_json
    from biorag_repro.evaluation.analysis_models import AnalysisInventory

    inv = read_project_json(
        tmp_path / inventory_ref.logical_ref, inventory_ref.sha256, AnalysisInventory
    )
    assert inv.row_count == 6
    by_state = {(e.condition, e.repeat_index): e for e in inv.not_evaluable}
    assert by_state[("JATS", 2)].answer_correct == "non_evaluable"
    assert by_state[("JATS", 3)].answer_correct == "missing"
    assert by_state[("PDF_DOCLING", 1)].answer_correct == "not_executed"
    assert ("JATS", 1) not in by_state
    assert inv.primary_repeat_index == 1
    assert [c.name for c in inv.columns] == COLUMNS
    assert {c.name: c.type for c in inv.columns}["repeat_index"] == "integer"
    assert {c.name: c.type for c in inv.columns}["primary_row"] == "logical"
    assert sum(inv.counts.values()) == 6
    assert read_table(build_ref, tmp_path) == table.rows


# ---- schemas and verification command -----------------------------------------------------


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

    def miscounted(*args):
        return honest(*args).model_copy(update={"row_count": 99})

    monkeypatch.setattr(analysis_storage, "make_inventory", miscounted)
    with pytest.raises(ValueError, match="inventory"):
        read_table(build_ref, tmp_path)


def test_entity_inside_a_sentence_is_exported_as_unresolved_with_its_reason():
    """An approved name inside longer text awaits adjudication; the table says why."""
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
        "non_evaluable",
        "approved_name_in_longer_text",
        "deterministic",
    )
    assert "approved_name_in_longer_text" in serialize_csv([r]).decode()


# ---- planned-observation roster (completeness and exclusions) ------------------------------


def planned(w, condition="PDF_DOCLING", repeat_index=1, arm="confirmatory", **plan):
    """A planned observation that has not been executed."""
    return observation(
        w,
        evidence=None,
        result=None,
        condition=condition,
        repeat_index=repeat_index,
        matrix_arm=arm,
        **plan,
    )


def full_pair(w, case, repeats=(1,)):
    obs = []
    for n in repeats:
        obs.append(
            observation(
                w,
                case=case,
                repeat_index=n,
                result=result_for(w.item, repeat_index=n, attempt_id=f"a{n}"),
            )
        )
        obs.append(planned(w, repeat_index=n))
    return obs


def test_one_jats_row_cannot_stand_for_the_planned_experiment():
    """The review's counterexample: a lone JATS repeat-1 row used to be published."""
    w = world()
    lone = observation(w, case=deliver(w, [w.chunk(P_A)]))
    pair_scope = _roster({"confirmatory": (["JATS", "PDF_DOCLING"], [1])}, [w.item])
    with pytest.raises(ValueError, match="without a row"):
        build_table([lone], pair_scope)  # PDF_DOCLING counterpart missing
    full = _roster({"confirmatory": (["JATS", "PDF_DOCLING"], [1, 2, 3])}, [w.item])
    with pytest.raises(ValueError, match="without a row"):
        build_table(full_pair(w, deliver(w, [w.chunk(P_A)])), full)  # repeats 2-3 omitted
    every_arm = _roster({a: (list(c), list(r)) for a, (c, r) in FROZEN_MATRIX.items()}, [w.item])
    with pytest.raises(ValueError, match="without a row"):
        build_table([lone], every_arm)


def test_declared_scope_with_every_planned_row_builds_without_inventing_attempts(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    obs = full_pair(w, case)
    table = build_table(obs, _roster({"confirmatory": (["JATS", "PDF_DOCLING"], [1])}, [w.item]))
    pdf = next(r for r in table.rows if r.condition == "PDF_DOCLING")
    assert (pdf.execution_status, pdf.run_id, pdf.attempt_id) == (
        "planned_not_executed",
        None,
        None,
    )
    assert (pdf.answer_correct, pdf.stage_c_status) == ("not_executed", None)
    _, inventory_ref, build_ref = publish_table(table, tmp_path)
    assert read_table(build_ref, tmp_path) == table.rows
    inv = _inventory(tmp_path, inventory_ref)
    assert (inv.scope, inv.planned_observations, inv.executed_observations) == (
        "partial_development",
        2,
        1,
    )


def test_unplanned_extra_and_wrong_identity_rows_are_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    jats_only = _roster({"confirmatory": (["JATS"], [1])}, [w.item])
    with pytest.raises(ValueError, match="outside the roster"):
        build_table(full_pair(w, case), jats_only)  # PDF row was never planned
    with pytest.raises(ValueError, match="outside the roster"):
        build_table([observation(w, case=case), closed_book(w)], jats_only)
    foreign = _roster({"confirmatory": (["JATS"], [1])}, [w.item]).model_copy(
        update={"phase": "qualification"}
    )
    with pytest.raises(ValueError, match="phase"):
        build_table([observation(w, case=case)], foreign)
    tampered = jats_only.model_copy(
        update={"items": [jats_only.items[0].model_copy(update={"item_sha256": "0" * 64})]}
    )
    with pytest.raises(ValueError, match="paper or item hash"):
        build_table([observation(w, case=case)], tampered)


def test_missing_item_and_duplicate_coordinate_are_rejected():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    scope = _roster({"confirmatory": (["JATS"], [1])}, [w.item])
    other = RosterItem(item_id="q_999", state="included", paper_id="paper_9", item_sha256="1" * 64)
    two_items = scope.model_copy(
        update={"items": sorted([*scope.items, other], key=lambda i: i.item_id)}
    )
    with pytest.raises(ValueError, match="without a row"):
        build_table([observation(w, case=case)], two_items)
    o = observation(w, case=case)
    with pytest.raises(ValueError, match="duplicate"):
        build_table([o, o], scope)


def test_exclusions_are_accounted_with_reasons():
    base = {"item_id": "q_900", "state": "excluded"}
    with pytest.raises(ValidationError):
        RosterItem(**base)  # no reason
    with pytest.raises(ValidationError):
        RosterItem(**base, reason="  ")
    with pytest.raises(ValidationError):
        RosterItem(**base, reason="withdrawn", paper_id="p1")
    with pytest.raises(ValidationError):
        RosterItem(item_id="q_900", state="included", paper_id="p1")  # no item hash
    w = world()
    scope = _roster({"confirmatory": (["JATS"], [1])}, [w.item])
    with pytest.raises(ValidationError, match="unique"):
        PlannedRoster.model_validate(
            scope.model_dump() | {"items": [*scope.model_dump()["items"]] * 2}
        )
    excluded = RosterItem(item_id="q_000", state="excluded", reason="item withdrawn in review")
    with_excluded = PlannedRoster(
        **(
            scope.model_dump()
            | {"items": [excluded.model_dump(), *[i.model_dump() for i in scope.items]]}
        )
    )
    # an excluded item contributes no planned observation, and a row for it is undeclared
    assert {c[1] for c in with_excluded.coordinates()} == {w.item.item_id}


def test_complete_matrix_claim_requires_the_whole_matrix_and_no_open_items():
    w = world()
    every_arm = {a: (list(c), list(r)) for a, (c, r) in FROZEN_MATRIX.items()}
    assert _roster(every_arm, [w.item], scope="complete_matrix").scope == "complete_matrix"
    for narrower in (
        {**every_arm, "confirmatory": (["JATS", "PDF_DOCLING"], [1])},
        {k: v for k, v in every_arm.items() if k != "closed_book"},
        {**every_arm, "confirmatory": (["JATS"], [1, 2, 3])},
    ):
        with pytest.raises(ValidationError, match="complete_matrix"):
            _roster(narrower, [w.item], scope="complete_matrix")
    open_item = RosterItem(item_id="q_999", state="unresolved", reason="gold under review")
    items = [i.model_dump() for i in _roster(every_arm, [w.item]).items]
    with pytest.raises(ValidationError, match="partial"):
        _roster(
            every_arm,
            [w.item],
            scope="complete_matrix",
            items=[RosterItem(**i) for i in items] + [open_item],
        )
    # the same open item is fine in a partial scope and is carried to the inventory
    partial = _roster(every_arm, [w.item], items=[RosterItem(**i) for i in items] + [open_item])
    assert partial.scope == "partial_development"
    assert {c[1] for c in partial.coordinates()} == {w.item.item_id}


@pytest.mark.parametrize(
    "arms",
    [
        {"confirmatory": (["PDF_DOCLING", "JATS"], [1])},  # not in matrix order
        {"confirmatory": (["JATS"], [1, 1])},
        {"confirmatory": (["JATS"], [4])},
        {"chunk_robustness": (["JATS"], [1, 2])},
        {"closed_book": (["JATS"], [1])},
    ],
)
def test_roster_arms_stay_inside_the_frozen_matrix(arms):
    with pytest.raises(ValidationError):
        _roster(arms, [world().item])


def _inventory(root, ref):
    from biorag_repro.core.storage import read_project_json
    from biorag_repro.evaluation.analysis_models import AnalysisInventory

    return read_project_json(root / ref.logical_ref, ref.sha256, AnalysisInventory)


def test_inventory_records_item_states_scope_and_roster_reference(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    scope = _roster({"confirmatory": (["JATS"], [1])}, [w.item])
    extra = [
        RosterItem(item_id="q_001a", state="missing", reason="no PDF counterpart ingested"),
        RosterItem(item_id="q_001b", state="unresolved", reason="gold adjudication pending"),
        RosterItem(item_id="q_001c", state="excluded", reason="split changed after review"),
    ]
    scope = scope.model_copy(
        update={"items": sorted([*scope.items, *extra], key=lambda i: i.item_id)}
    )
    scope = PlannedRoster.model_validate(scope.model_dump())
    _, inventory_ref, _ = publish_table(build_table([observation(w, case=case)], scope), tmp_path)
    inv = _inventory(tmp_path, inventory_ref)
    assert {(i.item_id, i.state, i.reason) for i in inv.items if i.state != "included"} == {
        ("q_001a", "missing", "no PDF counterpart ingested"),
        ("q_001b", "unresolved", "gold adjudication pending"),
        ("q_001c", "excluded", "split changed after review"),
    }
    assert [i.item_id for i in inv.items if i.state == "included"] == [w.item.item_id]
    assert inv.scope == "partial_development"
    assert (tmp_path / inv.roster.logical_ref).is_file() and inv.roster.sha256


def test_tampered_roster_is_detected_on_read_back(tmp_path):
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    scope = _roster({"confirmatory": (["JATS", "PDF_DOCLING"], [1])}, [w.item])
    table = build_table(full_pair(w, case), scope)
    _, inventory_ref, build_ref = publish_table(table, tmp_path)
    inv = _inventory(tmp_path, inventory_ref)
    path = tmp_path / inv.roster.logical_ref
    original = path.read_bytes()
    path.write_bytes(original.replace(b'"PDF_DOCLING"', b'"JATS"', 1))
    with pytest.raises(HashMismatchError):
        read_table(build_ref, tmp_path)
    path.write_bytes(original)
    assert read_table(build_ref, tmp_path) == table.rows


def test_published_rows_must_match_the_roster_on_read_back(tmp_path):
    """Dropping a planned row from a published table is caught by the roster, not only by hashes."""
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    scope = _roster({"confirmatory": (["JATS", "PDF_DOCLING"], [1])}, [w.item])
    table = build_table(full_pair(w, case), scope)
    smaller = Table(table.rows[:1], scope, table.inputs)
    with pytest.raises(ValueError, match="without a row"):
        publish_table(smaller, tmp_path)
    assert not list(tmp_path.rglob("*.csv"))


def test_publication_with_a_roster_never_replaces_an_existing_table(tmp_path):
    table = sample_table()
    publish_table(table, tmp_path)
    with pytest.raises(FileExistsError):
        publish_table(table, tmp_path)


# ---- primary comparison selection (R handoff) ----------------------------------------------


def mixed_arm_table():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    obs = [
        *full_pair(w, case, repeats=(1, 2)),
        observation(w, case=case, matrix_arm="chunk_robustness"),
        planned(w, arm="chunk_robustness"),
        closed_book(w),
    ]
    return tbl(obs)


def test_primary_row_alone_is_not_the_primary_comparison():
    table = mixed_arm_table()
    flagged = {(r.matrix_arm, r.condition) for r in table.rows if r.primary_row}
    assert flagged == {
        ("confirmatory", "JATS"),
        ("confirmatory", "PDF_DOCLING"),
        ("chunk_robustness", "JATS"),
        ("chunk_robustness", "PDF_DOCLING"),
        ("closed_book", "CLOSED_BOOK"),
    }


def test_primary_pairs_select_only_confirmatory_repeat_one_pairs():
    table = mixed_arm_table()
    pairs = primary_pairs(table.rows, "development")
    assert list(pairs) == [world().item.item_id]
    jats, pdf = pairs[world().item.item_id]
    assert (jats.matrix_arm, jats.condition, jats.repeat_index) == ("confirmatory", "JATS", 1)
    assert (pdf.matrix_arm, pdf.condition, pdf.repeat_index) == ("confirmatory", "PDF_DOCLING", 1)
    assert jats.paper_id == pdf.paper_id and jats.item_sha256 == pdf.item_sha256
    # development rows are never confirmatory results
    assert primary_pairs(table.rows, "confirmatory") == {}
    assert primary_pairs(table.rows, "secondary") == {}


def test_an_unpaired_representation_is_not_a_primary_pair():
    w = world()
    case = deliver(w, [w.chunk(P_A)])
    table = tbl([observation(w, case=case), closed_book(w)])
    assert primary_pairs(table.rows, "development") == {}


def test_inventory_states_the_primary_selection_rule(tmp_path):
    from biorag_repro.evaluation.analysis_models import PRIMARY_SELECTION

    _, inventory_ref, _ = publish_table(mixed_arm_table(), tmp_path)
    inv = _inventory(tmp_path, inventory_ref)
    assert inv.primary_selection == PRIMARY_SELECTION
    assert "matrix_arm = confirmatory" in inv.primary_selection
    assert "not primary" in inv.primary_selection
    assert "primary_row alone" in inv.pairing
