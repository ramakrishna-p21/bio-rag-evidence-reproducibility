# ruff: noqa: F811
"""Analysis handoff over several verified official-shaped development runs.

The runs are the scripted official runs of test_generation_official: simulated approvals and a
scripted backend, no model. They stand in for the official verification path; nothing here is a
scientific observation, and no mock result is promoted to one (every run goes through
verify_official_run before it can back a row).
"""

import dataclasses

import pytest
from test_analysis_handoff import CODE
from test_generation_official import (  # noqa: F401  (fixtures)
    anchored,
    build_world,
    recognised,
    verify,
    view,
)
from test_generation_prompt import needs_tokenizer
from test_generation_roster import make_scope

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_models import RosterArm
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import build_table
from biorag_repro.generation.handoff import official_observations

ARMS = ("confirmatory", "chunk_robustness")
BOTH = ["JATS", "PDF_DOCLING"]


def two_run_table(tmp_path, view, tiny):
    """Two independently verified runs in different arms, combined into one published table."""
    verified = []
    for arm in ARMS:
        world = build_world(tmp_path / arm, view, tiny, matrix_arm=arm)
        verified.append(verify(world, anchored(world)))
    items = read_development_view(*view)[1]
    scope = make_scope(
        *view,
        {"q_a", "q_b"},
        arms=[RosterArm(matrix_arm=arm, conditions=BOTH, repeats=[1]) for arm in ARMS],
    )
    observations = [
        o
        for run in verified
        for o in official_observations(run, items, lambda item, condition: None)
    ]
    root = tmp_path / "analysis_root"
    root.mkdir()
    _csv, _inventory, build = publish_table(build_table(observations, scope), root)
    return verified, root, build


@needs_tokenizer
def test_two_verified_runs_in_different_arms_publish_and_read_back_as_official(
    tmp_path, view, tiny, recognised
):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    assert len({run.run_id for run in verified}) == 2
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=verified)
    assert handoff.input_origin == "verified_official_run"
    assert [r.run_id for r in handoff.source_runs] == sorted(run.run_id for run in verified)
    by_run = {r.run_id: r for r in handoff.source_runs}
    for run in verified:
        completion = run.completed.completion
        source = by_run[run.run_id]
        assert (source.matrix_arm, source.completion.sha256) == (
            completion.matrix_arm,
            run.manifest_sha256,
        )
        assert source.observations == len(completion.observations) == 4
    assert {a.matrix_arm: a.resolved_config_sha256 for a in handoff.arms} == {
        run.completed.completion.matrix_arm: run.completed.completion.resolved_config_sha256
        for run in verified
    }
    assert len({a.resolved_config_sha256 for a in handoff.arms}) == 2

    # Only the confirmatory run's repeat-1 rows are primary; the other arm is never counted.
    primary_run = next(r for r in verified if r.completed.completion.matrix_arm == "confirmatory")
    for outcome in handoff.accounting:
        keys = {
            cell.observation_key
            for item in outcome.eligible_items
            for cell in (item.jats, item.pdf_docling)
            if cell is not None
        }
        assert keys == {r.observation_key for r in primary_run.results}

    _, built = ah.publish_handoff(handoff, root, official_runs=verified)
    inputs = read_manifest(root / built.logical_ref, built.sha256).inputs
    assert sorted(i.artifact.sha256 for i in inputs if i.role == "run_completion") == sorted(
        run.manifest_sha256 for run in verified
    )
    result = ah.read_handoff(built, root, official_runs=verified, require_official=True)
    assert result.official is True and result.handoff == handoff


@needs_tokenizer
def test_missing_run_evidence_is_refused(tmp_path, view, tiny, recognised):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    with pytest.raises(ah.HandoffError, match="differ from the supplied runs"):
        ah.build_handoff(build, root, code=CODE, official_runs=verified[:1])
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=verified)
    _, built = ah.publish_handoff(handoff, root, official_runs=verified)
    with pytest.raises(ah.HandoffError):
        ah.read_handoff(built, root, official_runs=verified[1:])
    with pytest.raises(ah.OfficialProvenanceNotVerified):
        ah.read_handoff(built, root)


@needs_tokenizer
def test_duplicate_supplied_runs_are_refused(tmp_path, view, tiny, recognised):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    with pytest.raises(ah.HandoffError, match="repeated"):
        ah.build_handoff(build, root, code=CODE, official_runs=[*verified, verified[0]])


@needs_tokenizer
def test_swapped_run_ids_are_refused(tmp_path, view, tiny, recognised):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    first, second = verified
    swapped = [
        dataclasses.replace(first, run_id=second.run_id),
        dataclasses.replace(second, run_id=first.run_id),
    ]
    with pytest.raises(ah.HandoffError):
        ah.build_handoff(build, root, code=CODE, official_runs=swapped)


@needs_tokenizer
def test_each_run_is_checked_against_its_own_results(tmp_path, view, tiny, recognised):
    """Results of one run cannot stand in for the other's, nor can a changed attempt."""
    verified, root, build = two_run_table(tmp_path, view, tiny)
    first, second = verified
    crossed = [
        dataclasses.replace(first, results=second.results),
        dataclasses.replace(second, results=first.results),
    ]
    with pytest.raises(ah.HandoffError):
        ah.build_handoff(build, root, code=CODE, official_runs=crossed)
    changed = second.results[0].model_copy(update={"attempt_id": "attempt_other"})
    altered = dataclasses.replace(second, results=[changed, *second.results[1:]])
    with pytest.raises(ah.HandoffError, match="not a result of run"):
        ah.build_handoff(build, root, code=CODE, official_runs=[first, altered])
    extra = second.results[0].model_copy(update={"observation_key": "obs_not_in_table"})
    padded = dataclasses.replace(second, results=[*second.results, extra])
    with pytest.raises(ah.HandoffError, match="absent from the analysis table"):
        ah.build_handoff(build, root, code=CODE, official_runs=[first, padded])


@needs_tokenizer
def test_a_synthetic_table_cannot_borrow_the_verified_runs(tmp_path, view, tiny, recognised):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    rehearsal = ah.build_handoff(build, root, code=CODE, rehearsal=True)
    assert rehearsal.input_origin == "synthetic_rehearsal" and rehearsal.source_runs == []
    with pytest.raises(ah.HandoffError):
        ah.publish_handoff(rehearsal, root, official_runs=verified)


@needs_tokenizer
@pytest.mark.parametrize(
    "change",
    [
        {"resolved_config_sha256": "f" * 64},
        {"matrix_arm": "dense_robustness"},
    ],
)
def test_each_run_is_checked_against_its_own_arm_and_configuration(
    tmp_path, view, tiny, recognised, change
):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    first, second = verified
    completion = second.completed.completion.model_copy(update=change)
    altered = dataclasses.replace(
        second, completed=dataclasses.replace(second.completed, completion=completion)
    )
    with pytest.raises(ah.HandoffError, match="differs from run"):
        ah.build_handoff(build, root, code=CODE, official_runs=[first, altered])


@needs_tokenizer
@pytest.mark.parametrize(
    "change",
    [
        {"status": "operational_failure"},  # status differs from the row's
        {"raw_output": '{"answer": "tampered", "abstained": false}'},  # same status and attempt
    ],
)
def test_a_result_that_differs_from_the_one_behind_the_row_is_refused(
    tmp_path, view, tiny, recognised, change
):
    verified, root, build = two_run_table(tmp_path, view, tiny)
    first, second = verified
    changed = second.results[0].model_copy(update=change)  # unvalidated copy of a decided result
    altered = dataclasses.replace(second, results=[changed, *second.results[1:]])
    # A status change is a handoff error; edited output fails result validation first.
    # Either way the handoff is refused (both are ValueError).
    with pytest.raises(ValueError):
        ah.build_handoff(build, root, code=CODE, official_runs=[first, altered])
