"""P10-C2C: all five frozen matrix arms through the guarded rehearsal lifecycle.

Synthetic content, disposable rehearsal roots, mock backend. Expected counts and keys come from
the campaign planner and the frozen matrix, not from the code under test.
"""

import json
from itertools import pairwise

import pytest
from finaltest_audit_support import CONFIGS, ITEMS, PLAN, settings_for
from finaltest_campaign_support import oracle_key
from finaltest_start_support import (
    FakeRenderer,
    MockBackend,
    SyntheticInputs,
    make_world,
    mock_environment,
)
from test_c2b_start import NOTE, audit, read_records, start
from test_generation_config import to_toml

from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest.campaign import ARMS, expected_keys
from biorag_repro.finaltest.execution import run_batch
from biorag_repro.finaltest.inventory import verify_root
from biorag_repro.finaltest.runrecord import FinalTestRunRecord
from biorag_repro.finaltest.session import open_session
from biorag_repro.finaltest.start import StartRefused


def use_arm(w, arm):
    w.request = w.with_request(
        matrix_arm=arm,
        config_toml=to_toml(settings_for(arm)).encode(),
        renderer=FakeRenderer(CONFIGS[arm]),
        inputs=SyntheticInputs(arm),
    )


def planned_count(arm):
    conditions, repeats = FROZEN_MATRIX[arm]
    return len(conditions) * len(repeats) * len(ITEMS)


def run_arm(w, arm):
    use_arm(w, arm)
    backend = MockBackend()
    with open_session(w.root) as session:
        started = start(w, session)
        summary = run_batch(started, backend, capture_environment=mock_environment)
    return started, summary, backend


def test_the_planner_and_the_frozen_matrix_agree_on_the_synthetic_population():
    counts = {arm: len(expected_keys(PLAN, ITEMS, arm)) for arm in ARMS}
    assert counts == {arm: planned_count(arm) for arm in ARMS}
    assert counts == {
        "confirmatory": 30,
        "closed_book": 15,
        "dense_robustness": 10,
        "chunk_robustness": 10,
        "scope_robustness": 10,
    }
    assert PLAN.total_expected == sum(counts.values()) == 75


@pytest.mark.parametrize("arm", ARMS)
def test_each_arm_runs_end_to_end_with_its_frozen_conditions(tmp_path, arm):
    w = make_world(tmp_path, arm=arm)
    started, summary, backend = run_arm(w, arm)
    keys = expected_keys(PLAN, ITEMS, arm)
    conditions, repeats = FROZEN_MATRIX[arm]

    assert (summary.complete, summary.terminal_failures) == (len(keys), 0)
    assert len(backend.calls) == len(keys)
    assert set(started.rows) == set(keys)

    manifest, binding, rosters, _ = read_records(w, started)
    assert manifest.matrix_arm == arm and set(rosters) == set(conditions)
    assert binding.resolved_config_sha256 == CONFIGS[arm].configuration_sha256
    owned = []
    for condition, roster in rosters.items():
        assert len(roster.rows) == len(ITEMS) * len(repeats)
        for row in roster.rows:
            assert (row.matrix_arm, row.condition) == (arm, condition)
            assert row.observation_key == oracle_key(
                row.item_id, arm, condition, row.repeat_index, CONFIGS[arm].configuration_sha256
            )
            owned.append(row.observation_key)
    assert sorted(owned) == keys and len(set(owned)) == len(owned)

    if arm == "closed_book":
        assert binding.retrieval_config_sha256 is None
        for roster in rosters.values():
            for row in roster.rows:
                assert (row.source_identity_sha256, row.context_sha256) == (None, None)
                assert row.evidence_sequence_sha256 is None
    else:
        for roster in rosters.values():
            for row in roster.rows:
                assert row.source_identity_sha256 and row.context_sha256
                assert row.evidence_sequence_sha256
        if len(conditions) == 2:  # JATS and PDF_DOCLING never share an identity
            by_condition = [
                {r.source_identity_sha256 for r in roster.rows} for roster in rosters.values()
            ]
            assert not by_condition[0] & by_condition[1]
            contexts = [{r.context_sha256 for r in roster.rows} for roster in rosters.values()]
            assert not contexts[0] & contexts[1]

    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,)
    assert verdict.batches[started.batch_id].state == "authorized"

    result = audit(w, started)
    assert result.structure_valid and result.claims_consistent and result.complete
    assert result.authorization == "not_established"
    assert sorted(s.key for s in result.observed.states) == keys
    assert all(s.status == "complete" for s in result.observed.states)
    record = FinalTestRunRecord.model_validate_json(
        (w.root / "runs" / started.run_id / "run.json").read_bytes()
    )
    assert (record.root_kind, record.backend_kind, record.official) == ("rehearsal", "mock", False)
    for key in keys:
        obs = w.root / "runs" / started.run_id / "observations" / key
        assert (obs / "final.build.json").exists()


@pytest.mark.parametrize("arm", ARMS)
def test_configuration_and_rows_are_deterministic_across_independent_roots(tmp_path, arm):
    def rows(base):
        w = make_world(base, arm=arm)
        with open_session(w.root) as session:
            started = start(w, session)
        _, binding, rosters, _ = read_records(w, started)
        return (
            binding.resolved_config_sha256,
            {c: [r.model_dump_json() for r in ro.rows] for c, ro in rosters.items()},
        )

    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir()
    second.mkdir()
    assert rows(first) == rows(second)


def test_one_campaign_in_one_root_keeps_every_arm_separate(tmp_path):
    w = make_world(tmp_path)
    started = {}
    for arm in ARMS:
        started[arm], summary, _ = run_arm(w, arm)
        assert summary.complete == planned_count(arm) and summary.terminal_failures == 0

    # one batch, run and configuration per arm
    assert len({s.batch_id for s in started.values()}) == 5
    assert len({s.run_id for s in started.values()}) == 5
    assert len({s.binding_sha256 for s in started.values()}) == 5
    configs = {a: CONFIGS[a].configuration_sha256 for a in ARMS}
    assert len(set(configs.values())) == 5

    # observation keys are unique across the campaign and owned by exactly the planned arm
    owner = {}
    for arm, batch in started.items():
        assert sorted(batch.rows) == expected_keys(PLAN, ITEMS, arm)
        for key in batch.rows:
            assert key not in owner, f"{key} is owned by {owner[key]} and {arm}"
            owner[key] = arm
    assert len(owner) == PLAN.total_expected == 75

    for arm, batch in started.items():
        manifest, binding, rosters, _ = read_records(w, batch)
        assert manifest.matrix_arm == arm
        assert binding.resolved_config_sha256 == configs[arm]
        for roster in rosters.values():
            assert {r.matrix_arm for r in roster.rows} == {arm}
            assert {r.resolved_config_sha256 for r in roster.rows} == {configs[arm]}

    # registry: genesis plus four events per batch, sequential, each batch authorized
    events = [json.loads(p.read_bytes()) for p in sorted((w.root / "registry").glob("0*"))]
    assert len(events) == 1 + 4 * 5
    assert [e["event_type"] for e in events[1:]] == [
        "batch_reserved",
        "batch_materialized",
        "batch_inputs_bound",
        "batch_authorized",
    ] * 5
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    for earlier, later in pairwise(events):
        assert later["previous_event_sha256"] != earlier.get("previous_event_sha256")
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,)
    assert {b.state for b in verdict.batches.values()} == {"authorized"}
    assert set(verdict.batches) == {s.batch_id for s in started.values()}

    # the auditor still does not authorize any of them, and per-run counts add up
    total = 0
    for arm, batch in started.items():
        result = audit(w, batch)
        assert result.structure_valid and result.complete
        assert result.authorization == "not_established"
        assert sorted(s.key for s in result.observed.states) == expected_keys(PLAN, ITEMS, arm)
        total += len(result.observed.states)
    assert total == PLAN.total_expected


def test_completed_observations_are_not_regenerated_after_later_batches(tmp_path):
    w = make_world(tmp_path)
    first, _, _ = run_arm(w, ARMS[0])
    snapshot = {
        p: p.read_bytes() for p in (w.root / "runs" / first.run_id).rglob("*") if p.is_file()
    }
    for arm in ARMS[1:]:
        run_arm(w, arm)
    after = {p: p.read_bytes() for p in (w.root / "runs" / first.run_id).rglob("*") if p.is_file()}
    assert after == snapshot  # later batches never touch an earlier run

    # a fresh session cannot restart the finished arm (no supersession, no resume in C2)
    use_arm(w, ARMS[0])
    backend = MockBackend()
    with open_session(w.root) as session, pytest.raises(StartRefused) as caught:
        start(w, session)
    assert caught.value.code == "arm_already_reserved"
    assert backend.calls == []


def test_a_crash_after_some_observations_cannot_be_resumed_or_repaired(tmp_path):
    from biorag_repro.generation.attempts import CorruptRecord, RunStore
    from biorag_repro.generation.finaltest_store import (
        FinalTestStore,
        GateRefused,
        StoreOperationRefused,
    )

    class Stop(BaseException):
        pass

    class DiesAfterThree(MockBackend):
        def generate(self, rendered, *, seed, max_new_tokens):
            if len(self.calls) == 3:
                raise Stop
            return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)

    w = make_world(tmp_path, arm="dense_robustness")
    with open_session(w.root) as session:
        started = start(w, session)
        with pytest.raises(Stop):
            run_batch(started, DiesAfterThree(), capture_environment=mock_environment)
    run_dir = w.root / "runs" / started.run_id
    done = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    assert sum(1 for p in done if p.name == "final.build.json") == 3

    # a later session cannot start the arm again, resume it, reopen the store or change a byte
    with open_session(w.root) as session:
        with pytest.raises(StartRefused) as caught:
            start(w, session)
        assert caught.value.code == "arm_already_reserved"
        with pytest.raises(StoreOperationRefused):
            FinalTestStore.open(w.root, started.run_id)
        with pytest.raises(CorruptRecord):
            RunStore.open(
                w.root, started.run_id, config=CONFIGS["dense_robustness"], backend_kind="mock"
            )
        with pytest.raises(GateRefused, match="session is closed"):
            started.store.begin_attempt(
                next(iter(started.rows)),
                environment=mock_environment(),
                rendered_input_sha256="0" * 64,
            )
    assert {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()} == done

    result = audit(w, started)
    assert result.authorization == "not_established" and not result.complete


@pytest.mark.parametrize("where", ["rosters", "citations"])
def test_a_directory_in_place_of_a_roster_or_citation_file_is_reported(tmp_path, where):
    w = make_world(tmp_path)
    with open_session(w.root) as session:
        started = start(w, session)
    base = (
        w.root
        / "batches"
        / started.batch_id
        / ("inputs/rosters" if where == "rosters" else "citations")
    )
    name = ("a" * 64 + ".json") if where == "rosters" else "000002.json"
    (base / name).mkdir()
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert [p for p in verdict.problems if p != NOTE]
