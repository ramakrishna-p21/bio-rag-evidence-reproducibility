"""P10-C4: all five frozen arms through start, interruption, resume and completion.

Two separate synthetic campaigns, each in one disposable rehearsal root with the mock backend.
Expected keys and counts come from the campaign planner and the frozen matrix, never from the
code under test.

Positive campaign: every arm reaches `batch_completed`, by each route a completion can take.
Negative campaign: the same root layout with batches left unfinished, interrupted, retryable,
reserved only or never reserved; nothing incomplete may be published as complete.
"""

import json
from itertools import pairwise

import pytest
from finaltest_audit_support import CONFIGS, ITEMS, PLAN
from finaltest_completion_support import FailsOn, assess_it, complete_fresh, in_session, verify_it
from finaltest_resume_support import DiesAfter, Stop, assess, renew, resume
from finaltest_start_support import (
    FakeRenderer,
    MockBackend,
    SyntheticInputs,
    make_world,
    mock_environment,
)
from test_c2b_start import audit
from test_c3_recovery import attempt
from test_generation_config import to_toml

from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.batches import batch_id_for
from biorag_repro.finaltest.campaign import ARMS, expected_keys, plan_sha256
from biorag_repro.finaltest.completeness import (
    MISSING_RECORDED,
    NEVER_ATTEMPTED,
    VERIFIED,
    check_campaign,
)
from biorag_repro.finaltest.execution import run_batch
from biorag_repro.finaltest.finish import CompletionRefused
from biorag_repro.finaltest.inventory import verify_root
from biorag_repro.finaltest.session import open_session
from biorag_repro.finaltest.start import _start_with

NOTE = inventory.HEAD_NOT_ESTABLISHED_TEXT
from finaltest_audit_support import settings_for


def planned_count(arm):
    conditions, repeats = FROZEN_MATRIX[arm]
    return len(conditions) * len(repeats) * len(ITEMS)


def prepare(w, arm, nonce):
    """Point the world at one arm and give it a fresh READY report that names its configuration."""
    w.request = w.with_request(
        matrix_arm=arm,
        config_toml=to_toml(settings_for(arm)).encode(),
        renderer=FakeRenderer(CONFIGS[arm]),
        inputs=SyntheticInputs(arm),
    )
    w.arm = arm
    renew(w, nonce)


def nonce(n):
    return f"{n:032x}"


def start(w, session):
    return _start_with(w.request, w.assess, session, now=w.clock)


def batch_of(w, arm):
    return batch_id_for(layout.root_id_for(w.auth), plan_sha256(PLAN), arm)


def stable(tree):
    return {k: v for k, v in tree.items() if not k.endswith(("session.holder", "session.lock"))}


def evidence_of(w, started):
    """Every byte of one batch: its registry-independent files and its whole run directory."""
    out = {}
    for top in (w.root / "batches" / started.batch_id, w.root / "runs" / started.run_id):
        for path in sorted(top.rglob("*")):
            if path.is_file():
                out[str(path)] = path.read_bytes()
    return out


# ---- the positive campaign --------------------------------------------------------------------

EXPECTED = {  # arm -> (route, complete, recorded missing); counts checked against the planner
    "confirmatory": ("same session", 30, 0),
    "closed_book": ("fresh session, one terminal failure", 14, 1),
    "dense_robustness": ("resumed session", 10, 0),
    "chunk_robustness": ("fresh session, retries exhausted", 9, 1),
    "scope_robustness": ("fresh session, nothing wrong", 10, 0),
}


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    w = make_world(tmp_path_factory.mktemp("positive"))
    started, snapshots, results = {}, {}, {}
    n = iter(range(1, 1000))

    def finish_arm(arm):
        snapshots[arm] = evidence_of(w, started[arm])

    # confirmatory: runs to the end and is completed by the session that ran it
    prepare(w, "confirmatory", nonce(next(n)))
    with open_session(w.root) as session:
        started["confirmatory"] = start(w, session)
        run_batch(started["confirmatory"], MockBackend(), capture_environment=mock_environment)
        results["confirmatory"] = in_session(w, started["confirmatory"], session)
    finish_arm("confirmatory")

    # closed book: one terminal failure, completed by a new session
    prepare(w, "closed_book", nonce(next(n)))
    with open_session(w.root) as session:
        started["closed_book"] = start(w, session)
        run_batch(started["closed_book"], FailsOn(3), capture_environment=mock_environment)
    results["closed_book"] = complete_fresh(w, nonce(next(n)))
    finish_arm("closed_book")

    # dense robustness: the worker dies, an authorized resume finishes it, the resumed session completes
    prepare(w, "dense_robustness", nonce(next(n)))
    with open_session(w.root) as session:
        started["dense_robustness"] = start(w, session)
        with pytest.raises(Stop):
            run_batch(
                started["dense_robustness"], DiesAfter(4), capture_environment=mock_environment
            )
    renew(w, nonce(next(n)))
    with open_session(w.root) as session:
        resumed = resume(w, session)
        run_batch(resumed, MockBackend(), capture_environment=mock_environment)
        results["dense_robustness"] = in_session(w, resumed, session)
    finish_arm("dense_robustness")

    # chunk robustness: an observation is interrupted until its retries are exhausted
    prepare(w, "chunk_robustness", nonce(next(n)))
    with open_session(w.root) as session:
        started["chunk_robustness"] = start(w, session)
        with pytest.raises(Stop):
            run_batch(
                started["chunk_robustness"], DiesAfter(4), capture_environment=mock_environment
            )
    with pytest.raises(Stop):
        attempt(w, started["chunk_robustness"], nonce(next(n)), DiesAfter(0))
    attempt(w, started["chunk_robustness"], nonce(next(n)))
    results["chunk_robustness"] = complete_fresh(w, nonce(next(n)))
    finish_arm("chunk_robustness")

    # scope robustness: finished, the original process exits, a new session completes it
    prepare(w, "scope_robustness", nonce(next(n)))
    with open_session(w.root) as session:
        started["scope_robustness"] = start(w, session)
        run_batch(started["scope_robustness"], MockBackend(), capture_environment=mock_environment)
    results["scope_robustness"] = complete_fresh(w, nonce(next(n)))
    finish_arm("scope_robustness")
    return w, started, results, snapshots


def test_the_planner_and_the_frozen_matrix_agree_on_the_campaign():
    counts = {arm: len(expected_keys(PLAN, ITEMS, arm)) for arm in ARMS}
    assert counts == {arm: planned_count(arm) for arm in ARMS}
    assert {arm: sum(v[1:]) for arm, v in EXPECTED.items()} == counts


def test_every_arm_is_completed_with_the_oracle_counts_and_keys(campaign):
    w, started, results, _ = campaign
    for arm in ARMS:
        record = results[arm].completion
        _, complete, missing = EXPECTED[arm]
        keys = expected_keys(PLAN, ITEMS, arm)
        assert [o.key for o in record.observations] == sorted(keys)
        assert (record.complete_count, record.missing_count) == (complete, missing)
        assert record.outcome == (
            "complete_with_recorded_missing" if missing else "complete_all_observed"
        )
        assert (record.batch_id, record.run_id) == (started[arm].batch_id, started[arm].run_id)
        assert record.matrix_arm == arm
        assert record.resolved_config_sha256 == CONFIGS[arm].configuration_sha256
        assert record.root_kind == "rehearsal"
        assert verify_it(w, started[arm].batch_id) == record
    modes = {arm: results[arm].completion.authorization.mode for arm in ARMS}
    assert modes == {
        "confirmatory": "same_session",
        "closed_book": "fresh_session",
        "dense_robustness": "same_session",
        "chunk_robustness": "fresh_session",
        "scope_robustness": "fresh_session",
    }


def test_missing_observations_are_recorded_failures_never_successes(campaign):
    _, _, results, _ = campaign
    failed = [
        o
        for arm in ("closed_book", "chunk_robustness")
        for o in results[arm].completion.observations
        if o.status != "complete"
    ]
    assert sorted(o.status for o in failed) == ["retries_exhausted", "terminal_failure"]
    terminal = next(o for o in failed if o.status == "terminal_failure")
    exhausted = next(o for o in failed if o.status == "retries_exhausted")
    assert (terminal.failure_kind, terminal.attempts) == ("unclassified_error", 1)
    assert (exhausted.final_state, exhausted.attempts) == ("interrupted", 2)
    assert exhausted.failure_kind is None  # an interruption names no failure and none is invented


def test_ownership_identities_and_contexts_are_the_planned_ones(campaign):
    w, started, _, _ = campaign
    owner = {}
    for arm in ARMS:
        for key in started[arm].rows:
            assert key not in owner
            owner[key] = arm
        assert sorted(started[arm].rows) == expected_keys(PLAN, ITEMS, arm)
        conditions, _ = FROZEN_MATRIX[arm]
        base = w.root / "batches" / started[arm].batch_id
        rosters = {
            json.loads(p.read_bytes())["condition"]: json.loads(p.read_bytes())
            for p in (base / "inputs" / "rosters").glob("*.json")
        }
        assert set(rosters) == set(conditions)
        for condition, roster in rosters.items():
            for row in roster["rows"]:
                assert (row["matrix_arm"], row["condition"]) == (arm, condition)
                if arm == "closed_book":
                    assert (row["source_identity_sha256"], row["context_sha256"]) == (None, None)
                else:
                    assert row["source_identity_sha256"] and row["context_sha256"]
    assert len(owner) == PLAN.total_expected == 75
    jats_pdf = [a for a in ARMS if len(FROZEN_MATRIX[a][0]) == 2]
    assert jats_pdf  # dense, chunk and scope robustness compare two representations
    for arm in jats_pdf:
        base = w.root / "batches" / started[arm].batch_id / "inputs" / "rosters"
        sources = [
            {r["source_identity_sha256"] for r in json.loads(p.read_bytes())["rows"]}
            for p in base.glob("*.json")
        ]
        assert not sources[0] & sources[1]


def test_the_registry_history_is_ordered_complete_and_witnessed(campaign):
    w, started, _, _ = campaign
    events = [json.loads(p.read_bytes()) for p in sorted((w.root / "registry").glob("0*"))]
    types = [e["event_type"] for e in events]
    assert types[0] == "root_initialized"
    assert (
        types.count("batch_completed") == 5 and types.count("batch_resumed") == 3
    )  # one for dense, two for chunk
    assert [e["sequence"] for e in events] == list(range(1, len(events) + 1))
    for before, after in pairwise(events):
        assert after["sequence"] == before["sequence"] + 1
    for arm in ARMS:  # each batch ends with its own completion, after everything it cites
        sequence = [
            e["sequence"]
            for e in events
            if any(c["sha256"] == started[arm].batch_id for c in e["evidence"])
        ]
        completed = next(
            e["sequence"]
            for e in events
            if e["event_type"] == "batch_completed"
            and any(c["sha256"] == started[arm].batch_id for c in e["evidence"])
        )
        assert completed == max(sequence)
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    assert verdict.problems == (NOTE,) and verdict.head_authority == "not_established"
    assert {b.state for b in verdict.batches.values()} == {"completed"}
    for e in events:
        assert (w.witnesses / f"{e['root_id']}.{e['sequence']:06d}.witness.json").is_file()


def test_the_campaign_is_complete_with_recorded_missing_and_the_audit_does_not_authorize(campaign):
    w, started, _, _ = campaign
    observed = {}
    for arm in ARMS:
        result = audit(w, started[arm])
        assert result.authorization == "not_established" and result.structure_valid
        observed[arm] = [result.observed]
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    campaign_verdict = check_campaign(PLAN, ITEMS, layout.root_id_for(w.auth), verdict, observed)
    assert campaign_verdict.verdict == "complete_with_recorded_missing"
    assert campaign_verdict.anomalies == ()
    for arm, (_, complete, missing) in EXPECTED.items():
        counts = campaign_verdict.per_arm[arm]
        assert counts.get(VERIFIED, 0) == complete and counts.get(MISSING_RECORDED, 0) == missing
        assert counts.get(NEVER_ATTEMPTED, 0) == 0


def test_a_later_batch_never_changes_an_earlier_batchs_evidence(campaign):
    w, started, _, snapshots = campaign
    for arm in ARMS:
        assert evidence_of(w, started[arm]) == snapshots[arm]


def test_the_citation_chain_of_the_resumed_batch_is_continuous(campaign):
    w, started, results, _ = campaign
    record = results["dense_robustness"].completion
    assert record.citation_count == 2
    assert record.authorization.basis_sha256 == record.last_citation_sha256
    assert record.first_citation_sha256 != record.last_citation_sha256
    assert assess(w, started["dense_robustness"].batch_id).classification == "completed"


def test_repeated_verification_changes_no_byte(campaign):
    w, started, _, _ = campaign
    before = w.tree()
    for _ in range(2):
        verify_root(w.root, w.auth, w.anchors, w.witnesses)
        for arm in ARMS:
            verify_it(w, started[arm].batch_id)
            assert assess_it(w, started[arm].batch_id).status == "completed"
            audit(w, started[arm])
    assert w.tree() == before


def test_no_completed_arm_can_be_completed_again(campaign):
    w, _, _, _ = campaign

    def evidence():
        return {
            k: v
            for k, v in w.tree().items()
            if any(part in k for part in ("/batches/", "/runs/", "/registry/", "/witnesses/"))
        }

    before = evidence()
    for arm in ARMS:
        prepare(w, arm, nonce(900 + ARMS.index(arm)))  # a new report each time: no authority gap
        with pytest.raises(CompletionRefused) as caught:
            complete_fresh(w, nonce(950 + ARMS.index(arm)), new_report=False)
        assert caught.value.code == "completed"
    assert evidence() == before


# ---- the negative campaign --------------------------------------------------------------------


@pytest.fixture
def partial(tmp_path):
    """One completed arm beside a crashed one, a retryable one, a reserved-only one and a missing one."""
    w = make_world(tmp_path)
    started = {}
    prepare(w, "confirmatory", nonce(1))
    with open_session(w.root) as session:
        started["confirmatory"] = start(w, session)
        run_batch(started["confirmatory"], MockBackend(), capture_environment=mock_environment)
        in_session(w, started["confirmatory"], session)

    prepare(w, "closed_book", nonce(2))  # the worker died; it has not been resumed
    with open_session(w.root) as session:
        started["closed_book"] = start(w, session)
        with pytest.raises(Stop):
            run_batch(started["closed_book"], DiesAfter(5), capture_environment=mock_environment)

    prepare(w, "dense_robustness", nonce(3))  # recovered but unfinished: retryable work remains
    with open_session(w.root) as session:
        started["dense_robustness"] = start(w, session)
        with pytest.raises(Stop):
            run_batch(
                started["dense_robustness"], DiesAfter(9), capture_environment=mock_environment
            )
    renew(w, nonce(4))
    with open_session(w.root) as session:
        resume(w, session)

    registry.reserve_batch(  # claimed and materialized, never started
        w.root, w.auth, w.anchors, w.witnesses, PLAN, "chunk_robustness"
    )
    registry.materialize_batch(
        w.root, w.auth, w.anchors, w.witnesses, batch_of(w, "chunk_robustness")
    )
    return w, started  # scope_robustness was never reserved


def test_unfinished_arms_are_not_completed_and_the_campaign_is_not_complete(partial):
    w, started = partial
    expected = {
        "closed_book": ("work_remains", "work_remains"),
        "dense_robustness": ("work_remains", "work_remains"),
    }
    for arm, (status, code) in expected.items():
        verdict = assess_it(w, started[arm].batch_id)
        assert (verdict.status, verdict.code) == (status, code)
    chunk = assess_it(w, batch_of(w, "chunk_robustness"))
    assert (chunk.status, chunk.code) == ("structurally_incomplete", "batch_materialized")
    assert assess_it(w, batch_of(w, "scope_robustness")).code == "unknown_batch"

    for arm in ("closed_book", "dense_robustness", "chunk_robustness", "scope_robustness"):
        prepare(w, arm, nonce(100 + ARMS.index(arm)))
        before = stable(w.tree())
        with pytest.raises(CompletionRefused):
            complete_fresh(w, nonce(200 + ARMS.index(arm)), new_report=False)
        after = stable(w.tree())
        assert {k: v for k, v in after.items() if "/batches/" in k or "/registry/" in k} == {
            k: v for k, v in before.items() if "/batches/" in k or "/registry/" in k
        }
    events = [json.loads(p.read_bytes()) for p in sorted((w.root / "registry").glob("0*"))]
    assert [e["event_type"] for e in events].count("batch_completed") == 1

    observed = {
        arm: [audit(w, started[arm]).observed]
        for arm in ("confirmatory", "closed_book", "dense_robustness")
    }
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    result = check_campaign(PLAN, ITEMS, layout.root_id_for(w.auth), verdict, observed)
    assert result.verdict == "incomplete"
    assert result.per_arm["confirmatory"] == {VERIFIED: 30}
    assert result.per_arm["scope_robustness"] == {NEVER_ATTEMPTED: 10}
    assert result.per_arm["chunk_robustness"] == {NEVER_ATTEMPTED: 10}
    assert result.per_arm["closed_book"].get(VERIFIED, 0) == 5
    assert all(
        key not in result.outcomes or result.outcomes[key] != VERIFIED
        for key in expected_keys(PLAN, ITEMS, "scope_robustness")
    )


def test_an_unfinished_neighbour_publication_blocks_completion_until_it_is_closed(tmp_path):
    """The whole-root rule: an arm cannot be completed while the root has an open publication."""
    w = make_world(tmp_path)
    prepare(w, "confirmatory", nonce(1))
    with open_session(w.root) as session:
        done = start(w, session)
        run_batch(done, MockBackend(), capture_environment=mock_environment)
    registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, PLAN, "closed_book")
    closed = batch_of(w, "closed_book")
    prepare(w, "confirmatory", nonce(2))
    before = stable(w.tree())
    assert assess_it(w, done.batch_id).code == "root_unverified"
    with pytest.raises(CompletionRefused) as caught:
        complete_fresh(w, nonce(3), new_report=False)
    assert caught.value.code == "structurally_incomplete"
    assert stable(w.tree()) == before  # the finished arm's evidence is intact
    registry.close_incomplete(w.root, w.auth, w.anchors, w.witnesses, closed)
    assert assess_it(w, done.batch_id).status == "completable"
    assert (
        complete_fresh(w, nonce(4), new_report=False).completion.outcome == "complete_all_observed"
    )


def test_a_same_session_start_of_another_arm_ends_the_earlier_authorization(tmp_path):
    w = make_world(tmp_path)
    prepare(w, "confirmatory", nonce(1))
    with open_session(w.root) as session:
        first = start(w, session)
        run_batch(first, MockBackend(), capture_environment=mock_environment)
        prepare(w, "closed_book", nonce(2))
        second = start(w, session)  # the registry head moves in the same session
        before = stable(w.tree())
        prepare(w, "confirmatory", nonce(3))
        with pytest.raises(CompletionRefused) as caught:
            in_session(w, first, session)
        assert caught.value.code == "session_unchanged_check"
        assert stable(w.tree()) == before
        assert second.batch_id != first.batch_id
    assert complete_fresh(w, nonce(4), new_report=False).completion.matrix_arm == "confirmatory"
