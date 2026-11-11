"""Completion eligibility, authorization, publication and read-back of one batch.

Synthetic rehearsal roots, mock backend. Expected keys come from the campaign planner, never
from the code under test. Every refusal test compares the whole storage before and after.
"""

import json
import stat

import pytest
import readiness_support as support
from finaltest_audit_support import CONFIGS
from finaltest_completion_support import (
    ARM,
    FailsOn,
    assess_it,
    complete_fresh,
    expected_for,
    finished,
    in_session,
    stopped,
    verify_it,
)
from finaltest_resume_support import DiesAfter, Stop, assess, loader, plan_only, renew, resume
from finaltest_start_support import MockBackend, make_world, mock_environment
from finaltest_support import NOTE

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.audit import AuditContext, audit_run
from biorag_repro.finaltest.batches import citation_sha256
from biorag_repro.finaltest.completion import BatchCompletion
from biorag_repro.finaltest.execution import run_batch
from biorag_repro.finaltest.finish import CompletionRefused, complete_in_new_session
from biorag_repro.finaltest.resume import ResumeRefused
from biorag_repro.finaltest.session import SessionBusy, SessionLost, open_session
from biorag_repro.finaltest.start import StartRefused, _start_with
from biorag_repro.finaltest.store import GateRefused
from biorag_repro.readiness.guard import AuthorizationRefused


def problems(w):
    verdict = inventory.verify_root(w.root, w.auth, w.anchors, w.witnesses)
    return [p for p in verdict.problems if p != NOTE]


def stored(w, batch):
    return BatchCompletion.model_validate_json(
        (w.root / "batches" / batch / layout.COMPLETION_NAME).read_bytes()
    )


def last_event(w):
    path = max((w.root / "registry").glob("0*.json"))
    return layout.RegistryEvent.model_validate_json(path.read_bytes())


def stable(tree):
    """The tree without the two volatile session files (the lock is empty, the holder is not evidence)."""
    return {k: v for k, v in tree.items() if not k.endswith(("session.holder", "session.lock"))}


def refused(w, code, call):
    """`call` must raise CompletionRefused(code) and leave the evidence unchanged."""
    before = stable(w.tree())
    with pytest.raises(CompletionRefused) as caught:
        call()
    assert caught.value.code == code, str(caught.value)
    assert stable(w.tree()) == before


# ---- the happy paths ---------------------------------------------------------------------------


def test_a_fully_observed_batch_is_completed_by_a_new_fresh_session(tmp_path):
    w, started = finished(tmp_path)
    assert assess_it(w, started.batch_id).status == "completable"
    result = complete_fresh(w)
    record = result.completion
    assert record.outcome == "complete_all_observed"
    assert (record.complete_count, record.missing_count) == (10, 0)
    assert [o.key for o in record.observations] == sorted(expected_for())
    assert {o.status for o in record.observations} == {"complete"}
    assert {o.failure_kind for o in record.observations} == {None}
    assert record.authorization.mode == "fresh_session"
    assert (record.batch_id, record.run_id) == (started.batch_id, started.run_id)
    assert problems(w) == []
    assert assess_it(w, started.batch_id).status == "completed"
    assert verify_it(w, started.batch_id) == record


def test_the_event_and_files_carry_exactly_the_cited_evidence(tmp_path):
    w, started = finished(tmp_path)
    result = complete_fresh(w)
    event = last_event(w)
    cited = {c.role: c.sha256 for c in event.evidence}
    assert [c.role for c in event.evidence] == layout.REHEARSAL_ROLES["batch_completed"]
    base = w.root / "batches" / started.batch_id
    assert cited["completion_record"] == sha256_bytes((base / layout.COMPLETION_NAME).read_bytes())
    assert cited["completion_record"] == result.sha256
    continuity = (base / layout.COMPLETION_CONTINUITY_NAME).read_bytes()
    assert cited["authorization_basis"] == sha256_bytes(continuity)
    assert cited["readiness_report"] == w.fixture.sha
    assert cited["batch_id"] == started.batch_id
    assert cited["run_record"] == started.run_record_sha256
    for name in (layout.COMPLETION_NAME, layout.COMPLETION_CONTINUITY_NAME):
        assert stat.S_IMODE((base / name).stat().st_mode) == 0o444
    # the record cites the registry event just before its own event, never its own
    head = sorted((w.root / "registry").glob("0*.json"))[-2]
    assert stored(w, started.batch_id).registry_head_sequence == event.sequence - 1
    assert stored(w, started.batch_id).registry_head_event_sha256 == sha256_bytes(head.read_bytes())
    # the witness chain covers the new event
    assert (w.witnesses / f"{event.root_id}.{event.sequence:06d}{layout.WITNESS_SUFFIX}").is_file()


def test_the_session_that_ran_the_batch_completes_it_on_its_own_authorization(tmp_path):
    w = _world(tmp_path)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
        calls = w.guard_calls
        result = in_session(w, started, session)
        assert w.guard_calls == calls  # the existing authorization is reused, not asked again
    record = result.completion
    last = _citations(w, started.batch_id)[-1]
    assert record.authorization.mode == "same_session"
    assert record.authorization.basis_sha256 == citation_sha256(last)
    assert record.authorization.report_sha256 == last.readiness_report_sha256
    assert not (w.root / "batches" / started.batch_id / layout.COMPLETION_CONTINUITY_NAME).exists()
    assert problems(w) == []
    assert verify_it(w, started.batch_id) == record


def test_a_resumed_session_completes_on_the_citation_of_its_resume(tmp_path):
    w, started = stopped(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        run_batch(resumed, MockBackend(), capture_environment=mock_environment)
        result = in_session(w, resumed, session)
    chain = _citations(w, started.batch_id)
    assert len(chain) == 2 and result.completion.citation_count == 2
    assert result.completion.authorization.basis_sha256 == citation_sha256(chain[1])
    assert result.completion.first_citation_sha256 == citation_sha256(chain[0])
    assert problems(w) == []


def test_the_original_process_exiting_before_completion_leaves_it_finishable(tmp_path):
    """All observations finished, the process died before batch_completed: a new session ends it."""
    w, _ = finished(tmp_path)
    before = {k: v for k, v in w.tree().items() if "/runs/" in k}
    result = complete_fresh(w)
    assert result.completion.authorization.mode == "fresh_session"
    # the run itself is byte-identical: completion writes nothing below runs/
    assert {k: v for k, v in w.tree().items() if "/runs/" in k} == before
    assert problems(w) == []


# ---- terminal failures and exhausted retries stay failures -----------------------------------


def test_recorded_terminal_failures_complete_the_batch_as_recorded_missing(tmp_path):
    w, started = finished(tmp_path, backend=FailsOn(2, 5))
    assert assess_it(w, started.batch_id).status == "completable"
    record = complete_fresh(w).completion
    assert record.outcome == "complete_with_recorded_missing"
    assert (record.complete_count, record.missing_count) == (8, 2)
    failed = [o for o in record.observations if o.status != "complete"]
    assert [o.status for o in failed] == ["terminal_failure"] * 2
    assert {o.failure_kind for o in failed} == {"unclassified_error"}
    assert all("synthetic terminal failure" in (o.failure_detail or "") for o in failed)
    assert all(o.attempts == 1 for o in failed)
    assert [o.key for o in record.observations] == sorted(expected_for())
    assert verify_it(w, started.batch_id) == record


def test_exhausted_retries_complete_the_batch_as_recorded_missing(tmp_path):
    from test_finaltest_resume_recovery import attempt

    w, started = stopped(tmp_path)
    with pytest.raises(Stop):
        attempt(w, started, "1" * 32, DiesAfter(0))
    attempt(w, started, "2" * 32)  # the second interruption exhausts the key; the rest finish
    assert assess(w, started.batch_id).classification == "terminally_closed"
    verdict = assess_it(w, started.batch_id)
    assert (verdict.status, verdict.outcome) == ("completable", "complete_with_recorded_missing")
    record = complete_fresh(w, "3" * 32).completion
    exhausted = [o for o in record.observations if o.status == "retries_exhausted"]
    assert len(exhausted) == 1 and exhausted[0].attempts == 2
    assert record.complete_count == 9 and record.missing_count == 1
    assert record.outcome == "complete_with_recorded_missing"


# ---- helpers local to this module ------------------------------------------------------------


def _world(tmp_path):
    return make_world(tmp_path, arm=ARM)


def _citations(w, batch):
    from biorag_repro.finaltest.batches import AuthorizationCitation

    folder = w.root / "batches" / batch / "citations"
    return [
        AuthorizationCitation.model_validate_json(p.read_bytes())
        for p in sorted(folder.glob("0*.json"))
        if "continuity" not in p.name
    ]


def test_the_read_only_audit_still_does_not_establish_authorization(tmp_path):
    w, started = finished(tmp_path)
    complete_fresh(w)
    audit = audit_run(
        w.root,
        started.run_id,
        AuditContext(
            root_id=layout.root_id_for(w.auth),
            root_kind="rehearsal",
            plan=__import__("finaltest_audit_support").PLAN,
            item_papers=dict(w.request.population.item_papers),
            manifest=None,
            binding=None,
            rosters={},
        ),
    )
    assert audit.authorization == "not_established"
    json.dumps(audit.authorization)


# ---- eligibility: work that remains, closure and invalid evidence -----------------------------


def attempt_fresh(w, **kwargs):
    return lambda: complete_fresh(w, **kwargs)


@pytest.mark.parametrize(
    ("build", "status", "code"),
    [
        (lambda p: stopped(p), "work_remains", "work_remains"),  # running key and absent keys
        (lambda p: plan_only(p), "work_remains", "work_remains"),  # planned, never attempted
        (lambda p: plan_only(p, planned=4), "work_remains", "work_remains"),  # 6 absent
    ],
)
def test_unfinished_observations_are_not_completable(tmp_path, build, status, code):
    w, started = build(tmp_path)
    verdict = assess_it(w, started.batch_id)
    assert (verdict.status, verdict.code, verdict.outcome) == (status, code, None)
    refused(w, status, attempt_fresh(w))


def test_a_retryable_observation_is_work_for_a_resume_not_a_completion(tmp_path):
    w, started = stopped(tmp_path, after=9)  # nine complete, the tenth left running
    renew(w, "1" * 32)
    with open_session(w.root) as session:
        resume(w, session)  # recovery only: the tenth becomes retryable
    assert started.store.observation(_the_open_key(started)).status == "retryable"
    assert assess_it(w, started.batch_id).status == "work_remains"
    refused(w, "work_remains", attempt_fresh(w, nonce="2" * 32))


def _the_open_key(started):
    return next(k for k in started.rows if started.store.observation(k).status != "complete")


def test_a_batch_closed_incomplete_is_never_completable(tmp_path):
    w = make_world(tmp_path, arm=ARM)
    registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    batch = registry.reserve_batch.__globals__["batch_id_for"](
        layout.root_id_for(w.auth),
        registry.reserve_batch.__globals__["plan_sha256"](w.request.plan),
        ARM,
    )
    registry.close_incomplete(w.root, w.auth, w.anchors, w.witnesses, batch)
    verdict = assess_it(w, batch)
    assert (verdict.status, verdict.code) == ("closed_incomplete", "batch_closed_incomplete")
    renew(w, "0" * 32)
    refused(w, "closed_incomplete", attempt_fresh(w, new_report=False))


def test_a_missing_seal_is_not_invented_and_blocks_completion(tmp_path):
    w, started = finished(tmp_path, backend=FailsOn(1))
    seal = next((w.root / "runs" / started.run_id / "observations").glob("*/final.build.json"))
    seal.unlink()
    verdict = assess_it(w, started.batch_id)
    assert (verdict.status, verdict.code) == ("structurally_incomplete", "seal_missing")
    assert not seal.exists()  # nothing recreated it
    refused(w, "structurally_incomplete", attempt_fresh(w))
    assert not seal.exists()


def test_a_missing_seal_of_a_terminal_failure_blocks_completion_too(tmp_path):
    w, started = finished(tmp_path, backend=FailsOn(1))
    failed = next(
        k for k in started.rows if started.store.observation(k).status == "terminal_failure"
    )
    (w.root / "runs" / started.run_id / "observations" / failed / "final.build.json").unlink()
    assert assess_it(w, started.batch_id).code == "seal_missing"
    refused(w, "structurally_incomplete", attempt_fresh(w))


def test_an_altered_observation_log_is_contradictory(tmp_path):
    w, started = finished(tmp_path)
    key = next(iter(started.rows))
    event = next(
        (w.root / "runs" / started.run_id / "observations" / key / "events").glob("0*.json")
    )
    event.chmod(0o644)
    event.write_bytes(event.read_bytes().replace(b'"seed"', b'"seeD"', 1))
    assert assess_it(w, started.batch_id).status in ("contradictory", "structurally_incomplete")
    refused(w, assess_it(w, started.batch_id).status, attempt_fresh(w))


def test_a_foreign_observation_in_the_run_blocks_completion(tmp_path):
    w, started = finished(tmp_path)
    key = next(iter(started.rows))
    folder = w.root / "runs" / started.run_id / "observations"
    import shutil

    shutil.copytree(folder / key, folder / ("obs_" + "0" * 64))
    assert assess_it(w, started.batch_id).status in ("contradictory", "structurally_incomplete")
    refused(w, assess_it(w, started.batch_id).status, attempt_fresh(w))


# ---- authorization ----------------------------------------------------------------------------


def test_a_not_ready_report_cannot_publish_a_completion(tmp_path):
    w, _ = finished(tmp_path)
    renew(w, blocked={"G12_shared_gpu_authorization": "pending"})
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(AuthorizationRefused):
        complete_in_new_session(w.request, w.assess, session, load_report=loader(w), now=w.clock)
    assert w.tree() == before


def test_a_historical_report_is_not_authorization_to_complete(tmp_path):
    w, started = finished(tmp_path)  # the only report is the one the batch started on
    refused(w, "report_reused", attempt_fresh(w, new_report=False))
    assert assess_it(w, started.batch_id).status == "completable"


def test_the_lock_the_registry_and_the_chain_alone_do_not_authorize(tmp_path):
    """A fully verifying batch and a held session lock, with a report that blocks: nothing."""
    w, started = finished(tmp_path)
    assert problems(w) == [] and assess_it(w, started.batch_id).status == "completable"
    renew(w, "1" * 32, candidate=support.candidate(
        configuration_sha256_by_arm={w.arm: CONFIGS[w.arm].configuration_sha256},
        scientific_source_identity_sha256="f" * 64,
    ))  # fmt: skip
    refused(w, "not_continuous", attempt_fresh(w, new_report=False))


def test_a_report_that_is_not_the_one_the_batch_started_on_is_refused(tmp_path):
    w, _ = finished(tmp_path)
    renew(w)
    elsewhere = renew(
        w,
        "5" * 32,
        candidate=support.candidate(
            configuration_sha256_by_arm={w.arm: CONFIGS[w.arm].configuration_sha256},
            scientific_source_identity_sha256="e" * 64,
        ),
    )
    other = loader(w)(elsewhere)
    renew(w, "6" * 32)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(CompletionRefused) as caught:
        complete_in_new_session(w.request, w.assess, session, load_report=lambda _s: other)
    assert caught.value.code == "initial_report_mismatch" and w.tree() == before


def test_an_unreadable_first_report_is_refused(tmp_path):
    w, _ = finished(tmp_path)
    renew(w)

    def missing(_sha):
        raise FileNotFoundError

    before = w.tree()
    with open_session(w.root) as session, pytest.raises(CompletionRefused) as caught:
        complete_in_new_session(w.request, w.assess, session, load_report=missing)
    assert caught.value.code == "initial_report_unavailable" and w.tree() == before


def test_a_changed_policy_blocks_the_new_session(tmp_path):
    w, _ = finished(tmp_path)
    first = w.fixture.policy
    w.fixture.policy = support.fixture_policy("another-policy")
    renew(w)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(CompletionRefused) as caught:
        complete_in_new_session(
            w.request, w.assess, session, load_report=loader(w, first), now=w.clock
        )
    assert caught.value.code == "not_continuous" and "policy" in str(caught.value)
    assert w.tree() == before


def test_a_closed_session_cannot_complete(tmp_path):
    w = make_world(tmp_path, arm=ARM)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
    before = w.tree()
    with pytest.raises(SessionLost):
        in_session(w, started, session)
    assert w.tree() == before


def test_another_session_cannot_complete_on_the_old_sessions_authorization(tmp_path):
    w, started = finished(tmp_path)
    with open_session(w.root) as later:
        refused(w, "not_the_running_session", lambda: in_session(w, started, later))


def test_a_moved_head_ends_the_same_session_authorization(tmp_path):
    w = make_world(tmp_path, arm=ARM)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
        registry._append(
            w.root,
            w.witnesses,
            layout.root_id_for(w.auth),
            "batch_completed",
            [layout.EvidenceCitation(role="x", sha256="a" * 64)],
            w.clock,
        )
        before = w.tree()
        with pytest.raises(CompletionRefused) as caught:
            in_session(w, started, session)
        assert w.tree() == before
    assert caught.value.code in ("root_unverified", "contradictory", "structurally_incomplete")


def test_a_changed_bound_file_ends_the_same_session_authorization(tmp_path):
    w = make_world(tmp_path, arm=ARM)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
        binding = w.root / "batches" / started.batch_id / "inputs" / "binding.json"
        binding.chmod(0o644)
        binding.write_bytes(binding.read_bytes() + b" ")
        with pytest.raises(CompletionRefused):
            in_session(w, started, session)
    assert not (w.root / "batches" / started.batch_id / layout.COMPLETION_NAME).exists()


# ---- after completion -------------------------------------------------------------------------


def test_a_completed_batch_cannot_be_completed_resumed_started_or_extended(tmp_path):
    w = make_world(tmp_path, arm=ARM)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, MockBackend(), capture_environment=mock_environment)
        done = in_session(w, started, session)
        # the store of the session that ran it refuses every further write
        key = next(iter(started.rows))
        with pytest.raises(GateRefused):
            started.store.begin_attempt(
                key,
                environment=mock_environment(),
                rendered_input_sha256=started.rows[key].rendered_input_sha256,
            )
    assert assess_it(w, started.batch_id).status == "completed"
    assert assess(w, started.batch_id).classification == "completed"
    after_completion = w.tree()
    # no second completion, in either mode
    renew(w, "1" * 32)
    refused(w, "completed", attempt_fresh(w, new_report=False))
    with open_session(w.root) as session:
        refused(w, "completed", lambda: in_session(w, started, session))
    # no resume
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == "completed"
    # no second start, no new reservation of the arm
    with open_session(w.root) as session, pytest.raises(StartRefused):
        _start_with(w.request, w.assess, session, now=w.clock)
    with pytest.raises(registry.ProtocolRefused):
        registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    assert stable(w.tree()) == stable(after_completion)
    assert verify_it(w, started.batch_id) == done.completion


def test_read_back_changes_no_byte_and_needs_no_authorization(tmp_path):
    w, started = finished(tmp_path)
    complete_fresh(w)
    before = w.tree()
    guard_calls = w.guard_calls
    first = verify_it(w, started.batch_id)
    assert verify_it(w, started.batch_id) == first
    assert assess_it(w, started.batch_id).status == "completed"
    assert w.tree() == before and w.guard_calls == guard_calls


def test_a_completion_that_no_longer_matches_the_run_fails_read_back(tmp_path):
    w, started = finished(tmp_path)
    complete_fresh(w)
    key = next(iter(started.rows))
    folder = w.root / "runs" / started.run_id / "observations" / key
    (folder / "final.build.json").unlink()
    with pytest.raises(CompletionRefused):
        verify_it(w, started.batch_id)


# ---- crashes at each publication boundary ----------------------------------------------------


class Crash(Exception):
    pass


def crash_on(monkeypatch, target, name, *, nth=1):
    real = getattr(target, name)
    seen = {"n": 0}

    def wrapper(*args, **kwargs):
        seen["n"] += 1
        if seen["n"] == nth:
            raise Crash
        return real(*args, **kwargs)

    monkeypatch.setattr(target, name, wrapper)


def test_a_crash_before_the_event_writes_nothing_and_completion_can_be_retried(
    tmp_path, monkeypatch
):
    w, _ = finished(tmp_path)
    renew(w)
    crash_on(monkeypatch, registry, "append_event_locked")
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(Crash):
        complete_in_new_session(w.request, w.assess, session, load_report=loader(w), now=w.clock)
    assert w.tree() == before
    monkeypatch.undo()
    with open_session(w.root) as session:
        # the report of the crashed attempt was never cited, so it may still be used
        result = complete_in_new_session(
            w.request, w.assess, session, load_report=loader(w), now=w.clock
        )
    assert result.completion.outcome == "complete_all_observed"


@pytest.mark.parametrize("nth", [1, 2])
def test_a_crash_after_the_event_leaves_an_incomplete_batch_that_is_never_repaired(
    tmp_path, monkeypatch, nth
):
    """nth 1: the continuity file is missing; nth 2: the completion file is missing."""
    w, started = finished(tmp_path)
    renew(w)
    crash_on(monkeypatch, registry, "publish_batch_file", nth=nth)
    with open_session(w.root) as session, pytest.raises(Crash):
        complete_in_new_session(w.request, w.assess, session, load_report=loader(w), now=w.clock)
    monkeypatch.undo()
    left = w.tree()
    assert any("has no completion record" in p or "lacks its continuity" in p for p in problems(w))
    verdict = assess_it(w, started.batch_id)
    assert verdict.status == "structurally_incomplete"
    # a second attempt, even on a new report, writes, adopts and deletes nothing
    renew(w, "2" * 32)
    left = w.tree()
    with pytest.raises(CompletionRefused) as caught:
        complete_fresh(w, "3" * 32, new_report=False)
    assert caught.value.code == "structurally_incomplete"
    assert w.tree() == left
    with open_session(w.root) as session, pytest.raises(ResumeRefused):
        resume(w, session)
    assert w.tree() == left


def test_an_intact_completion_whose_readback_was_interrupted_verifies_without_republishing(
    tmp_path, monkeypatch
):
    from biorag_repro.finaltest import finish

    w, started = finished(tmp_path)
    renew(w)
    crash_on(monkeypatch, finish, "verify_completion")
    with open_session(w.root) as session, pytest.raises(Crash):
        complete_in_new_session(w.request, w.assess, session, load_report=loader(w), now=w.clock)
    monkeypatch.undo()
    published = w.tree()
    assert problems(w) == []
    record = verify_it(w, started.batch_id)  # read-only read-back succeeds
    assert w.tree() == published
    assert record.outcome == "complete_all_observed"
    refused(w, "completed", attempt_fresh(w, nonce="4" * 32))  # and nothing is published twice


# ---- concurrency ------------------------------------------------------------------------------


def test_a_competing_session_blocks_completion(tmp_path):
    w, _ = finished(tmp_path)
    renew(w)
    with open_session(w.root):
        before = w.tree()
        with pytest.raises(SessionBusy), open_session(w.root) as other:
            complete_in_new_session(w.request, w.assess, other, load_report=loader(w))
        assert w.tree() == before


def test_a_competing_registry_writer_blocks_completion(tmp_path):
    import multiprocessing

    from biorag_repro.finaltest.inventory import RegistryBusy, registry_lock

    w, started = finished(tmp_path)
    renew(w)
    ctx = multiprocessing.get_context("fork")
    held, release = ctx.Event(), ctx.Event()

    def holder():
        with registry_lock(w.root):
            held.set()
            release.wait(30)

    process = ctx.Process(target=holder)
    process.start()
    try:
        assert held.wait(30)
        before = w.tree()
        with open_session(w.root) as session, pytest.raises(RegistryBusy):
            complete_in_new_session(w.request, w.assess, session, load_report=loader(w))
        assert stable(w.tree()) == stable(before)
    finally:
        release.set()
        process.join(30)
    assert assess_it(w, started.batch_id).status == "completable"


# ---- eligibility does not come from the classification -----------------------------------------


@pytest.mark.parametrize(
    ("damage", "code"),
    [
        ("drop", "observation_absent"),
        ("open", "observation_open"),
        ("foreign", "foreign_observation"),
        ("duplicate", "duplicate_observation"),
    ],
)
def test_eligibility_is_decided_from_the_observations_not_from_the_classification(
    tmp_path, monkeypatch, damage, code
):
    """Even when the resume classification says the batch is complete, the states decide."""
    import dataclasses

    from biorag_repro.finaltest import finish

    w, started = finished(tmp_path)
    real = finish.assess_resume

    def misreporting(*args, **kwargs):
        found = real(*args, **kwargs)
        assert found.classification == "already_complete"
        states = list(found.states)
        if damage == "drop":
            states = states[:-1]
        elif damage == "open":
            states[0] = dataclasses.replace(states[0], status="retryable")
        elif damage == "foreign":
            states[0] = dataclasses.replace(states[0], key="obs_" + "0" * 64)
        else:
            states.append(states[0])
        return dataclasses.replace(found, states=tuple(states))

    monkeypatch.setattr(finish, "assess_resume", misreporting)
    verdict = assess_it(w, started.batch_id)
    assert (verdict.status, verdict.code) == ("work_remains", code)
    refused(w, "work_remains", attempt_fresh(w))


def test_read_back_refuses_a_batch_that_is_not_completed(tmp_path):
    w, started = finished(tmp_path)
    with pytest.raises(CompletionRefused) as caught:
        verify_it(w, started.batch_id)
    assert caught.value.code == "completable"


def test_read_back_refuses_a_published_record_that_the_run_does_not_support(tmp_path, monkeypatch):
    """The verifier cannot see the run, so only the read-back can catch a record that is wrong."""
    from biorag_repro.finaltest import finish

    w, _ = finished(tmp_path)
    renew(w)
    real = finish.build_completion
    calls = []

    def first_call_lies(**kwargs):
        record = real(**kwargs)
        calls.append(1)
        if len(calls) > 1:
            return record
        first = record.observations[0].model_copy(update={"attempts": 7})
        return record.model_copy(update={"observations": [first, *record.observations[1:]]})

    monkeypatch.setattr(finish, "build_completion", first_call_lies)
    with open_session(w.root) as session, pytest.raises(CompletionRefused) as caught:
        complete_in_new_session(w.request, w.assess, session, load_report=loader(w), now=w.clock)
    assert caught.value.code == "completion_differs_from_run"


def test_open_work_after_completion_makes_the_batch_contradictory(tmp_path, monkeypatch):
    import dataclasses

    from biorag_repro.finaltest import resume as resume_module
    from biorag_repro.finaltest.inputs import Finding

    w, started = finished(tmp_path)
    complete_fresh(w)
    real = resume_module.audit_run

    def with_open_observation(*args, **kwargs):
        audit = real(*args, **kwargs)
        extra = Finding("incomplete", "never_attempted", next(iter(started.rows)))
        return dataclasses.replace(audit, findings=(*audit.findings, extra))

    monkeypatch.setattr(resume_module, "audit_run", with_open_observation)
    verdict = assess_it(w, started.batch_id)
    assert (verdict.status, verdict.code) == ("contradictory", "open_observation_after_completion")
