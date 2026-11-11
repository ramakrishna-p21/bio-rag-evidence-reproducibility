"""Attempt recovery, evidence integrity, sessions, and the stopped-worker rule.

The retry limit of the synthetic configuration is two attempts per observation. A run
directory is damaged only after a real interrupted run, so each refusal names the check that the
damaged fact is protected by.
"""

import json
import multiprocessing
import os
import re
import signal
import socket
import subprocess
from datetime import timedelta

import pytest
from finaltest_resume_support import (
    DiesAfter,
    Stop,
    assess,
    interrupted,
    renew,
    resume,
)
from finaltest_start_support import MockBackend, mock_environment

from biorag_repro.finaltest import layout
from biorag_repro.finaltest.execution import run_batch
from biorag_repro.finaltest.resume import ResumeRefused
from biorag_repro.finaltest.session import SessionLost, _process_start, open_session
from biorag_repro.finaltest.store import GateRefused
from biorag_repro.generation.attempts import InvalidTransition


def states(started, key):
    return [e.state for e in started.store.observation(key).events]


def running_key(started):
    return next(k for k in started.rows if started.store.observation(k).status == "running")


def attempt(w, started, nonce, backend=None):
    """One resumed session that runs the batch with `backend` (default: to the end)."""
    renew(w, nonce)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        summary = run_batch(resumed, backend or MockBackend(), capture_environment=mock_environment)
    return resumed, summary


# ---- retry limit and recovery -----------------------------------------------------------------


def test_an_interrupted_attempt_counts_toward_the_limit_and_is_never_run_again(tmp_path):
    w, started = interrupted(tmp_path)
    key = running_key(started)
    # resume 1: the second attempt of the same observation dies too
    attempt_one = DiesAfter(0)
    with pytest.raises(Stop):
        attempt(w, started, "1" * 32, attempt_one)
    assert states(started, key) == ["planned", "running", "interrupted", "running"]
    # resume 2: the limit (two attempts) is reached; recovery closes the log, nothing reruns it
    counting = MockBackend()
    resumed, summary = attempt(w, started, "2" * 32, counting)
    assert states(started, key) == ["planned", "running", "interrupted", "running", "interrupted"]
    state = started.store.observation(key)
    assert (state.status, state.attempts) == ("retries_exhausted", 2)
    assert (summary.attempts, summary.retries_exhausted, summary.complete) == (6, 1, 9)
    assert len(counting.calls) == 6
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("terminally_closed", "no_eligible_observation")
    renew(w, "3" * 32)
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == "terminally_closed"
    assert resumed.run_id == started.run_id


def test_recovery_keeps_attempt_history_and_adds_only_the_interrupted_event(tmp_path):
    w, started = interrupted(tmp_path)
    key = running_key(started)
    before = {
        p: p.read_bytes() for p in (w.root / "runs" / started.run_id).rglob("*") if p.is_file()
    }
    renew(w)
    with open_session(w.root) as session:
        resume(w, session)  # recovers and returns without running
    after = {
        p: p.read_bytes() for p in (w.root / "runs" / started.run_id).rglob("*") if p.is_file()
    }
    added = set(after) - set(before)
    assert {p.name for p in added} == {"000003.json"} and all(key in str(p) for p in added)
    assert all(after[p] == before[p] for p in before)
    assert states(started, key) == ["planned", "running", "interrupted"]


def test_a_completed_observation_is_never_replayed_by_a_resumed_store(tmp_path):
    w, started = interrupted(tmp_path)
    done = next(k for k in started.rows if started.store.observation(k).status == "complete")
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        row = resumed.rows[done]
        with pytest.raises(InvalidTransition, match="never regenerated"):
            resumed.store.begin_attempt(
                done,
                environment=mock_environment(),
                rendered_input_sha256=row.rendered_input_sha256,
            )
        with pytest.raises(GateRefused, match="not an observation of this batch"):
            resumed.store.begin_attempt(
                "f" * 64, environment=mock_environment(), rendered_input_sha256="e" * 64
            )
        with pytest.raises(GateRefused, match="differs from the bound roster"):
            resumed.store.begin_attempt(
                next(iter(resumed.rows)),
                environment=mock_environment(),
                rendered_input_sha256="e" * 64,
            )


def test_a_failed_attempt_is_not_a_new_scientific_repeat(tmp_path):
    w, started = interrupted(tmp_path)
    key = running_key(started)
    coordinates = started.store.observation(key).coordinates
    attempt(w, started, "1" * 32)
    state = started.store.observation(key)
    assert state.coordinates == coordinates and state.key == key
    assert state.attempts == 2 and state.status == "complete"
    assert {c.repeat_index for c in (s.coordinates for s in started.store.verify())} == {1}


# ---- damaged evidence is refused, never repaired ----------------------------------------------


def damage_and_refuse(w, started, classification, code=None):
    found = assess(w, started.batch_id)
    assert found.classification == classification, (found.code, found.detail)
    if code is not None:
        assert found.code == code
    renew(w)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == classification
    after = {k: v for k, v in w.tree().items() if not k.endswith("session.holder")}
    assert after == {k: v for k, v in before.items() if not k.endswith("session.holder")}


def complete_key(started):
    return next(k for k in started.rows if started.store.observation(k).status == "complete")


def test_a_missing_seal_is_reported_and_not_silently_published(tmp_path):
    w, started = interrupted(tmp_path)
    seal = started.store._seal_path(complete_key(started))
    seal.chmod(0o644)
    seal.unlink()
    damage_and_refuse(w, started, "structurally_incomplete", "seal_missing")
    assert not seal.exists()


def test_an_altered_seal_is_contradictory(tmp_path):
    w, started = interrupted(tmp_path)
    seal = started.store._seal_path(complete_key(started))
    seal.chmod(0o644)
    text = seal.read_text()
    digest = re.search(r"[0-9a-f]{64}", text).group()
    seal.write_text(text.replace(digest, "0" * 64, 1))  # a different sealed event, same form
    damage_and_refuse(w, started, "contradictory")


def test_a_truncated_event_file_is_contradictory(tmp_path):
    w, started = interrupted(tmp_path)
    events = sorted(
        (started.store.run_dir / "observations" / running_key(started) / "events").glob("0*.json")
    )
    events[-1].chmod(0o644)
    events[-1].write_bytes(events[-1].read_bytes()[:40])
    damage_and_refuse(w, started, "contradictory")


def test_a_gap_in_an_attempt_log_is_contradictory(tmp_path):
    w, started = interrupted(tmp_path)
    events = sorted(
        (started.store.run_dir / "observations" / running_key(started) / "events").glob("0*.json")
    )
    events[0].chmod(0o644)
    events[0].unlink()
    damage_and_refuse(w, started, "contradictory")


def test_an_observation_that_the_batch_does_not_own_is_contradictory(tmp_path):
    import shutil

    w, started = interrupted(tmp_path)
    source = started.store.run_dir / "observations" / complete_key(started)
    shutil.copytree(source, started.store.run_dir / "observations" / ("f" * 64))
    damage_and_refuse(w, started, "contradictory")


def test_an_already_complete_batch_is_refused_and_not_run_again(tmp_path):
    w, started = interrupted(tmp_path)
    attempt(w, started, "1" * 32)
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("already_complete", "all_observations_complete")
    renew(w, "2" * 32)
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == "already_complete"


# ---- sessions ---------------------------------------------------------------------------------


def test_a_closed_session_cannot_resume(tmp_path):
    w, _ = interrupted(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        pass
    with pytest.raises(SessionLost):
        resume(w, session)


def test_authorization_ends_with_the_session(tmp_path):
    w, _ = interrupted(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
    key = next(iter(resumed.rows))
    with pytest.raises(GateRefused, match="session is closed"):
        resumed.store.begin_attempt(
            key,
            environment=mock_environment(),
            rendered_input_sha256=resumed.rows[key].rendered_input_sha256,
        )
    with pytest.raises(SessionLost):
        run_batch(resumed, MockBackend(), capture_environment=mock_environment)


def test_a_lost_session_lock_stops_the_resumed_run_before_any_backend_call(tmp_path):
    w, _ = interrupted(tmp_path)
    renew(w)
    backend = MockBackend()
    with open_session(w.root) as session:
        resumed = resume(w, session)
        (w.root / layout.SESSION_LOCK_FILE).unlink()  # the lock file is gone from its path
        with pytest.raises(SessionLost):
            run_batch(resumed, backend, capture_environment=mock_environment)
    assert backend.calls == []


def test_a_moved_registry_head_stops_the_resumed_run(tmp_path):
    from biorag_repro.finaltest import registry
    from biorag_repro.finaltest.layout import EvidenceCitation

    w, _ = interrupted(tmp_path)
    renew(w)
    backend = MockBackend()
    with open_session(w.root) as session:
        resumed = resume(w, session)
        registry._append(
            w.root,
            w.witnesses,
            layout.root_id_for(w.auth),
            "batch_completed",
            [EvidenceCitation(role="x", sha256="a" * 64)],
            w.clock,
        )
        with pytest.raises(GateRefused, match="registry head"):
            run_batch(resumed, backend, capture_environment=mock_environment)
    assert backend.calls == []


def test_only_one_session_can_resume_at_a_time(tmp_path):
    from biorag_repro.finaltest.session import SessionBusy

    w, _ = interrupted(tmp_path)
    with open_session(w.root):
        with pytest.raises(SessionBusy), open_session(w.root):
            pass
        outcome = {}

        def other_process():
            try:
                with open_session(w.root):
                    os._exit(0)
            except SessionBusy:
                os._exit(7)

        process = multiprocessing.get_context("fork").Process(target=other_process)
        process.start()
        process.join(30)
        outcome["code"] = process.exitcode
    assert outcome["code"] == 7


# ---- the previous worker must be shown to have stopped ----------------------------------------


def write_holder(w, **record):
    path = w.root / layout.HOLDER_FILE
    path.write_bytes(json.dumps(record, sort_keys=True).encode())


def refuse_for_worker(w, detail):
    renew(w)
    before = {k: v for k, v in w.tree().items() if not k.endswith("session.holder")}
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session)
    assert caught.value.code == "worker_not_stopped" and detail in str(caught.value)
    assert {k: v for k, v in w.tree().items() if not k.endswith("session.holder")} == before


def test_no_record_of_the_previous_worker_is_not_proof(tmp_path):
    w, _ = interrupted(tmp_path)
    (w.root / layout.HOLDER_FILE).unlink()
    refuse_for_worker(w, "no record")


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"not json",
        b'{"state": "open"}',
        b'{"state": "sleeping", "pid": 1, "host": "h", "start": "1"}',
    ],
)
def test_an_unreadable_record_of_the_previous_worker_is_not_proof(tmp_path, content):
    w, _ = interrupted(tmp_path)
    (w.root / layout.HOLDER_FILE).write_bytes(content)
    refuse_for_worker(w, "unreadable")


def test_a_worker_on_another_host_is_not_proof(tmp_path):
    w, _ = interrupted(tmp_path)
    write_holder(w, state="open", pid=1, start="1", host="another-host")
    refuse_for_worker(w, "another host")


def test_a_still_running_worker_process_is_not_stopped_even_without_the_lock(tmp_path):
    w, _ = interrupted(tmp_path)
    process = subprocess.Popen(["sleep", "60"])
    try:
        write_holder(
            w,
            state="open",
            pid=process.pid,
            start=_process_start(process.pid),
            host=socket.gethostname(),
        )
        refuse_for_worker(w, "still running")
    finally:
        process.kill()
        process.wait()


def test_a_reused_pid_is_a_different_process(tmp_path):
    w, started = interrupted(tmp_path)
    key = running_key(started)
    process = subprocess.Popen(["sleep", "60"])
    try:
        # same pid, another start time: the recorded worker is gone and the pid is someone else's
        write_holder(w, state="open", pid=process.pid, start="0", host=socket.gethostname())
        renew(w)
        with open_session(w.root) as session:
            resumed = resume(w, session)
        assert states(started, key) == ["planned", "running", "interrupted"]
        assert resumed.run_id == started.run_id
    finally:
        process.kill()
        process.wait()


def test_a_killed_worker_process_is_recovered_once_it_is_gone(tmp_path):
    w, started = interrupted(tmp_path)
    key = running_key(started)
    renew(w, "1" * 32)

    class KillsItself(MockBackend):
        def generate(self, rendered, *, seed, max_new_tokens):
            os.kill(os.getpid(), signal.SIGKILL)

    def worker():
        with open_session(w.root) as session:
            run_batch(resume(w, session), KillsItself(), capture_environment=mock_environment)

    process = multiprocessing.get_context("fork").Process(target=worker)
    process.start()
    process.join(60)
    assert process.exitcode == -signal.SIGKILL
    holder = json.loads((w.root / layout.HOLDER_FILE).read_bytes())
    assert holder["state"] == "open" and holder["pid"] == process.pid  # it died holding the record
    w.clock.value += timedelta(days=1)  # the child's clock ticks are not visible here
    assert states(started, key) == ["planned", "running", "interrupted", "running"]

    resumed, summary = attempt(w, started, "2" * 32)
    # two attempts were already used by the key, so the dead worker's attempt is the last one
    assert states(started, key)[-1] == "interrupted"
    assert started.store.observation(key).status == "retries_exhausted"
    assert (summary.complete, summary.retries_exhausted) == (9, 1)
    assert resumed.run_id == started.run_id


def test_no_proof_is_needed_when_no_attempt_is_left_running(tmp_path):
    w, started = interrupted(tmp_path)
    renew(w, "1" * 32)
    with open_session(w.root) as session:
        resume(w, session)  # recovers the running attempt and runs nothing
    (w.root / layout.HOLDER_FILE).unlink()
    assert assess(w, started.batch_id).running == ()
    renew(w, "2" * 32)
    with open_session(w.root) as session:
        assert resume(w, session).run_id == started.run_id


# ---- interruption after the resume records are published --------------------------------------


def test_a_fault_while_recovering_leaves_a_batch_that_can_be_resumed_again(tmp_path):
    from biorag_repro.generation import attempts

    class Fault(Exception):
        pass

    w, started = interrupted(tmp_path)
    key = running_key(started)
    renew(w, "1" * 32)
    patch = pytest.MonkeyPatch()
    original = attempts.publish_bytes

    def failing(*args, **kwargs):
        raise Fault

    patch.setattr(attempts, "publish_bytes", failing)
    try:
        with open_session(w.root) as session, pytest.raises(Fault):
            resume(w, session)
    finally:
        patch.undo()
    assert attempts.publish_bytes is original
    assert states(started, key) == ["planned", "running"]  # nothing was recovered
    found = assess(w, started.batch_id)
    assert found.classification == "resumable" and len(found.citations) == 2
    resumed, summary = attempt(w, started, "2" * 32)
    assert len(assess(w, started.batch_id).citations) == 3
    assert (summary.complete, started.store.observation(key).attempts) == (10, 2)
    assert resumed.run_id == started.run_id


def test_a_crash_at_a_completion_publication_leaves_a_batch_that_can_be_resumed(tmp_path):
    from biorag_repro.generation import attempts

    w, started = interrupted(tmp_path)
    renew(w, "1" * 32)
    calls = {"n": 0}
    original = attempts.publish_bytes

    def dies_on_the_third_write(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Stop
        return original(*args, **kwargs)

    patch = pytest.MonkeyPatch()
    patch.setattr(attempts, "publish_bytes", dies_on_the_third_write)
    try:
        with open_session(w.root) as session, pytest.raises(Stop):
            run_batch(resume(w, session), MockBackend(), capture_environment=mock_environment)
    finally:
        patch.undo()
    resumed, summary = attempt(w, started, "2" * 32)
    final = assess(w, started.batch_id)
    assert final.classification in ("already_complete", "terminally_closed")
    assert resumed.run_id == started.run_id and summary.attempts >= 1


# ---- which unfinished states are resumable ----------------------------------------------------


def test_a_batch_that_never_began_an_attempt_resumes_without_a_worker_proof(tmp_path):
    from finaltest_resume_support import plan_only

    w, started = plan_only(tmp_path)
    found = assess(w, started.batch_id)
    assert found.classification == "resumable" and len(found.pending) == 10
    assert found.running == () and not (w.root / layout.HOLDER_FILE).exists()
    _, summary = attempt(w, started, "1" * 32)
    assert (summary.complete, summary.attempts) == (10, 10)


def test_observations_that_were_never_planned_are_planned_by_the_resume(tmp_path):
    from finaltest_resume_support import plan_only

    w, started = plan_only(tmp_path, planned=4)
    assert len(started.store.observation_keys()) == 4
    found = assess(w, started.batch_id)
    assert found.classification == "resumable" and len(found.pending) == 10  # 6 absent, 4 planned
    _, summary = attempt(w, started, "1" * 32)
    assert (summary.complete, len(started.store.observation_keys())) == (10, 10)


def test_a_retryable_observation_alone_keeps_the_batch_resumable(tmp_path):
    w, started = interrupted(tmp_path, after=9)  # nine complete, the tenth left running
    renew(w, "1" * 32)
    with open_session(w.root) as session:
        resume(w, session)  # recovery only: the tenth becomes retryable
    found = assess(w, started.batch_id)
    assert found.classification == "resumable" and len(found.pending) == 1 and found.running == ()
    _, summary = attempt(w, started, "2" * 32)
    assert (summary.complete, summary.attempts) == (10, 1)


def test_a_terminal_failure_is_recorded_and_closes_the_rest(tmp_path):
    w, started = interrupted(tmp_path, after=9)
    key = running_key(started)
    renew(w, "1" * 32)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        attempt_id = resumed.store.begin_attempt(
            key,
            environment=mock_environment(),
            rendered_input_sha256=resumed.rows[key].rendered_input_sha256,
        )
        resumed.store.fail(key, attempt_id, kind="invalid_request", detail="x", retryable=False)
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("terminally_closed", "no_eligible_observation")


def test_a_resumed_store_refuses_an_attempt_on_another_host_or_gpu(tmp_path):
    w, _ = interrupted(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        key = next(k for k in resumed.rows if resumed.store.observation(k).status == "planned")
        elsewhere = mock_environment().model_copy(update={"hostname": "elsewhere"})
        with pytest.raises(GateRefused, match="host or GPU"):
            resumed.store.begin_attempt(
                key,
                environment=elsewhere,
                rendered_input_sha256=resumed.rows[key].rendered_input_sha256,
            )


def test_the_registry_moving_during_the_guard_is_refused(tmp_path):
    from finaltest_resume_support import loader

    from biorag_repro.finaltest import registry
    from biorag_repro.finaltest.layout import EvidenceCitation
    from biorag_repro.finaltest.resume import _resume_with

    w, _ = interrupted(tmp_path)
    renew(w)

    def assess_and_move():
        registry._append(
            w.root,
            w.witnesses,
            layout.root_id_for(w.auth),
            "batch_completed",
            [EvidenceCitation(role="x", sha256="a" * 64)],
            w.clock,
        )
        return w.assess()

    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        _resume_with(
            w.request,
            assess_and_move,
            session,
            load_report=loader(w),
            capture_environment=mock_environment,
            now=w.clock,
        )
    assert caught.value.code == "head_changed"
