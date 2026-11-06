"""Run and attempt lifecycle. Backends are mocks and every configuration is synthetic."""

import json
import os
import shutil
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError
from test_generation_config import QWEN_REVISION, resolve, synthetic_settings
from test_generation_environment import cpu_probes

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.attempts import (
    AttemptEvent,
    Coordinates,
    CorruptRecord,
    DuplicateCompletion,
    InvalidTransition,
    LifecycleError,
    NonOfficialBackend,
    PublicationConflict,
    RetryPolicy,
    RunMismatch,
    RunRecord,
    RunStore,
    UnsafeResume,
    check_no_duplicate_completions,
    official_completions,
)
from biorag_repro.generation.environment import BackendIdentity, capture_environment

RENDERED = "c" * 64
OTHER_RENDERED = "d" * 64
POLICY = RetryPolicy(max_attempts=3, retryable_failures=["out_of_memory", "transport_error"])
GOOD_OUTPUT = json.dumps({"answer": "42.5 mg/L", "abstained": False})

# Different papers and answer types; none refers to a real corpus paper.
PAPERS = [
    (
        "item-q1",
        "paper-a",
        "quantitative_value",
        json.dumps({"answer": "0.31", "abstained": False}),
    ),
    ("item-e1", "paper-b", "biological_entity", json.dumps({"answer": "TP53", "abstained": False})),
    (
        "item-s1",
        "paper-c",
        "bounded_entity_set",
        json.dumps({"answer": ["a", "b"], "abstained": False}),
    ),
    (
        "item-c1",
        "paper-d",
        "categorical_conclusion",
        json.dumps({"answer": None, "abstained": True}),
    ),
]


class Clock:
    def __init__(self):
        self.value = datetime(2026, 11, 5, 12, 0, 0, tzinfo=UTC)

    def __call__(self) -> str:
        self.value += timedelta(seconds=1)
        return self.value.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def config_for(kind, **overrides):
    settings = synthetic_settings()
    settings["purpose"] = "development_candidate" if kind == "official" else "synthetic_test"
    settings["retry_max_attempts"] = 3
    settings["retryable_failures"] = list(POLICY.retryable_failures)
    settings.update(overrides)
    return resolve(settings)


def new_store(tmp_path, kind="mock", sample=False, clock=None, **overrides):
    config = config_for(kind, **overrides)
    store = RunStore.create(
        tmp_path,
        config=config,
        phase="development",
        benchmark_version="synthetic-bench-0",
        backend_kind=kind,
        now=clock or Clock(),
    )
    return store, config


def coordinates(index=0, repeat=1, seed=None, arm="confirmatory", condition="JATS"):
    item, paper, _, _ = PAPERS[index]
    return Coordinates(
        item_id=item,
        paper_id=paper,
        matrix_arm=arm,
        representation_or_condition=condition,
        repeat_index=repeat,
        seed=seed,
    )


def environment(kind="mock", model=("Qwen/Qwen3-8B", QWEN_REVISION)):
    return capture_environment(
        backend=BackendIdentity(kind=kind, name="synthetic", version=None),
        model=model,
        tokenizer=model,
        packages=("pydantic",),
        probes=cpu_probes(),
        captured_utc="2026-11-05T12:00:00Z",
    )


def start(store, key, kind="mock", rendered=RENDERED):
    return store.begin_attempt(key, environment=environment(kind), rendered_input_sha256=rendered)


def finish(store, key, raw=GOOD_OUTPUT, kind="mock"):
    attempt = start(store, key, kind)
    store.complete(key, attempt, raw)
    return attempt


def event_files(store, key):
    return sorted((store.run_dir / "observations" / key / "events").glob("0*.json"))


# --- identity -------------------------------------------------------------------------


def test_run_and_attempt_ids_are_opaque_and_unique(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="synthetic", retryable=True)
    b = start(store, key)
    assert len({store.run_id, a, b}) == 3
    assert all(len(x) == 32 and int(x, 16) >= 0 for x in (store.run_id, a, b))


def test_observation_key_uses_existing_identity_function(tmp_path):
    store, config = new_store(tmp_path)
    coords = coordinates(index=1, repeat=2)
    key = store.plan(coords)
    assert key == observation_key(
        phase="development",
        benchmark_version="synthetic-bench-0",
        item_id="item-e1",
        matrix_arm="confirmatory",
        representation_or_condition="JATS",
        repeat_index=2,
        resolved_config_sha256=config.configuration_sha256,
    )


def test_planning_twice_is_idempotent_and_distinct_coordinates_get_distinct_keys(tmp_path):
    store, _ = new_store(tmp_path)
    assert store.plan(coordinates()) == store.plan(coordinates())
    keys = {store.plan(coordinates(i, condition=c)) for i in range(4) for c in ("JATS", "PDF")}
    assert len(keys) == 8
    assert len(event_files(store, store.plan(coordinates()))) == 1


def test_matrix_arm_must_match_the_configuration(tmp_path):
    store, _ = new_store(tmp_path)
    with pytest.raises(LifecycleError, match="matrix arm"):
        store.plan(coordinates(arm="dense_robustness"))


def test_seed_is_required_exactly_when_decoding_samples(tmp_path):
    greedy, _ = new_store(tmp_path / "g")
    with pytest.raises(LifecycleError, match="seed"):
        greedy.plan(coordinates(seed=5))
    sampling, _ = new_store(
        tmp_path / "s", decoding={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    )
    with pytest.raises(LifecycleError, match="seed"):
        sampling.plan(coordinates())
    key = sampling.plan(coordinates(seed=77))
    attempt = start(sampling, key)
    assert sampling.observation(key).events[1].seed == 77
    sampling.fail(key, attempt, kind="transport_error", detail="x", retryable=True)
    start(sampling, key)
    assert [e.seed for e in sampling.observation(key).events if e.state == "running"] == [77, 77]


# --- normal completion and the distinct raw / parsed / status axes -------------------------


@pytest.mark.parametrize("index", range(len(PAPERS)))
def test_normal_completion_keeps_raw_parsed_and_status_separate(tmp_path, index):
    store, _ = new_store(tmp_path)
    _, _, _, raw = PAPERS[index]
    key = store.plan(coordinates(index))
    attempt = finish(store, key, raw)
    obs = store.observation(key)
    assert obs.status == "complete" and obs.attempts == 1
    done = obs.events[-1]
    assert done.state == "complete" and done.attempt_id == attempt
    assert (store.root / done.raw_output.logical_ref).read_text(encoding="utf-8") == raw
    assert done.raw_output.sha256 == sha256_bytes(raw.encode())
    assert done.parse_status == "parsed" and done.run_record_sha256 == store.run_sha256
    expected = json.loads(raw)
    assert (done.answer, done.abstained) == (expected["answer"], expected["abstained"])


def test_unparseable_output_still_completes_and_is_not_repaired(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    raw = '```json\n{"answer": "x", "abstained": false}\n```'
    finish(store, key, raw)
    done = store.observation(key).events[-1]
    assert done.state == "complete" and done.parse_status == "unparseable"
    assert done.answer is None and done.abstained is None and done.parse_reason
    assert (store.root / done.raw_output.logical_ref).read_text() == raw


def test_empty_raw_output_is_kept_as_a_completed_unparseable_attempt(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    finish(store, key, "")
    done = store.observation(key).events[-1]
    assert done.raw_output.sha256 == sha256_bytes(b"") and done.parse_status == "unparseable"


def test_timestamps_are_utc_and_ordered(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    finish(store, key)
    stamps = [e.at_utc for e in store.observation(key).events]
    assert all(s.endswith("Z") for s in stamps) and stamps == sorted(stamps)
    with pytest.raises(ValidationError):
        RunRecord.model_validate(
            {**json.loads(store.run_dir.joinpath("run.json").read_text()), "created_utc": "x"}
        )


def test_attempt_provenance_is_the_actual_environment_used(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    start(store, key)
    running = store.observation(key).events[-1]
    path = store.run_dir / "environments" / f"{running.environment_sha256}.json"
    assert sha256_bytes(path.read_bytes()) == running.environment_sha256
    assert running.backend_kind == "mock" and running.rendered_input_sha256 == RENDERED


# --- state machine ------------------------------------------------------------------------


def test_transitions_are_enforced(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    with pytest.raises(InvalidTransition):
        store.complete(key, "0" * 32, GOOD_OUTPUT)  # nothing running
    with pytest.raises(InvalidTransition):
        store.fail(key, "0" * 32, kind="transport_error", detail="x", retryable=True)
    attempt = start(store, key)
    with pytest.raises(InvalidTransition, match="in progress"):
        start(store, key)
    with pytest.raises(InvalidTransition):
        store.complete(key, "f" * 32, GOOD_OUTPUT)  # wrong attempt
    store.complete(key, attempt, GOOD_OUTPUT)
    with pytest.raises(DuplicateCompletion):
        store.complete(key, attempt, GOOD_OUTPUT)
    with pytest.raises(InvalidTransition, match="complete"):
        start(store, key)


def test_event_model_rejects_inconsistent_state_fields():
    base = {
        "run_id": "1" * 32,
        "run_record_sha256": "a" * 64,
        "observation_key": "obs_" + "b" * 64,
        "sequence": 2,
        "previous_event_sha256": "c" * 64,
        "at_utc": "2026-11-05T12:00:00.000000Z",
    }
    with pytest.raises(ValidationError):
        AttemptEvent(**base, state="running")  # no attempt, environment or input
    with pytest.raises(ValidationError):
        AttemptEvent(**base, state="terminal_failure", attempt_id="2" * 32, attempt_number=1)
    with pytest.raises(ValidationError):
        AttemptEvent(**base, state="planned")  # no coordinates
    with pytest.raises(ValidationError):
        AttemptEvent(
            **{**base, "sequence": 1}, state="interrupted", attempt_id="2" * 32, attempt_number=1
        )


# --- failure, retry, limits -----------------------------------------------------------------


def test_retryable_failure_then_success_preserves_failed_history(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    first = start(store, key)
    store.fail(key, first, kind="transport_error", detail="connection reset", retryable=True)
    assert store.observation(key).status == "retryable"
    second = start(store, key)
    store.complete(key, second, GOOD_OUTPUT)
    obs = store.observation(key)
    assert obs.status == "complete" and obs.attempts == 2
    assert [e.state for e in obs.events] == [
        "planned",
        "running",
        "retryable_failure",
        "running",
        "complete",
    ]
    failed = obs.events[2]
    assert failed.attempt_id == first and failed.attempt_number == 1
    assert failed.failure_kind == "transport_error" and failed.failure_detail == "connection reset"
    assert obs.events[3].attempt_number == 2


def test_retry_must_use_identical_rendered_input(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    with pytest.raises(InvalidTransition, match="identical"):
        start(store, key, rendered=OTHER_RENDERED)
    start(store, key)


def test_failure_kind_must_be_allowed_by_the_retry_policy(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    with pytest.raises(InvalidTransition, match="retryable"):
        store.fail(key, a, kind="invalid_prompt", detail="x", retryable=True)
    store.fail(key, a, kind="invalid_prompt", detail="x", retryable=False)
    obs = store.observation(key)
    assert obs.status == "terminal_failure"
    with pytest.raises(InvalidTransition):
        start(store, key)


def test_retry_limit_comes_from_configuration(tmp_path):
    store, config = new_store(tmp_path)
    assert config.settings.retry_max_attempts == 3 == store.retry_policy.max_attempts
    key = store.plan(coordinates())
    for _ in range(3):
        a = start(store, key)
        store.fail(key, a, kind="out_of_memory", detail="x", retryable=True)
    obs = store.observation(key)
    assert obs.status == "retries_exhausted" and obs.attempts == 3
    assert obs.events[-1].state == "retryable_failure"  # the record is not rewritten
    with pytest.raises(InvalidTransition, match="limit"):
        start(store, key)
    assert key not in store.pending()


def test_retry_policy_must_match_configuration_limit(tmp_path):
    config = config_for("mock")
    with pytest.raises(RunMismatch, match="retry"):
        RunStore.create(
            tmp_path,
            config=config,
            phase="development",
            benchmark_version="b",
            backend_kind="mock",
            retry_policy=RetryPolicy(max_attempts=2, retryable_failures=[]),
        )


def test_retry_policy_validation():
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0, retryable_failures=[])
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=1, retryable_failures=["a", "a"])
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=1, retryable_failures=[" "])


def test_failed_attempt_holds_no_output(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    failed = store.observation(key).events[-1]
    assert failed.raw_output is None and failed.parse_status is None
    assert not (store.run_dir / "observations" / key / "raw").exists()


# --- interruption and resume ---------------------------------------------------------------


def test_running_attempt_is_detected_as_interrupted_on_resume(tmp_path):
    store, config = new_store(tmp_path)
    done_key = store.plan(coordinates(0))
    finish(store, done_key)
    crashed_key = store.plan(coordinates(1))
    crashed = start(store, crashed_key)
    untouched_key = store.plan(coordinates(2))

    resumed = RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")
    assert resumed.run_id == store.run_id
    assert resumed.observation(crashed_key).status == "running"
    assert resumed.pending() == [untouched_key]  # a running attempt is not silently restarted
    assert resumed.recover_interrupted(previous_worker_stopped=True) == [crashed_key]
    obs = resumed.observation(crashed_key)
    assert obs.status == "retryable" and obs.events[-1].state == "interrupted"
    assert obs.events[-1].attempt_id == crashed
    assert set(resumed.pending()) == {crashed_key, untouched_key}


def test_recovery_requires_explicit_confirmation_that_no_worker_is_active(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    start(store, key)
    with pytest.raises(UnsafeResume):
        store.recover_interrupted(previous_worker_stopped=False)
    assert store.observation(key).status == "running"


def test_interrupted_attempt_can_be_retried_and_counts_toward_the_limit(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    for _ in range(2):
        start(store, key)
        store.recover_interrupted(previous_worker_stopped=True)
    finish(store, key)
    obs = store.observation(key)
    assert obs.status == "complete" and obs.attempts == 3
    start_events = [e for e in obs.events if e.state == "running"]
    assert [e.attempt_number for e in start_events] == [1, 2, 3]
    with pytest.raises(InvalidTransition):
        start(store, key)


def test_late_completion_of_an_interrupted_attempt_is_refused(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    attempt = start(store, key)
    store.recover_interrupted(previous_worker_stopped=True)
    with pytest.raises(InvalidTransition):
        store.complete(key, attempt, GOOD_OUTPUT)


def test_resume_never_regenerates_completed_observations(tmp_path):
    store, config = new_store(tmp_path)
    keys = [store.plan(coordinates(i)) for i in range(4)]
    for key, (_, _, _, raw) in zip(keys[:2], PAPERS[:2], strict=True):
        finish(store, key, raw)
    before = {k: [p.read_bytes() for p in event_files(store, k)] for k in keys[:2]}
    resumed = RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")
    assert resumed.pending() == keys[2:]
    for key in keys[:2]:
        with pytest.raises(InvalidTransition):
            start(resumed, key)
        assert [p.read_bytes() for p in event_files(resumed, key)] == before[key]


def test_resume_refuses_a_different_configuration_backend_or_unknown_run(tmp_path):
    store, config = new_store(tmp_path)
    other = config_for("mock", scoring_version="other-scoring")
    with pytest.raises(RunMismatch, match="configuration"):
        RunStore.open(tmp_path, store.run_id, config=other, backend_kind="mock")
    with pytest.raises(RunMismatch, match="backend"):
        RunStore.open(tmp_path, store.run_id, config=config, backend_kind="smoke")
    with pytest.raises(LifecycleError, match="no such run"):
        RunStore.open(tmp_path, "9" * 32, config=config, backend_kind="mock")


def test_resumed_run_keeps_phase_benchmark_and_policy(tmp_path):
    store, config = new_store(tmp_path)
    resumed = RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")
    assert resumed.record == store.record
    assert resumed.retry_policy == POLICY and resumed.record.phase == "development"


def test_only_development_runs_are_supported(tmp_path):
    for phase in ("final_test", "qualification"):
        with pytest.raises(ValidationError):
            RunRecord.model_validate({**_run_dict(tmp_path), "phase": phase})


def _run_dict(tmp_path):
    store, _ = new_store(tmp_path)
    return json.loads((store.run_dir / "run.json").read_text())


# --- publication, atomicity, conflicts ---------------------------------------------------


def test_published_records_are_not_clobbered_by_a_racing_writer(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    stale = store._load(key)  # a second worker has read the same state
    rival = RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")
    start(rival, key)
    winner = [p.read_bytes() for p in event_files(store, key)]
    store._load = lambda _key: stale  # type: ignore[method-assign]
    with pytest.raises(PublicationConflict):
        start(store, key)
    assert [p.read_bytes() for p in event_files(store, key)] == winner
    assert len(winner) == 2


def test_interrupted_write_leaves_no_partial_event_and_can_be_repeated(tmp_path, monkeypatch):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    real_link = os.link

    def crash(*args, **kwargs):
        raise OSError("simulated crash before the record became visible")

    monkeypatch.setattr(os, "link", crash)
    with pytest.raises(OSError, match="simulated"):
        start(store, key)
    monkeypatch.setattr(os, "link", real_link)
    events = store.run_dir / "observations" / key / "events"
    assert [p.name for p in events.iterdir()] == ["000001.json"]
    assert store.observation(key).status == "planned"
    finish(store, key)
    assert store.observation(key).status == "complete"


def test_leftover_temporary_files_are_ignored(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    events = store.run_dir / "observations" / key / "events"
    (events / ".000002.json.abcd.tmp").write_bytes(b'{"partial":')
    assert store.observation(key).status == "planned"
    finish(store, key)


def test_crash_between_raw_output_and_completion_event_does_not_complete(tmp_path, monkeypatch):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    attempt = start(store, key)
    real = store._append

    def crash(*args, **kwargs):
        raise OSError("simulated crash after raw output was written")

    monkeypatch.setattr(store, "_append", crash)
    with pytest.raises(OSError):
        store.complete(key, attempt, GOOD_OUTPUT)
    monkeypatch.setattr(store, "_append", real)
    assert (store.run_dir / "observations" / key / "raw" / f"{attempt}.txt").exists()
    resumed = RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")
    assert resumed.observation(key).status == "running"
    resumed.recover_interrupted(previous_worker_stopped=True)
    assert resumed.observation(key).status == "retryable"
    assert resumed.pending() == [key]


# --- corruption ------------------------------------------------------------------------------


def completed_store(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    finish(store, key)
    return store, config, key


def reopen(store, config):
    return RunStore.open(store.root, store.run_id, config=config, backend_kind="mock")


def test_modified_middle_event_breaks_the_chain(tmp_path):
    store, config, key = completed_store(tmp_path)
    path = event_files(store, key)[2]
    data = json.loads(path.read_text())
    data["failure_detail"] = "edited"
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(CorruptRecord, match="chain"):
        reopen(store, config)


def test_missing_middle_event_is_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    event_files(store, key)[1].unlink()
    with pytest.raises(CorruptRecord, match="sequence"):
        reopen(store, config)


def test_garbage_and_unexpected_files_are_corruption(tmp_path):
    store, config, key = completed_store(tmp_path)
    event_files(store, key)[-1].write_bytes(b"not json")
    with pytest.raises(CorruptRecord):
        reopen(store, config)
    shutil.rmtree(store.root)
    store, config, key = completed_store(tmp_path)
    (event_files(store, key)[0].parent / "notes.txt").write_text("x")
    with pytest.raises(CorruptRecord, match="unexpected"):
        reopen(store, config)


def test_raw_output_hash_mismatch_is_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    done = store.observation(key).events[-1]
    (store.root / done.raw_output.logical_ref).write_text(GOOD_OUTPUT.replace("42.5", "42.6"))
    with pytest.raises(CorruptRecord, match="missing or altered"):
        reopen(store, config)


def test_missing_raw_output_is_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    done = store.observation(key).events[-1]
    (store.root / done.raw_output.logical_ref).unlink()
    with pytest.raises(CorruptRecord, match="missing or altered"):
        reopen(store, config)


def test_parsed_fields_that_disagree_with_the_raw_output_are_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    path = event_files(store, key)[-1]
    data = json.loads(path.read_text())
    data["answer"] = "forged"
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(CorruptRecord, match="parsed"):
        reopen(store, config)


def test_changed_run_record_or_configuration_file_is_detected(tmp_path):
    store, config, _ = completed_store(tmp_path)
    run_path = store.run_dir / "run.json"
    original = run_path.read_bytes()
    data = json.loads(original)
    data["benchmark_version"] = "edited"
    run_path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(CorruptRecord):
        reopen(store, config)
    run_path.write_bytes(original)
    (store.run_dir / "config.canonical.json").write_text("{}")
    with pytest.raises(CorruptRecord, match="configuration"):
        reopen(store, config)


def test_changed_environment_record_is_detected(tmp_path):
    store, config, _ = completed_store(tmp_path)
    env = min((store.run_dir / "environments").glob("*.json"))
    env.write_text(env.read_text().replace("synthetic-host", "elsewhere"))
    with pytest.raises(CorruptRecord, match="environment"):
        reopen(store, config)


def test_event_for_another_observation_or_run_is_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    other = store.plan(coordinates(1))
    stray = event_files(store, other)[0]
    target = event_files(store, key)[0]
    target.write_bytes(stray.read_bytes())
    with pytest.raises(CorruptRecord):
        reopen(store, config)


def forge_after_last(store, key, **changes):
    files = event_files(store, key)
    last = AttemptEvent.model_validate_json(files[-1].read_text())
    forged = last.model_copy(
        update={
            "sequence": last.sequence + 1,
            "previous_event_sha256": sha256_bytes(files[-1].read_bytes()),
            **changes,
        }
    )
    files[-1].with_name(f"{forged.sequence:06d}.json").write_bytes(
        serialize_project_json(forged, AttemptEvent)
    )


def test_duplicate_completion_in_one_observation_is_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    forge_after_last(store, key)
    with pytest.raises(DuplicateCompletion):
        reopen(store, config)


def test_events_after_a_terminal_state_or_with_illegal_transitions_are_detected(tmp_path):
    store, config, key = completed_store(tmp_path)
    forge_after_last(
        store,
        key,
        state="interrupted",
        raw_output=None,
        parse_status=None,
        answer=None,
        abstained=None,
        parse_reason=None,
    )
    with pytest.raises(CorruptRecord, match="transition"):
        reopen(store, config)


def test_attempt_number_beyond_the_limit_is_detected(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    for _ in range(3):
        a = start(store, key)
        store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    forge_after_last(
        store,
        key,
        state="running",
        attempt_id="e" * 32,
        attempt_number=4,
        failure_kind=None,
        failure_detail=None,
        backend_kind="mock",
        environment_sha256=store.observation(key).events[1].environment_sha256,
        rendered_input_sha256=RENDERED,
    )
    with pytest.raises(CorruptRecord, match="limit"):
        reopen(store, config)


def test_same_observation_completed_in_two_runs_is_a_conflict(tmp_path):
    first, _ = new_store(tmp_path)
    second, _ = new_store(tmp_path)
    for store in (first, second):
        finish(store, store.plan(coordinates()))
    with pytest.raises(DuplicateCompletion):
        check_no_duplicate_completions(tmp_path)


def test_distinct_observations_in_separate_runs_are_not_conflicts(tmp_path):
    first, _ = new_store(tmp_path)
    second, _ = new_store(tmp_path)
    finish(first, first.plan(coordinates(0)))
    finish(second, second.plan(coordinates(1)))
    check_no_duplicate_completions(tmp_path)


# --- official, smoke and mock backends ---------------------------------------------------------


def test_backend_and_purpose_must_agree(tmp_path):
    cases = [
        ("official", "synthetic_test"),
        ("mock", "development_candidate"),
    ]
    for kind, purpose in cases:
        config = resolve({**synthetic_settings(), "purpose": purpose, "retry_max_attempts": 3})
        with pytest.raises(LifecycleError, match="backend"):
            RunStore.create(
                tmp_path,
                config=config,
                phase="development",
                benchmark_version="b",
                backend_kind=kind,
                retry_policy=POLICY,
            )


def test_smoke_runs_may_name_a_candidate_configuration_but_are_never_official(tmp_path):
    config = config_for("official")
    store = RunStore.create(
        tmp_path,
        config=config,
        phase="development",
        benchmark_version="b",
        backend_kind="smoke",
        retry_policy=POLICY,
    )
    finish(store, store.plan(coordinates()), kind="smoke")
    assert not store.record.official
    with pytest.raises(NonOfficialBackend):
        official_completions(store)


@pytest.mark.parametrize("kind", ["mock", "smoke"])
def test_non_official_completions_are_never_accepted(tmp_path, kind):
    store, _ = new_store(tmp_path, kind=kind)
    finish(store, store.plan(coordinates()), kind=kind)
    with pytest.raises(NonOfficialBackend, match="official"):
        official_completions(store)


def test_official_run_yields_only_complete_observations(tmp_path):
    store, _ = new_store(tmp_path, kind="official")
    done = store.plan(coordinates(0))
    finish(store, done, kind="official")
    store.plan(coordinates(1))
    failed = store.plan(coordinates(2))
    a = start(store, failed, "official")
    store.fail(failed, a, kind="invalid_prompt", detail="x", retryable=False)
    assert store.record.official
    assert [o.key for o in official_completions(store)] == [done]


def test_environment_backend_must_match_the_run(tmp_path):
    store, _ = new_store(tmp_path, kind="mock")
    key = store.plan(coordinates())
    with pytest.raises(NonOfficialBackend, match="backend"):
        store.begin_attempt(
            key, environment=environment("official"), rendered_input_sha256=RENDERED
        )
    official, _ = new_store(tmp_path / "o", kind="official")
    key = official.plan(coordinates())
    with pytest.raises(NonOfficialBackend, match="backend"):
        official.begin_attempt(key, environment=environment("mock"), rendered_input_sha256=RENDERED)


def test_official_environment_must_load_the_configured_model_and_tokenizer(tmp_path):
    store, _ = new_store(tmp_path, kind="official")
    key = store.plan(coordinates())
    wrong = environment("official", model=("Qwen/Qwen3-8B", "0" * 40))
    with pytest.raises(NonOfficialBackend, match="model"):
        store.begin_attempt(key, environment=wrong, rendered_input_sha256=RENDERED)


def test_relabeling_a_mock_run_as_official_on_disk_is_detected(tmp_path):
    store, config = new_store(tmp_path, kind="mock")
    finish(store, store.plan(coordinates()))
    path = store.run_dir / "run.json"
    data = json.loads(path.read_text())
    data["backend_kind"] = "official"
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(LifecycleError):
        RunStore.open(tmp_path, store.run_id, config=config, backend_kind="official")


def test_run_record_rejects_inconsistent_official_claims():
    base = {
        "run_id": "1" * 32,
        "created_utc": "2026-11-05T12:00:00.000000Z",
        "phase": "development",
        "benchmark_version": "b",
        "resolved_config_sha256": "a" * 64,
        "raw_config_sha256": "b" * 64,
        "matrix_arm": "confirmatory",
        "retry": {"max_attempts": 1, "retryable_failures": []},
    }
    with pytest.raises(ValidationError):
        RunRecord(**base, backend_kind="official", purpose="synthetic_test")
    with pytest.raises(ValidationError):
        RunRecord(**base, backend_kind="mock", purpose="development_candidate")
    assert RunRecord(**base, backend_kind="official", purpose="development_candidate").official


# --- retry policy comes from the hashed configuration ------------------------------------------


def test_run_retry_policy_is_derived_from_the_configuration(tmp_path):
    store, config = new_store(tmp_path)
    assert store.retry_policy == POLICY
    assert store.record.retry.retryable_failures == config.settings.retryable_failures


def test_explicit_retry_policy_that_contradicts_the_configuration_is_refused(tmp_path):
    config = config_for("mock")
    for policy in (
        RetryPolicy(max_attempts=3, retryable_failures=["transport_error"]),
        RetryPolicy(max_attempts=2, retryable_failures=list(POLICY.retryable_failures)),
    ):
        with pytest.raises(RunMismatch, match="retry"):
            RunStore.create(
                tmp_path,
                config=config,
                phase="development",
                benchmark_version="b",
                backend_kind="mock",
                retry_policy=policy,
            )
    same = RunStore.create(
        tmp_path,
        config=config,
        phase="development",
        benchmark_version="b",
        backend_kind="mock",
        retry_policy=POLICY,
    )
    assert same.retry_policy == POLICY


def test_different_retry_categories_mean_a_different_run_identity(tmp_path):
    store, config = new_store(tmp_path)
    other = config_for("mock", retryable_failures=["transport_error"])
    assert other.configuration_sha256 != config.configuration_sha256
    with pytest.raises(RunMismatch, match="configuration"):
        RunStore.open(tmp_path, store.run_id, config=other, backend_kind="mock")
    first = store.plan(coordinates())
    second_store, _ = new_store(tmp_path / "other", retryable_failures=["transport_error"])
    assert second_store.plan(coordinates()) != first  # the observation key carries the identity


def test_run_record_retry_edited_on_disk_cannot_contradict_the_configuration(tmp_path):
    store, config = new_store(tmp_path)
    path = store.run_dir / "run.json"
    data = json.loads(path.read_text())
    data["retry"]["retryable_failures"] = ["invalid_prompt"]
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
    with pytest.raises(RunMismatch, match="retry"):
        reopen(store, config)


def test_failure_categories_follow_the_configuration(tmp_path):
    store, _ = new_store(tmp_path, retryable_failures=["timeout"])
    key = store.plan(coordinates())
    a = start(store, key)
    with pytest.raises(InvalidTransition, match="retryable"):
        store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    store.fail(key, a, kind="timeout", detail="x", retryable=True)


def test_a_planned_observation_consumes_no_attempt(tmp_path):
    store, _ = new_store(tmp_path)
    keys = [store.plan(coordinates(i)) for i in range(4)]
    assert all(store.observation(k).attempts == 0 for k in keys)
    start(store, keys[0])
    assert [store.observation(k).attempts for k in keys] == [1, 0, 0, 0]


# --- terminal-event integrity (seal) ---------------------------------------------------------------


def seal_path(store, key):
    return store.run_dir / "observations" / key / "final.build.json"


def test_final_states_are_sealed_with_an_existing_build_manifest(tmp_path):
    from biorag_repro.core.manifest import BuildManifest

    store, _ = new_store(tmp_path)
    done = store.plan(coordinates(0))
    finish(store, done)
    failed = store.plan(coordinates(1))
    a = start(store, failed)
    store.fail(failed, a, kind="invalid_prompt", detail="x", retryable=False)
    for key in (done, failed):
        manifest = BuildManifest.model_validate_json(seal_path(store, key).read_text())
        final = event_files(store, key)[-1]
        assert manifest.status == "complete"
        assert manifest.outputs[0].artifact.sha256 == sha256_bytes(final.read_bytes())
        assert manifest.outputs[0].artifact.logical_ref.endswith(final.name)


def test_exhausted_retries_are_sealed_and_open_observations_are_not(tmp_path):
    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    for _ in range(3):
        a = start(store, key)
        assert not seal_path(store, key).exists()
        store.fail(key, a, kind="out_of_memory", detail="x", retryable=True)
    assert seal_path(store, key).exists()


@pytest.mark.parametrize("final", ["complete", "terminal_failure"])
def test_removing_a_sealed_final_event_is_detected(tmp_path, final):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    if final == "complete":
        store.complete(key, a, GOOD_OUTPUT)
    else:
        store.fail(key, a, kind="invalid_prompt", detail="x", retryable=False)
    event_files(store, key)[-1].unlink()  # the observation now looks as if it were still running
    with pytest.raises(CorruptRecord, match="seal"):
        reopen(store, config)


def test_replacing_a_sealed_final_event_is_detected(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.complete(key, a, GOOD_OUTPUT)
    last = event_files(store, key)[-1]
    forged = json.loads(last.read_text())
    forged["abstained"] = True
    forged["answer"] = None
    last.write_text(json.dumps(forged, sort_keys=True, indent=2) + "\n")
    with pytest.raises(CorruptRecord):
        reopen(store, config)


def test_altered_or_foreign_seal_is_detected(tmp_path):
    store, config = new_store(tmp_path)
    key, other = store.plan(coordinates(0)), store.plan(coordinates(1))
    finish(store, key)
    finish(store, other)
    original = seal_path(store, key).read_bytes()
    seal_path(store, key).write_text("{}")
    with pytest.raises(CorruptRecord, match="seal"):
        reopen(store, config)
    seal_path(store, key).write_bytes(seal_path(store, other).read_bytes())
    with pytest.raises(CorruptRecord, match="seal"):
        reopen(store, config)
    seal_path(store, key).write_bytes(original)
    reopen(store, config)


def test_a_seal_on_an_unfinished_observation_is_detected(tmp_path):
    store, config = new_store(tmp_path)
    done = store.plan(coordinates(0))
    finish(store, done)
    other = store.plan(coordinates(1))
    start(store, other)
    seal_path(store, other).write_bytes(seal_path(store, done).read_bytes())
    with pytest.raises(CorruptRecord, match="seal"):
        reopen(store, config)


def test_a_missing_seal_on_a_valid_final_log_is_completed_on_resume(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    finish(store, key)  # crash between the final event and its seal looks like this
    before = seal_path(store, key).read_bytes()
    seal_path(store, key).unlink()
    reopen(store, config)
    assert seal_path(store, key).read_bytes() == before


def test_known_limit_newest_unsealed_event_can_be_truncated_undetected(tmp_path):
    """Characterization, not a guarantee: no later record mentions an open log's newest event."""
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    event_files(store, key)[-1].unlink()
    resumed = reopen(store, config)  # no error: the log is a valid shorter log
    assert resumed.observation(key).status == "running"
    assert resumed.observation(key).attempts == 1  # the attempt still counts toward the limit


def test_a_wellformed_seal_for_a_still_open_observation_is_detected(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    a = start(store, key)
    store.fail(key, a, kind="transport_error", detail="x", retryable=True)
    store._seal(store.observation(key))  # a seal that matches the newest event but is premature
    with pytest.raises(CorruptRecord, match="seal"):
        reopen(store, config)


# --- findings of the independent review of the execution boundary ---------------------------


def test_an_observation_that_lost_its_first_event_is_corrupt_not_absent(tmp_path):
    store, config = new_store(tmp_path)
    key = store.plan(coordinates())
    finish(store, key)
    event_files(store, key)[0].unlink()
    with pytest.raises(CorruptRecord):
        RunStore.open(tmp_path, store.run_id, config=config, backend_kind="mock")


def test_a_clock_that_moved_backwards_writes_nothing(tmp_path):
    clock = Clock()
    store, _ = new_store(tmp_path, clock=clock)
    key = store.plan(coordinates())
    attempt = start(store, key)
    clock.value -= timedelta(hours=1)
    with pytest.raises(LifecycleError, match="clock"):
        store.complete(key, attempt, GOOD_OUTPUT)
    assert len(event_files(store, key)) == 2  # planned, running: no unreadable event is left
    assert store.observation(key).status == "running"


def test_an_attempt_ended_by_interruption_gets_a_failure_reason(tmp_path):
    from biorag_repro.generation.runner import decided_results

    store, _ = new_store(tmp_path)
    key = store.plan(coordinates())
    for _ in range(3):
        start(store, key)
        store.recover_interrupted(previous_worker_stopped=True)
    assert store.observation(key).status == "retries_exhausted"
    (result,) = decided_results(store)
    assert result.failure is not None and "None" not in result.failure
    assert "interrupted" in result.failure
