"""Qualification report over stored generation records. Everything is synthetic.

Mock and smoke runs are engineering evidence of the reporting code; none of it says anything
about Qwen3-8B, a GPU, or whether a Phase 7 gate is met.
"""

# ruff: noqa: F811  (fixtures imported from the official-run tests)

import json

import pytest
from pydantic import ValidationError
from synthetic_environment import LOCK
from test_generation_completion import finished_run, read_back, record_for
from test_generation_official import (  # noqa: F401  (recognised, view are fixtures)
    CONTEXTS,
    anchored,
    build_world,
    recognised,
    verify,
    view,
)
from test_generation_runner import (
    Crash,
    MockBackend,
    go,
    good,
    make_roster,
    make_store,
    reopen,
)

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import HashMismatchError
from biorag_repro.generation.attempts import CorruptRecord, NonOfficialBackend
from biorag_repro.generation.completion import RunIncomplete, complete_run
from biorag_repro.generation.qualification import (
    QualificationReport,
    official_qualification_report,
    publish_report,
    qualification_report,
    read_report,
    report_sha256,
)
from biorag_repro.generation.receipt import (
    CudaMemoryMeasured,
    CudaMemoryUnavailable,
    ExecutionMeasurements,
)
from biorag_repro.generation.runner import BackendFailure


def report_for(store, config, anchor, kind="mock"):
    return qualification_report(store.root, store.run_id, anchor, config=config, backend_kind=kind)


def anchored_run(tmp_path, script=None, repeats=(1,)):
    store, config, _, record = finished_run(tmp_path, script, repeats)
    return store, config, complete_run(store, record)


# ---- a complete synthetic run ----------------------------------------------------------


def test_complete_run_counts_and_labels(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    report = report_for(store, config, anchor)
    assert report.evidence_class == "synthetic_engineering"
    assert report.scientific_gate == "not_evaluated"
    assert report.run.run_id == store.run_id
    assert report.run.completion_manifest_sha256 == anchor
    assert report.run.resolved_config_sha256 == config.configuration_sha256
    n = report.observations
    assert (n.planned, n.attempted, n.completed, n.missing) == (6, 6, 6, 0)
    assert (n.terminal_failure, n.retries_exhausted) == (0, 0)
    assert report.parse.parsed.model_dump() == {"numerator": 6, "denominator": 6}
    assert report.parse.unparseable.model_dump() == {"numerator": 0, "denominator": 6}
    assert report.parse.abstained.model_dump() == {"numerator": 0, "denominator": 6}
    assert report.environment_verification.status == "not_applicable"


def test_mock_run_has_no_model_measurements_and_says_not_applicable(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    report = report_for(store, config, anchor)
    for summary in (report.input_tokens_received, report.output_tokens):
        assert summary.status == "not_applicable" and summary.n == 0
        assert summary.total is None and summary.reason
    for series in (
        report.generation_duration_ns,
        report.peak_allocated_bytes,
        report.peak_reserved_bytes,
        report.device_used_after_bytes,
    ):
        assert series.status == "not_applicable" and series.n == 0 and series.reason
    assert report.stop_reasons == {}
    assert report.receipts.model_dump() == {"numerator": 0, "denominator": 6}
    assert report.measurements.model_dump() == {"numerator": 0, "denominator": 0}
    assert report.generation_throughput.status == "not_applicable"
    assert report.attempt_throughput.status == "not_applicable"
    assert report.run.phase == "development"


def test_planned_input_tokens_are_derived_from_the_roster(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    planned = report_for(store, config, anchor).planned_input_tokens
    assert (planned.status, planned.n, planned.total) == ("derived", 6, 120)
    assert (planned.minimum, planned.maximum) == (20, 20)


def test_attempt_durations_come_from_event_timestamps(tmp_path):
    # The test clock advances one second per recorded event.
    store, config, anchor = anchored_run(tmp_path)
    duration = report_for(store, config, anchor).attempt_duration_us
    assert duration.status == "derived" and duration.n == 6
    assert (duration.total, duration.minimum, duration.maximum) == (6_000_000, 1_000_000, 1_000_000)
    assert "wall-clock" in duration.reason


# ---- failures, retries, parse states ---------------------------------------------------


def mixed_run(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    script = {
        ("q_a", "JATS"): ["not json at all"],  # unparseable, still a completed attempt
        ("q_b", "JATS"): [good(None, abstained=True)],  # explicit abstention
        ("q_c", "JATS"): [BackendFailure("transport_error", "reset")],  # retry, then answer
        ("q_a", "PDF_DOCLING"): [BackendFailure("invalid_prompt", "bad")],  # terminal
        ("q_b", "PDF_DOCLING"): [BackendFailure("out_of_memory", "oom")] * 3,  # exhausted
    }
    go(store, config, roster, MockBackend(roster, script))
    anchor = complete_run(store, record_for(config, roster))
    return store, config, anchor


def test_failed_attempts_are_counted_apart_from_answers(tmp_path):
    store, config, anchor = mixed_run(tmp_path)
    report = report_for(store, config, anchor)
    n = report.observations
    assert (n.planned, n.attempted, n.completed, n.missing) == (6, 6, 4, 2)
    assert (n.terminal_failure, n.retries_exhausted) == (1, 1)
    a = report.attempts
    assert a.total == 9  # six first tries, one retry, two more for the exhausted observation
    assert (a.retryable_failure_events, a.terminal_failure_events) == (1 + 3, 1)
    assert a.failure_kinds == {"invalid_prompt": 1, "out_of_memory": 3, "transport_error": 1}
    assert a.observations_with_retries == 2


def test_parse_and_abstention_denominators(tmp_path):
    store, config, anchor = mixed_run(tmp_path)
    parse = report_for(store, config, anchor).parse
    assert parse.parsed.model_dump() == {"numerator": 3, "denominator": 4}
    assert parse.unparseable.model_dump() == {"numerator": 1, "denominator": 4}
    # Abstention is only defined for output that parsed.
    assert parse.abstained.model_dump() == {"numerator": 1, "denominator": 3}


def test_interrupted_attempts_are_counted_and_not_timed(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    with pytest.raises(Crash):
        go(store, config, roster, MockBackend(roster, {("q_b", "JATS"): [Crash()]}))
    resumed = reopen(store, config)
    resumed.recover_interrupted(previous_worker_stopped=True)
    go(resumed, config, roster, MockBackend(roster))
    anchor = complete_run(resumed, record_for(config, roster))
    report = report_for(resumed, config, anchor)
    assert report.attempts.interrupted_events == 1
    assert report.attempts.total == 4
    assert report.attempt_duration_us.n == 3  # the interrupted attempt has no trustworthy end
    assert report.attempts.untimed_attempts == 1
    assert "wall-clock" in report.attempt_duration_us.reason


# ---- identity, determinism, integrity --------------------------------------------------


def test_report_is_deterministic_and_round_trips(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    first = report_for(store, config, anchor)
    again = report_for(store, config, anchor)
    assert first == again and report_sha256(first) == report_sha256(again)
    path = tmp_path / "out" / "qualification.json"
    digest = publish_report(path, first)
    assert digest == report_sha256(first)
    assert read_report(path, digest) == first
    assert json.loads(path.read_text())["scientific_gate"] == "not_evaluated"


def test_different_runs_have_different_report_hashes(tmp_path):
    one = anchored_run(tmp_path / "a")
    two = anchored_run(tmp_path / "b", script={("q_a", "JATS"): [good(None, True)]})
    assert report_sha256(report_for(*one)) != report_sha256(report_for(*two))


def test_stored_report_with_altered_bytes_is_refused(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    report = report_for(store, config, anchor)
    path = tmp_path / "report.json"
    digest = publish_report(path, report)
    path.write_bytes(path.read_bytes().replace(b"synthetic_engineering", b"official_development"))
    with pytest.raises(HashMismatchError):
        read_report(path, digest)


def test_tampered_source_record_refuses_a_report(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    event = next(store.run_dir.glob("observations/*/events/000003.json"))
    event.write_bytes(event.read_bytes().replace(b"parsed", b"unparseable"))
    with pytest.raises((CorruptRecord, ValueError)):
        report_for(store, config, anchor)


def test_a_wrong_anchor_refuses_a_report(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    with pytest.raises((CorruptRecord, ValueError, OSError)):
        report_for(store, config, "0" * 64)
    assert read_back(store, config, anchor)  # the run itself is still fine


def test_run_that_is_not_fully_decided_has_no_report(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    with pytest.raises(Crash):
        go(store, config, roster, MockBackend(roster, {("q_b", "JATS"): [Crash()]}))
    with pytest.raises(RunIncomplete):  # nothing to anchor, so nothing to report on
        complete_run(reopen(store, config), record_for(config, roster))


# ---- scientific status cannot be forged by the model -----------------------------------


def test_mock_evidence_cannot_be_labelled_official(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    data = report_for(store, config, anchor).model_dump(mode="json")
    data["evidence_class"] = "official_development"
    with pytest.raises(ValidationError):
        QualificationReport.model_validate(data)


def test_a_report_cannot_declare_a_gate_passed(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    data = report_for(store, config, anchor).model_dump(mode="json")
    data["scientific_gate"] = "passed"
    with pytest.raises(ValidationError):
        QualificationReport.model_validate(data)


def test_no_gpu_memory_value_can_be_invented(tmp_path):
    store, config, anchor = anchored_run(tmp_path)
    data = report_for(store, config, anchor).model_dump(mode="json")
    # A mock run measured nothing: an "observed" series with values is not allowed to appear.
    data["peak_allocated_bytes"] = {
        "status": "unavailable", "reason": "made up", "n": 1,
        "total": 20000, "minimum": 20000, "maximum": 20000,
    }  # fmt: skip
    with pytest.raises(ValidationError):
        QualificationReport.model_validate(data)


# ---- official runs: only through verification ---------------------------------------------


def test_official_report_comes_only_from_a_verified_run(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    with pytest.raises(NonOfficialBackend, match="official"):
        qualification_report(
            world.store.root,
            world.store.run_id,
            anchor,
            config=world.config,
            backend_kind="official",
        )
    report = official_qualification_report(verify(world, anchor))
    assert report.evidence_class == "official_development"
    assert report.scientific_gate == "not_evaluated"
    ref = report.environment_verification
    assert ref.status == "verified" and ref.lock_sha256 == sha256_bytes(LOCK)
    assert ref.verification_sha256 is not None


def test_receipts_give_observed_tokens_stop_reasons_and_derived_throughput(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.receipts.model_dump() == {"numerator": 4, "denominator": 4}
    assert report.stop_reasons == {"eos": 4}
    assert report.output_tokens.status == "observed" and report.output_tokens.n == 4
    assert report.input_tokens_received.status == "observed"
    assert report.input_tokens_received.total == report.planned_input_tokens.total
    assert report.attempt_throughput.status == "derived"
    assert report.attempt_throughput.output_tokens == report.output_tokens.total
    # This scripted backend attached no measurements: nothing is timed or sized, nothing is zero.
    assert report.measurements.model_dump() == {"numerator": 0, "denominator": 4}
    for series in (
        report.generation_duration_ns,
        report.peak_allocated_bytes,
        report.peak_reserved_bytes,
        report.device_used_after_bytes,
    ):
        assert series.status == "unavailable" and series.n == 0 and series.total is None
    assert report.generation_throughput.status == "unavailable"
    env = report.environments[0]
    assert (env.backend_kind, env.device, env.dtype) == ("official", "cuda:0", "bfloat16")
    assert env.checkpoint_manifest_sha256 is not None


def test_failed_official_attempts_reduce_receipts_not_scientific_correctness(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny, fail=[CONTEXTS[("q_a", "PDF_DOCLING")]])
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.observations.missing == 1 and report.observations.retries_exhausted == 0
    assert report.observations.terminal_failure == 1
    assert report.receipts.model_dump() == {"numerator": 3, "denominator": 3}
    assert report.parse.parsed.denominator == 3


def test_official_report_is_deterministic(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    first, second = official_qualification_report(verified), official_qualification_report(verified)
    assert report_sha256(first) == report_sha256(second)


def test_a_closing_event_without_a_start_is_corruption_not_a_duration():
    from biorag_repro.generation.attempts import CorruptRecord
    from biorag_repro.generation.qualification import _attempts

    class Event:
        state = "complete"

    with pytest.raises(CorruptRecord):
        _attempts([Event()])


# ---- recorded measurements ---------------------------------------------------------------


def measuring(memory):
    """Successive calls take 1000, 2000, ... ns; `memory` is a function of the call number."""
    count = iter(range(1, 100))

    def measure():
        n = next(count)
        return ExecutionMeasurements(
            clock="perf_counter_ns", generation_ns=1000 * n, cuda_memory=memory(n)
        )

    return measure


def measured_memory(n):
    return CudaMemoryMeasured(
        device_index=0,
        allocated_before_bytes=100,
        peak_allocated_bytes=1000 * n,
        peak_reserved_bytes=2000 * n,
        device_total_bytes=100_000,
        device_used_after_bytes=5000 * n,
    )


def test_recorded_durations_and_memory_are_summarised_exactly(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, measure=measuring(measured_memory))
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.measurements.model_dump() == {"numerator": 4, "denominator": 4}
    duration = report.generation_duration_ns
    assert (duration.status, duration.n) == ("observed", 4)
    assert (duration.total, duration.minimum, duration.maximum) == (10_000, 1000, 4000)
    assert report.peak_allocated_bytes.maximum == 4000
    assert report.peak_reserved_bytes.total == 20_000
    assert report.device_used_after_bytes.maximum == 20_000
    throughput = report.generation_throughput
    assert throughput.status == "derived"
    assert (throughput.attempts, throughput.output_tokens, throughput.duration_ns) == (
        4,
        report.output_tokens.total,
        10_000,
    )
    # the wall-clock quantity stays separate from the monotonic one
    assert report.attempt_throughput.duration_ns != throughput.duration_ns


def test_a_duration_without_a_memory_reading_leaves_memory_unavailable(
    tmp_path, view, tiny, recognised
):
    cpu = lambda n: CudaMemoryUnavailable(reason="device cpu is not CUDA")
    world = build_world(tmp_path, view, tiny, measure=measuring(cpu))
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.generation_duration_ns.status == "observed"
    for series in (
        report.peak_allocated_bytes,
        report.peak_reserved_bytes,
        report.device_used_after_bytes,
    ):
        assert series.status == "unavailable" and series.n == 0 and series.total is None
        assert "device cpu is not CUDA" in series.reason


def test_a_report_with_measurements_is_deterministic(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, measure=measuring(measured_memory))
    verified = verify(world, anchored(world))
    first, second = official_qualification_report(verified), official_qualification_report(verified)
    assert report_sha256(first) == report_sha256(second)


def test_a_measurement_without_a_memory_reading_is_not_counted_as_a_memory_reading(
    tmp_path, view, tiny, recognised
):
    def cpu(n):
        return CudaMemoryUnavailable(reason="device cpu is not CUDA")

    world = build_world(tmp_path, view, tiny, measure=measuring(cpu))
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.measurements.model_dump() == {"numerator": 4, "denominator": 4}
    assert report.memory_readings.model_dump() == {"numerator": 0, "denominator": 4}
    assert report.memory_device_indices == []


def test_memory_readings_name_every_device_they_came_from(tmp_path, view, tiny, recognised):
    def two_devices(n):
        return measured_memory(n).model_copy(update={"device_index": n % 2})

    world = build_world(tmp_path, view, tiny, measure=measuring(two_devices))
    report = official_qualification_report(verify(world, anchored(world)))
    assert report.memory_readings.model_dump() == {"numerator": 4, "denominator": 4}
    assert report.memory_device_indices == [0, 1]  # pooled series are not hidden as one device
