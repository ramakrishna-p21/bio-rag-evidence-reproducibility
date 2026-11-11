"""Execution measurements carried by a receipt.

Clocks and CUDA calls are injected; no model runs and no GPU is used. A measurement that was
not taken stays explicitly unavailable.
"""

from types import SimpleNamespace

import pytest
from generation_support import needs_tokenizer
from pydantic import ValidationError
from test_generation_hf_backend import (  # noqa: F401  (fixture)
    NEW_TOKENS,
    backend_for,
    rendered_for,
    renderer,
)

from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation import hf_backend
from biorag_repro.generation.receipt import (
    CudaMemoryMeasured,
    CudaMemoryUnavailable,
    ExecutionFacts,
    ExecutionMeasurements,
)

MEASURED = dict(  # noqa: C408
    device_index=0,
    allocated_before_bytes=100,
    peak_allocated_bytes=300,
    peak_reserved_bytes=400,
    device_total_bytes=1000,
    device_used_after_bytes=500,
)


def test_a_measurement_states_its_clock_and_a_nonnegative_duration():
    measured = ExecutionMeasurements(
        clock="perf_counter_ns",
        generation_ns=750,
        cuda_memory=CudaMemoryUnavailable(reason="the device is cpu"),
    )
    assert measured.generation_ns == 750
    with pytest.raises(ValidationError):
        ExecutionMeasurements(
            clock="perf_counter_ns",
            generation_ns=-1,
            cuda_memory=CudaMemoryUnavailable(reason="x"),
        )
    with pytest.raises(ValidationError):
        ExecutionMeasurements(
            clock="time.time",
            generation_ns=1,
            cuda_memory=CudaMemoryUnavailable(reason="x"),
        )


def test_an_unavailable_memory_reading_has_a_reason_and_no_value():
    with pytest.raises(ValidationError):
        CudaMemoryUnavailable(reason=" ")
    with pytest.raises(ValidationError):
        CudaMemoryUnavailable(reason="x", peak_allocated_bytes=1)


@pytest.mark.parametrize(
    "change",
    [
        {"peak_allocated_bytes": 50},  # a peak below what was already allocated
        {"peak_reserved_bytes": 200},  # reserved memory cannot be below the allocated peak
        {"device_used_after_bytes": 1001},  # more used than the device has
        {"device_total_bytes": 0},
    ],
)
def test_physically_impossible_memory_readings_are_refused(change):
    with pytest.raises(ValidationError):
        CudaMemoryMeasured(**(MEASURED | change))


def test_measurements_are_optional_on_facts_and_leave_old_bytes_alone():
    assert ExecutionFacts.model_fields["measurements"].default is None


# ---- the timed interval ---------------------------------------------------------------------


def test_the_device_is_synchronized_before_the_timer_starts_and_after_the_call_returns():
    log = []
    ticks = iter([1_000, 1_750])

    def clock():
        log.append("clock")
        return next(ticks)

    result, elapsed = hf_backend.timed_call(
        lambda: log.append("generate") or "output", clock, lambda: log.append("sync")
    )
    assert (result, elapsed) == ("output", 750)
    assert log == ["sync", "clock", "generate", "sync", "clock"]


def test_a_failing_call_gives_no_measurement():
    def boom():
        raise RuntimeError("x")

    with pytest.raises(RuntimeError):
        hf_backend.timed_call(boom, lambda: 0, lambda: None)


# ---- CUDA memory ----------------------------------------------------------------------------


def fake_torch(*, free=500, total=1000, allocated=100, peak=300, reserved=400, log=None):
    log = log if log is not None else []
    cuda = SimpleNamespace(
        memory_allocated=lambda d: log.append(("allocated", d)) or allocated,
        reset_peak_memory_stats=lambda d: log.append(("reset", d)),
        max_memory_allocated=lambda d: log.append(("peak_allocated", d)) or peak,
        max_memory_reserved=lambda d: log.append(("peak_reserved", d)) or reserved,
        mem_get_info=lambda d: log.append(("mem_get_info", d)) or (free, total),
        synchronize=lambda d: log.append(("sync", d)),
        current_device=lambda: 0,
    )
    return SimpleNamespace(cuda=cuda)


def test_process_peaks_and_the_device_wide_snapshot_are_kept_apart():
    log = []
    torch_like = fake_torch(log=log)
    before = hf_backend.cuda_allocated_before(torch_like, 1)
    memory = hf_backend.cuda_memory_after(torch_like, 1, before)
    assert before == 100
    assert log[:2] == [("allocated", 1), ("reset", 1)]  # the baseline precedes the reset
    assert memory == CudaMemoryMeasured(
        **(MEASURED | {"device_index": 1, "device_used_after_bytes": 500})
    )
    # device-wide used memory is total minus free: it includes other processes on the GPU
    assert memory.device_used_after_bytes == 1000 - 500
    assert memory.peak_allocated_bytes != memory.device_used_after_bytes


def test_a_cpu_device_never_gets_a_memory_value():
    reading = hf_backend.cpu_memory_reading("cpu")
    assert isinstance(reading, CudaMemoryUnavailable)
    assert "cpu" in reading.reason


# ruff: noqa: F811
# ---- the backend ----------------------------------------------------------------------------


@needs_tokenizer
def test_the_backend_reports_the_duration_between_its_clock_readings(renderer, tiny):
    ticks = iter([5_000, 5_750])
    backend = backend_for(renderer, tiny, clock=lambda: next(ticks))
    output = backend.generate(rendered_for(renderer), seed=None, max_new_tokens=NEW_TOKENS)
    measurements = output.execution.measurements
    assert measurements.clock == "perf_counter_ns"
    assert measurements.generation_ns == 750
    assert isinstance(measurements.cuda_memory, CudaMemoryUnavailable)
    assert b"measurements" in serialize_project_json(output.execution, ExecutionFacts)


@needs_tokenizer
def test_the_default_clock_is_monotonic_and_nonnegative(renderer, tiny):
    backend = backend_for(renderer, tiny)
    output = backend.generate(rendered_for(renderer), seed=None, max_new_tokens=NEW_TOKENS)
    assert output.execution.measurements.generation_ns >= 0


# ---- the CUDA branch of generate, driven with injected torch.cuda functions ---------------


@needs_tokenizer
def test_memory_is_read_straight_after_generate_and_before_the_output_is_processed(
    renderer, tiny, monkeypatch
):
    import torch

    log = []
    fake = fake_torch(log=log).cuda
    for name in (
        "memory_allocated",
        "reset_peak_memory_stats",
        "max_memory_allocated",
        "max_memory_reserved",
        "mem_get_info",
        "synchronize",
    ):
        monkeypatch.setattr(torch.cuda, name, getattr(fake, name), raising=False)
    monkeypatch.setattr(hf_backend, "_cuda_index", lambda torch_module, device: 0)
    real_decode = hf_backend.decode_output

    def decode(*args, **kwargs):
        log.append(("decode", 0))
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(hf_backend, "decode_output", decode)
    backend = backend_for(renderer, tiny)
    output = backend.generate(rendered_for(renderer), seed=None, max_new_tokens=NEW_TOKENS)
    names = [name for name, _ in log]
    assert names.index("mem_get_info") < names.index("decode")
    assert names.count("sync") == 2  # before the clock starts and before it is read
    assert names.index("sync") < names.index("peak_allocated")
    assert output.execution.measurements.cuda_memory.status == "measured"
