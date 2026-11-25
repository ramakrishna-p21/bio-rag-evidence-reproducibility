"""Environment capture with injected probes. No hardware is required."""

import pytest
from generation_support import QWEN_REVISION
from pydantic import ValidationError

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.environment import (
    BackendIdentity,
    EnvironmentProbes,
    EnvironmentRecord,
    GitState,
    GpuReading,
    capture_environment,
    environment_sha256,
)

NOW = "2026-11-05T20:00:00Z"
MODEL = ("Qwen/Qwen3-8B", QWEN_REVISION)


def cpu_probes(**changes):
    values = {
        "python_version": lambda: ("3.11.9", "CPython"),
        "package_version": lambda name: {"pydantic": "2.13.5"}.get(name),
        "operating_system": lambda: ("Linux", "6.1.0", "x86_64"),
        "hostname": lambda: "synthetic-host",
        "cpu_count": lambda: 8,
        "ram_total_mib": lambda: 32000,
        "gpus": lambda: GpuReading(status="no_nvidia_smi", devices=[]),
        "cuda_runtime": lambda: None,
        "git": lambda: GitState(status="ok", commit="1" * 40, dirty=False, dirty_paths=0),
    }
    values.update(changes)
    return EnvironmentProbes(**values)


def capture(probes=None, kind="mock", packages=("pydantic", "absent-package")):
    return capture_environment(
        backend=BackendIdentity(kind=kind, name="synthetic", version=None),
        model=MODEL,
        tokenizer=MODEL,
        packages=packages,
        probes=probes or cpu_probes(),
        captured_utc=NOW,
    )


def test_cpu_environment_records_absence_of_gpu_explicitly():
    record = capture()
    assert record.gpu_status == "no_nvidia_smi"
    assert record.gpus == []
    assert record.cuda_runtime is None
    assert record.python_version == "3.11.9"
    assert record.operating_system == "Linux"
    assert record.cpu_count == 8


def test_gpu_environment_records_identity_memory_driver_and_cuda():
    gpus = GpuReading(
        status="ok",
        devices=[
            {"index": 0, "name": "RTX 3090", "memory_total_mib": 24576, "driver_version": "550.1"},
            {"index": 1, "name": "RTX 3090", "memory_total_mib": 24576, "driver_version": "550.1"},
        ],
    )
    record = capture(cpu_probes(gpus=lambda: gpus, cuda_runtime=lambda: "12.4"))
    assert record.gpu_status == "ok"
    assert [g.index for g in record.gpus] == [0, 1]
    assert record.gpus[0].memory_total_mib == 24576
    assert record.cuda_runtime == "12.4"


def test_failed_gpu_query_is_not_reported_as_no_gpu():
    record = capture(cpu_probes(gpus=lambda: GpuReading(status="failed", devices=[])))
    assert record.gpu_status == "failed"


def test_ok_status_without_devices_is_rejected():
    with pytest.raises(ValidationError):
        GpuReading(status="ok", devices=[])
    with pytest.raises(ValidationError):
        GpuReading(
            status="failed",
            devices=[{"index": 0, "name": "x", "memory_total_mib": 1, "driver_version": "1"}],
        )


def test_missing_package_is_recorded_as_missing_not_dropped():
    record = capture()
    assert record.packages == {"pydantic": "2.13.5", "absent-package": None}


def test_git_states_stay_distinct():
    dirty = GitState(status="ok", commit="2" * 40, dirty=True, dirty_paths=3)
    assert capture(cpu_probes(git=lambda: dirty)).git.dirty is True
    not_repo = GitState(status="not_a_repository", commit=None, dirty=None, dirty_paths=None)
    assert capture(cpu_probes(git=lambda: not_repo)).git.status == "not_a_repository"
    with pytest.raises(ValidationError):
        GitState(status="ok", commit=None, dirty=None, dirty_paths=None)
    with pytest.raises(ValidationError):
        GitState(status="unavailable", commit="1" * 40, dirty=False, dirty_paths=0)
    with pytest.raises(ValidationError):
        GitState(status="ok", commit="1" * 40, dirty=False, dirty_paths=2)


def test_model_tokenizer_and_backend_come_from_the_caller():
    record = capture(kind="smoke")
    assert record.backend.kind == "smoke"
    assert (record.model_repository, record.model_revision) == MODEL
    assert (record.tokenizer_repository, record.tokenizer_revision) == MODEL


def test_unknown_backend_kind_is_rejected():
    with pytest.raises(ValidationError):
        BackendIdentity(kind="production", name="x", version=None)


@pytest.mark.parametrize(
    "stamp", ["2026-11-05T20:00:00", "2026-11-05T20:00:00+01:00", "not a time"]
)
def test_capture_time_must_be_utc(stamp):
    with pytest.raises(ValidationError):
        EnvironmentRecord.model_validate(
            {**capture().model_dump(mode="json"), "captured_utc": stamp}
        )


def test_hash_is_of_stored_bytes_and_stable():
    record = capture()
    assert environment_sha256(record) == sha256_bytes(
        serialize_project_json(record, EnvironmentRecord)
    )
    assert environment_sha256(record) == environment_sha256(capture())
    assert environment_sha256(record) != environment_sha256(
        capture(cpu_probes(cpu_count=lambda: 16))
    )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        EnvironmentRecord.model_validate({**capture().model_dump(mode="json"), "docker": True})


def test_real_probes_produce_a_valid_record_without_gpu_or_scheduler(tmp_path):
    from biorag_repro.generation.environment import system_probes

    record = capture_environment(
        backend=BackendIdentity(kind="mock", name="synthetic", version=None),
        model=MODEL,
        tokenizer=MODEL,
        packages=("pydantic", "no-such-distribution"),
        probes=system_probes(tmp_path),
    )
    assert record.packages["no-such-distribution"] is None
    assert record.packages["pydantic"]
    assert record.git.status == "not_a_repository"  # tmp_path is not inside a repository
    assert record.captured_utc.endswith("Z")
