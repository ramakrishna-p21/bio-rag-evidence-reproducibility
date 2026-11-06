"""Execution environment record for generation attempts.

Every observation about the machine comes from an injected probe, so tests can simulate
CPU-only and GPU hosts. States that mean different things stay different: no GPU tool,
a failed GPU query, a package that is not installed, a directory that is not a Git
repository. Nothing here assumes Docker, a scheduler or a particular filesystem.
"""

import os
import platform
import socket
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import serialize_project_json

BackendKind = Literal["official", "smoke", "mock"]


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def check_utc(value: str) -> str:
    """RFC 3339 UTC with a Z suffix; the project records no local or offset times."""
    if not value.endswith("Z"):
        raise ValueError("timestamp must be UTC with a Z suffix")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError("timestamp is not RFC 3339") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be UTC with a Z suffix")
    return value


class BackendIdentity(Record):
    kind: BackendKind
    name: str = Field(min_length=1)
    version: str | None


class GpuDevice(Record):
    index: int = Field(ge=0)
    name: str = Field(min_length=1)
    memory_total_mib: int = Field(gt=0)
    driver_version: str = Field(min_length=1)


class GpuReading(Record):
    status: Literal["ok", "no_nvidia_smi", "failed"]
    devices: list[GpuDevice]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.status == "ok") != bool(self.devices):
            raise ValueError("devices are listed exactly when the GPU query succeeded")
        return self


class GitState(Record):
    status: Literal["ok", "not_a_repository", "unavailable"]
    commit: str | None
    dirty: bool | None
    dirty_paths: int | None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        known = (self.commit, self.dirty, self.dirty_paths)
        if self.status != "ok":
            if any(value is not None for value in known):
                raise ValueError("Git details are recorded only when Git state was read")
            return self
        if any(value is None for value in known):
            raise ValueError("Git state needs commit, dirty flag and changed-path count")
        assert self.dirty_paths is not None and self.commit is not None
        if len(self.commit) != 40 or self.dirty_paths < 0 or self.dirty != (self.dirty_paths > 0):
            raise ValueError("inconsistent Git state")
        return self


class EnvironmentRecord(Record):
    schema_version: Literal["1"] = "1"
    captured_utc: str
    backend: BackendIdentity
    model_repository: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    tokenizer_repository: str = Field(min_length=1)
    tokenizer_revision: str = Field(min_length=1)
    python_version: str
    python_implementation: str
    packages: dict[str, str | None]  # None: not installed
    operating_system: str
    os_release: str
    machine: str
    hostname: str
    cpu_count: int | None
    ram_total_mib: int | None
    gpu_status: Literal["ok", "no_nvidia_smi", "failed"]
    gpus: list[GpuDevice]
    cuda_runtime: str | None
    git: GitState

    _utc = field_validator("captured_utc")(check_utc)


@dataclass(frozen=True)
class EnvironmentProbes:
    python_version: Callable[[], tuple[str, str]]
    package_version: Callable[[str], str | None]
    operating_system: Callable[[], tuple[str, str, str]]
    hostname: Callable[[], str]
    cpu_count: Callable[[], int | None]
    ram_total_mib: Callable[[], int | None]
    gpus: Callable[[], GpuReading]
    cuda_runtime: Callable[[], str | None]
    git: Callable[[], GitState]


def capture_environment(
    *,
    backend: BackendIdentity,
    model: tuple[str, str],
    tokenizer: tuple[str, str],
    packages: Sequence[str],
    probes: EnvironmentProbes,
    captured_utc: str | None = None,
) -> EnvironmentRecord:
    """Model and tokenizer references are supplied by the caller from the configuration."""
    version, implementation = probes.python_version()
    system, release, machine = probes.operating_system()
    gpus = probes.gpus()
    return EnvironmentRecord(
        captured_utc=captured_utc or utc_now(),
        backend=backend,
        model_repository=model[0],
        model_revision=model[1],
        tokenizer_repository=tokenizer[0],
        tokenizer_revision=tokenizer[1],
        python_version=version,
        python_implementation=implementation,
        packages={name: probes.package_version(name) for name in sorted(set(packages))},
        operating_system=system,
        os_release=release,
        machine=machine,
        hostname=probes.hostname(),
        cpu_count=probes.cpu_count(),
        ram_total_mib=probes.ram_total_mib(),
        gpu_status=gpus.status,
        gpus=gpus.devices,
        cuda_runtime=probes.cuda_runtime(),
        git=probes.git(),
    )


def environment_sha256(record: EnvironmentRecord) -> str:
    return sha256_bytes(serialize_project_json(record, EnvironmentRecord))


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ram_total_mib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return None


def _gpus() -> GpuReading:
    query = "--query-gpu=index,name,memory.total,driver_version"
    try:
        done = subprocess.run(
            ["nvidia-smi", query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return GpuReading(status="no_nvidia_smi", devices=[])
    except (OSError, subprocess.TimeoutExpired):
        return GpuReading(status="failed", devices=[])
    try:
        if done.returncode != 0:
            raise ValueError("nvidia-smi failed")
        devices = []
        for line in done.stdout.strip().splitlines():
            index, name, memory, driver = (part.strip() for part in line.split(","))
            devices.append(
                GpuDevice(
                    index=int(index), name=name, memory_total_mib=int(memory), driver_version=driver
                )
            )
        return GpuReading(status="ok", devices=devices)
    except ValueError:
        return GpuReading(status="failed", devices=[])


def _cuda_runtime() -> str | None:
    """CUDA runtime the installed torch was built for; None if torch is absent or CPU-only."""
    if _package_version("torch") is None:
        return None
    import torch

    return torch.version.cuda


def _git(repository: Path) -> Callable[[], GitState]:
    def read() -> GitState:
        def run(*args: str) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(repository), *args],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

        try:
            head = run("rev-parse", "HEAD")
            if head.returncode != 0:
                if "not a git repository" in head.stderr.lower():
                    return GitState(
                        status="not_a_repository", commit=None, dirty=None, dirty_paths=None
                    )
                return GitState(status="unavailable", commit=None, dirty=None, dirty_paths=None)
            changed = run("status", "--porcelain")
            if changed.returncode != 0:
                return GitState(status="unavailable", commit=None, dirty=None, dirty_paths=None)
        except (OSError, subprocess.TimeoutExpired):
            return GitState(status="unavailable", commit=None, dirty=None, dirty_paths=None)
        count = len(changed.stdout.splitlines())
        return GitState(status="ok", commit=head.stdout.strip(), dirty=count > 0, dirty_paths=count)

    return read


def system_probes(repository: Path) -> EnvironmentProbes:
    """Probes for the machine this process is running on."""
    return EnvironmentProbes(
        python_version=lambda: (platform.python_version(), platform.python_implementation()),
        package_version=_package_version,
        operating_system=lambda: (platform.system(), platform.release(), platform.machine()),
        hostname=socket.gethostname,
        cpu_count=os.cpu_count,
        ram_total_mib=_ram_total_mib,
        gpus=_gpus,
        cuda_runtime=_cuda_runtime,
        git=_git(repository),
    )
