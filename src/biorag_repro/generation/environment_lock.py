"""Compare the intended execution environment with the one actually installed.

The intended environment is a TOML specification (`configs/environment.toml`). It is only an
approved lock when its status says so and the caller supplies the lock file's SHA-256. This
module never approves anything: a provisional specification, an absent lock, a missing
probe or an unresolved choice can never produce `matches_approved_lock`.

CUDA compatibility is judged from what the machine reports (the torch build, whether torch
can use a device, the device names torch and the driver query list), never from package
names alone.
"""

import argparse
import re
import sys
import tomllib
from collections.abc import Callable
from importlib import metadata
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, ValidationError, model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.generation.environment import (
    EnvironmentProbes,
    GpuDevice,
    Sha256,
    system_probes,
)

Outcome = Literal[
    "matches_approved_lock",
    "differs_from_approved_lock",
    "not_approved",
    "unavailable",
    "unverifiable",
]
CheckStatus = Literal["pass", "fail", "unavailable", "unverifiable", "unresolved"]
EXIT_CODES: dict[str, int] = {
    "matches_approved_lock": 0,
    "differs_from_approved_lock": 2,
    "not_approved": 3,
    "unavailable": 4,
    "unverifiable": 5,
}


class EnvironmentLockError(ValueError):
    """The specification or lock file itself is unusable."""


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


class GpuRequirement(Record):
    min_count: int = Field(ge=1)
    name_contains: str = Field(min_length=1)
    nvidia_driver: str | None = None


class EnvironmentSpec(Record):
    schema_version: Literal["1"]
    status: Literal["provisional", "approved_lock"]
    python: str = Field(pattern=r"^3\.\d+(\.\d+)?$")  # major.minor; a lock needs the patch too
    python_implementation: Literal["CPython"]
    system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    torch_cuda: str | None = None
    unresolved: list[str]
    gpu: GpuRequirement
    packages: dict[str, str]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        names = [normalise(n) for n in self.packages]
        if len(names) != len(set(names)):
            raise ValueError("package names repeat after normalisation")
        if self.status == "approved_lock":
            if self.unresolved or self.torch_cuda is None or self.gpu.nvidia_driver is None:
                raise ValueError("an approved lock leaves no choice unresolved")
            if self.python.count(".") != 2:
                raise ValueError("an approved lock names the exact Python patch version")
            if "+" not in dict(zip(names, self.packages.values(), strict=True)).get("torch", ""):
                raise ValueError("an approved lock names the exact torch build, e.g. 2.x.y+cuNNN")
        return self


class TorchReading(Record):
    status: Literal["ok", "not_installed", "failed"]
    cuda_build: str | None
    cuda_available: bool | None
    device_names: list[str]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if (self.status == "ok") != (self.cuda_available is not None):
            raise ValueError("CUDA availability is recorded exactly when torch was queried")
        if self.status != "ok" and (self.cuda_build or self.device_names):
            raise ValueError("no torch details without a successful torch query")
        return self


class EnvironmentInventory(Record):
    """Machine facts only: no hostname, path or timestamp, so equal machines hash equal."""

    schema_version: Literal["1"] = "1"
    python_version: str
    python_implementation: str
    system: str
    machine: str
    packages: dict[str, str]
    gpu_status: Literal["ok", "no_nvidia_smi", "failed"]
    gpus: list[GpuDevice]
    torch_status: Literal["ok", "not_installed", "failed"]
    torch_cuda_build: str | None
    torch_cuda_available: bool | None
    torch_device_names: list[str]
    package_problems: list[str]  # duplicate or unreadable distributions found while listing

    @model_validator(mode="after")
    def consistent(self) -> Self:
        TorchReading(
            status=self.torch_status,
            cuda_build=self.torch_cuda_build,
            cuda_available=self.torch_cuda_available,
            device_names=self.torch_device_names,
        )
        if any(name != normalise(name) for name in self.packages):
            raise ValueError("inventory package names must be normalised")
        if (self.gpu_status == "ok") != bool(self.gpus):
            raise ValueError("GPUs are listed exactly when the GPU query succeeded")
        return self


class Check(Record):
    name: str
    status: CheckStatus
    expected: str | None
    actual: str | None


class EnvironmentVerification(Record):
    schema_version: Literal["1"] = "1"
    outcome: Outcome
    spec_status: Literal["provisional", "approved_lock"] | None
    spec_sha256: Sha256 | None
    inventory_sha256: Sha256
    inventory_source: Literal["live", "stored"]  # only a live inventory describes this machine now
    checks: list[Check]


def inventory_sha256(inventory: EnvironmentInventory) -> str:
    return sha256_bytes(serialize_project_json(inventory, EnvironmentInventory))


def verification_sha256(report: EnvironmentVerification) -> str:
    return sha256_bytes(serialize_project_json(report, EnvironmentVerification))


def _installed() -> list[tuple[str | None, str]]:
    return [(d.metadata["Name"], d.version) for d in metadata.distributions()]


def _torch() -> TorchReading:
    try:
        metadata.version("torch")
    except metadata.PackageNotFoundError:
        return TorchReading(
            status="not_installed", cuda_build=None, cuda_available=None, device_names=[]
        )
    try:
        import torch

        available = bool(torch.cuda.is_available())
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return TorchReading(
            status="ok",
            cuda_build=torch.version.cuda,
            cuda_available=available,
            device_names=names if available else [],
        )
    except (
        ImportError,
        OSError,
        RuntimeError,
        AssertionError,
        AttributeError,
    ):  # a broken install reads as "cannot verify"
        return TorchReading(status="failed", cuda_build=None, cuda_available=None, device_names=[])


def capture_inventory(
    probes: EnvironmentProbes,
    installed: Callable[[], list[tuple[str | None, str]]] = _installed,
    torch: Callable[[], TorchReading] = _torch,
) -> EnvironmentInventory:
    version, implementation = probes.python_version()
    system, _, machine = probes.operating_system()
    gpus = probes.gpus()
    reading = torch()
    packages: dict[str, str] = {}
    problems: set[str] = set()
    for raw_name, package_version in installed():
        if not raw_name:
            problems.add("unreadable-distribution")
            continue
        name = normalise(raw_name)
        if name in packages:  # which copy Python imports is then unknown
            problems.add(f"duplicate:{name}")
        packages[name] = package_version
    return EnvironmentInventory(
        python_version=version,
        python_implementation=implementation,
        system=system,
        machine=machine,
        packages=dict(sorted(packages.items())),
        gpu_status=gpus.status,
        gpus=gpus.devices,
        torch_status=reading.status,
        torch_cuda_build=reading.cuda_build,
        torch_cuda_available=reading.cuda_available,
        torch_device_names=reading.device_names,
        package_problems=sorted(problems),
    )


def parse_spec(content: bytes) -> EnvironmentSpec:
    try:
        return EnvironmentSpec.model_validate(tomllib.loads(content.decode("utf-8")))
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, ValidationError) as error:
        raise EnvironmentLockError(f"unusable environment specification: {error}") from error


def _python_matches(pinned: str, actual: str) -> bool:
    return actual == pinned or (pinned.count(".") == 1 and actual.startswith(pinned + "."))


def _check_packages(spec: EnvironmentSpec, actual: dict[str, str]) -> list[Check]:
    checks = []
    pins = {normalise(n): v for n, v in spec.packages.items()}
    for name, pin in pins.items():
        found = actual.get(name)
        # A provisional pin without a build label (torch "2.15.0") leaves the build open.
        compared = (
            found.split("+")[0]
            if found and "+" not in pin and spec.status != "approved_lock"
            else found
        )
        status: CheckStatus = (
            "unavailable" if found is None else "pass" if compared == pin else "fail"
        )
        checks.append(Check(name=f"package:{name}", status=status, expected=pin, actual=found))
    if spec.status == "approved_lock":
        for name in sorted(set(actual) - set(pins)):
            checks.append(
                Check(name=f"package:{name}", status="fail", expected=None, actual=actual[name])
            )
    return checks


def _check_hardware(spec: EnvironmentSpec, inv: EnvironmentInventory) -> list[Check]:
    checks: list[Check] = []
    wanted = spec.gpu
    if inv.torch_status == "ok":
        if spec.torch_cuda is None:
            status: CheckStatus = "unresolved"
        else:
            status = "pass" if inv.torch_cuda_build == spec.torch_cuda else "fail"
        checks.append(
            Check(
                name="torch_cuda_build",
                status=status,
                expected=spec.torch_cuda,
                actual=inv.torch_cuda_build,
            )
        )
        checks.append(
            Check(
                name="torch_cuda_available",
                status="pass" if inv.torch_cuda_available else "unavailable",
                expected="True",
                actual=str(inv.torch_cuda_available),
            )
        )
    else:
        status = "unverifiable" if inv.torch_status == "failed" else "unavailable"
        checks.append(
            Check(name="torch_cuda_build", status=status, expected=spec.torch_cuda, actual=None)
        )

    matching = [g for g in inv.gpus if wanted.name_contains in g.name]
    if inv.gpu_status == "ok":
        status = "pass" if len(matching) >= wanted.min_count else "fail"
        actual = ", ".join(g.name for g in inv.gpus)
    else:
        status = "unavailable" if inv.gpu_status == "no_nvidia_smi" else "unverifiable"
        actual = inv.gpu_status
    checks.append(
        Check(
            name="gpu",
            status=status,
            expected=f"{wanted.min_count} x *{wanted.name_contains}*",
            actual=actual,
        )
    )

    # The driver is machine-wide: every listed GPU must report the pinned version.
    drivers = sorted({g.driver_version for g in inv.gpus})
    if wanted.nvidia_driver is None:
        status = "unresolved"
    elif inv.gpu_status != "ok":
        status = "unavailable" if inv.gpu_status == "no_nvidia_smi" else "unverifiable"
    else:
        status = "pass" if drivers == [wanted.nvidia_driver] else "fail"
    checks.append(
        Check(
            name="nvidia_driver",
            status=status,
            expected=wanted.nvidia_driver,
            actual=", ".join(drivers) or None,
        )
    )

    if inv.torch_cuda_available and inv.gpu_status == "ok":
        listed = {g.name for g in inv.gpus}
        seen = set(inv.torch_device_names)
        ok = bool(seen) and seen <= listed and all(wanted.name_contains in n for n in seen)
        checks.append(
            Check(
                name="torch_devices",
                status="pass" if ok else "fail",
                expected=f"only driver-listed *{wanted.name_contains}* devices",
                actual=", ".join(sorted(seen)),
            )
        )
    return checks


def verify_environment(
    spec_bytes: bytes | None,
    inventory: EnvironmentInventory,
    *,
    expected_spec_sha256: str | None,
    inventory_source: Literal["live", "stored"],
) -> EnvironmentVerification:
    """`spec_bytes` is None when no lock exists. Raises only if the file is unusable."""
    inv_hash = inventory_sha256(inventory)
    if spec_bytes is None:
        check = Check(
            name="spec", status="unavailable", expected="an environment lock", actual=None
        )
        return EnvironmentVerification(
            outcome="not_approved",
            spec_status=None,
            spec_sha256=None,
            inventory_sha256=inv_hash,
            inventory_source=inventory_source,
            checks=[check],
        )
    spec = parse_spec(spec_bytes)
    spec_hash = sha256_bytes(spec_bytes)
    checks: list[Check] = []
    if expected_spec_sha256 is not None:
        status: CheckStatus = "pass" if spec_hash == expected_spec_sha256 else "fail"
        checks.append(
            Check(
                name="spec_sha256", status=status, expected=expected_spec_sha256, actual=spec_hash
            )
        )
    elif spec.status == "approved_lock":
        checks.append(
            Check(name="spec_sha256", status="unverifiable", expected=None, actual=spec_hash)
        )
    checks += [
        Check(
            name="python",
            status="pass"
            if _python_matches(spec.python, inventory.python_version)
            and inventory.python_implementation == spec.python_implementation
            else "fail",
            expected=f"{spec.python_implementation} {spec.python}",
            actual=f"{inventory.python_implementation} {inventory.python_version}",
        ),
        Check(
            name="platform",
            status="pass"
            if (inventory.system, inventory.machine) == (spec.system, spec.machine)
            else "fail",
            expected=f"{spec.system} {spec.machine}",
            actual=f"{inventory.system} {inventory.machine}",
        ),
    ]
    if inventory.package_problems:
        checks.append(
            Check(
                name="package_inventory",
                status="unverifiable",
                expected="every distribution listed once",
                actual=", ".join(inventory.package_problems),
            )
        )
    checks += _check_packages(spec, inventory.packages)
    checks += _check_hardware(spec, inventory)
    checks += [
        Check(name=f"unresolved:{item}", status="unresolved", expected=None, actual=None)
        for item in spec.unresolved
    ]

    found = {c.status for c in checks}
    outcome: Outcome
    if spec.status != "approved_lock":
        outcome = "not_approved"
    elif "fail" in found:
        outcome = "differs_from_approved_lock"
    elif "unavailable" in found:
        outcome = "unavailable"
    elif found & {"unverifiable", "unresolved"}:
        outcome = "unverifiable"
    else:
        outcome = "matches_approved_lock"
    return EnvironmentVerification(
        outcome=outcome,
        spec_status=spec.status,
        spec_sha256=spec_hash,
        inventory_sha256=inv_hash,
        inventory_source=inventory_source,
        checks=checks,
    )


def write_inventory(inventory: EnvironmentInventory, path: Path) -> str:
    content = serialize_project_json(inventory, EnvironmentInventory)
    return publish_bytes(path, content, sha256_bytes(content))


def read_inventory(path: Path, expected_sha256: str) -> EnvironmentInventory:
    return read_project_json(path, expected_sha256, EnvironmentInventory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("capture", help="write an inventory of this machine")
    make.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("verify", help="compare an inventory with a specification")
    check.add_argument("--spec", type=Path, help="omit when no lock exists")
    check.add_argument("--spec-sha256")
    check.add_argument("--inventory", type=Path, help="stored inventory; default: this machine")
    check.add_argument("--inventory-sha256")
    check.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "capture":
            inventory = capture_inventory(system_probes(Path.cwd()))
            print(f"inventory sha256 {write_inventory(inventory, args.output)}")
            return 0
        if args.inventory is not None:
            if args.inventory_sha256 is None:
                raise ValueError("--inventory needs --inventory-sha256")
            inventory = read_inventory(args.inventory, args.inventory_sha256)
        else:
            inventory = capture_inventory(system_probes(Path.cwd()))
        if args.spec is None and args.spec_sha256 is not None:
            raise ValueError("--spec-sha256 needs --spec")
        spec_bytes = args.spec.read_bytes() if args.spec is not None else None
        report = verify_environment(
            spec_bytes,
            inventory,
            expected_spec_sha256=args.spec_sha256,
            inventory_source="live" if args.inventory is None else "stored",
        )
        if args.output is not None:
            content = serialize_project_json(report, EnvironmentVerification)
            publish_bytes(args.output, content, sha256_bytes(content))
    except (ValueError, OSError) as error:
        print(f"environment verification failed: {error}", file=sys.stderr)
        return 1
    print(f"{report.outcome} (inventory {report.inventory_sha256})")
    for item in report.checks:
        if item.status != "pass":
            print(f"  {item.status}: {item.name} expected={item.expected} actual={item.actual}")
    return EXIT_CODES[report.outcome]


if __name__ == "__main__":
    raise SystemExit(main())
