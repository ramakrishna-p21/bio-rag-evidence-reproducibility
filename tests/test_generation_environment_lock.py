"""Intended-versus-actual environment comparison. Synthetic inventories only; no GPU."""

import json
import tomllib
from pathlib import Path

import pytest

from biorag_repro.core.hashing import HashMismatchError, sha256_bytes
from biorag_repro.generation.environment import EnvironmentProbes, GitState, GpuDevice, GpuReading
from biorag_repro.generation.environment_lock import (
    EnvironmentInventory,
    EnvironmentLockError,
    TorchReading,
    capture_inventory,
    inventory_sha256,
    main,
    read_inventory,
    verification_sha256,
    verify_environment,
    write_inventory,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = {"torch": "2.15.0+cu128", "transformers": "4.57.6", "numpy": "2.4.6"}
GPU = GpuDevice(
    index=0, name="NVIDIA GeForce RTX 3090", memory_total_mib=24576, driver_version="580.1"
)


def spec_toml(status="approved_lock", packages=None, **changes):
    values = {
        "python": "3.11.15",
        "system": "Linux",
        "machine": "x86_64",
        "torch_cuda": "12.8",
        "driver": "580.1",
        "unresolved": [],
    }
    values.update(changes)
    pins = "\n".join(f'{name} = "{version}"' for name, version in (packages or PACKAGES).items())
    lines = [
        'schema_version = "1"',
        f'status = "{status}"',
        f'python = "{values["python"]}"',
        'python_implementation = "CPython"',
        f'system = "{values["system"]}"',
        f'machine = "{values["machine"]}"',
        f"unresolved = {json.dumps(values['unresolved'])}",
    ]
    if values["torch_cuda"]:
        lines.append(f'torch_cuda = "{values["torch_cuda"]}"')
    lines += ["", "[gpu]", "min_count = 1", 'name_contains = "RTX 3090"']
    if values["driver"]:
        lines.append(f'nvidia_driver = "{values["driver"]}"')
    return ("\n".join(lines) + "\n\n[packages]\n" + pins + "\n").encode()


def inventory(**changes):
    values = {
        "python_version": "3.11.15",
        "python_implementation": "CPython",
        "system": "Linux",
        "machine": "x86_64",
        "packages": dict(PACKAGES),
        "gpu_status": "ok",
        "gpus": [GPU],
        "torch_status": "ok",
        "torch_cuda_build": "12.8",
        "torch_cuda_available": True,
        "torch_device_names": [GPU.name],
        "package_problems": [],
    }
    values.update(changes)
    return EnvironmentInventory(**values)


def verify(spec=None, inv=None, spec_hash="match"):
    spec = spec_toml() if spec is None else spec
    expected = sha256_bytes(spec) if spec_hash == "match" else spec_hash
    return verify_environment(
        spec, inv or inventory(), expected_spec_sha256=expected, inventory_source="live"
    )


def statuses(report):
    return {check.name: check.status for check in report.checks}


def test_exact_match_with_an_approved_lock():
    report = verify()
    assert report.outcome == "matches_approved_lock"
    assert {check.status for check in report.checks} == {"pass"}


def test_package_version_drift_differs():
    report = verify(inv=inventory(packages={**PACKAGES, "transformers": "4.57.5"}))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["package:transformers"] == "fail"


def test_missing_required_package_is_unavailable_not_a_match():
    packages = {k: v for k, v in PACKAGES.items() if k != "numpy"}
    report = verify(inv=inventory(packages=packages))
    assert report.outcome == "unavailable"
    assert statuses(report)["package:numpy"] == "unavailable"


def test_spec_package_names_are_compared_after_normalisation():
    spec = spec_toml(
        packages={"Torch": PACKAGES["torch"], "Transformers": "4.57.6", "NUMPY": "2.4.6"}
    )
    assert verify(spec).outcome == "matches_approved_lock"


def test_inventory_refuses_unnormalised_package_names():
    with pytest.raises(ValueError):
        inventory(packages={"Torch": PACKAGES["torch"]})


def test_package_not_in_the_approved_lock_is_drift():
    report = verify(inv=inventory(packages={**PACKAGES, "extra-thing": "1"}))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["package:extra-thing"] == "fail"


def test_unsupported_python_version_differs():
    report = verify(inv=inventory(python_version="3.12.1"))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["python"] == "fail"


def test_other_operating_system_or_machine_differs():
    assert verify(inv=inventory(system="Darwin")).outcome == "differs_from_approved_lock"
    assert verify(inv=inventory(machine="aarch64")).outcome == "differs_from_approved_lock"


def test_cpu_only_torch_in_a_gpu_environment_differs():
    inv = inventory(torch_cuda_build=None, torch_cuda_available=False, torch_device_names=[])
    report = verify(inv=inv)
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["torch_cuda_build"] == "fail"


def test_cuda_build_mismatch_differs():
    report = verify(inv=inventory(torch_cuda_build="12.1"))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["torch_cuda_build"] == "fail"


def test_matching_package_names_do_not_prove_cuda_works():
    inv = inventory(torch_cuda_available=False, torch_device_names=[])
    report = verify(inv=inv)
    assert report.outcome == "unavailable"
    assert statuses(report)["torch_cuda_available"] == "unavailable"


def test_missing_gpu_is_unavailable():
    inv = inventory(
        gpu_status="no_nvidia_smi", gpus=[], torch_cuda_available=False, torch_device_names=[]
    )
    report = verify(inv=inv)
    assert report.outcome == "unavailable"
    assert statuses(report)["gpu"] == "unavailable"


def test_failed_gpu_query_cannot_be_verified():
    inv = inventory(gpu_status="failed", gpus=[])
    report = verify(inv=inv)
    assert report.outcome == "unverifiable"
    assert statuses(report)["gpu"] == "unverifiable"


def test_torch_failing_to_import_cannot_be_verified():
    inv = inventory(
        torch_status="failed",
        torch_cuda_build=None,
        torch_cuda_available=None,
        torch_device_names=[],
    )
    assert verify(inv=inv).outcome == "unverifiable"


def test_wrong_gpu_model_differs():
    other = GPU.model_copy(update={"name": "NVIDIA A100"})
    report = verify(inv=inventory(gpus=[other], torch_device_names=[other.name]))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["gpu"] == "fail"


def test_driver_mismatch_differs():
    other = GPU.model_copy(update={"driver_version": "535.0"})
    assert verify(inv=inventory(gpus=[other])).outcome == "differs_from_approved_lock"


def test_torch_sees_a_device_the_driver_query_does_not_list():
    report = verify(inv=inventory(torch_device_names=["NVIDIA GeForce RTX 3090", "Mystery GPU"]))
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["torch_devices"] == "fail"


def test_torch_sees_only_a_subset_of_listed_gpus_when_one_is_selected():
    second = GPU.model_copy(update={"index": 1})
    inv = inventory(gpus=[GPU, second], torch_device_names=[GPU.name])
    assert verify(inv=inv).outcome == "matches_approved_lock"


def test_lock_file_altered_after_hashing_differs():
    spec = spec_toml()
    altered = spec.replace(b"4.57.6", b"4.57.7")
    report = verify_environment(
        altered, inventory(), expected_spec_sha256=sha256_bytes(spec), inventory_source="live"
    )
    assert report.outcome == "differs_from_approved_lock"
    assert statuses(report)["spec_sha256"] == "fail"


def test_approved_lock_without_an_expected_hash_cannot_be_verified():
    report = verify_environment(
        spec_toml(), inventory(), expected_spec_sha256=None, inventory_source="live"
    )
    assert report.outcome == "unverifiable"
    assert statuses(report)["spec_sha256"] == "unverifiable"


def test_absent_lock_is_not_approved():
    report = verify_environment(
        None, inventory(), expected_spec_sha256=None, inventory_source="live"
    )
    assert report.outcome == "not_approved"
    assert report.spec_sha256 is None


def test_provisional_spec_is_never_approved_even_when_everything_matches():
    spec = spec_toml(
        status="provisional", torch_cuda="", driver="", unresolved=["torch build", "driver"]
    )
    report = verify(spec, inventory())
    assert report.outcome == "not_approved"
    assert statuses(report)["torch_cuda_build"] == "unresolved"
    assert statuses(report)["nvidia_driver"] == "unresolved"


def test_approved_lock_cannot_leave_choices_unresolved():
    with pytest.raises(EnvironmentLockError):
        verify(spec_toml(unresolved=["driver"]))
    with pytest.raises(EnvironmentLockError):
        verify(spec_toml(torch_cuda=""))
    with pytest.raises(EnvironmentLockError):
        verify(spec_toml(driver=""))
    with pytest.raises(EnvironmentLockError):  # a lock names the exact torch build
        verify(spec_toml(packages={**PACKAGES, "torch": "2.15.0"}))


def test_malformed_or_unknown_spec_content_is_refused():
    with pytest.raises(EnvironmentLockError):
        verify_environment(
            b"not = [toml", inventory(), expected_spec_sha256=None, inventory_source="live"
        )
    with pytest.raises(EnvironmentLockError):
        verify_environment(
            spec_toml() + b"\n[surprise]\nx = 1\n",
            inventory(),
            expected_spec_sha256=None,
            inventory_source="live",
        )


def test_repeated_verification_has_a_stable_identity():
    first, second = verify(), verify()
    assert first == second
    assert verification_sha256(first) == verification_sha256(second)
    assert inventory_sha256(inventory()) == inventory_sha256(inventory())
    changed = verify(inv=inventory(packages={**PACKAGES, "numpy": "2.4.7"}))
    assert verification_sha256(changed) != verification_sha256(first)


def test_inventory_round_trip_and_tamper_detection(tmp_path):
    inv = inventory()
    digest = write_inventory(inv, tmp_path / "inventory.json")
    assert read_inventory(tmp_path / "inventory.json", digest) == inv
    path = tmp_path / "inventory.json"
    path.write_bytes(path.read_bytes().replace(b"4.57.6", b"4.57.7"))
    with pytest.raises(HashMismatchError):
        read_inventory(path, digest)


def test_inventory_refuses_inconsistent_torch_reading():
    with pytest.raises(ValueError):
        inventory(torch_status="not_installed", torch_cuda_available=True)
    with pytest.raises(ValueError):
        inventory(torch_status="ok", torch_cuda_available=None)


def probes(gpus=None):
    return EnvironmentProbes(
        python_version=lambda: ("3.11.15", "CPython"),
        package_version=lambda name: None,
        operating_system=lambda: ("Linux", "6.1", "x86_64"),
        hostname=lambda: "host",
        cpu_count=lambda: 1,
        ram_total_mib=lambda: 1,
        gpus=lambda: gpus or GpuReading(status="no_nvidia_smi", devices=[]),
        cuda_runtime=lambda: None,
        git=lambda: GitState(status="ok", commit="1" * 40, dirty=False, dirty_paths=0),
    )


def test_capture_normalises_names_and_omits_host_identity():
    inv = capture_inventory(
        probes(GpuReading(status="ok", devices=[GPU])),
        installed=lambda: [("Huggingface_Hub", "0.36.2"), ("torch", "2.15.0+cpu")],
        torch=lambda: TorchReading(
            status="ok", cuda_build=None, cuda_available=False, device_names=[]
        ),
    )
    assert inv.packages == {"huggingface-hub": "0.36.2", "torch": "2.15.0+cpu"}
    assert "host" not in inv.model_dump_json()
    assert inv.gpus == [GPU]


def test_committed_spec_is_provisional_and_pins_match_pyproject():
    spec = tomllib.loads((ROOT / "configs/environment.toml").read_text())
    assert spec["status"] == "provisional" and spec["unresolved"]
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]
    declared = {}
    for group in ("generation", "dense"):
        for requirement in extras[group]:
            name, _, version = requirement.partition("==")
            declared[name.lower().replace("_", "-")] = version
    assert declared.items() <= spec["packages"].items()
    report = verify_environment(
        (ROOT / "configs/environment.toml").read_bytes(),
        inventory(),
        expected_spec_sha256=None,
        inventory_source="live",
    )
    assert report.outcome == "not_approved"


def test_cli_writes_report_and_uses_distinct_exit_codes(tmp_path):
    spec = tmp_path / "lock.toml"
    spec.write_bytes(spec_toml())
    path = tmp_path / "inventory.json"
    digest = write_inventory(inventory(), path)
    base = ["verify", "--inventory", str(path), "--inventory-sha256", digest]
    out = tmp_path / "report.json"
    args = [*base, "--spec", str(spec), "--spec-sha256", sha256_bytes(spec.read_bytes())]
    assert main([*args, "--output", str(out)]) == 0
    assert json.loads(out.read_text())["outcome"] == "matches_approved_lock"
    assert main([*args, "--output", str(out)]) == 1  # refuses to overwrite a report
    assert main([*base, "--spec", str(spec), "--spec-sha256", "0" * 64]) == 2
    assert main(base) == 3  # no --spec: no lock exists
    assert main([*base, "--spec", str(tmp_path / "typo.toml")]) == 1  # a wrong path is an error
    assert main([*base, "--spec-sha256", "0" * 64]) == 1  # a hash for no file is an error
    bad = ["verify", "--inventory", str(path), "--inventory-sha256", "1" * 64, "--spec", str(spec)]
    assert main(bad) == 1  # an altered inventory is an error, not a verdict


def test_report_says_whether_the_inventory_was_live_or_stored():
    spec = spec_toml()
    stored = verify_environment(
        spec, inventory(), expected_spec_sha256=sha256_bytes(spec), inventory_source="stored"
    )
    assert stored.inventory_source == "stored" and verify().inventory_source == "live"
    assert verification_sha256(stored) != verification_sha256(verify())


def test_cli_marks_a_stored_inventory_as_stored(tmp_path):
    spec = tmp_path / "lock.toml"
    spec.write_bytes(spec_toml())
    path = tmp_path / "inventory.json"
    digest = write_inventory(inventory(), path)
    out = tmp_path / "report.json"
    args = ["verify", "--inventory", str(path), "--inventory-sha256", digest, "--spec", str(spec)]
    main([*args, "--spec-sha256", sha256_bytes(spec.read_bytes()), "--output", str(out)])
    assert json.loads(out.read_text())["inventory_source"] == "stored"


def test_approved_lock_pins_the_exact_python_patch_version():
    with pytest.raises(EnvironmentLockError):
        verify(spec_toml(python="3.11"))
    exact = spec_toml(python="3.11.15")
    assert verify(exact).outcome == "matches_approved_lock"
    assert verify(exact, inventory(python_version="3.11.9")).outcome == "differs_from_approved_lock"


def test_duplicate_distributions_are_recorded_not_merged():
    inv = capture_inventory(
        probes(),
        installed=lambda: [("torch", "1"), ("Torch", "2"), ("numpy", "2")],
        torch=lambda: TorchReading(
            status="not_installed", cuda_build=None, cuda_available=None, device_names=[]
        ),
    )
    assert inv.package_problems == ["duplicate:torch"]
    assert verify(inv=inventory(package_problems=["duplicate:torch"])).outcome == "unverifiable"


def test_unreadable_distribution_is_recorded_not_a_crash():
    inv = capture_inventory(
        probes(),
        installed=lambda: [(None, "1"), ("numpy", "2")],
        torch=lambda: TorchReading(
            status="not_installed", cuda_build=None, cuda_available=None, device_names=[]
        ),
    )
    assert inv.package_problems == ["unreadable-distribution"]
    assert inv.packages == {"numpy": "2"}


def test_torch_seeing_a_non_matching_gpu_differs():
    other = GPU.model_copy(update={"index": 1, "name": "NVIDIA A100"})
    inv = inventory(gpus=[GPU, other], torch_device_names=[GPU.name, other.name])
    assert verify(inv=inv).outcome == "differs_from_approved_lock"


def test_driver_check_is_present_and_unverifiable_when_the_gpu_query_failed():
    report = verify(inv=inventory(gpu_status="failed", gpus=[]))
    assert statuses(report)["nvidia_driver"] == "unverifiable"


def test_driver_check_covers_every_listed_gpu():
    other = GPU.model_copy(update={"index": 1, "driver_version": "535.0"})
    inv = inventory(gpus=[GPU, other])
    assert verify(inv=inv).outcome == "differs_from_approved_lock"


@pytest.mark.parametrize("error", [AssertionError, AttributeError, RuntimeError, ImportError])
def test_broken_torch_reads_as_failed_not_a_crash(monkeypatch, error):
    import sys
    import types

    from biorag_repro.generation import environment_lock

    fake = types.ModuleType("torch")

    def explode():
        raise error("broken")

    fake.cuda = types.SimpleNamespace(is_available=explode)
    fake.version = types.SimpleNamespace(cuda=None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setattr(environment_lock.metadata, "version", lambda name: "2.15.0")
    assert environment_lock._torch().status == "failed"
