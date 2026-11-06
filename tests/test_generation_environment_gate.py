"""Environment evidence an official run must carry. All locks and inventories are synthetic."""

# ruff: noqa: F811  (fixtures imported from the official-run tests)

import json
from pathlib import Path

import pytest
from synthetic_environment import LOCK, approved_locks, evidence, inventory
from test_generation_official import (  # noqa: F401  (fixtures)
    anchored,
    build_world,
    gpu_probes,
    recognised,
    verify,
    view,
)

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.attempts import CorruptRecord, NonOfficialBackend
from biorag_repro.generation.environment_gate import (
    EnvironmentEvidence,
    EnvironmentNotApproved,
    check_approved,
    publish_environment_evidence,
)
from biorag_repro.generation.environment_lock import verify_environment

PROVISIONAL = Path("configs/environment.toml").read_bytes()


@pytest.fixture
def registered(monkeypatch):
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", approved_locks())


def test_nothing_is_approved_by_default():
    assert generator_config.APPROVED_ENVIRONMENT_LOCKS == frozenset()
    with pytest.raises(EnvironmentNotApproved, match="not an approved lock"):
        check_approved(evidence())


def test_a_matching_live_verification_of_a_registered_lock_is_accepted(registered):
    check_approved(evidence())


def test_a_lock_that_declares_itself_approved_is_not_approved(monkeypatch):
    # The report is a perfect match and the caller supplied the lock's own hash; neither is
    # approval. Only the registry, edited by the maintainer, is.
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", frozenset({"0" * 64}))
    with pytest.raises(EnvironmentNotApproved, match="not an approved lock"):
        check_approved(evidence())


def test_the_provisional_specification_is_never_accepted(monkeypatch):
    monkeypatch.setattr(
        generator_config,
        "APPROVED_ENVIRONMENT_LOCKS",
        frozenset({sha256_bytes(PROVISIONAL)}),
    )
    report = verify_environment(
        PROVISIONAL, inventory(), expected_spec_sha256=None, inventory_source="live"
    )
    with pytest.raises(EnvironmentNotApproved, match="approved_lock"):
        check_approved(EnvironmentEvidence(lock_bytes=PROVISIONAL, report=report))


def test_a_report_about_another_lock_is_refused(registered):
    other = evidence(lock=LOCK.replace(b"3.11.9", b"3.11.8"))
    with pytest.raises(EnvironmentNotApproved, match="another lock"):
        check_approved(EnvironmentEvidence(lock_bytes=LOCK, report=other.report))


def test_a_stored_inventory_does_not_describe_the_execution_machine(registered):
    with pytest.raises(EnvironmentNotApproved, match="live"):
        check_approved(evidence(source="stored"))


@pytest.mark.parametrize(
    "changes",
    [
        {"gpus": []},
        {"packages": {"torch": "2.15.0+cu000"}},
        {"torch_cuda_available": False, "torch_device_names": []},
        {"python_version": "3.11.10"},
        {"package_problems": ["duplicate: torch"]},
    ],
)
def test_any_difference_from_the_lock_is_refused(registered, changes):
    if "gpus" in changes:
        changes = changes | {"gpu_status": "no_nvidia_smi"}
    with pytest.raises(EnvironmentNotApproved, match="matches_approved_lock|outcome"):
        check_approved(evidence(**changes))


def test_a_report_edited_to_claim_a_match_is_refused(registered):
    bad = evidence(python_version="3.11.10")
    forged = bad.report.model_copy(update={"outcome": "matches_approved_lock"})
    with pytest.raises(EnvironmentNotApproved, match="check"):
        check_approved(EnvironmentEvidence(lock_bytes=LOCK, report=forged))


# ---- the evidence a run carries ----------------------------------------------------------


def test_an_official_run_with_registered_matching_evidence_verifies(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny)
    verified = verify(world, anchored(world))
    assert verified.environment.report.outcome == "matches_approved_lock"
    assert verified.environment.lock_bytes == LOCK


def test_an_official_run_without_environment_evidence_is_refused(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny, evidence=False)
    with pytest.raises(NonOfficialBackend, match="environment evidence"):
        verify(world, anchored(world))


def test_evidence_for_a_lock_the_maintainer_has_not_approved_is_refused(
    tmp_path, view, tiny, recognised, monkeypatch
):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", frozenset())
    with pytest.raises(EnvironmentNotApproved):
        verify(world, anchor)


def test_evidence_that_does_not_match_is_refused_even_if_stored_with_the_run(
    tmp_path, view, tiny, recognised
):
    world = build_world(tmp_path, view, tiny, evidence=False)
    publish_environment_evidence(world.store.run_dir, evidence(python_version="3.11.10"))
    with pytest.raises(EnvironmentNotApproved):
        verify(world, anchored(world))


def test_evidence_altered_after_the_run_was_anchored_is_found(tmp_path, view, tiny, recognised):
    world = build_world(tmp_path, view, tiny)
    anchor = anchored(world)
    path = world.store.run_dir / "environment_evidence" / "verification.json"
    data = json.loads(path.read_text())
    data["inventory_source"] = "stored"
    path.write_text(json.dumps(data))
    with pytest.raises(CorruptRecord):
        verify(world, anchor)


def test_evidence_that_disagrees_with_the_run_s_own_environment_is_refused(
    tmp_path, view, tiny, recognised, monkeypatch
):
    import test_generation_official as module

    from biorag_repro.generation.environment import GpuDevice, GpuReading

    other = GpuDevice(index=0, name="Synthetic GPU", memory_total_mib=24576, driver_version="1")
    original = module.gpu_probes
    monkeypatch.setattr(
        module,
        "gpu_probes",
        lambda status="ok": original(status).__class__(
            **(vars(original(status)) | {"gpus": lambda: GpuReading(status="ok", devices=[other])})
        ),
    )
    world = build_world(tmp_path, view, tiny)
    with pytest.raises(NonOfficialBackend, match="driver"):
        verify(world, anchored(world))


# ---- review fixes -------------------------------------------------------------------------


def test_a_report_with_no_checks_is_refused(registered):
    empty = evidence().report.model_copy(update={"checks": []})
    with pytest.raises(EnvironmentNotApproved, match="checks"):
        check_approved(EnvironmentEvidence(lock_bytes=LOCK, report=empty))


def test_a_registered_file_that_is_not_a_lock_is_a_refusal_not_a_crash(monkeypatch):
    junk = b"not a toml specification ["
    monkeypatch.setattr(
        generator_config, "APPROVED_ENVIRONMENT_LOCKS", frozenset({sha256_bytes(junk)})
    )
    report = evidence().report.model_copy(update={"spec_sha256": sha256_bytes(junk)})
    with pytest.raises(EnvironmentNotApproved, match="unusable"):
        check_approved(EnvironmentEvidence(lock_bytes=junk, report=report))


def test_resuming_keeps_the_first_verification_and_requires_the_same_lock(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    publish_environment_evidence(run_dir, evidence())
    first = (run_dir / "environment_evidence" / "verification.json").read_bytes()
    publish_environment_evidence(run_dir, evidence())
    again = evidence()
    again = EnvironmentEvidence(
        lock_bytes=LOCK, report=again.report.model_copy(update={"inventory_sha256": "9" * 64})
    )
    publish_environment_evidence(run_dir, again)  # a fresh capture on resume is not an error
    assert (run_dir / "environment_evidence" / "verification.json").read_bytes() == first
    with pytest.raises(CorruptRecord):
        publish_environment_evidence(run_dir, evidence(lock=LOCK.replace(b"3.11.9", b"3.11.8")))


def test_an_attempt_record_without_gpu_evidence_is_incompatible_with_a_gpu_lock():
    from synthetic_environment import GPU
    from test_generation_environment import capture

    from biorag_repro.generation.environment_gate import _disagreements
    from biorag_repro.generation.environment_lock import parse_spec

    record = capture()  # CPU-only record: no GPU read, no execution block
    assert any("gpu" in p for p in _disagreements(parse_spec(LOCK), record))
    assert GPU
