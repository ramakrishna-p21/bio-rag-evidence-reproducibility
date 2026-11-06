"""The execution driver: dry run, mock or smoke execution, anchoring and verification.

Everything is synthetic and weight-free. The mock backend answers with fixed text and its
run is labelled a mock run throughout; nothing here is an observation of Qwen3-8B.
"""

import json
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
from test_generation_attempts import environment
from test_generation_config import to_toml
from test_generation_hf_backend import REV
from test_generation_official import SCRIPT, Scripted
from test_generation_roster import make_scope
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation import config as generator_config
from biorag_repro.generation import driver, official
from biorag_repro.generation.attempts import CorruptRecord, LifecycleError, UnsafeResume
from biorag_repro.generation.driver import (
    Anchor,
    ConcurrentRun,
    DriverError,
    DriverPlan,
    MockBackend,
    OfficialDisabled,
    dry_run,
    execute,
    load_plan,
    verify,
)
from biorag_repro.generation.environment import BackendIdentity
from biorag_repro.generation.hf_backend import HF_BACKEND_NAME
from biorag_repro.generation.model_snapshot import inventory_snapshot
from biorag_repro.generation.official import VerifiedOfficialRun
from biorag_repro.generation.runner import NonOfficialResults


@pytest.fixture
def world(roster_world):  # noqa: F811
    return roster_world


class Killed(BaseException):
    """Stands in for a killed process."""


def make_plan(tmp_path, world, *, settings=None, seeds=None, anchor_dir=None, scope=None):
    directory, record_sha, _sources, renderer, _ = world
    scope = scope or make_scope(directory, record_sha, {"q_a", "q_b"})
    config = tmp_path / "generator.toml"
    config.write_bytes(
        to_toml(settings).encode() if settings is not None else renderer.config.raw_toml
    )
    prompt = tmp_path / "prompt.json"
    prompt.write_text(renderer.prompt.model_dump_json())
    scope_path = tmp_path / "scope.json"
    scope_path.write_text(scope.model_dump_json())
    seeds_path = None
    if seeds is not None:
        seeds_path = tmp_path / "seeds.json"
        seeds_path.write_text(seeds.model_dump_json())
    return DriverPlan(
        run_root=tmp_path / "runs_root",
        anchor_dir=anchor_dir or tmp_path / "anchors",
        config=config,
        prompt=prompt,
        seeds=seeds_path,
        tokenizer_root=_tokenizer_root(),
        view_directory=directory,
        view_record_sha256=record_sha,
        scope=scope_path,
        repository=Path(__file__).parent,
    )


def _tokenizer_root():
    from test_generation_prompt import ROOT

    return ROOT


def mock_engine(backend):
    return lambda _prepared: (backend, lambda: environment("mock"))


def never_started(_prepared):
    pytest.fail("no backend may be created for a refused run")


def run_mock(plan, world, **kwargs):
    backend = kwargs.pop("backend", None) or MockBackend()
    return execute(plan, world[2], mode="mock", start=mock_engine(backend), **kwargs)


def tree(path):
    return sorted(p.relative_to(path).as_posix() for p in path.rglob("*")) if path.exists() else []


# ---- dry run -------------------------------------------------------------------------------


def test_dry_run_checks_everything_and_writes_nothing(world, tmp_path):
    plan = make_plan(tmp_path, world)
    report = dry_run(plan, world[2], mode="mock")
    assert report.observations == 2 and report.items == 2
    assert report.conditions == ["JATS"] and report.repeats == [1]
    assert 0 < report.min_input_tokens <= report.max_input_tokens
    assert report.max_input_tokens <= report.input_token_limit
    assert not plan.run_root.exists() and not plan.anchor_dir.exists()


def test_dry_run_reports_why_official_execution_is_not_possible(world, tmp_path):
    report = dry_run(make_plan(tmp_path, world), world[2], mode="official")
    assert any("approved" in blocker for blocker in report.official_blockers)
    assert any("purpose" in blocker for blocker in report.official_blockers)


def test_missing_scientific_setting_fails_clearly(world, tmp_path):
    settings = tomllib.loads(world[3].config.raw_toml.decode())
    del settings["decoding"]
    plan = make_plan(tmp_path, world, settings=settings)
    with pytest.raises(DriverError, match="decoding"):
        dry_run(plan, world[2], mode="mock")


def test_sampling_without_a_seed_registry_fails_before_anything_is_written(world, tmp_path):
    settings = tomllib.loads(world[3].config.raw_toml.decode())
    settings["decoding"] = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    plan = make_plan(tmp_path, world, settings=settings)
    with pytest.raises(DriverError, match="seed"):
        dry_run(plan, world[2], mode="mock")
    with pytest.raises(DriverError, match="seed"):
        run_mock(plan, world)
    assert not plan.run_root.exists()


def test_plan_file_with_unknown_or_missing_keys_is_refused(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"run_root": "x"}))
    with pytest.raises(DriverError, match="missing"):
        load_plan(path)
    path.write_text(json.dumps({"surprise": 1}))
    with pytest.raises(DriverError):
        load_plan(path)


def test_plan_file_round_trip(world, tmp_path):
    plan = make_plan(tmp_path, world)
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({k: (str(v) if v is not None else None) for k, v in vars(plan).items()})
    )
    assert load_plan(path) == plan


def plan_json(plan, **extra):
    data = {k: (str(v) if v is not None else None) for k, v in vars(plan).items()}
    return json.dumps({**data, **extra})


def test_context_sources_and_their_hash_come_together_or_not_at_all(world, tmp_path):
    plan = make_plan(tmp_path, world)
    path = tmp_path / "plan.json"
    path.write_text(plan_json(plan, context_sources="sources.json"))
    with pytest.raises(DriverError, match="together"):
        load_plan(path)
    path.write_text(
        plan_json(plan, context_sources="sources.json", context_sources_sha256="a" * 64)
    )
    loaded = load_plan(path)
    assert loaded.context_sources == tmp_path / "sources.json"
    assert loaded.context_sources_sha256 == "a" * 64


def test_sources_must_be_given_exactly_one_way(world, tmp_path):
    plan = make_plan(tmp_path, world)
    with pytest.raises(DriverError, match="needed"):
        driver.resolve_sources(plan, None)
    pinned = replace(plan, context_sources=tmp_path / "s.json", context_sources_sha256="a" * 64)
    with pytest.raises(DriverError, match="not both"):
        driver.resolve_sources(pinned, "pkg.mod:fn")
    with pytest.raises(DriverError, match="unreadable"):  # a missing spec is a refusal, no fallback
        driver.resolve_sources(pinned, None)


def test_command_line_refuses_without_context_sources(world, tmp_path, capsys):
    plan = make_plan(tmp_path, world)
    path = tmp_path / "plan.json"
    path.write_text(plan_json(plan))
    assert driver.main(["dry-run", "--plan", str(path), "--mode", "mock"]) == 1
    assert "context sources are needed" in capsys.readouterr().err
    assert not plan.run_root.exists()


# ---- execution, anchor and verification ------------------------------------------------------


def test_mock_run_completes_is_anchored_outside_the_run_and_verifies(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    assert outcome.summary.complete == 2 and outcome.mode == "mock"
    anchor_file = plan.anchor_dir / f"{outcome.run_id}.anchor.json"
    assert anchor_file.is_file() and not anchor_file.is_relative_to(plan.run_root)
    assert anchor_file.stat().st_mode & 0o222 == 0  # stored read-only
    stored = Anchor.model_validate_json(anchor_file.read_bytes())
    assert stored.manifest_sha256 == outcome.manifest_sha256
    assert outcome.anchor_file_sha256 == sha256_bytes(anchor_file.read_bytes())
    verified = verify(
        plan, outcome.run_id, mode="mock", anchor_file_sha256=outcome.anchor_file_sha256
    )
    assert isinstance(verified, NonOfficialResults) and verified.backend_kind == "mock"
    assert {r.status for r in verified.results} == {"completed"}


def test_a_mock_run_is_never_verified_as_official(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    with pytest.raises(DriverError):
        verify(plan, outcome.run_id, mode="official")


def test_anchor_inside_the_run_root_is_refused_before_anything_is_written(world, tmp_path):
    plan = make_plan(tmp_path, world)
    inside = DriverPlan(**{**vars(plan), "anchor_dir": plan.run_root / "anchors"})
    with pytest.raises(DriverError, match="outside"):
        run_mock(inside, world)
    assert not plan.run_root.exists()
    around = DriverPlan(**{**vars(plan), "run_root": plan.anchor_dir / "runs"})
    with pytest.raises(DriverError, match="outside"):
        run_mock(around, world)
    assert not plan.anchor_dir.exists()


def test_deleted_or_altered_anchor_is_found(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    anchor_file = plan.anchor_dir / f"{outcome.run_id}.anchor.json"
    with pytest.raises(DriverError, match="anchor file"):
        verify(plan, outcome.run_id, mode="mock", anchor_file_sha256="0" * 64)
    forged = Anchor.model_validate_json(anchor_file.read_bytes()).model_copy(
        update={"manifest_sha256": "1" * 64}
    )
    anchor_file.chmod(0o644)
    anchor_file.write_bytes(forged.model_dump_json().encode())
    with pytest.raises(Exception):  # noqa: B017  the run no longer matches its anchor
        verify(plan, outcome.run_id, mode="mock")
    anchor_file.unlink()
    with pytest.raises(DriverError, match="no anchor"):
        verify(plan, outcome.run_id, mode="mock")


def test_a_different_anchor_is_never_overwritten(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    anchor_file = plan.anchor_dir / f"{outcome.run_id}.anchor.json"
    before = anchor_file.read_bytes()
    other = Anchor.model_validate_json(before).model_copy(update={"manifest_sha256": "2" * 64})
    with pytest.raises(DriverError, match="different anchor"):
        driver.write_anchor(plan.anchor_dir, plan.run_root, other)
    assert anchor_file.read_bytes() == before


def test_a_second_driver_on_the_same_root_is_refused(world, tmp_path):
    plan = make_plan(tmp_path, world)
    plan.run_root.mkdir()
    with driver.single_writer(plan.run_root), pytest.raises(ConcurrentRun):
        run_mock(plan, world)
    assert tree(plan.run_root) == ["driver.lock"]  # nothing else was written
    run_mock(plan, world)  # the lock was released


def test_killed_run_resumes_only_after_explicit_recovery(world, tmp_path):
    plan = make_plan(tmp_path, world)

    class KillsOnSecond(MockBackend):
        def generate(self, rendered, *, seed, max_new_tokens):
            if self.calls:
                raise Killed()
            return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)

    with pytest.raises(Killed):
        run_mock(plan, world, backend=KillsOnSecond())
    (run_id,) = [p.name for p in (plan.run_root / "runs").iterdir()]
    assert not plan.anchor_dir.exists()  # an unfinished run is never anchored
    with pytest.raises(UnsafeResume):
        run_mock(plan, world, resume=run_id)
    again = MockBackend()
    outcome = run_mock(plan, world, resume=run_id, recover_interrupted=True, backend=again)
    assert outcome.run_id == run_id and outcome.summary.complete == 2
    assert len(again.calls) == 1  # the finished observation was not generated again
    assert [p.name for p in (plan.run_root / "runs").iterdir()] == [run_id]


def test_finished_run_can_be_resumed_without_changing_anything(world, tmp_path):
    plan = make_plan(tmp_path, world)
    first = run_mock(plan, world)
    again = MockBackend()
    second = run_mock(plan, world, resume=first.run_id, backend=again)
    assert again.calls == [] and second.manifest_sha256 == first.manifest_sha256


def test_backend_kind_must_match_the_requested_mode(world, tmp_path):
    plan = make_plan(tmp_path, world)

    class Smoke(MockBackend):
        identity = BackendIdentity(kind="smoke", name="smoke", version=None)

    with pytest.raises(DriverError, match="mode"):
        run_mock(plan, world, backend=Smoke())
    assert tree(plan.run_root) == ["driver.lock"] and not plan.anchor_dir.exists()


# ---- official execution ---------------------------------------------------------------------


def test_official_mode_is_disabled_until_a_checkpoint_is_approved(world, tmp_path):
    plan = make_plan(tmp_path, world)
    assert generator_config.APPROVED_CHECKPOINTS == {}
    with pytest.raises(OfficialDisabled):
        execute(plan, world[2], mode="official", start=never_started)
    assert not plan.run_root.exists() and not plan.anchor_dir.exists()


def test_official_mode_refuses_a_synthetic_configuration_even_when_approved(
    world, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): "a" * 64}
    )
    with pytest.raises(OfficialDisabled, match="purpose"):
        execute(make_plan(tmp_path, world), world[2], mode="official", start=never_started)


def test_official_flow_end_to_end_with_a_scripted_backend(world, tmp_path, tiny, monkeypatch):
    """With every approval simulated, the driver reaches a verified official handoff."""
    monkeypatch.setattr(official, "OFFICIAL_BACKENDS", frozenset({HF_BACKEND_NAME, SCRIPT}))
    monkeypatch.setattr(
        generator_config,
        "APPROVED_CHECKPOINTS",
        {("Qwen/Qwen3-8B", REV): inventory_snapshot(tiny, REV).manifest_sha256},
    )
    settings = tomllib.loads(world[3].config.raw_toml.decode())
    settings["purpose"] = "development_candidate"
    settings["model"]["revision"] = REV
    settings["budgets"]["max_new_tokens"] = 64
    plan = make_plan(tmp_path, world, settings=settings)

    def start(prepared):
        backend = Scripted(prepared.renderer, tiny, "official")
        return backend, backend.environment

    outcome = execute(plan, world[2], mode="official", start=start)
    assert outcome.mode == "official" and outcome.summary.complete == 2
    verified = verify(plan, outcome.run_id, mode="official")
    assert isinstance(verified, VerifiedOfficialRun) and len(verified.results) == 2


# ---- generality -----------------------------------------------------------------------------


def test_driver_contains_no_item_or_paper_specific_logic():
    source = Path(driver.__file__).read_text().lower()
    for word in ("p027", "q_a", "tp53", "pmc"):
        assert word not in source


def test_anchor_that_does_not_read_back_as_written_is_refused(world, tmp_path, monkeypatch):
    plan = make_plan(tmp_path, world)
    monkeypatch.setattr(
        driver, "publish_bytes", lambda path, content, digest: path.write_bytes(content + b" ")
    )
    with pytest.raises(DriverError, match="retained hash"):
        run_mock(plan, world)


def test_execution_verifies_the_run_against_the_anchor_it_reads_back(world, tmp_path, monkeypatch):
    plan = make_plan(tmp_path, world)
    real = driver.read_anchor
    monkeypatch.setattr(
        driver,
        "read_anchor",
        lambda *a, **k: real(*a, **k).model_copy(update={"manifest_sha256": "3" * 64}),
    )
    with pytest.raises(CorruptRecord):
        run_mock(plan, world)


def test_verification_with_another_configuration_is_refused(world, tmp_path):
    plan = make_plan(tmp_path, world)
    outcome = run_mock(plan, world)
    settings = tomllib.loads(world[3].config.raw_toml.decode())
    settings["budgets"]["max_new_tokens"] = 64
    (tmp_path / "other").mkdir()
    other = make_plan(tmp_path / "other", world, settings=settings)
    other = DriverPlan(**{**vars(other), "run_root": plan.run_root, "anchor_dir": plan.anchor_dir})
    with pytest.raises(LifecycleError):
        verify(other, outcome.run_id, mode="mock")


# ---- command line -------------------------------------------------------------------------


def write_plan_file(plan, path):
    fields = {k: (None if v is None else str(v)) for k, v in vars(plan).items()}
    path.write_text(json.dumps(fields))
    return path


def test_command_line_dry_run_run_and_verify(world, tmp_path, monkeypatch, capsys):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "fake_sources", SimpleNamespace(provide=lambda: world[2]))
    plan = make_plan(tmp_path, world)
    plan_file = write_plan_file(plan, tmp_path / "plan.json")
    base = ["--plan", str(plan_file), "--mode", "mock"]
    sources = ["--sources", "fake_sources:provide"]

    assert driver.main(["dry-run", *base, *sources]) == 0
    assert json.loads(capsys.readouterr().out)["observations"] == 2
    assert not plan.run_root.exists()

    assert driver.main(["run", *base, *sources]) == 0
    out = capsys.readouterr().out
    run_id = next(p.name for p in (plan.run_root / "runs").iterdir())
    assert run_id in out and "retain the anchor file hash" in out
    assert driver.main(["verify", *base, "--run-id", run_id]) == 0
    assert (
        driver.main(["verify", "--plan", str(plan_file), "--mode", "official", "--run-id", run_id])
        == 1
    )


def test_command_line_refuses_official_without_loading_anything(
    world, tmp_path, monkeypatch, capsys
):
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "fake_sources", SimpleNamespace(provide=lambda: world[2]))
    plan_file = write_plan_file(make_plan(tmp_path, world), tmp_path / "plan.json")
    code = driver.main(
        ["run", "--plan", str(plan_file), "--mode", "official", "--sources", "fake_sources:provide",
         "--model-dir", str(tmp_path / "absent"), "--device", "cuda:0"]
    )  # fmt: skip
    assert code == 1 and "official execution is disabled" in capsys.readouterr().err
    assert not (tmp_path / "runs_root").exists()
