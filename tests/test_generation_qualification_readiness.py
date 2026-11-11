"""Local qualification readiness: the qualification label, the largest legal input, a restart.

Everything is synthetic. Nothing here qualifies a model, a GPU or a configuration, and no
acceptance threshold is stated: the tests check that records are identified, bounded and
reported honestly.
"""

# ruff: noqa: F811  (fixtures imported from other test modules)

import pytest
from generation_support import needs_tokenizer, resolve
from pydantic import ValidationError
from synthetic_environment import LOCK, approved_locks
from synthetic_environment import evidence as synthetic_evidence
from test_generation_attempts import Clock, coordinates
from test_generation_completion import record_for
from test_generation_driver import (
    Killed,
    make_plan,
    official_setup,
    run_mock,
    verify,
    world,  # noqa: F401  (fixture)
)
from test_generation_hf_backend import (
    FakeModel,
    backend_for,
    make_renderer,
    rendered_for,
)
from test_generation_official import Scripted
from test_generation_qualification import measured_memory, measuring
from test_generation_roster import make_scope
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)
from test_generation_runner import BENCHMARK, make_roster, settings_for

from biorag_repro.evaluation.analysis_models import RosterItem
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.attempts import CorruptRecord, RunRecord, RunStore
from biorag_repro.generation.completion import RosterMismatch, complete_run
from biorag_repro.generation.driver import DriverError, execute
from biorag_repro.generation.handoff import official_observations
from biorag_repro.generation.official import NotVerifiedOfficial, VerifiedOfficialRun
from biorag_repro.generation.qualification import (
    official_qualification_report,
    qualification_report,
)
from biorag_repro.generation.runner import BackendFailure

pytestmark = pytest.mark.usefixtures("driver_clock")

# ---- the qualification label ------------------------------------------------------------


def create(tmp_path, phase):
    config = resolve(settings_for("mock"))
    store = RunStore.create(
        tmp_path / phase,
        config=config,
        phase=phase,
        benchmark_version=BENCHMARK,
        backend_kind="mock",
        now=Clock(),
    )
    return store, config


def test_the_same_configuration_gives_different_run_and_observation_identities_per_phase(
    tmp_path,
):
    development, _ = create(tmp_path, "development")
    qualification, _ = create(tmp_path, "qualification")
    assert qualification.record.phase == "qualification"
    assert development.run_sha256 != qualification.run_sha256
    assert development.key_for(coordinates()) != qualification.key_for(coordinates())


def test_the_final_test_label_is_still_refused(tmp_path):
    with pytest.raises(ValidationError):
        create(tmp_path, "final_test")
    assert RunRecord.model_fields["phase"].annotation.__args__ == ("development", "qualification")


def test_completion_refuses_a_roster_of_the_other_phase(tmp_path):
    store, config = create(tmp_path, "qualification")
    roster = make_roster(config)
    record = record_for(config, roster)  # a development roster
    with pytest.raises(RosterMismatch):
        complete_run(store, record)


def qualification_scope(world):
    directory, record_sha = world[0], world[1]
    return make_scope(directory, record_sha, {"q_a", "q_b"}, phase="qualification")


def test_a_mock_qualification_run_is_labelled_through_the_driver_and_the_report(world, tmp_path):
    plan = make_plan(tmp_path, world, scope=qualification_scope(world))
    outcome = run_mock(plan, world)
    store = RunStore.open(
        plan.run_root,
        outcome.run_id,
        config=world[3].config,
        backend_kind="mock",
    )
    assert store.record.phase == "qualification"
    assert {s.coordinates.item_id for s in store.verify()} == {"q_a", "q_b"}
    report = qualification_report(
        plan.run_root,
        outcome.run_id,
        outcome.manifest_sha256,
        config=world[3].config,
        backend_kind="mock",
    )
    assert report.run.phase == "qualification"
    assert report.evidence_class == "synthetic_engineering"  # mock stays synthetic evidence
    results = verify(plan, outcome.run_id, mode="mock").results
    assert {r.phase for r in results} == {"qualification"}


def test_a_qualification_scope_must_match_the_development_view_exactly(world, tmp_path):
    scope = qualification_scope(world)
    foreign = RosterItem(item_id="q_not_in_view", state="excluded", reason="not a development item")
    for changed in (
        scope.model_copy(update={"items": [*scope.items, foreign]}),  # an item outside the view
        scope.model_copy(update={"items": scope.items[:-1]}),  # a development item left out
    ):
        plan = make_plan(tmp_path, world, scope=changed)
        with pytest.raises(DriverError, match="development"):
            run_mock(plan, world)


def test_a_qualification_run_never_becomes_a_stage_c_observation(
    world, tmp_path, tiny, monkeypatch
):
    plan = official_setup(world, tmp_path, tiny, monkeypatch)
    plan = plan.__class__(
        **{**vars(plan), "scope": _write_scope(tmp_path, qualification_scope(world))}
    )

    def start(prepared):
        backend = Scripted(prepared.renderer, tiny, "official")
        return backend, backend.environment

    outcome = execute(
        plan, world[2], mode="official", start=start, environment_evidence=synthetic_evidence()
    )
    verified = verify(plan, outcome.run_id, mode="official")
    assert isinstance(verified, VerifiedOfficialRun)
    assert official_qualification_report(verified).run.phase == "qualification"
    with pytest.raises(NotVerifiedOfficial, match="qualification"):
        official_observations(verified, [], lambda item_id, condition: None)


def _write_scope(tmp_path, scope):
    path = tmp_path / "qualification_scope.json"
    path.write_text(scope.model_dump_json())
    return path


# ---- the largest legal input ------------------------------------------------------------


@needs_tokenizer
def test_a_full_evidence_input_is_counted_exactly_and_nothing_is_cut(tiny):
    r = make_renderer(new_tokens=64)
    budgets = r.config.settings.budgets
    context = " x" * budgets.evidence_tokens
    counted = r.counter(context)
    assert counted == budgets.evidence_tokens == 2048
    rendered = r.render("What was measured?", context, counted)
    # independent count of the exact text the model would be given
    ids = r.tokenizer(rendered.rendered_text, add_special_tokens=False)["input_ids"]
    assert rendered.input_tokens == len(ids)
    assert context in rendered.rendered_text  # the whole context is present
    assert rendered.input_tokens + budgets.max_new_tokens <= budgets.context_window_tokens
    assert rendered.input_tokens <= budgets.max_input_tokens

    over = " x" * (budgets.evidence_tokens + 1)
    with pytest.raises(ValueError, match="evidence"):
        r.render("What was measured?", over, r.counter(over))


@needs_tokenizer
def test_an_input_one_token_over_the_limit_is_refused_by_the_renderer_and_the_backend(tiny):
    probe = make_renderer(new_tokens=64)
    context = " x" * probe.config.settings.budgets.evidence_tokens
    exact = probe.render("Q?", context, probe.counter(context)).input_tokens
    tight = make_renderer(new_tokens=64)
    tight.config.settings.budgets.max_input_tokens = exact - 1  # type: ignore[misc]
    with pytest.raises(ValueError, match="input"):
        tight.render("Q?", context, tight.counter(context))

    # the backend independently refuses a call that would pass the model's positions
    model = FakeModel()
    backend = backend_for(probe, tiny, load_model=lambda directory, device: model)
    rendered = rendered_for(probe)
    backend.max_positions = rendered.input_tokens + 64 - 1
    with pytest.raises(BackendFailure) as refusal:
        backend.generate(rendered, seed=None, max_new_tokens=64)
    assert refusal.value.kind == "input_too_long"
    assert model.seen == {}  # the model was never called, so nothing was shortened


# ---- interruption and restart of an official-shaped run ---------------------------------


def killing_start(tiny, after, measure):
    """A backend that completes `after` calls, then dies the way a killed process does."""

    def start(prepared):
        class Dies(Scripted):
            calls = 0

            def generate(self, *args, **kwargs):
                if self.calls >= after:
                    raise Killed()
                self.calls += 1
                return super().generate(*args, **kwargs)

        backend = Dies(prepared.renderer, tiny, "official", measure=measure)
        return backend, backend.environment

    return start


def test_a_killed_official_shaped_run_resumes_under_its_first_verification(
    world, tmp_path, tiny, monkeypatch
):
    plan = official_setup(world, tmp_path, tiny, monkeypatch)
    evidence = synthetic_evidence()
    measure = measuring(measured_memory)
    with pytest.raises(Killed):
        execute(
            plan, world[2], mode="official", start=killing_start(tiny, 1, measure),
            environment_evidence=evidence,
        )  # fmt: skip
    (run_id,) = [p.name for p in (plan.run_root / "runs").iterdir()]
    resumed = execute(
        plan, world[2], mode="official", start=killing_start(tiny, 99, measure),
        environment_evidence=evidence, resume=run_id, recover_interrupted=True,
    )  # fmt: skip
    assert resumed.run_id == run_id and resumed.summary.complete == 2
    assert resumed.summary.attempts == 1  # the decided observation was not generated again

    verified = verify(plan, run_id, mode="official")
    store = verified.completed.store
    cited = {
        e.environment_verification_sha256
        for key in store.observation_keys()
        for e in store.observation(key).events
        if e.state == "running"
    }
    assert len(cited) == 1 and None not in cited  # the interrupted attempt cites it too
    report = official_qualification_report(verified)
    assert report.measurements.model_dump() == {"numerator": 2, "denominator": 2}
    assert report.attempts.interrupted_events == 1
    assert report.generation_duration_ns.n == 2


def test_a_restart_cannot_substitute_another_environment_lock(world, tmp_path, tiny, monkeypatch):
    plan = official_setup(world, tmp_path, tiny, monkeypatch)
    with pytest.raises(Killed):
        execute(
            plan, world[2], mode="official", start=killing_start(tiny, 1, None),
            environment_evidence=synthetic_evidence(),
        )  # fmt: skip
    (run_id,) = [p.name for p in (plan.run_root / "runs").iterdir()]
    other = LOCK.replace(b"3.11.9", b"3.11.8")
    monkeypatch.setattr(generator_config, "APPROVED_ENVIRONMENT_LOCKS", approved_locks(other))
    substitute = synthetic_evidence(lock=other, python_version="3.11.8")
    with pytest.raises(CorruptRecord, match="different environment lock"):
        execute(
            plan, world[2], mode="official", start=killing_start(tiny, 99, None),
            environment_evidence=substitute, resume=run_id, recover_interrupted=True,
        )  # fmt: skip
