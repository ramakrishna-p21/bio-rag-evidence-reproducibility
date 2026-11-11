"""Synthetic end-to-end path from an authorized roster to a non-official analysis table.

authorized roster -> exact rendered input -> run store -> scripted mock execution ->
raw-output publication -> read-back -> parser -> Stage C -> analysis table.

Nothing here is an observation of Qwen3-8B. Every answer is scripted by the test, and the
run is labelled as a mock run throughout.
"""

import json

import pytest
from generation_support import ROOT
from test_generation_attempts import Clock, environment
from test_generation_roster import (
    build,
    make_renderer,
    make_scope,
)
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.analysis_table import (
    Observation,
    build_table,
    parse_csv,
    serialize_csv,
)
from biorag_repro.generation import official
from biorag_repro.generation.attempts import (
    CorruptRecord,
    DuplicateCompletion,
    NonOfficialBackend,
    RunStore,
    UnsafeResume,
)
from biorag_repro.generation.completion import complete_run, verify_completed_run
from biorag_repro.generation.environment import BackendIdentity
from biorag_repro.generation.official import (
    NotVerifiedOfficial,
    VerifiedOfficialRun,
    check_official_lineage,
    verify_official_run,
)
from biorag_repro.generation.runner import BackendFailure, nonofficial_results, run_roster

pytestmark = pytest.mark.usefixtures("driver_clock")


@pytest.fixture
def world(roster_world):  # noqa: F811
    return roster_world


class Killed(BaseException):
    """Stands in for a killed process."""


def answer(value, abstained=False):
    return json.dumps({"answer": value, "abstained": abstained})


# What the scripted backend says for each item, first attempt first. A string is raw model
# text; an exception is raised by the backend. The text values are the test's own.
CORRECT = {"q_a": answer("0.05"), "q_b": answer("p53")}


class Scripted:
    identity = BackendIdentity(kind="mock", name="scripted", version=None)

    def __init__(self, entries, script=None):
        self.item_of = {e.rendered.rendered_sha256: e.item_id for e in entries}
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls = []

    def generate(self, rendered, *, seed, max_new_tokens):
        item = self.item_of[rendered.rendered_sha256]
        self.calls.append((item, rendered.rendered_sha256, seed))
        queue = self.script.get(item)
        action = queue.pop(0) if queue else CORRECT[item]
        if isinstance(action, BaseException):
            raise action
        return action


def new_store(tmp_path, config):
    return RunStore.create(
        tmp_path,
        config=config,
        phase="development",
        benchmark_version="benchmark_v1",
        backend_kind="mock",
        now=Clock(),
    )


def execute(store, built, backend):
    return run_roster(
        store, built.entries, backend, capture_environment=lambda: environment("mock")
    )


def reopen(store, config):
    return RunStore.open(store.root, store.run_id, config=config, backend_kind="mock")


def setup(world, tmp_path, repeats=(1,), script=None):
    directory, record_sha, _, renderer, _ = world
    scope = make_scope(directory, record_sha, {"q_a", "q_b"}, repeats=repeats)
    built = build(world, scope=scope)
    store = new_store(tmp_path, renderer.config)
    return store, built, scope, Scripted(built.entries, script)


def score_all(world, store):
    items = {i.item_id: i for i in read_development_view(world[0], world[1])[1]}
    results = nonofficial_results(store).results
    return results, {r.observation_key: stage_c.score(items[r.item_id], r) for r in results}, items


def table_for(world, store, scope, config):
    results, scores, items = score_all(world, store)
    observations = [
        Observation(
            item=items[r.item_id],
            phase="development",
            matrix_arm=r.matrix_arm,
            condition=r.representation_or_condition,
            repeat_index=r.repeat_index,
            resolved_config_sha256=config.configuration_sha256,
            retrieval_config_sha256=config.settings.retrieval_config_sha256,
            result=r,
            score=scores[r.observation_key],
        )
        for r in results
    ]
    return build_table(observations, scope)


# ---- the successful path ---------------------------------------------------------------


def test_authorized_roster_runs_scores_and_tabulates(world, tmp_path):
    store, built, scope, backend = setup(world, tmp_path)
    summary = execute(store, built, backend)
    assert summary.complete == 2 and summary.attempts == 2
    anchor = complete_run(store, built.record)
    done = verify_completed_run(
        tmp_path, store.run_id, anchor, config=store.config, backend_kind="mock"
    )
    assert done.roster == built.record
    _results, scores, _ = score_all(world, store)
    assert {s.status for s in scores.values()} == {"correct"}
    table = table_for(world, store, scope, store.config)
    assert [(r.item_id, r.execution_status, r.stage_c_status) for r in table.rows] == [
        ("q_a", "completed", "correct"),
        ("q_b", "completed", "correct"),
    ]
    assert parse_csv(serialize_csv(table.rows)) == table.rows
    assert {r.run_id for r in table.rows} == {store.run_id}


def test_every_attempt_received_the_rostered_input_and_context(world, tmp_path):
    store, built, _, backend = setup(world, tmp_path)
    execute(store, built, backend)
    rostered = {r.item_id: r.rendered_input_sha256 for r in built.record.rows}
    for state in store.verify():
        started = [e for e in state.events if e.state == "running"]
        assert {e.rendered_input_sha256 for e in started} == {rostered[state.coordinates.item_id]}
    for entry in built.entries:
        assert entry.rendered.context  # the exact assembled evidence, not a summary


def test_repeats_are_separate_observations_over_one_input(world, tmp_path):
    store, built, scope, backend = setup(world, tmp_path, repeats=(1, 2, 3))
    execute(store, built, backend)
    assert len(store.observation_keys()) == 6
    assert len({call[1] for call in backend.calls}) == 2  # one rendered input per item
    table = table_for(world, store, scope, store.config)
    assert sorted({r.repeat_index for r in table.rows}) == [1, 2, 3]
    assert [r.primary_row for r in table.rows if r.item_id == "q_a"] == [True, False, False]


# ---- interruption, retry, failure --------------------------------------------------------


def test_killed_run_resumes_without_regenerating_finished_observations(world, tmp_path):
    store, built, _scope, first = setup(world, tmp_path, script={"q_b": [Killed()]})
    with pytest.raises(Killed):
        execute(store, built, first)
    resumed = reopen(store, store.config)
    second = Scripted(built.entries)
    with pytest.raises(UnsafeResume):
        execute(resumed, built, second)
    resumed.recover_interrupted(previous_worker_stopped=True)
    execute(resumed, built, second)
    assert [call[0] for call in second.calls] == ["q_b"]  # q_a was not generated again
    read_back = verify_completed_run(
        tmp_path,
        store.run_id,
        complete_run(resumed, built.record),
        config=store.config,
        backend_kind="mock",
    )
    q_b = next(o for o in read_back.completion.observations if o.item_id == "q_b")
    assert q_b.attempts == 2


def test_retryable_failure_is_retried_with_the_identical_input(world, tmp_path):
    store, built, _, backend = setup(
        world, tmp_path, script={"q_a": [BackendFailure("transport_error", "reset")]}
    )
    summary = execute(store, built, backend)
    assert summary.attempts == 3 and summary.complete == 2
    q_a_calls = [call for call in backend.calls if call[0] == "q_a"]
    assert len(q_a_calls) == 2 and q_a_calls[0] == q_a_calls[1]


def test_terminal_failure_is_missing_not_incorrect(world, tmp_path):
    store, built, scope, backend = setup(
        world, tmp_path, script={"q_a": [BackendFailure("bad_request", "refused")]}
    )
    execute(store, built, backend)
    complete_run(store, built.record)  # a decided failure still closes the run
    results, scores, _ = score_all(world, store)
    failed = next(r for r in results if r.item_id == "q_a")
    assert failed.status == "operational_failure"
    score = scores[failed.observation_key]
    assert (score.status, score.reason) == ("missing", "operational_failure")
    rows = {r.item_id: r for r in table_for(world, store, scope, store.config).rows}
    assert rows["q_a"].execution_status == "operational_failure"
    assert rows["q_a"].completion_kind is None


@pytest.mark.parametrize(
    ("raw", "completion", "status", "reason"),
    [
        ("The answer is 0.05", "unparseable", "incorrect", "unparseable"),
        ("{not json", "unparseable", "incorrect", "unparseable"),
        (answer(""), "answer", "incorrect", "empty_answer"),
        (answer(None, abstained=True), "abstained", "incorrect", "abstained"),
        (answer("0.5"), "answer", "incorrect", "value_mismatch"),
    ],
)
def test_malformed_and_wrong_outputs_are_scored_not_dropped(
    world, tmp_path, raw, completion, status, reason
):
    store, built, scope, backend = setup(world, tmp_path, script={"q_a": [raw]})
    execute(store, built, backend)
    results, scores, _ = score_all(world, store)
    result = next(r for r in results if r.item_id == "q_a")
    assert result.status == "completed" and result.raw_output == raw
    assert (scores[result.observation_key].status, scores[result.observation_key].reason) == (
        status,
        reason,
    )
    rows = {r.item_id: r for r in table_for(world, store, scope, store.config).rows}
    assert rows["q_a"].completion_kind == completion


# ---- integrity ---------------------------------------------------------------------------------


def test_corrupted_raw_output_stops_resume_and_read_back(world, tmp_path):
    store, built, _, backend = setup(world, tmp_path)
    execute(store, built, backend)
    anchor = complete_run(store, built.record)
    raw = next(store.run_dir.glob("observations/*/raw/*.txt"))
    raw.chmod(0o644)
    raw.write_text(answer("0.99"))
    with pytest.raises(CorruptRecord):
        verify_completed_run(
            tmp_path, store.run_id, anchor, config=store.config, backend_kind="mock"
        )
    untouched = Scripted(built.entries)
    with pytest.raises(ValueError):
        execute(reopen_unchecked(store), built, untouched)
    assert untouched.calls == []


def reopen_unchecked(store):
    return RunStore.open(store.root, store.run_id, config=store.config, backend_kind="mock")


def test_the_same_observations_cannot_be_completed_in_two_runs(world, tmp_path):
    first, built, _, backend = setup(world, tmp_path)
    execute(first, built, backend)
    complete_run(first, built.record)
    second = new_store(tmp_path, first.config)
    execute(second, built, Scripted(built.entries))
    with pytest.raises(DuplicateCompletion):
        complete_run(second, built.record)


def test_a_mock_run_is_refused_at_the_official_boundary(world, tmp_path, monkeypatch):
    monkeypatch.setattr(official, "OFFICIAL_BACKENDS", frozenset({"scripted", "synthetic"}))
    store, built, _, backend = setup(world, tmp_path)
    execute(store, built, backend)
    anchor = complete_run(store, built.record)
    with pytest.raises(NonOfficialBackend, match="mock run"):
        verify_official_run(
            tmp_path,
            store.run_id,
            anchor,
            config=store.config,
            view_directory=world[0],
            view_record_sha256=world[1],
            tokenizer_root=ROOT,
        )
    never = VerifiedOfficialRun("x", "0" * 64, [], frozenset(), None, None)  # type: ignore[arg-type]
    with pytest.raises(NotVerifiedOfficial):
        check_official_lineage(nonofficial_results(store).results, never)


# ---- closed book --------------------------------------------------------------------------------


def test_closed_book_path_has_no_context_anywhere(world, tmp_path):
    directory, record_sha, _, _, _ = world
    renderer = make_renderer(None, arm="closed_book")
    scope = make_scope(
        directory,
        record_sha,
        {"q_a", "q_b"},
        arm="closed_book",
        conditions=("CLOSED_BOOK",),
        repeats=(1, 2, 3),
    )
    built = build(world, scope=scope, sources=[], renderer=renderer)
    store = new_store(tmp_path, renderer.config)
    execute(store, built, Scripted(built.entries))
    complete_run(store, built.record)
    assert all(e.rendered.context is None for e in built.entries)
    table = table_for(world, store, scope, renderer.config)
    assert {r.context_provenance for r in table.rows} == {"not_applicable"}
    assert {r.condition for r in table.rows} == {"CLOSED_BOOK"}


# ---- stored inputs and execution receipts over the authoritative roster ------------------------


class ScriptedReceipts(Scripted):
    """Scripted answers with the execution account a real backend would give."""

    def __init__(self, entries, config, script=None):
        super().__init__(entries, script)
        self.config = config

    def generate(self, rendered, *, seed, max_new_tokens):
        from types import SimpleNamespace

        from test_generation_receipts import facts_for

        from biorag_repro.generation.runner import GenerationOutput

        text = super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)
        return GenerationOutput(text, facts_for(SimpleNamespace(rendered=rendered), self.config))


def receipt_setup(world, tmp_path, script=None, repeats=(1,)):
    store, built, scope, _ = setup(world, tmp_path, repeats=repeats)
    return store, built, scope, ScriptedReceipts(built.entries, store.config, script)


def row_of(built, item_id, repeat=1):
    return next(r for r in built.record.rows if (r.item_id, r.repeat_index) == (item_id, repeat))


def test_each_completed_observation_has_a_receipt_for_its_rostered_input(world, tmp_path):
    from biorag_repro.core.hashing import sha256_bytes
    from biorag_repro.generation.official import simulated_generator_input

    store, built, scope, backend = receipt_setup(world, tmp_path)
    execute(store, built, backend)
    anchor = complete_run(store, built.record)
    done = verify_completed_run(
        tmp_path, store.run_id, anchor, config=store.config, backend_kind="mock"
    )
    results, _, _ = score_all(world, store)
    for state in done.states:
        receipt = store.receipt_of(state)
        row = row_of(built, state.coordinates.item_id)
        assert receipt is not None and receipt.input_record_sha256 == row.input_record.sha256
    for result in results:
        link = simulated_generator_input(done, result)
        row = row_of(built, result.item_id)
        source = next(x for x in world[2] if x.selection.query.query_id == result.item_id)
        assert link.provenance == "simulated" and link.context == source.context
        entry = next(e for e in built.entries if e.item_id == result.item_id)
        assert link.exact_context_sha256 == sha256_bytes(entry.rendered.context.encode())
        assert link.input.sha256 == row.input_record.sha256
    # A receipt alone does not upgrade a row: only an attached, verified link does.
    table = table_for(world, store, scope, store.config)
    assert {r.context_provenance for r in table.rows} == {"assembled_only"}


def test_killed_run_keeps_finished_receipts_and_gives_the_retried_attempt_its_own(world, tmp_path):
    store, built, _scope, first = receipt_setup(world, tmp_path, script={"q_b": [Killed()]})
    with pytest.raises(Killed):
        execute(store, built, first)
    q_a_state = next(s for s in store.verify() if s.coordinates.item_id == "q_a")
    receipt_file = store.run_dir / "observations" / q_a_state.key / "receipts"
    before = {p.name: p.read_bytes() for p in receipt_file.iterdir()}
    resumed = reopen(store, store.config)
    resumed.recover_interrupted(previous_worker_stopped=True)
    second = ScriptedReceipts(built.entries, store.config)
    execute(resumed, built, second)
    assert {p.name: p.read_bytes() for p in receipt_file.iterdir()} == before
    q_b_state = next(s for s in resumed.verify() if s.coordinates.item_id == "q_b")
    receipts = list((store.run_dir / "observations" / q_b_state.key / "receipts").iterdir())
    assert [p.stem for p in receipts] == [q_b_state.events[-1].attempt_id]  # not the killed attempt
    complete_run(resumed, built.record)


def test_a_retry_after_a_backend_failure_receives_a_receipt_only_when_it_completes(world, tmp_path):
    store, built, _, backend = receipt_setup(
        world, tmp_path, script={"q_a": [BackendFailure("transport_error", "reset")]}
    )
    execute(store, built, backend)
    state = next(s for s in store.verify() if s.coordinates.item_id == "q_a")
    assert state.attempts == 2
    receipts = list((store.run_dir / "observations" / state.key / "receipts").iterdir())
    assert [p.stem for p in receipts] == [state.events[-1].attempt_id]


def test_malformed_and_abstained_outputs_are_still_accounted_for_and_scored(world, tmp_path):
    store, built, scope, backend = receipt_setup(
        world, tmp_path, script={"q_a": ["not json at all"], "q_b": [answer(None, abstained=True)]}
    )
    execute(store, built, backend)
    complete_run(store, built.record)
    results, scores, _ = score_all(world, store)
    by_item = {r.item_id: r for r in results}
    assert by_item["q_a"].parse_status == "unparseable" and by_item["q_b"].abstained is True
    assert all(store.receipt_of(s) is not None for s in store.verify())
    assert scores[by_item["q_a"].observation_key].status != "correct"
    assert len(table_for(world, store, scope, store.config).rows) == 2


@pytest.mark.parametrize("damage", ["alter", "delete"])
def test_a_damaged_stored_input_stops_read_back_and_resume(world, tmp_path, damage):
    store, built, _, backend = receipt_setup(world, tmp_path)
    execute(store, built, backend)
    anchor = complete_run(store, built.record)
    path = store.run_dir / built.record.rows[0].input_record.logical_ref
    path.chmod(0o644)
    if damage == "delete":
        path.unlink()
    else:
        path.write_bytes(path.read_bytes().replace(b"Question", b"Questiox", 1))
    with pytest.raises(CorruptRecord):
        verify_completed_run(
            tmp_path, store.run_id, anchor, config=store.config, backend_kind="mock"
        )
    with pytest.raises(CorruptRecord):
        reopen_unchecked(store)


@pytest.mark.model_smoke
def test_hugging_face_smoke_run_over_the_authoritative_roster(world, tmp_path):
    """Tiny random weights on CPU. The outputs are noise and are never scientific."""
    from generation_support import QWEN_REVISION
    from test_generation_environment import cpu_probes
    from tiny_model import build_tiny, load_once

    from biorag_repro.generation.hf_backend import HuggingFaceBackend
    from biorag_repro.generation.official import simulated_generator_input

    renderer = world[3]
    model_dir = tmp_path / "model" / QWEN_REVISION
    model_dir.mkdir(parents=True)
    build_tiny(model_dir)
    scope = make_scope(world[0], world[1], {"q_a", "q_b"})
    built = build(world, scope=scope)
    store = RunStore.create(
        tmp_path / "runs",
        config=renderer.config,
        phase="development",
        benchmark_version="benchmark_v1",
        backend_kind="smoke",
        now=Clock(),
    )
    backend = HuggingFaceBackend(
        renderer.config,
        model_dir=model_dir,
        tokenizer_root=ROOT,
        device="cpu",
        load_model=load_once,
    )
    assert backend.identity.kind == "smoke"
    summary = run_roster(
        store, built.entries, backend, capture_environment=lambda: backend.environment(cpu_probes())
    )
    assert summary.complete == 2
    anchor = complete_run(store, built.record)
    done = verify_completed_run(
        tmp_path / "runs", store.run_id, anchor, config=renderer.config, backend_kind="smoke"
    )
    handoff = nonofficial_results(store)
    assert handoff.backend_kind == "smoke"
    assert {simulated_generator_input(done, r).provenance for r in handoff.results} == {"simulated"}
    for state in done.states:
        receipt = store.receipt_of(state)
        assert receipt.facts.stop_reason in ("eos", "max_new_tokens")
        assert (
            receipt.facts.generation.max_new_tokens
            == renderer.config.settings.budgets.max_new_tokens
        )
    table = table_for(world, store, scope, renderer.config)
    assert {r.execution_status for r in table.rows} == {"completed"}
    # Random weights do not write the answer schema, so nothing here can be scored correct.
    assert "correct" not in {r.stage_c_status for r in table.rows}
    monkey_manifest = backend.environment(cpu_probes()).execution.snapshot.manifest_sha256
    from biorag_repro.generation import config as generator_config

    assert (QWEN_REVISION and generator_config.APPROVED_CHECKPOINTS == {}) and monkey_manifest
    with pytest.raises(NonOfficialBackend):
        verify_official_run(
            tmp_path / "runs",
            store.run_id,
            anchor,
            config=renderer.config,
            view_directory=world[0],
            view_record_sha256=world[1],
            tokenizer_root=ROOT,
        )
