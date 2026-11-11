"""Mock-backed runner. Backends, configurations, papers and answers are all synthetic."""

import json
import random

import pytest
from generation_support import resolve, synthetic_settings
from test_generation_attempts import POLICY, Clock, environment
from test_generation_seeds import registry
from test_stage_c import SET_GOLD, entity, make_item, quantity

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation.attempts import (
    InvalidTransition,
    LifecycleError,
    NonOfficialBackend,
    PublicationConflict,
    RunMismatch,
    RunStore,
    UnsafeResume,
)
from biorag_repro.generation.environment import BackendIdentity
from biorag_repro.generation.prompt import RenderedInput
from biorag_repro.generation.runner import (
    BackendFailure,
    RosterEntry,
    RosterMismatch,
    nonofficial_results,
    run_roster,
)
from biorag_repro.generation.seeds import seed_registry_sha256

BENCHMARK = "benchmark_v1"  # the version make_item uses


class Crash(BaseException):
    """Stands in for a killed process: not an Exception, so nothing may record it."""


def good(answer, abstained=False):
    return json.dumps({"answer": answer, "abstained": abstained})


# Three synthetic papers, three answer classes; gold and the correct raw answer for each.
CASES = [
    ("q_a", "paperA", quantity("0.05"), good("0.05")),
    ("q_b", "paperB", entity("TP53", ["p53"]), good("p53")),
    ("q_c", "paperC", SET_GOLD, good(["TP53", "KRAS", "EGFR"])),
]
CONDITIONS = ("JATS", "PDF_DOCLING")


def item_for(item_id, paper, gold) -> BenchmarkItem:
    base = make_item(gold)
    text = base.model_dump_json().replace("P001", paper).replace("q_001", item_id)
    return BenchmarkItem.model_validate_json(text)


ITEMS = {case[0]: item_for(case[0], case[1], case[2]) for case in CASES}


def settings_for(kind="mock", **overrides):
    settings = synthetic_settings()
    settings["purpose"] = "development_candidate" if kind == "official" else "synthetic_test"
    settings["retry_max_attempts"] = POLICY.max_attempts
    settings["retryable_failures"] = list(POLICY.retryable_failures)
    settings.update(overrides)
    return settings


def make_store(tmp_path, kind="mock", **overrides):
    config = resolve(settings_for(kind, **overrides))
    store = RunStore.create(
        tmp_path,
        config=config,
        phase="development",
        benchmark_version=BENCHMARK,
        backend_kind=kind,
        now=Clock(),
    )
    return store, config


def rendered(config, question, context):
    text = f"Question: {question}\n" + (f"Context: {context}\n" if context else "")
    return RenderedInput(
        resolved_config_sha256=config.configuration_sha256,
        prompt_sha256=config.settings.prompt_sha256,
        chat_template_sha256="e" * 64,
        question=question,
        context=context,
        context_sha256=sha256_bytes(context.encode()) if context else None,
        context_tokens=7 if context else None,
        rendered_text=text,
        rendered_sha256=sha256_bytes(text.encode()),
        input_tokens=20,
        input_token_ids_sha256="f" * 64,
    )


def make_roster(config, repeats=(1,), conditions=CONDITIONS, closed_book=False):
    roster = []
    for item_id, paper, _, _ in CASES:
        for condition in conditions:
            for repeat in repeats:
                context = None if closed_book else f"evidence for {item_id} via {condition}"
                roster.append(
                    RosterEntry(
                        item_id=item_id,
                        paper_id=paper,
                        representation_or_condition=condition,
                        repeat_index=repeat,
                        rendered=rendered(config, f"question {item_id}?", context),
                    )
                )
    return roster


class MockBackend:
    """Scripted backend. `script[(item, condition)]` is a list of actions, used in order."""

    identity = BackendIdentity(kind="mock", name="scripted", version=None)

    def __init__(self, roster, script=None):
        self.by_input = {e.rendered.rendered_sha256: e for e in roster}
        self.answers = {c[0]: c[3] for c in CASES}
        self.script = script or {}
        self.calls = []

    def generate(self, rendered, *, seed, max_new_tokens):
        entry = self.by_input[rendered.rendered_sha256]
        name = (entry.item_id, entry.representation_or_condition)
        self.calls.append((*name, seed, max_new_tokens))
        queue = self.script.get(name)
        if queue:
            action = queue.pop(0)
            if isinstance(action, BaseException):
                raise action
            return action
        return self.answers[entry.item_id]

    def calls_for(self, item_id):
        return [c for c in self.calls if c[0] == item_id]


def go(store, config, roster, backend, kind="mock", seeds=None):
    return run_roster(
        store, roster, backend, capture_environment=lambda: environment(kind), seeds=seeds
    )


def event_bytes(store):
    return {
        str(p.relative_to(store.run_dir)): p.read_bytes()
        for p in store.run_dir.glob("observations/*/*")
        if p.is_file()
    } | {
        str(p.relative_to(store.run_dir)): p.read_bytes()
        for p in store.run_dir.glob("observations/*/events/*.json")
    }


def reopen(store, config, kind="mock"):
    return RunStore.open(store.root, store.run_id, config=config, backend_kind=kind)


# --- normal execution and the Stage C handoff ---------------------------------------------------


def test_full_roster_completes_and_scores_through_stage_c(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    backend = MockBackend(roster)
    summary = go(store, config, roster, backend)
    assert summary.complete == 6 and summary.attempts == 6 and len(backend.calls) == 6
    handoff = nonofficial_results(store)
    assert handoff.backend_kind == "mock" and len(handoff.results) == 6
    for result in handoff.results:
        assert isinstance(result, GenerationResult) and result.status == "completed"
        score = stage_c.score(ITEMS[result.item_id], result)
        assert (score.status, score.reason) == ("correct", None)
        assert result.run_id == store.run_id and result.phase == "development"
        assert result.resolved_config_sha256 == config.configuration_sha256


def test_result_lineage_matches_the_stored_records(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster, MockBackend(roster))
    for result in nonofficial_results(store).results:
        state = store.observation(result.observation_key)
        done = state.events[-1]
        assert result.attempt_id == done.attempt_id
        assert result.output == done.raw_output
        assert (store.root / result.output.logical_ref).read_text() == result.raw_output
        assert (result.answer, result.abstained) == (done.answer, done.abstained)
        assert result.paper_id == state.coordinates.paper_id
        assert result.matrix_arm == "confirmatory"
        assert result.representation_or_condition == state.coordinates.representation_or_condition


def test_planning_and_execution_order_is_independent_of_roster_order(tmp_path):
    orders = []
    for seed in (1, 2):
        store, config = make_store(tmp_path / str(seed))
        roster = make_roster(config)
        random.Random(seed).shuffle(roster)
        backend = MockBackend(roster)
        go(store, config, roster, backend)
        orders.append(([c[:2] for c in backend.calls], sorted(store.observation_keys())))
    assert orders[0][0] == orders[1][0] == sorted(orders[0][0])
    assert orders[0][1] == orders[1][1]


def test_generation_limits_come_from_the_configuration(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    backend = MockBackend(roster)
    go(store, config, roster, backend)
    assert {c[3] for c in backend.calls} == {config.settings.budgets.max_new_tokens}
    assert {c[2] for c in backend.calls} == {None}  # greedy: no seed supplied


# --- restart, resume, no regeneration --------------------------------------------------------------


def test_second_pass_makes_no_backend_calls_and_changes_nothing(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster, MockBackend(roster))
    before = event_bytes(store)
    resumed = reopen(store, config)
    again = MockBackend(roster)
    summary = go(resumed, config, roster, again)
    assert again.calls == [] and summary.attempts == 0 and summary.complete == 6
    assert event_bytes(resumed) == before


def test_killed_run_resumes_under_the_same_run_id_without_regeneration(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    first = MockBackend(roster, {("q_b", "JATS"): [Crash()]})
    with pytest.raises(Crash):
        go(store, config, roster, first)
    finished = [c[:2] for c in first.calls[:-1]]  # everything before the crash completed

    resumed = reopen(store, config)
    assert resumed.run_id == store.run_id
    second = MockBackend(roster)
    with pytest.raises(UnsafeResume):  # the interrupted attempt must be recovered explicitly
        go(resumed, config, roster, second)
    assert second.calls == []
    assert len(resumed.recover_interrupted(previous_worker_stopped=True)) == 1
    go(resumed, config, roster, second)

    regenerated = [c[:2] for c in second.calls if c[:2] in finished]
    assert regenerated == []
    assert len(second.calls) == 6 - len(finished)
    final = nonofficial_results(resumed).results
    assert len(final) == 6 and all(r.status == "completed" for r in final)
    crashed = resumed.observation(
        next(
            r.observation_key
            for r in final
            if r.item_id == "q_b" and r.representation_or_condition == "JATS"
        )
    )
    assert [e.state for e in crashed.events] == [
        "planned", "running", "interrupted", "running", "complete",
    ]  # fmt: skip
    assert crashed.attempts == 2


# --- retry and failure -------------------------------------------------------------------------------


def test_retryable_failure_is_retried_with_identical_input_and_history_is_kept(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    script = {("q_a", "PDF_DOCLING"): [BackendFailure("transport_error", "reset")]}
    backend = MockBackend(roster, script)
    summary = go(store, config, roster, backend)
    assert summary.complete == 6 and summary.attempts == 7
    key = next(
        r.observation_key
        for r in nonofficial_results(store).results
        if r.item_id == "q_a" and r.representation_or_condition == "PDF_DOCLING"
    )
    state = store.observation(key)
    running = [e for e in state.events if e.state == "running"]
    assert [e.state for e in state.events] == [
        "planned", "running", "retryable_failure", "running", "complete",
    ]  # fmt: skip
    assert len({e.rendered_input_sha256 for e in running}) == 1
    assert state.events[2].failure_kind == "transport_error"
    assert len(backend.calls_for("q_a")) == 3  # JATS once, PDF twice


def test_retry_exhaustion_is_an_operational_failure_not_an_incorrect_answer(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    failures = [BackendFailure("out_of_memory", "oom")] * 5
    backend = MockBackend(roster, {("q_c", "JATS"): failures})
    summary = go(store, config, roster, backend)
    assert summary.complete == 5 and summary.retries_exhausted == 1
    assert len(backend.calls_for("q_c")) == 4  # three attempts for JATS, one for PDF
    failed = next(r for r in nonofficial_results(store).results if r.status != "completed")
    assert failed.status == "operational_failure" and failed.output is None
    assert failed.raw_output is None and failed.answer is None
    assert "out_of_memory" in failed.failure
    score = stage_c.score(ITEMS["q_c"], failed)
    assert (score.status, score.reason) == ("missing", "operational_failure")


def test_terminal_and_unclassified_failures_are_not_retried(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    script = {
        ("q_a", "JATS"): [BackendFailure("invalid_prompt", "bad")],  # not a retryable category
        ("q_b", "JATS"): [RuntimeError("unexpected")],
    }
    backend = MockBackend(roster, script)
    summary = go(store, config, roster, backend)
    assert summary.terminal_failures == 2 and summary.complete == 1
    assert len(backend.calls_for("q_a")) == 1 and len(backend.calls_for("q_b")) == 1
    kinds = {
        e.failure_kind
        for k in store.observation_keys()
        for e in store.observation(k).events
        if e.state == "terminal_failure"
    }
    assert kinds == {"invalid_prompt", "unclassified_error"}
    statuses = {r.item_id: r.status for r in nonofficial_results(store).results}
    assert statuses == {
        "q_a": "operational_failure",
        "q_b": "operational_failure",
        "q_c": "completed",
    }


def test_non_text_backend_output_is_a_terminal_failure(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    backend = MockBackend(roster, {("q_a", "JATS"): [None]})
    go(store, config, roster, backend)
    kinds = [
        e.failure_kind
        for k in store.observation_keys()
        for e in store.observation(k).events
        if e.state == "terminal_failure"
    ]
    assert kinds == ["invalid_backend_output"]


def test_failed_observation_is_not_rerun_on_resume(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    go(
        store,
        config,
        roster,
        MockBackend(roster, {("q_a", "JATS"): [BackendFailure("invalid_prompt", "x")]}),
    )
    again = MockBackend(roster)
    go(reopen(store, config), config, roster, again)
    assert again.calls == []


# --- malformed output ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw", ["", "The answer is 0.05", '```json\n{"answer": "0.05", "abstained": false}\n```', "{"]
)
def test_malformed_output_is_kept_raw_and_scored_incorrect_not_missing(tmp_path, raw):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    go(store, config, roster, MockBackend(roster, {("q_a", "JATS"): [raw]}))
    result = next(r for r in nonofficial_results(store).results if r.item_id == "q_a")
    assert result.status == "completed" and result.parse_status == "unparseable"
    assert result.raw_output == raw and result.answer is None
    score = stage_c.score(ITEMS["q_a"], result)
    assert (score.status, score.reason) == ("incorrect", "unparseable")


def test_abstention_is_a_completed_answer_scored_incorrect(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    go(store, config, roster, MockBackend(roster, {("q_b", "JATS"): [good(None, True)]}))
    result = next(r for r in nonofficial_results(store).results if r.item_id == "q_b")
    assert (result.parse_status, result.abstained) == ("parsed", True)
    assert stage_c.score(ITEMS["q_b"], result).reason == "abstained"


# --- sampling seeds ---------------------------------------------------------------------------------------


def sampling_setup(tmp_path):
    reg = registry()
    store, config = make_store(
        tmp_path,
        decoding={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20},
        seed_registry_version=reg.registry_version,
        seed_registry_sha256=seed_registry_sha256(reg),
    )
    return store, config, reg


def test_named_repeat_seed_is_the_same_for_both_representations(tmp_path):
    store, config, reg = sampling_setup(tmp_path)
    roster = make_roster(config, repeats=(1, 2))
    backend = MockBackend(roster)
    go(store, config, roster, backend, seeds=reg)
    seeds: dict[tuple[str, int], set[int]] = {}
    for state in store.verify():
        c = state.coordinates
        seeds.setdefault((c.item_id, c.repeat_index), set()).add(c.seed)
    assert all(len(v) == 1 for v in seeds.values()) and len(seeds) == 6
    assert {c[2] for c in backend.calls} == {11, 22}


def test_sampling_requires_the_registry_the_configuration_names(tmp_path):
    store, config, _ = sampling_setup(tmp_path)
    roster = make_roster(config)
    backend = MockBackend(roster)
    with pytest.raises(ValueError):
        go(store, config, roster, backend, seeds=None)
    with pytest.raises(ValueError, match="registry"):
        go(store, config, roster, backend, seeds=registry(seeds=(5, 6, 7)))
    assert backend.calls == [] and store.observation_keys() == []


def test_retry_uses_the_same_seed(tmp_path):
    store, config, reg = sampling_setup(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    backend = MockBackend(roster, {("q_a", "JATS"): [BackendFailure("transport_error", "x")]})
    go(store, config, roster, backend, seeds=reg)
    assert [c[2] for c in backend.calls_for("q_a")] == [11, 11]


# --- contradictions are refused before anything runs ---------------------------------------------------------


def test_backend_identity_must_match_the_run(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    backend = MockBackend(roster)
    backend.identity = BackendIdentity(kind="smoke", name="x", version=None)
    with pytest.raises(RunMismatch, match="backend"):
        go(store, config, roster, backend)
    assert backend.calls == [] and store.observation_keys() == []


def test_rendered_input_from_another_configuration_is_refused(tmp_path):
    store, config = make_store(tmp_path)
    other = resolve(settings_for(scoring_version="elsewhere"))
    roster = make_roster(config)
    roster[3] = roster[3].model_copy(update={"rendered": make_roster(other)[3].rendered})
    backend = MockBackend(roster)
    with pytest.raises(RosterMismatch, match="configuration"):
        go(store, config, roster, backend)
    assert backend.calls == [] and store.observation_keys() == []


def test_rendered_input_with_another_prompt_is_refused(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    changed = roster[0].rendered.model_copy(update={"prompt_sha256": "9" * 64})
    roster[0] = roster[0].model_copy(update={"rendered": changed})
    with pytest.raises(RosterMismatch, match="prompt"):
        go(store, config, roster, MockBackend(roster))


def test_context_presence_must_match_the_matrix_arm(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, closed_book=True)
    with pytest.raises(RosterMismatch, match="context"):
        go(store, config, roster, MockBackend(roster))
    settings = settings_for()
    del settings["retrieval_config_sha256"]
    settings["matrix_arm"] = "closed_book"
    closed = resolve(settings)
    closed_store = RunStore.create(
        tmp_path / "cb",
        config=closed,
        phase="development",
        benchmark_version=BENCHMARK,
        backend_kind="mock",
        now=Clock(),
    )
    roster = make_roster(closed, closed_book=True, conditions=("closed_book",))
    summary = go(closed_store, closed, roster, MockBackend(roster))
    assert summary.complete == 3
    with_context = make_roster(closed, conditions=("closed_book",))
    with pytest.raises(RosterMismatch, match="context"):
        go(closed_store, closed, with_context, MockBackend(with_context))


def test_duplicate_roster_entries_are_refused(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    roster.append(roster[0])
    with pytest.raises(RosterMismatch, match="duplicate"):
        go(store, config, roster, MockBackend(roster))


def test_stored_observation_outside_the_roster_is_a_contradiction(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster[:2], MockBackend(roster))
    with pytest.raises(RosterMismatch, match="not in the roster"):
        go(reopen(store, config), config, roster[2:], MockBackend(roster))


def test_same_observation_with_another_paper_is_refused(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster[:1], MockBackend(roster))
    changed = roster[0].model_copy(update={"paper_id": "otherPaper"})
    with pytest.raises(PublicationConflict):
        go(reopen(store, config), config, [changed], MockBackend([changed]))


def test_resume_with_a_changed_rendered_input_is_refused_at_the_retry(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    backend = MockBackend(roster, {("q_a", "JATS"): [Crash()]})
    with pytest.raises(Crash):
        go(store, config, roster, backend)
    resumed = reopen(store, config)
    resumed.recover_interrupted(previous_worker_stopped=True)
    roster[0] = roster[0].model_copy(
        update={"rendered": rendered(config, "a different question?", "different evidence")}
    )
    with pytest.raises(InvalidTransition, match="identical"):
        go(resumed, config, roster, MockBackend(roster))


def test_corrupt_stored_records_stop_resume_before_any_generation(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config)
    go(store, config, roster[:3], MockBackend(roster))
    key = store.observation_keys()[0]
    done = next(s for s in store.verify() if s.status == "complete")
    (store.root / done.events[-1].raw_output.logical_ref).write_text("tampered")
    backend = MockBackend(roster)
    with pytest.raises(LifecycleError):
        go(reopen(store, config), config, roster, backend)
    assert backend.calls == [] and key


# --- official versus mock, smoke ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["mock", "smoke"])
def test_nonofficial_results_are_labelled_with_their_backend(tmp_path, kind):
    store, config = make_store(tmp_path, kind=kind)
    roster = make_roster(config, conditions=("JATS",))
    backend = MockBackend(roster)
    backend.identity = BackendIdentity(kind=kind, name="x", version=None)
    go(store, config, roster, backend, kind=kind)
    handoff = nonofficial_results(store)
    assert handoff.backend_kind == kind and len(handoff.results) == 3


def test_official_run_results_are_not_available_through_the_nonofficial_path(tmp_path):
    store, _ = make_store(tmp_path, kind="official")
    with pytest.raises(NonOfficialBackend, match="official run"):
        nonofficial_results(store)


def test_mock_backend_cannot_drive_an_official_run(tmp_path):
    store, config = make_store(tmp_path, kind="official")
    roster = make_roster(config)
    backend = MockBackend(roster)  # identifies itself as mock
    with pytest.raises(RunMismatch, match="backend"):
        go(store, config, roster, backend, kind="official")
    assert backend.calls == []


def test_undecided_observations_produce_no_result(tmp_path):
    store, config = make_store(tmp_path)
    roster = make_roster(config, conditions=("JATS",))
    backend = MockBackend(roster, {("q_b", "JATS"): [Crash()]})
    with pytest.raises(Crash):
        go(store, config, roster, backend)
    results = nonofficial_results(store).results
    assert {r.item_id for r in results} == {"q_a"}  # q_b is running, q_c was never reached
