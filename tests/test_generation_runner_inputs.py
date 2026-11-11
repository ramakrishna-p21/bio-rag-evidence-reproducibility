"""The runner stores each exact input before the call and keeps what a backend reports.

Backends are scripted. A stored input and a rendering are never treated as execution.
"""

import json

import pytest
from generation_support import ROOT, artifact, needs_tokenizer, resolve, synthetic_settings
from synthetic_environment import evidence as synthetic_evidence
from test_generation_attempts import Clock, environment
from test_generation_receipts import CONTEXT, CONTEXT_REF, facts_for

from biorag_repro.generation.attempts import Coordinates, RunStore
from biorag_repro.generation.environment import BackendIdentity
from biorag_repro.generation.environment_gate import publish_environment_evidence
from biorag_repro.generation.inputs import (
    CorruptInput,
    build_input_record,
    input_record_ref,
)
from biorag_repro.generation.prompt import InputRenderer, output_schema_sha256, prompt_sha256
from biorag_repro.generation.runner import (
    BackendFailure,
    GenerationOutput,
    RosterEntry,
    RosterMismatch,
    decided_results,
    run_roster,
)

pytestmark = needs_tokenizer

ANSWER = json.dumps({"answer": "4.2 hours", "abstained": False})


@pytest.fixture(scope="module")
def renderers():
    def make(purpose):
        prompt = artifact()
        settings = synthetic_settings()
        settings["purpose"] = purpose
        settings["prompt_sha256"] = prompt_sha256(prompt)
        settings["output_schema_sha256"] = output_schema_sha256(prompt)
        return InputRenderer(ROOT, resolve(settings), prompt)

    return {
        "synthetic_test": make("synthetic_test"),
        "development_candidate": make("development_candidate"),
    }


def new_store(tmp_path, renderer, kind="smoke"):
    store = RunStore.create(
        tmp_path,
        config=renderer.config,
        phase="development",
        benchmark_version="synthetic-bench-0",
        backend_kind=kind,
        now=Clock(),
    )
    if kind == "official":  # an official attempt cites the stored environment verification
        publish_environment_evidence(store.run_dir, synthetic_evidence())
    return store


def entry_for(renderer, store, item="q1", paper="paperA", *, with_record=True, question=None):
    coordinates = Coordinates(
        item_id=item,
        paper_id=paper,
        matrix_arm="confirmatory",
        representation_or_condition="JATS",
        repeat_index=1,
    )
    rendered = renderer.render(
        question or f"Question for {item}?", CONTEXT, renderer.counter(CONTEXT)
    )
    record = build_input_record(
        renderer, rendered, observation_key=store.key_for(coordinates), context=CONTEXT_REF
    )
    return RosterEntry(
        item_id=item,
        paper_id=paper,
        representation_or_condition="JATS",
        repeat_index=1,
        rendered=rendered,
        input_record=record if with_record else None,
    )


class Backend:
    def __init__(self, renderer, kind="smoke", action=None, with_facts=True):
        self.identity = BackendIdentity(kind=kind, name="synthetic", version=None)
        self.renderer, self.action, self.with_facts = renderer, action, with_facts
        self.calls = []

    def generate(self, rendered, *, seed, max_new_tokens):
        self.calls.append(rendered)
        if isinstance(self.action, BaseException):
            raise self.action
        if not self.with_facts:
            return ANSWER
        facts = facts_for(
            type("R", (), {"rendered": rendered})(), self.renderer.config, **(self.action or {})
        )
        return GenerationOutput(ANSWER, facts)


def run(store, entries, backend, kind="smoke"):
    return run_roster(store, entries, backend, capture_environment=lambda: environment(kind))


def test_input_is_stored_before_the_call_and_the_receipt_follows_the_call(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    published_before_call = []

    class Watching(Backend):
        def generate(self, rendered, *, seed, max_new_tokens):
            ref = input_record_ref(entry.input_record)
            published_before_call.append((store.run_dir / ref.logical_ref).is_file())
            return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)

    summary = run(store, [entry], Watching(renderer))
    assert summary.complete == 1 and published_before_call == [True]
    key = store.observation_keys()[0]
    final = store.observation(key).events[-1]
    assert final.execution_receipt_sha256 is not None
    assert decided_results(store)[0].status == "completed"


def test_a_backend_that_reports_nothing_leaves_an_unproven_completion(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    assert run(store, [entry], Backend(renderer, with_facts=False)).complete == 1
    assert (
        store.observation(store.observation_keys()[0]).events[-1].execution_receipt_sha256 is None
    )
    assert (store.run_dir / input_record_ref(entry.input_record).logical_ref).is_file()


def test_a_failed_call_stores_the_input_but_no_receipt(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    run(store, [entry], Backend(renderer, action=BackendFailure("model_crash", "boom")))
    key = store.observation_keys()[0]
    assert store.observation(key).status == "terminal_failure"
    assert (store.run_dir / input_record_ref(entry.input_record).logical_ref).is_file()
    assert not (store.run_dir / "observations" / key / "receipts").exists()


def test_input_that_disagrees_with_the_entry_is_refused_before_anything_runs(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    other = entry_for(renderer, store, question="A different question?")
    mixed = entry.model_copy(update={"input_record": other.input_record})
    backend = Backend(renderer)
    with pytest.raises(RosterMismatch, match="input"):
        run(store, [mixed], backend)
    assert backend.calls == [] and store.observation_keys() == []
    assert not (store.run_dir / "inputs").exists()


def test_input_for_another_observation_is_refused(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    wrong = entry.input_record.model_copy(update={"observation_key": "obs_" + "7" * 64})
    with pytest.raises(RosterMismatch, match="observation"):
        run(store, [entry.model_copy(update={"input_record": wrong})], Backend(renderer))


def test_input_from_another_configuration_is_refused(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    other_store = new_store(tmp_path / "other", renderers["development_candidate"], kind="smoke")
    foreign = entry_for(renderers["development_candidate"], other_store)
    with pytest.raises(RosterMismatch, match="configuration"):
        run(store, [foreign], Backend(renderer))


def test_failed_input_publication_stops_before_any_attempt(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    ref = input_record_ref(entry.input_record)
    (store.run_dir / "inputs").mkdir()
    (store.run_dir / ref.logical_ref).write_bytes(b"something else")
    backend = Backend(renderer)
    with pytest.raises(CorruptInput):
        run(store, [entry], backend)
    assert backend.calls == []
    assert all(
        e.state != "running" for k in store.observation_keys() for e in store.observation(k).events
    )


def test_substituted_token_ids_are_recorded_as_a_failure_not_an_answer(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    bad = {"input_token_ids_sha256": "0" * 64}
    run(store, [entry], Backend(renderer, action=bad))
    key = store.observation_keys()[0]
    state = store.observation(key)
    assert state.status == "terminal_failure"
    assert state.events[-1].failure_kind == "invalid_execution_receipt"
    assert not (store.run_dir / "observations" / key / "raw").exists()
    assert decided_results(store)[0].status == "operational_failure"


def test_facts_without_a_stored_input_are_a_failure(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store, with_record=False)
    run(store, [entry], Backend(renderer))
    state = store.observation(store.observation_keys()[0])
    assert state.status == "terminal_failure"
    assert state.events[-1].failure_kind == "receipt_without_input_record"


def test_official_run_needs_the_recorded_input_of_every_observation(tmp_path, renderers):
    renderer = renderers["development_candidate"]
    store = new_store(tmp_path, renderer, kind="official")
    entry = entry_for(renderer, store, with_record=False)
    backend = Backend(renderer, kind="official")
    with pytest.raises(RosterMismatch, match="official"):
        run(store, [entry], backend, kind="official")
    assert backend.calls == []


def test_official_run_does_not_complete_an_observation_without_a_receipt(tmp_path, renderers):
    renderer = renderers["development_candidate"]
    store = new_store(tmp_path, renderer, kind="official")
    entry = entry_for(renderer, store)
    run(store, [entry], Backend(renderer, kind="official", with_facts=False), kind="official")
    state = store.observation(store.observation_keys()[0])
    assert state.status == "terminal_failure"
    assert state.events[-1].failure_kind == "missing_execution_receipt"
    assert not (store.run_dir / "observations" / state.key / "raw").exists()


def test_rerun_after_completion_stores_nothing_new(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    entry = entry_for(renderer, store)
    backend = Backend(renderer)
    run(store, [entry], backend)
    before = sorted(p.relative_to(store.run_dir) for p in store.run_dir.rglob("*"))
    assert run(store, [entry], backend).attempts == 0
    assert sorted(p.relative_to(store.run_dir) for p in store.run_dir.rglob("*")) == before
    assert len(backend.calls) == 1


def test_publishing_helper_refuses_other_configurations(tmp_path, renderers):
    renderer = renderers["synthetic_test"]
    store = new_store(tmp_path, renderer)
    other = renderers["development_candidate"]
    other_store = new_store(tmp_path / "x", other)
    foreign = entry_for(other, other_store).input_record
    with pytest.raises(Exception, match="configuration"):
        store.publish_input(foreign)
