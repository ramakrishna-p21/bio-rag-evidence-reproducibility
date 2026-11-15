"""Execution receipts in the attempt store.

A stored input proves nothing about execution. These tests show what a receipt must agree
with before an attempt can be completed, and what read-back finds when a record is damaged.
Everything is synthetic; no model runs.
"""

import json

import pytest
from generation_support import (
    ROOT,
    artifact,
    make_renderer,
    needs_tokenizer,
    resolve,
    synthetic_settings,
)
from test_generation_attempts import GOOD_OUTPUT, Clock, coordinates, environment

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.generation.attempts import AttemptEvent, CorruptRecord, InvalidReceipt, RunStore
from biorag_repro.generation.inputs import build_input_record, publish_input_record
from biorag_repro.generation.prompt import InputRenderer, output_schema_sha256, prompt_sha256
from biorag_repro.generation.receipt import (
    ExecutionFacts,
    ExecutionReceipt,
    GenerationSettings,
    token_ids_sha256,
)

pytestmark = needs_tokenizer

CONTEXT = "The assay measured a half-life of 4.2 hours in the treated cells."
CONTEXT_REF = ArtifactRef(artifact_id="ctx-1", logical_ref="contexts/ctx-1.json", sha256="c" * 64)
EOS = 151645


@pytest.fixture(scope="module")
def renderer():
    return make_renderer("a" * 64)


def sampling_renderer():
    prompt = artifact()
    settings = synthetic_settings()
    settings["prompt_sha256"] = prompt_sha256(prompt)
    settings["output_schema_sha256"] = output_schema_sha256(prompt)
    settings["decoding"] = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    return InputRenderer(ROOT, resolve(settings), prompt)


def new_store(tmp_path, renderer, sampling=False):
    return RunStore.create(
        tmp_path,
        config=renderer.config,
        phase="development",
        benchmark_version="synthetic-bench-0",
        backend_kind="smoke",
        now=Clock(),
    )


def planned(store, seed=None):
    return store.plan(coordinates(seed=seed))


def record_for(renderer, store, key, *, question="What was the half-life?"):
    rendered = renderer.render(question, CONTEXT, renderer.counter(CONTEXT))
    return build_input_record(renderer, rendered, observation_key=key, context=CONTEXT_REF)


def facts_for(record, config, *, output=(11, 12, EOS), stop="eos", seed=None, **changes):
    decoding = config.settings.decoding
    generation = {
        "max_new_tokens": config.settings.budgets.max_new_tokens,
        "do_sample": decoding.do_sample,
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "top_k": decoding.top_k,
        "eos_token_ids": [EOS],
        "pad_token_id": 151643,
        "seed": seed,
        "effective": {"do_sample": decoding.do_sample},
    }
    generation.update(changes.pop("generation", {}))
    fields = {
        "input_token_ids_sha256": record.rendered.input_token_ids_sha256,
        "input_token_count": record.rendered.input_tokens,
        "generation": GenerationSettings(**generation),
        "output_token_ids": list(output),
        "stop_reason": stop,
    }
    return ExecutionFacts(**{**fields, **changes})


def attempt_with_input(store, renderer, key, *, publish=True, **record_changes):
    record = record_for(renderer, store, key, **record_changes)
    ref = publish_input_record(store.run_dir, record) if publish else None
    attempt = store.begin_attempt(
        key, environment=environment("smoke"), rendered_input_sha256=record.rendered.rendered_sha256
    )
    return attempt, record, ref


def complete(store, key, attempt, record, ref, facts):
    store.complete(key, attempt, GOOD_OUTPUT, execution=facts, input_record_sha256=ref.sha256)


def reopen(store):
    return RunStore.open(store.root, store.run_id, config=store.config, backend_kind="smoke")


def receipt_path(store, key, attempt):
    return store.run_dir / "observations" / key / "receipts" / f"{attempt}.json"


def test_receipt_is_stored_and_cited_by_the_completion_event(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, ref = attempt_with_input(store, renderer, key)
    complete(store, key, attempt, record, ref, facts_for(record, renderer.config))

    final = store.observation(key).events[-1]
    path = receipt_path(store, key, attempt)
    assert final.execution_receipt_sha256 == sha256_bytes(path.read_bytes())
    receipt = ExecutionReceipt.model_validate_json(path.read_bytes())
    running = store.observation(key).events[-2]
    assert (receipt.run_id, receipt.observation_key, receipt.attempt_id) == (
        store.run_id,
        key,
        attempt,
    )
    assert receipt.environment_sha256 == running.environment_sha256
    assert receipt.rendered_input_sha256 == running.rendered_input_sha256
    assert receipt.raw_output_sha256 == final.raw_output.sha256
    assert receipt.input_record_sha256 == ref.sha256
    assert receipt.facts.input_token_ids_sha256 == record.rendered.input_token_ids_sha256
    assert reopen(store).observation(key).status == "complete"


def test_completion_without_a_receipt_is_still_a_valid_unproven_record(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, _, _ = attempt_with_input(store, renderer, key)
    store.complete(key, attempt, GOOD_OUTPUT)
    assert store.observation(key).events[-1].execution_receipt_sha256 is None
    assert not (store.run_dir / "observations" / key / "receipts").exists()


def test_facts_without_an_input_record_are_refused(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, _ = attempt_with_input(store, renderer, key)
    with pytest.raises(ValueError, match="input record"):
        store.complete(key, attempt, GOOD_OUTPUT, execution=facts_for(record, renderer.config))
    assert store.observation(key).status == "running"


def test_an_unpublished_input_record_cannot_support_a_receipt(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, _ = attempt_with_input(store, renderer, key, publish=False)
    with pytest.raises(InvalidReceipt, match="input"):
        store.complete(
            key,
            attempt,
            GOOD_OUTPUT,
            execution=facts_for(record, renderer.config),
            input_record_sha256="e" * 64,
        )
    assert store.observation(key).status == "running"


def refused(tmp_path, renderer, make_facts, match):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, ref = attempt_with_input(store, renderer, key)
    facts = make_facts(record, renderer.config)
    with pytest.raises(InvalidReceipt, match=match):
        complete(store, key, attempt, record, ref, facts)
    # Nothing was written: the attempt is still running and can still be closed honestly.
    assert store.observation(key).status == "running"
    assert not receipt_path(store, key, attempt).exists()
    store.fail(key, attempt, kind="invalid_execution_receipt", detail="refused", retryable=False)
    return store


def test_substituted_input_token_ids_are_refused(tmp_path, renderer):
    refused(
        tmp_path,
        renderer,
        lambda r, c: facts_for(r, c, input_token_ids_sha256=token_ids_sha256([1, 2, 3])),
        match="token",
    )


def test_wrong_input_token_count_is_refused(tmp_path, renderer):
    refused(
        tmp_path,
        renderer,
        lambda r, c: facts_for(r, c, input_token_count=r.rendered.input_tokens + 1),
        match="token",
    )


def test_changed_output_allowance_is_refused(tmp_path, renderer):
    refused(
        tmp_path,
        renderer,
        lambda r, c: facts_for(r, c, generation={"max_new_tokens": 64}, output=(11, EOS)),
        match="generation",
    )


def test_sampling_that_the_configuration_does_not_name_is_refused(tmp_path, renderer):
    refused(
        tmp_path,
        renderer,
        lambda r, c: facts_for(
            r, c, generation={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
        ),
        match="generation",
    )


def test_changed_temperature_is_refused(tmp_path):
    renderer = sampling_renderer()
    store = new_store(tmp_path, renderer)
    key = planned(store, seed=5)
    attempt, record, ref = attempt_with_input(store, renderer, key)
    bad = facts_for(record, renderer.config, seed=5, generation={"temperature": 0.9})
    with pytest.raises(InvalidReceipt, match="generation"):
        complete(store, key, attempt, record, ref, bad)
    good = facts_for(record, renderer.config, seed=5)
    complete(store, key, attempt, record, ref, good)
    assert store.observation(key).status == "complete"


def test_a_seed_other_than_the_planned_seed_is_refused(tmp_path):
    renderer = sampling_renderer()
    store = new_store(tmp_path, renderer)
    key = planned(store, seed=5)
    attempt, record, ref = attempt_with_input(store, renderer, key)
    with pytest.raises(InvalidReceipt, match="seed"):
        complete(store, key, attempt, record, ref, facts_for(record, renderer.config, seed=6))


def test_input_record_of_another_observation_is_refused(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, _ = attempt_with_input(store, renderer, key)
    other_key = "obs_" + "9" * 64
    other = record.model_copy(update={"observation_key": other_key})
    ref = publish_input_record(store.run_dir, other)
    with pytest.raises(InvalidReceipt, match="observation"):
        complete(store, key, attempt, other, ref, facts_for(record, renderer.config))


def test_a_stale_input_record_for_other_rendered_text_is_refused(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, _record, _ = attempt_with_input(store, renderer, key)
    stale = record_for(renderer, store, key, question="Which cell line was used?")
    ref = publish_input_record(store.run_dir, stale)
    with pytest.raises(InvalidReceipt, match="rendered"):
        complete(store, key, attempt, stale, ref, facts_for(stale, renderer.config))


def test_an_input_record_from_another_configuration_is_refused(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, _record, _ = attempt_with_input(store, renderer, key)
    other_renderer = make_renderer("a" * 64, max_new_tokens=64)
    foreign = record_for(other_renderer, store, key)
    ref = publish_input_record(store.run_dir, foreign)
    with pytest.raises(InvalidReceipt, match="configuration"):
        complete(store, key, attempt, foreign, ref, facts_for(foreign, renderer.config))


# ---- read-back after damage -------------------------------------------------------------------


def completed(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    key = planned(store)
    attempt, record, ref = attempt_with_input(store, renderer, key)
    complete(store, key, attempt, record, ref, facts_for(record, renderer.config))
    return store, key, attempt, ref


def test_missing_receipt_is_found_on_read_back(tmp_path, renderer):
    store, key, attempt, _ = completed(tmp_path, renderer)
    receipt_path(store, key, attempt).unlink()
    with pytest.raises(CorruptRecord, match="receipt"):
        reopen(store)


def test_edited_receipt_is_found_on_read_back(tmp_path, renderer):
    store, key, attempt, _ = completed(tmp_path, renderer)
    path = receipt_path(store, key, attempt)
    data = json.loads(path.read_text())
    data["facts"]["output_token_ids"] = [11, 99, EOS]
    data["output_token_ids_sha256"] = token_ids_sha256([11, 99, EOS])
    path.write_text(json.dumps(data))
    with pytest.raises(CorruptRecord, match="receipt"):
        reopen(store)


def test_replaced_input_record_is_found_on_read_back(tmp_path, renderer):
    store, _, _, ref = completed(tmp_path, renderer)
    path = store.run_dir / ref.logical_ref
    path.write_bytes(path.read_bytes().replace(b"half-life", b"half-lifx", 1))
    with pytest.raises(CorruptRecord, match="input"):
        reopen(store)


def test_deleted_input_record_is_found_on_read_back(tmp_path, renderer):
    store, _, _, ref = completed(tmp_path, renderer)
    (store.run_dir / ref.logical_ref).unlink()
    with pytest.raises(CorruptRecord, match="input"):
        reopen(store)


def test_event_citing_a_receipt_on_a_failure_is_refused_by_the_model(tmp_path, renderer):
    store, key, _, _ = completed(tmp_path, renderer)
    final = store.observation(key).events[-1]
    data = final.model_dump(mode="json")
    data.update(state="terminal_failure", failure_kind="x", failure_detail="y")
    for name in ("raw_output", "parse_status", "answer", "abstained", "parse_reason"):
        data[name] = None
    with pytest.raises(ValueError, match="receipt"):
        AttemptEvent.model_validate(data)


# ---- the receipt must describe a generation that could have happened ----------------------


def test_a_stop_token_inside_the_output_is_impossible(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    record = record_for(renderer, store, planned(store))
    with pytest.raises(ValueError, match="stop token before the end"):
        facts_for(record, renderer.config, output=(11, EOS, 12, EOS))


def test_a_length_stop_cannot_end_on_a_stop_token(tmp_path, renderer):
    store = new_store(tmp_path, renderer)
    record = record_for(renderer, store, planned(store))
    limit = renderer.config.settings.budgets.max_new_tokens
    with pytest.raises(ValueError, match="stop token before the end"):
        facts_for(record, renderer.config, output=[11] * (limit - 1) + [EOS], stop="max_new_tokens")
