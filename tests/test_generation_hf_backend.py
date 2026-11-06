"""Hugging Face backend, tested on a tiny random-weight Qwen3 model built locally.

No weights are downloaded. The tiny model, its tokenizer pin and every setting here are
synthetic; whatever text it produces is noise and is never a scientific observation.
"""

import json
import shutil
from importlib import metadata

import pytest
import torch
from test_generation_attempts import Clock
from test_generation_config import resolve, synthetic_settings
from test_generation_environment import cpu_probes
from test_generation_prompt import ROOT, artifact, needs_tokenizer
from test_generation_receipts import CONTEXT, CONTEXT_REF
from tiny_model import EOS, PAD, load_once

from biorag_repro.generation import config as generator_config
from biorag_repro.generation import hf_backend
from biorag_repro.generation.attempts import Coordinates, RunStore
from biorag_repro.generation.hf_backend import HF_BACKEND_NAME, HuggingFaceBackend
from biorag_repro.generation.inputs import build_input_record
from biorag_repro.generation.model_snapshot import SnapshotError
from biorag_repro.generation.prompt import InputRenderer, output_schema_sha256, prompt_sha256
from biorag_repro.generation.receipt import token_ids_sha256
from biorag_repro.generation.runner import BackendFailure, GenerationOutput, RosterEntry, run_roster

pytestmark = needs_tokenizer

REV = "1" * 40
NEW_TOKENS = 12
smoke = pytest.mark.model_smoke


class FakeModel(torch.nn.Module):
    """Default test model: bf16 parameter, no forward pass. It appends a fixed tail."""

    def __init__(self, tail=None, error=None, prefix_change=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.tail, self.error, self.prefix_change = tail, error, prefix_change
        self.seen = {}

    def generate(self, input_ids, attention_mask, generation_config, use_model_defaults):
        if self.error:
            raise self.error
        self.seen = {
            "ids": input_ids.tolist(),
            "mask": attention_mask.tolist(),
            "config": generation_config,
            "defaults": use_model_defaults,
        }
        ids = input_ids.clone()
        if self.prefix_change:
            ids[0, 0] += 1
        tail = (
            self.tail
            if self.tail is not None
            else list(range(11, 11 + generation_config.max_new_tokens))
        )
        return torch.cat([ids, torch.tensor([tail])], dim=1)


def fake_loader(**options):
    return lambda directory, device: FakeModel(**options)


def make_renderer(*, purpose="synthetic_test", repository="synthetic/tiny-qwen3", decoding=None,
                  window=8192, revision=REV, new_tokens=NEW_TOKENS):  # fmt: skip
    prompt = artifact()
    settings = synthetic_settings()
    settings["purpose"] = purpose
    settings["model"] = {"repository": repository, "revision": revision, "dtype": "bfloat16"}
    settings["prompt_sha256"] = prompt_sha256(prompt)
    settings["output_schema_sha256"] = output_schema_sha256(prompt)
    settings["budgets"].update(max_new_tokens=new_tokens, context_window_tokens=window)
    if decoding:
        settings["decoding"] = decoding
    return InputRenderer(ROOT, resolve(settings), prompt)


@pytest.fixture(scope="module")
def renderer():
    return make_renderer()


def backend_for(renderer, directory, device="cpu", **kwargs):
    kwargs.setdefault("load_model", fake_loader())
    return HuggingFaceBackend(
        renderer.config, model_dir=directory, tokenizer_root=ROOT, device=device, **kwargs
    )


def rendered_for(renderer, question="What was the half-life?"):
    return renderer.render(question, CONTEXT, renderer.counter(CONTEXT))


def oracle(model, ids, count):
    """Greedy decoding written out by hand: no generate(), no cache, no model defaults."""
    sequence, new = list(ids), []
    with torch.inference_mode():
        for _ in range(count):
            token = int(model(torch.tensor([sequence])).logits[0, -1].argmax())
            new.append(token)
            sequence.append(token)
    return new


# ---- identity -------------------------------------------------------------------------------


def test_backend_identity_is_derived_from_the_implementation(renderer, tiny):
    backend = backend_for(renderer, tiny)
    cls = type(backend)
    assert backend.identity.name == f"{cls.__module__}.{cls.__qualname__}" == HF_BACKEND_NAME
    assert backend.identity.version == metadata.version("transformers")
    assert backend.identity.kind == "smoke"


def test_a_subclass_has_another_name_and_so_is_not_the_recognised_backend(renderer, tiny):
    class Variant(HuggingFaceBackend):
        pass

    assert backend_for_cls(Variant, renderer, tiny).identity.name != HF_BACKEND_NAME


def backend_for_cls(cls, renderer, directory, device="cpu"):
    return cls(
        renderer.config,
        model_dir=directory,
        tokenizer_root=ROOT,
        device=device,
        load_model=fake_loader(),
    )


@pytest.fixture
def approved(monkeypatch, tiny):
    """Pretend the project approved the tiny checkpoint, so the kind rule can be exercised."""
    from biorag_repro.generation.model_snapshot import inventory_snapshot

    manifest = inventory_snapshot(tiny, REV).manifest_sha256
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): manifest}
    )
    monkeypatch.setattr(hf_backend, "is_official_device", lambda device: True)
    return manifest


def candidate():
    return make_renderer(purpose="development_candidate", repository="Qwen/Qwen3-8B")


def test_official_kind_needs_every_condition(approved, tiny, monkeypatch):
    assert backend_for(candidate(), tiny).identity.kind == "official"
    # not a development candidate configuration
    synthetic = make_renderer(repository="Qwen/Qwen3-8B")
    assert backend_for(synthetic, tiny).identity.kind == "smoke"
    # checkpoint not on the approved list
    monkeypatch.setattr(generator_config, "APPROVED_CHECKPOINTS", {})
    assert backend_for(candidate(), tiny).identity.kind == "smoke"
    # a different manifest for the same revision
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): "0" * 64}
    )
    assert backend_for(candidate(), tiny).identity.kind == "smoke"


def test_an_approved_checkpoint_on_cpu_is_still_not_official(tiny, monkeypatch):
    from biorag_repro.generation.model_snapshot import inventory_snapshot

    manifest = inventory_snapshot(tiny, REV).manifest_sha256
    monkeypatch.setattr(
        generator_config, "APPROVED_CHECKPOINTS", {("Qwen/Qwen3-8B", REV): manifest}
    )
    assert backend_for(candidate(), tiny, device="cpu").identity.kind == "smoke"


def test_cpu_is_never_official_and_nothing_is_approved_by_default(tiny):
    assert generator_config.APPROVED_CHECKPOINTS == {}
    assert not hf_backend.is_official_device("cpu")
    assert not hf_backend.is_official_device("cuda")
    assert not hf_backend.is_official_device("cuda:0,1")
    assert hf_backend.is_official_device("cuda:1")
    assert backend_for(candidate(), tiny).identity.kind == "smoke"


# ---- checkpoint verification ----------------------------------------------------------------


def test_environment_records_the_inventory_device_dtype_and_tokenizer(renderer, tiny):
    backend = backend_for(renderer, tiny)
    environment = backend.environment(cpu_probes())
    execution = environment.execution
    settings = renderer.config.settings
    assert environment.backend == backend.identity
    assert (environment.model_repository, environment.model_revision) == (
        "synthetic/tiny-qwen3",
        REV,
    )
    assert execution.device == "cpu" and execution.dtype == "bfloat16"
    assert execution.device_name is None
    assert execution.tokenizer_file_sha256 == settings.tokenizer.file_sha256
    assert execution.max_position_embeddings == 8192
    names = {f.path for f in execution.snapshot.files}
    assert {"config.json", "generation_config.json", "model.safetensors"} <= names
    assert {"torch", "transformers", "safetensors"} <= set(environment.packages)


def test_wrong_directory_name_is_refused(renderer, tiny, tmp_path):
    other = tmp_path / ("2" * 40)
    shutil.copytree(tiny, other)
    with pytest.raises(SnapshotError, match="revision"):
        backend_for(renderer, other)


def test_missing_weights_are_refused(renderer, tiny, tmp_path):
    copy = tmp_path / REV
    shutil.copytree(tiny, copy)
    (copy / "model.safetensors").unlink()
    with pytest.raises(SnapshotError, match="weight"):
        backend_for(renderer, copy)


def test_an_extra_unlisted_weight_file_is_refused(renderer, tiny, tmp_path):
    copy = tmp_path / REV
    shutil.copytree(tiny, copy)
    shutil.copy(copy / "model.safetensors", copy / "other.safetensors")
    with pytest.raises(SnapshotError, match="weight"):
        backend_for(renderer, copy)


def test_a_context_window_beyond_the_model_is_refused(tiny):
    wide = make_renderer(window=8192)  # the model has exactly 8192 positions
    backend_for(wide, tiny)
    too_wide = make_renderer(window=8193)
    with pytest.raises(ValueError, match="context window"):
        backend_for(too_wide, tiny)


def test_missing_stop_tokens_are_refused(renderer, tiny, tmp_path):
    copy = tmp_path / REV
    shutil.copytree(tiny, copy)
    (copy / "generation_config.json").write_text(json.dumps({"pad_token_id": PAD}))
    with pytest.raises(SnapshotError, match="stop"):
        backend_for(renderer, copy)


def test_a_model_that_does_not_stop_on_the_chat_end_token_is_refused(renderer, tiny, tmp_path):
    copy = tmp_path / REV
    shutil.copytree(tiny, copy)
    (copy / "generation_config.json").write_text(json.dumps({"eos_token_id": 151643}))
    with pytest.raises(SnapshotError, match="stop"):
        backend_for(renderer, copy)


def test_loader_must_deliver_the_configured_precision(renderer, tiny):
    def float32(directory, device):
        return FakeModel().float()

    with pytest.raises(ValueError, match="dtype"):
        backend_for(renderer, tiny, load_model=float32)


def test_loader_must_place_every_weight_on_the_requested_device(renderer, tiny):
    def elsewhere(directory, device):
        return FakeModel().to("meta")

    with pytest.raises(ValueError, match="device"):
        backend_for(renderer, tiny, load_model=elsewhere)


def test_files_changed_while_loading_are_detected(renderer, tiny, tmp_path):
    copy = tmp_path / REV
    shutil.copytree(tiny, copy)

    def tamper(directory, device):
        model = FakeModel()
        (directory / "config.json").write_text(
            (directory / "config.json").read_text().replace("2", "3", 1)
        )
        return model

    with pytest.raises(SnapshotError, match="changed"):
        backend_for(renderer, copy, load_model=tamper)


@smoke
def test_default_loader_is_offline_safetensors_only_unquantized_and_unmapped(
    renderer, real_tiny, monkeypatch
):
    seen = {}
    from transformers import AutoModelForCausalLM

    real = AutoModelForCausalLM.from_pretrained

    def spy(*args, **kwargs):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", spy)
    backend_for(renderer, real_tiny, load_model=hf_backend.load_local_model)
    assert seen["local_files_only"] is True
    assert seen["trust_remote_code"] is False
    assert seen["use_safetensors"] is True
    assert seen["dtype"] is torch.bfloat16
    assert seen.get("device_map") is None
    assert "quantization_config" not in seen


# ---- generation -----------------------------------------------------------------------------


@smoke
def test_greedy_generation_matches_a_hand_written_decoder(renderer, real_tiny):
    backend = backend_for(renderer, real_tiny, load_model=load_once)
    rendered = rendered_for(renderer)
    ids = renderer.tokenizer(rendered.rendered_text, add_special_tokens=False)["input_ids"]
    output = backend.generate(rendered, seed=None, max_new_tokens=NEW_TOKENS)
    expected = oracle(backend.model, ids, NEW_TOKENS)
    assert isinstance(output, GenerationOutput)
    assert output.execution.output_token_ids == expected
    assert output.execution.stop_reason == "max_new_tokens"
    assert output.text == renderer.tokenizer.decode(
        expected, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def test_receipt_facts_describe_the_call_that_was_made(renderer, tiny):
    backend = backend_for(renderer, tiny)
    rendered = rendered_for(renderer)
    facts = backend.generate(rendered, seed=None, max_new_tokens=NEW_TOKENS).execution
    assert facts.input_token_ids_sha256 == rendered.input_token_ids_sha256
    assert facts.input_token_count == rendered.input_tokens
    generation = facts.generation
    assert (generation.max_new_tokens, generation.do_sample) == (NEW_TOKENS, False)
    assert (generation.temperature, generation.top_p, generation.top_k) == (None, None, None)
    assert generation.eos_token_ids == [EOS] and generation.pad_token_id == PAD
    assert generation.seed is None
    assert generation.effective["do_sample"] is False
    assert generation.effective["max_new_tokens"] == NEW_TOKENS
    assert generation.effective["eos_token_id"] == [EOS]


@smoke
def test_the_models_own_sampling_defaults_do_not_change_greedy_decoding(
    renderer, real_tiny, tmp_path
):
    sampling_defaults = {
        "eos_token_id": [EOS, PAD],
        "pad_token_id": PAD,
        "do_sample": True,
        "temperature": 0.9,
        "top_k": 5,
        "top_p": 0.5,
        "repetition_penalty": 1.3,
    }
    copy = tmp_path / REV
    shutil.copytree(real_tiny, copy)
    (copy / "generation_config.json").write_text(json.dumps(sampling_defaults))
    rendered = rendered_for(renderer)
    real = load_once
    plain = backend_for(renderer, real_tiny, load_model=real).generate(
        rendered, seed=None, max_new_tokens=NEW_TOKENS
    )
    other = backend_for(renderer, copy, load_model=real).generate(
        rendered, seed=None, max_new_tokens=NEW_TOKENS
    )
    assert plain.execution.output_token_ids == other.execution.output_token_ids
    assert other.execution.generation.effective["repetition_penalty"] == 1.0
    assert other.execution.generation.eos_token_ids == [EOS, PAD]


@smoke
def test_sampling_follows_the_configuration_and_the_seed(real_tiny):
    decoding = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    renderer = make_renderer(decoding=decoding)
    backend = backend_for(renderer, real_tiny, load_model=load_once)
    rendered = rendered_for(renderer)
    first = backend.generate(rendered, seed=11, max_new_tokens=NEW_TOKENS)
    again = backend.generate(rendered, seed=11, max_new_tokens=NEW_TOKENS)
    assert first.execution == again.execution and first.text == again.text
    settings = first.execution.generation
    assert (settings.do_sample, settings.temperature, settings.top_p, settings.top_k) == (
        True,
        0.7,
        0.9,
        20,
    )
    assert settings.seed == 11
    others = {
        tuple(
            backend.generate(rendered, seed=s, max_new_tokens=NEW_TOKENS).execution.output_token_ids
        )
        for s in range(5)
    }
    assert len(others) > 1  # the seed matters; the random model has no preferred token
    with pytest.raises(BackendFailure) as raised:
        backend.generate(rendered, seed=None, max_new_tokens=NEW_TOKENS)
    assert raised.value.kind == "missing_seed"


def test_the_exact_recorded_ids_reach_the_model_and_the_prefix_is_checked(renderer, tiny):
    seen = {}

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        def generate(self, input_ids, attention_mask, generation_config, use_model_defaults):
            seen.update(
                ids=input_ids.tolist(),
                mask=attention_mask.tolist(),
                config=generation_config,
                defaults=use_model_defaults,
            )
            tail = torch.tensor([[7, 8, EOS]])
            return torch.cat([input_ids, tail], dim=1)

    backend = backend_for(renderer, tiny, load_model=lambda directory, device: Stub())
    rendered = rendered_for(renderer)
    expected = renderer.tokenizer(rendered.rendered_text, add_special_tokens=False)["input_ids"]
    output = backend.generate(rendered, seed=None, max_new_tokens=NEW_TOKENS)
    assert seen["ids"] == [expected] and seen["mask"] == [[1] * len(expected)]
    assert seen["defaults"] is False
    assert output.execution.stop_reason == "eos"
    assert output.execution.output_token_ids == [7, 8, EOS]
    assert output.text == renderer.tokenizer.decode([7, 8], skip_special_tokens=False)
    assert output.execution.input_token_ids_sha256 == token_ids_sha256(expected)


def scripted(tail=None, prefix_change=False, error=None):
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))

        def generate(self, input_ids, attention_mask, generation_config, use_model_defaults):
            if error:
                raise error
            ids = input_ids.clone()
            if prefix_change:
                ids[0, 0] += 1
            return torch.cat([ids, torch.tensor([tail])], dim=1)

    return lambda directory, device: Stub()


def test_unexplained_stops_and_altered_prompts_are_failures_not_answers(renderer, tiny):
    rendered = rendered_for(renderer)
    cases = {
        "unexpected_stop": scripted(tail=[7, 8]),  # neither a stop token nor the length limit
        "early_stop_token": scripted(tail=[EOS, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        "prompt_altered": scripted(tail=[7, EOS], prefix_change=True),
    }
    for kind, loader in cases.items():
        with pytest.raises(BackendFailure) as raised:
            backend_for(renderer, tiny, load_model=loader).generate(
                rendered, seed=None, max_new_tokens=NEW_TOKENS
            )
        assert raised.value.kind == kind


def test_out_of_memory_is_reported_as_a_classified_failure(renderer, tiny):
    loader = scripted(error=torch.cuda.OutOfMemoryError("synthetic"))
    with pytest.raises(BackendFailure) as raised:
        backend_for(renderer, tiny, load_model=loader).generate(
            rendered_for(renderer), seed=None, max_new_tokens=NEW_TOKENS
        )
    assert raised.value.kind == "out_of_memory"


def test_a_substituted_token_hash_is_refused_before_the_model_is_called(renderer, tiny):
    backend = backend_for(
        renderer, tiny, load_model=scripted(error=AssertionError("must not be called"))
    )
    wrong = rendered_for(renderer).model_copy(update={"input_token_ids_sha256": "0" * 64})
    with pytest.raises(BackendFailure) as raised:
        backend.generate(wrong, seed=None, max_new_tokens=NEW_TOKENS)
    assert raised.value.kind == "input_token_mismatch"


def test_other_configurations_and_limits_are_refused(renderer, tiny):
    backend = backend_for(renderer, tiny)
    rendered = rendered_for(renderer)
    with pytest.raises(BackendFailure) as raised:
        backend.generate(rendered, seed=None, max_new_tokens=NEW_TOKENS + 1)
    assert raised.value.kind == "configuration_mismatch"
    foreign = rendered.model_copy(update={"resolved_config_sha256": "0" * 64})
    with pytest.raises(BackendFailure) as raised:
        backend.generate(foreign, seed=None, max_new_tokens=NEW_TOKENS)
    assert raised.value.kind == "configuration_mismatch"


# ---- through the runner and the attempt store ----------------------------------------------------


@smoke
def test_smoke_run_end_to_end_on_cpu(renderer, real_tiny, tmp_path):
    """Smoke only: random weights, so the text is noise and the observation is not scientific."""
    store = RunStore.create(
        tmp_path,
        config=renderer.config,
        phase="development",
        benchmark_version="synthetic-bench-0",
        backend_kind="smoke",
        now=Clock(),
    )
    backend = backend_for(renderer, real_tiny, load_model=load_once)
    coordinates = Coordinates(
        item_id="q1",
        paper_id="paperA",
        matrix_arm="confirmatory",
        representation_or_condition="JATS",
        repeat_index=1,
    )
    rendered = rendered_for(renderer)
    record = build_input_record(
        renderer, rendered, observation_key=store.key_for(coordinates), context=CONTEXT_REF
    )
    entry = RosterEntry(
        item_id="q1",
        paper_id="paperA",
        representation_or_condition="JATS",
        repeat_index=1,
        rendered=rendered,
        input_record=record,
    )
    summary = run_roster(
        store, [entry], backend, capture_environment=lambda: backend.environment(cpu_probes())
    )
    assert summary.complete == 1 and not store.record.official
    key = store.observation_keys()[0]
    final = store.observation(key).events[-1]
    assert final.parse_status == "unparseable"  # random weights do not write the answer schema
    assert final.execution_receipt_sha256 is not None
    reopened = RunStore.open(tmp_path, store.run_id, config=renderer.config, backend_kind="smoke")
    assert reopened.observation(key).status == "complete"
    environment_files = list((store.run_dir / "environments").glob("*.json"))
    assert len(environment_files) == 1
    stored = json.loads(environment_files[0].read_text())
    assert stored["execution"]["snapshot"]["manifest_sha256"] == (
        backend.environment(cpu_probes()).execution.snapshot.manifest_sha256
    )
