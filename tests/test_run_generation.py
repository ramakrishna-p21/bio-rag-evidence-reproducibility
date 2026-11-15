"""Scientific behaviour of the candidate runner (scripts/run_generation.py) on a tiny CPU model.

Everything is synthetic; no output is a scientific result.
"""

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest
from generation_support import (
    QWEN_REVISION,
    ROOT,
    SHA_A,
    artifact,
    needs_tokenizer,
    synthetic_settings,
    to_toml,
)
from tiny_model import build_tiny

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.generation.prompt import output_schema_sha256, prompt_sha256
from biorag_repro.generation.seeds import GenerationSeedRegistry, seed_registry_sha256
from biorag_repro.retrieval.context import QwenCounter, tokenizer_snapshot
from scripts import run_generation as rg

pytestmark = needs_tokenizer

SEEDS = GenerationSeedRegistry(
    registry_version="synthetic-seeds-1",
    entries=[
        {"name": f"generation_repeat_{n}", "seed": 100 + n, "consumer": "transformers.set_seed"}
        for n in (1, 2, 3)
    ],
)


def record(item, arm, condition, context, count):
    return {
        "item_id": item, "paper_id": "P001", "benchmark_version": "benchmark_v1",
        "split": "development", "question": f"What is measured in {item}?",
        "matrix_arm": arm, "condition": condition, "context": context,
        "context_sha256": None if context is None else sha256_bytes(context.encode()),
        "context_tokens": None if context is None else count(context),
        "context_stop": None, "included_chunk_ids": [], "candidate_chunk_ids": [],
        "retrieval_config_sha256": None if context is None else SHA_A,
    }  # fmt: skip


@pytest.fixture(scope="module")
def model_root(tmp_path_factory):
    return build_tiny(tmp_path_factory.mktemp("tiny") / QWEN_REVISION)


@pytest.fixture
def batch(tmp_path, model_root):
    count = QwenCounter(ROOT, tokenizer_snapshot())
    records = [
        record("q_1", "confirmatory", "JATS", "Expression rose 2.5-fold (p < 0.01).", count),
        record("q_1", "confirmatory", "PDF_DOCLING", "Expression rose 2.5-fold (p<0.01).", count),
        record("q_1", "scope_robustness", "JATS", "Expression rose 2.5-fold.", count),
        record("q_1", "scope_robustness", "PDF_DOCLING", "Expression rose 2.5 fold.", count),
        record("q_1", "closed_book", "CLOSED_BOOK", None, count),
    ]
    contexts = tmp_path / "contexts.json"
    header = {"item_ids": ["q_1"], "arms": ["confirmatory", "scope_robustness", "closed_book"]}
    contexts.write_bytes(json.dumps({"header": header, "records": records}).encode())
    prompt = artifact()
    (tmp_path / "prompt.json").write_text(prompt.model_dump_json())
    (tmp_path / "seeds.json").write_text(SEEDS.model_dump_json())
    configs = tmp_path / "configs"
    configs.mkdir()
    for arm in ("confirmatory", "scope_robustness", "closed_book"):
        settings = synthetic_settings() | {
            "matrix_arm": arm,
            "prompt_sha256": prompt_sha256(prompt),
            "output_schema_sha256": output_schema_sha256(prompt),
            "seed_registry_version": SEEDS.registry_version,
            "seed_registry_sha256": seed_registry_sha256(SEEDS),
            "retry_max_attempts": 2,
        }
        if arm == "closed_book":
            del settings["retrieval_config_sha256"]
        (configs / f"{arm}.toml").write_text(to_toml(settings))
    return Namespace(
        contexts=contexts, contexts_sha256=sha256_bytes(contexts.read_bytes()), configs=configs,
        prompt=tmp_path / "prompt.json", seeds=tmp_path / "seeds.json", tokenizer_root=ROOT,
        model_root=model_root, device="cpu", phase="development", out=tmp_path / "batch",
        anchor=None, anchor_sha256=None, stop_after=None,
    )  # fmt: skip


def results(out: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_bytes()) for p in (out / "results").glob("obs_*.json")}


def events(out: Path) -> list[dict]:
    return [json.loads(p.read_bytes()) for p in sorted((out / "attempts").glob("*.json"))]


class FailingModel:
    """Stands in for a model whose generate call fails operationally."""

    def __init__(self, error):
        import torch

        self.error, self.weight = error, torch.zeros(1)

    def parameters(self):
        yield self.weight

    def generate(self, **_):
        raise self.error


def test_plan_follows_the_frozen_matrix_and_the_shared_identity(batch):
    _package, plan, _seeds, _prompt = rg.load_inputs(batch)
    counts = {}
    for p in plan:
        counts[(p.record["matrix_arm"], p.record["condition"])] = (
            counts.get((p.record["matrix_arm"], p.record["condition"]), 0) + 1
        )
        assert p.key == observation_key(
            phase="development", benchmark_version="benchmark_v1", item_id="q_1",
            matrix_arm=p.record["matrix_arm"], representation_or_condition=p.record["condition"],
            repeat_index=p.repeat, resolved_config_sha256=p.config.configuration_sha256,
        )  # fmt: skip
    assert counts == {
        ("confirmatory", "JATS"): 3, ("confirmatory", "PDF_DOCLING"): 3,
        ("scope_robustness", "JATS"): 1, ("scope_robustness", "PDF_DOCLING"): 1,
        ("closed_book", "CLOSED_BOOK"): 3,
    }  # fmt: skip
    pair = {p.config.configuration_sha256 for p in plan if p.record["matrix_arm"] == "confirmatory"}
    assert len(pair) == 1  # JATS and PDF_DOCLING share the configuration


@pytest.mark.parametrize(
    "change", ["final_test", "not_development", "wrong_hash", "missing_context", "extra_record"]
)
def test_inputs_outside_the_development_scope_are_refused(batch, change):
    if change == "final_test":
        batch.phase = "final_test"
    elif change == "not_development":
        data = json.loads(batch.contexts.read_bytes())
        data["records"][0]["split"] = "final_test"
        batch.contexts.write_bytes(json.dumps(data).encode())
        batch.contexts_sha256 = sha256_bytes(batch.contexts.read_bytes())
    elif change == "wrong_hash":
        batch.contexts_sha256 = "0" * 64
    else:  # a missing context never shrinks the plan; an unplanned record is not accepted
        data = json.loads(batch.contexts.read_bytes())
        if change == "missing_context":
            data["records"].pop(1)
        else:
            data["records"].append(dict(data["records"][0], item_id="q_2"))
        batch.contexts.write_bytes(json.dumps(data).encode())
        batch.contexts_sha256 = sha256_bytes(batch.contexts.read_bytes())
    with pytest.raises(rg.Refused):
        rg.load_inputs(batch)
    assert not batch.out.exists()


def test_closed_book_renders_without_context(batch):
    _package, plan, _seeds, prompt = rg.load_inputs(batch)
    closed = next(p for p in plan if p.record["matrix_arm"] == "closed_book")
    renderer = rg.InputRenderer(ROOT, closed.config, prompt)
    rendered = renderer.render(closed.record["question"], None, None)
    assert rendered.context is None and "evidence below" not in rendered.rendered_text


def test_an_interrupted_batch_resumes_only_its_missing_observations(batch):
    assert rg.run(batch_with(batch, stop_after=4)) == 4
    first = json.loads((batch.out / "batch.json").read_bytes())
    # a crash during an attempt leaves a started event without an end
    rg.append_event(batch.out, events(batch.out), run_id=first["run_id"], attempt_id="att_x",
                    observation_key=next(iter(results(batch.out))), event="started")  # fmt: skip
    assert rg.run(batch) == 7
    done = results(batch.out)
    assert len(done) == 11
    assert {r["generation_result"]["run_id"] for r in done.values()} == {first["run_id"]}
    assert json.loads((batch.out / "batch.json").read_bytes()) == first
    log = events(batch.out)
    assert [e["seq"] for e in log] == list(range(1, len(log) + 1))
    assert any(e["event"] == "interrupted" and e["attempt_id"] == "att_x" for e in log)
    assert sum(e["event"] == "completed" for e in log) == 11
    assert rg.run(batch) == 0  # nothing is regenerated


def test_a_tampered_result_stops_the_batch_and_is_not_regenerated(batch):
    rg.run(batch_with(batch, stop_after=1))
    path = next((batch.out / "results").glob("obs_*.json"))
    value = json.loads(path.read_bytes())
    value["generation_result"]["answer"] = "a better answer"
    os.chmod(path, 0o644)
    path.write_text(json.dumps(value))
    with pytest.raises(rg.Refused):
        rg.run(batch)
    assert json.loads(path.read_bytes())["generation_result"]["answer"] == "a better answer"


def test_a_completed_result_cannot_be_overwritten(tmp_path):
    rg.write_once(tmp_path / "r.json", b"first\n")
    with pytest.raises(FileExistsError):
        rg.write_once(tmp_path / "r.json", b"second\n")
    assert (tmp_path / "r.json").read_bytes() == b"first\n"


def test_operational_failures_are_retried_by_the_rule_and_recorded_as_missing(batch):
    import torch

    oom = FailingModel(torch.cuda.OutOfMemoryError("synthetic"))
    assert rg.run(batch_with(batch, stop_after=1), model=oom) == 1
    log = events(batch.out)
    assert [e["event"] for e in log] == ["started", "failed", "started", "failed"]
    assert [e["final"] for e in log if e["event"] == "failed"] == [False, True]
    (only,) = results(batch.out).values()
    result = only["generation_result"]
    assert result["status"] == "operational_failure" and result["answer"] is None
    assert result["failure"].startswith("out_of_memory")


def test_a_changed_configuration_cannot_resume_the_batch(batch):
    rg.run(batch_with(batch, stop_after=1))
    path = batch.configs / "confirmatory.toml"
    text = path.read_text()
    path.write_text(text.replace('"max_new_tokens" = 128', '"max_new_tokens" = 64'))
    assert path.read_text() != text
    with pytest.raises(rg.Refused, match="batch identity"):
        rg.run(batch)


def test_the_batch_records_its_provenance_and_a_sealed_anchor_detects_tampering(batch):
    rg.run(batch_with(batch, stop_after=2))
    batch_record = json.loads((batch.out / "batch.json").read_bytes())
    identity = batch_record["identity"]
    assert set(identity) == {
        "phase", "contexts_sha256", "planned_keys_sha256", "generator_configurations",
        "prompt_sha256", "seed_registry_sha256", "model_manifest_sha256", "git", "environment",
    }  # fmt: skip
    assert batch_record["model_snapshot"]["manifest_sha256"] == identity["model_manifest_sha256"]
    digest = rg.seal(batch.out)
    anchor = batch.out / "anchors" / "000001.json"
    rg.check_anchor(batch.out, anchor, digest)
    event = min((batch.out / "attempts").glob("*.json"))
    os.chmod(event, 0o644)
    event.write_text(event.read_text().replace("started", "completed"))
    with pytest.raises(rg.Refused, match="differs from the sealed anchor"):
        rg.check_anchor(batch.out, anchor, digest)


def batch_with(batch: Namespace, **changes) -> Namespace:
    return Namespace(**(vars(batch) | changes))
