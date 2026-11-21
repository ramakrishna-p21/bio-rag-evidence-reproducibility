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


def test_synthetic_final_test_runner_lifecycle(batch, tmp_path, monkeypatch):
    """Exercise the guarded runner's private seam with all frozen matrix coordinates."""
    from types import SimpleNamespace

    from test_finaltest_history_v2 import root_fixture

    from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
    from biorag_repro.finaltest.custody import SyntheticCustodian
    from biorag_repro.finaltest.history_v2 import verify_history

    (tmp_path / "custody").mkdir()
    root, storage, anchors, genesis = root_fixture(tmp_path / "custody")
    package = json.loads(batch.contexts.read_bytes())
    package["header"] = {
        "item_ids": [f"synthetic_{n:03d}" for n in range(305)],
        "arms": list(FROZEN_MATRIX),
    }
    base = package["records"][0]
    package["records"] = [
        {
            **base,
            "item_id": item,
            "paper_id": f"synthetic_paper_{index % 40:02d}",
            "matrix_arm": arm,
            "condition": condition,
            "split": "final_test",
            "context": None if arm == "closed_book" else base["context"],
            "context_sha256": None if arm == "closed_book" else base["context_sha256"],
            "context_tokens": None if arm == "closed_book" else base["context_tokens"],
            "retrieval_config_sha256": None if arm == "closed_book" else SHA_A,
        }
        for index, item in enumerate(package["header"]["item_ids"])
        for arm, (conditions, _) in FROZEN_MATRIX.items()
        for condition in conditions
    ]
    batch.contexts.write_bytes(json.dumps(package).encode())
    for arm in ("dense_robustness", "chunk_robustness"):
        settings = synthetic_settings() | {
            "matrix_arm": arm,
            "prompt_sha256": prompt_sha256(artifact()),
            "output_schema_sha256": output_schema_sha256(artifact()),
            "seed_registry_version": SEEDS.registry_version,
            "seed_registry_sha256": seed_registry_sha256(SEEDS),
        }
        (batch.configs / f"{arm}.toml").write_text(to_toml(settings))
    authorization = SimpleNamespace(
        source_revision="synthetic",
        policy_sha256="a" * 64,
        evidence_sha256="b" * 64,
        storage_root=root,
        storage_authorization=storage,
        anchor_dir=anchors,
        history_head_sha256=genesis,
        campaign_id="rk_capstone_final_test_2026",
    )
    custody = SyntheticCustodian(genesis)
    seam = SimpleNamespace(authorization=authorization, custodian=custody, genesis_head=genesis)
    args = batch_with(
        batch,
        contexts_sha256=sha256_bytes(batch.contexts.read_bytes()),
        phase="final_test",
        out=root / "batches" / ("run_" + "d" * 32),
        custody_head_sha256=custody.authoritative_head(),
        expected_head_sha256=genesis,
        drive_root=tmp_path / "synthetic_drive",
    )

    def acknowledge():
        from biorag_repro.core.hashing import sha256_file

        head = authorization.history_head_sha256
        request = args.drive_root / "requests" / f"{head}.json"
        retained = tmp_path / "retained_heads"
        retained.mkdir(exist_ok=True)
        target = retained / f"simulated_{head}.json"
        target.write_text(
            json.dumps(
                {
                    "schema_version": "researcher_head_ack_v1",
                    "researcher": "Rama Krishna Pudota",
                    "request_sha256": sha256_file(request),
                    "campaign_id": authorization.campaign_id,
                    "batch_id": args.out.name,
                    "new_head_sha256": head,
                }
            )
        )
        args.expected_head_sha256 = head
        args.researcher_acknowledgment = target

    def synthetic_generate(_model, _tokenizer, rendered, _ids, _planned, _seeds, **_):
        return {
            "raw": '{"answer":"synthetic","abstained":false}',
            "rendered_sha256": rendered.rendered_sha256,
            "input_tokens": rendered.input_tokens,
            "input_token_ids_sha256": rendered.input_token_ids_sha256,
            "output_tokens": 1,
            "output_token_ids_sha256": sha256_bytes(b"synthetic"),
            "stop_reason": "eos",
            "seed": None,
            "generation_ns": 1,
            "cuda_memory": {"status": "synthetic"},
        }

    monkeypatch.setattr(rg, "generate", synthetic_generate)
    with pytest.raises(rg.Refused, match="researcher acknowledgment required"):
        rg._run(batch_with(args, stop_after=2), model=object(), synthetic=seam)
    acknowledge()
    args.custody_head_sha256 = custody.authoritative_head()
    original_write = rg.write_once

    def interrupted_write(path, data):
        original_write(path, data)
        if path.parent.name == "results":
            raise KeyboardInterrupt("synthetic termination after result publication")

    monkeypatch.setattr(rg, "write_once", interrupted_write)
    with pytest.raises(KeyboardInterrupt, match="synthetic termination"):
        rg._run(batch_with(args, stop_after=1), model=object(), synthetic=seam)
    monkeypatch.setattr(rg, "write_once", original_write)
    args.custody_head_sha256 = custody.authoritative_head()
    with pytest.raises(rg.Refused, match="researcher acknowledgment required"):
        rg._run(args, model=object(), synthetic=seam)
    acknowledge()
    args.custody_head_sha256 = custody.authoritative_head()
    assert rg._run(args, model=object(), synthetic=seam) == 0
    assert len(list((args.out / "results").glob("obs_*.json"))) == 4575
    assert len(verify_history(root, storage, anchors, authorization.history_head_sha256)) == 2


def test_small_amended_drive_runner_ack_and_restart(batch, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from test_finaltest_history_v2 import root_fixture

    from biorag_repro.core.hashing import sha256_file
    from biorag_repro.finaltest.custody import SyntheticCustodian

    root, storage, anchors, genesis = root_fixture(tmp_path)
    package = json.loads(batch.contexts.read_bytes())
    for record in package["records"]:
        record["split"] = "final_test"
    batch.contexts.write_text(json.dumps(package))
    monkeypatch.setattr(rg, "validate_final_test_plan", lambda _package, _plan: None)
    authorization = SimpleNamespace(
        source_revision="synthetic",
        policy_sha256="a" * 64,
        evidence_sha256="b" * 64,
        storage_root=root,
        storage_authorization=storage,
        anchor_dir=anchors,
        history_head_sha256=genesis,
        campaign_id="rk_capstone_final_test_2026",
    )
    custodian = SyntheticCustodian(genesis)
    seam = SimpleNamespace(authorization=authorization, custodian=custodian, genesis_head=genesis)
    drive = Path(os.environ.get("BIORAG_SYNTHETIC_DRIVE_ROOT", tmp_path / "drive"))
    retained = tmp_path / "retained"
    retained.mkdir()

    def fake_generate(_model, _tokenizer, rendered, _ids, _planned, _seeds, **_):
        return {
            "raw": '{"answer":"synthetic","abstained":false}',
            "rendered_sha256": rendered.rendered_sha256,
            "input_tokens": rendered.input_tokens,
            "input_token_ids_sha256": rendered.input_token_ids_sha256,
            "output_tokens": 1,
            "output_token_ids_sha256": sha256_bytes(b"synthetic"),
            "stop_reason": "eos",
            "seed": None,
            "generation_ns": 1,
            "cuda_memory": {"status": "synthetic"},
        }

    monkeypatch.setattr(rg, "generate", fake_generate)
    first = batch_with(
        batch,
        phase="final_test",
        contexts_sha256=sha256_bytes(batch.contexts.read_bytes()),
        out=root / "batches" / ("run_" + "a" * 32),
        drive_root=drive,
        custody_head_sha256=custodian.authoritative_head(),
        stop_after=1,
        expected_head_sha256=genesis,
    )
    with pytest.raises(rg.Refused, match="researcher acknowledgment required"):
        rg._run(first, model=object(), synthetic=seam)
    head = authorization.history_head_sha256
    request = drive / "requests" / f"{head}.json"
    acknowledgment = retained / f"simulated_{head}.json"
    acknowledgment.write_text(
        json.dumps(
            {
                "schema_version": "researcher_head_ack_v1",
                "researcher": "Rama Krishna Pudota",
                "request_sha256": sha256_file(request),
                "campaign_id": authorization.campaign_id,
                "batch_id": first.out.name,
                "new_head_sha256": head,
            }
        )
    )
    second_package = json.loads(first.contexts.read_bytes())
    second_package["header"]["item_ids"] = ["q_2"]
    for record in second_package["records"]:
        record["item_id"] = "q_2"
        record["question"] = "What is measured in q_2?"
    second_contexts = tmp_path / "second_contexts.json"
    second_contexts.write_text(json.dumps(second_package))
    second = batch_with(
        first,
        contexts=second_contexts,
        contexts_sha256=sha256_bytes(second_contexts.read_bytes()),
        out=root / "batches" / ("run_" + "b" * 32),
        researcher_acknowledgment=acknowledgment,
        custody_head_sha256=custodian.authoritative_head(),
        expected_head_sha256=head,
    )
    with pytest.raises(rg.Refused, match="observation in two batches"):
        rg._run(
            batch_with(
                first,
                out=second.out,
                researcher_acknowledgment=acknowledgment,
                expected_head_sha256=head,
            ),
            model=object(),
            synthetic=seam,
        )
    original = rg.write_once

    def interrupt(path, data):
        original(path, data)
        if path.parent.name == "results":
            raise KeyboardInterrupt("synthetic interruption")

    monkeypatch.setattr(rg, "write_once", interrupt)
    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        rg._run(second, model=object(), synthetic=seam)
    monkeypatch.setattr(rg, "write_once", original)
    from biorag_repro.finaltest import history_v2

    original_publish = history_v2.publish_bytes

    def interrupt_anchor(path, *args, **kwargs):
        if path.parent == anchors:
            raise KeyboardInterrupt("synthetic interruption after Drive manifest")
        return original_publish(path, *args, **kwargs)

    monkeypatch.setattr(history_v2, "publish_bytes", interrupt_anchor)
    with pytest.raises(KeyboardInterrupt, match="after Drive manifest"):
        rg._run(second, model=object(), synthetic=seam)
    monkeypatch.setattr(history_v2, "publish_bytes", original_publish)
    with pytest.raises(rg.Refused, match="researcher acknowledgment required"):
        rg._run(second, model=object(), synthetic=seam)
    assert len(list((second.out / "results").glob("obs_*.json"))) == 2
    second_head = authorization.history_head_sha256
    second_request = drive / "requests" / f"{second_head}.json"
    second_ack = retained / f"simulated_{second_head}.json"
    second_ack.write_text(
        json.dumps(
            {
                "schema_version": "researcher_head_ack_v1",
                "researcher": "Rama Krishna Pudota",
                "request_sha256": sha256_file(second_request),
                "campaign_id": authorization.campaign_id,
                "batch_id": second.out.name,
                "new_head_sha256": second_head,
            }
        )
    )
    second = batch_with(
        second,
        expected_head_sha256=second_head,
        researcher_acknowledgment=second_ack,
        stop_after=None,
    )
    from biorag_repro.finaltest import amended_custody

    original_request = amended_custody.publish_request

    def interrupt_request(*_args, **_kwargs):
        raise KeyboardInterrupt("synthetic interruption after external anchor")

    monkeypatch.setattr(amended_custody, "publish_request", interrupt_request)
    with pytest.raises(KeyboardInterrupt, match="after external anchor"):
        rg._run(second, model=object(), synthetic=seam)
    monkeypatch.setattr(amended_custody, "publish_request", original_request)
    pending = authorization.history_head_sha256
    authorization.history_head_sha256 = second_head
    authorization.pending_head_sha256 = pending
    with pytest.raises(rg.Refused, match="researcher acknowledgment required"):
        rg._run(second, model=object(), synthetic=seam)
    assert len(list((second.out / "results").glob("obs_*.json"))) == 11


@pytest.mark.parametrize("field", ["input_tokens", "input_token_ids_sha256"])
def test_a_generation_input_unlike_the_rendered_input_stops_before_any_attempt(
    batch, monkeypatch, field
):
    render = rg.InputRenderer.render

    def drifted(self, *args):
        rendered = render(self, *args)
        wrong = rendered.input_tokens + 1 if field == "input_tokens" else "0" * 64
        return rendered.model_copy(update={field: wrong})

    monkeypatch.setattr(rg.InputRenderer, "render", drifted)
    with pytest.raises(rg.Refused, match="token-id hashes differ"):
        rg.run(batch_with(batch, stop_after=1))
    assert events(batch.out) == [] and results(batch.out) == {}
    assert not (batch.out / "batch.lock").exists()


def test_a_cpu_call_records_tokens_timing_and_no_cuda_memory(batch):
    rg.run(batch_with(batch, stop_after=3))
    _package, plan, _seeds, prompt = rg.load_inputs(batch)
    by_key = {p.key: p for p in plan}
    for key, value in results(batch.out).items():
        planned = by_key[key]
        rendered = rg.InputRenderer(ROOT, planned.config, prompt).render(
            planned.record["question"], planned.record["context"], planned.record["context_tokens"]
        )
        assert value["input_tokens"] == rendered.input_tokens
        assert value["input_token_ids_sha256"] == rendered.input_token_ids_sha256
        assert len(value["output_token_ids_sha256"]) == 64
        assert isinstance(value["generation_ns"], int) and value["generation_ns"] > 0
        assert value["cuda_memory"] == {"status": "unavailable", "reason": "device cpu is not CUDA"}
        assert "latency_s" not in value


def test_output_token_hashes_repeat_across_independent_batches(batch, tmp_path):
    rg.run(batch_with(batch, stop_after=4))
    rg.run(batch_with(batch, stop_after=4, out=tmp_path / "again"))
    first, second = results(batch.out), results(tmp_path / "again")
    assert first.keys() == second.keys()
    for key in first:
        assert first[key]["output_token_ids_sha256"] == second[key]["output_token_ids_sha256"]
    # the hash rule is SHA-256 of the compact JSON array, checked against hashlib directly
    import hashlib

    assert rg.token_ids_sha256([151645, 7, 0]) == hashlib.sha256(b"[151645,7,0]").hexdigest()


def test_a_cuda_call_is_synchronized_and_reads_this_calls_peak_memory():
    import torch

    order = []

    class Cuda:
        def synchronize(self, index):
            order.append(("sync", index))

        def memory_allocated(self, index):
            return 100

        def reset_peak_memory_stats(self, index):
            order.append(("reset", index))

        def max_memory_allocated(self, index):
            return 300

        def max_memory_reserved(self, index):
            return 400

        def mem_get_info(self, index):
            return 600, 1000

        def get_device_name(self, index):
            return "Stub GPU"

    fake = Namespace(cuda=Cuda())
    result, elapsed, memory = rg.measured_call(
        fake, torch.device("cuda:1"), lambda: order.append("call") or "out"
    )
    assert result == "out" and elapsed >= 0
    assert order == [("sync", 1), ("reset", 1), "call", ("sync", 1)]
    assert memory == {
        "status": "measured",
        "scope": "this process, peak since reset immediately before model.generate",
        "device_index": 1, "device_name": "Stub GPU", "allocated_before_bytes": 100,
        "peak_allocated_bytes": 300, "peak_reserved_bytes": 400, "device_total_bytes": 1000,
        "device_used_after_bytes": 400,
    }  # fmt: skip


def long_contexts(batch: Namespace) -> Namespace:
    """The fixture batch with development-length JATS contexts (about 1,400 tokens each)."""
    count = QwenCounter(ROOT, tokenizer_snapshot())
    data = json.loads(batch.contexts.read_bytes())
    for n, rec in enumerate(data["records"]):
        if rec["condition"] == "JATS":
            text = " ".join(
                f"In cohort {n}-{i}, expression rose 2.{i}-fold (p < 0.01)." for i in range(55)
            )
            rec.update(context=text, context_sha256=sha256_bytes(text.encode()),
                       context_tokens=count(text))  # fmt: skip
            assert 1000 < rec["context_tokens"] < 2048
    batch.contexts.write_bytes(json.dumps(data).encode())
    return batch_with(batch, contexts_sha256=sha256_bytes(batch.contexts.read_bytes()),
                      phase="qualification", item_id="q_1", condition="JATS",
                      out=batch.out.parent / "probes" / "probe.json")  # fmt: skip


def test_the_probe_fills_the_exact_evidence_budget_and_the_whole_output_allowance(batch):
    args = long_contexts(batch)
    result = rg.probe(args)
    stored = json.loads(args.out.read_bytes())
    assert stored == json.loads(rg.dump(result))
    assert stored["kind"] == "boundary_probe" and stored["not_an_observation"] is True
    assert stored["context_tokens"] == 2048 == stored["budgets"]["evidence_tokens"]
    assert stored["output_tokens"] == 128 and stored["stop_reason"] == "length"
    assert stored["probe_status"] == "complete"
    assert stored["context_origin"] == rg.BOUNDARY_ORIGIN
    assert stored["context_sources"][0] == ["q_1", "confirmatory", "JATS"]
    assert all(source[2] == "JATS" for source in stored["context_sources"])
    assert stored["checkpoint_positions"] == {"max_position_embeddings": 8192, "rope_scaling": None}
    assert stored["input_tokens"] + 128 <= stored["budgets"]["context_window_tokens"]
    assert "observation_key" not in stored and "generation_result" not in stored
    assert not batch.out.exists()  # no batch, attempt or result was created
    with pytest.raises(FileExistsError):
        rg.probe(args)


@pytest.mark.parametrize(
    ("problem", "reason"),
    [
        ("inside_batch", "never written into a batch"),
        ("window_beyond_checkpoint", "exceeds the checkpoint's positions"),
        ("final_test", "development or qualification"),
    ],
)
def test_the_probe_refuses_unsafe_targets_and_configurations(batch, problem, reason):
    args = long_contexts(batch)
    if problem == "inside_batch":
        rg.run(batch_with(args, phase="development", out=batch.out, stop_after=1))
        args = batch_with(args, out=batch.out / "probe.json")
    elif problem == "window_beyond_checkpoint":
        path = batch.configs / "confirmatory.toml"
        path.write_text(path.read_text().replace(
            '"context_window_tokens" = 8192', '"context_window_tokens" = 16384'))  # fmt: skip
    else:
        args = batch_with(args, phase="final_test")
    with pytest.raises(rg.Refused, match=reason):
        rg.probe(args)
    assert not args.out.exists()


def test_the_gpu_inventory_reads_nvidia_smi_and_states_when_it_is_absent(monkeypatch):
    def missing(*_args, **_kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(rg.subprocess, "run", missing)
    assert rg.gpu_inventory() == {"status": "unavailable", "reason": "nvidia-smi: nvidia-smi"}

    full = "Driver Version : 550.54.15\nCUDA Version : 12.4\n"
    listing = "0, NVIDIA L4, GPU-1234, 23034 MiB, 550.54.15, 8.9, 00000000:00:04.0\n"

    def fake(command, **_kwargs):
        return Namespace(stdout=full if command[1] == "-q" else listing)

    monkeypatch.setattr(rg.subprocess, "run", fake)
    found = rg.gpu_inventory()
    assert found["gpus"] == [{
        "index": "0", "name": "NVIDIA L4", "uuid": "GPU-1234", "memory.total": "23034 MiB",
        "driver_version": "550.54.15", "compute_cap": "8.9", "pci.bus_id": "00000000:00:04.0",
    }]  # fmt: skip
    assert found["driver_cuda_version"] == "12.4"
    assert found["nvidia_smi_q_sha256"] == sha256_bytes(full.encode())


def test_an_inventory_is_written_once_and_refuses_a_non_cuda_device(tmp_path):
    out = tmp_path / "inventory.json"
    rg.main(["inventory", "--out", str(out)])
    record = json.loads(out.read_bytes())
    assert record["torch_device"] is None and record["pip_freeze_sha256"]
    with pytest.raises(FileExistsError):
        rg.main(["inventory", "--out", str(out)])
    with pytest.raises(rg.Refused):
        rg.inventory("cpu")


def test_chunk_seals_survive_a_kill_and_resume_without_rewriting_or_regenerating(batch):
    """The Colab sequence: chunk, seal, kill mid-attempt, seal, resume against the anchor, seal."""
    rg.run(batch_with(batch, stop_after=3))
    first = rg.seal(batch.out)
    chunk = {k: (batch.out / "results" / f"{k}.json").read_bytes() for k in results(batch.out)}
    run_id = json.loads((batch.out / "batch.json").read_bytes())["run_id"]
    rg.append_event(batch.out, events(batch.out), run_id=run_id, attempt_id="att_killed",
                    observation_key="obs_unfinished", event="started")  # fmt: skip
    second = rg.seal(batch.out)  # sealing after a kill records the open attempt as it stands
    anchors = sorted((batch.out / "anchors").glob("*.json"))
    sealed = {p: p.read_bytes() for p in anchors}
    rg.run(batch_with(batch, anchor=anchors[-1], anchor_sha256=second))
    third = rg.seal(batch.out)

    for key, data in chunk.items():  # nothing completed before the kill was regenerated
        assert (batch.out / "results" / f"{key}.json").read_bytes() == data
    assert len(results(batch.out)) == 11
    log = events(batch.out)
    assert sum(e["event"] == "completed" for e in log) == 11
    assert [e["event"] for e in log if e["attempt_id"] == "att_killed"] == [
        "started",
        "interrupted",
    ]
    for path, data in sealed.items():  # finalized seals are never rewritten
        assert path.read_bytes() == data
    paths = sorted((batch.out / "anchors").glob("*.json"))
    assert [p.name for p in paths] == ["000001.json", "000002.json", "000003.json"]
    for path, digest in zip(paths, (first, second, third), strict=True):
        rg.check_anchor(batch.out, path, digest)  # every earlier anchor still verifies
    assert json.loads(paths[2].read_bytes())["previous_anchor_sha256"] == second
    assert rg.run(batch_with(batch, anchor=paths[2], anchor_sha256=third)) == 0


def test_published_execution_refuses_unbacked_contexts_before_loading_model(batch):
    batch.published_context_root = batch.out.parent
    with pytest.raises(rg.Refused, match="published ContextRecord"):
        rg.load_inputs(batch)


def test_published_closed_book_keeps_input_and_call_evidence(batch):
    package = json.loads(batch.contexts.read_bytes())
    package["header"]["arms"] = ["closed_book"]
    package["records"] = [r for r in package["records"] if r["matrix_arm"] == "closed_book"]
    batch.contexts.write_text(json.dumps(package))
    batch.contexts_sha256 = sha256_bytes(batch.contexts.read_bytes())
    batch.published_context_root = batch.out.parent
    assert rg.run(batch) == 3
    result_path = next((batch.out / "results").glob("*.json"))
    saved = json.loads(result_path.read_bytes())
    assert saved["generator_input"]["logical_ref"].startswith("inputs/")
    assert saved["input_token_ids"] and saved["output_token_ids"]
    from biorag_repro.core.storage import read_project_json
    from biorag_repro.generation.receipt import ExecutionReceipt

    receipt = read_project_json(
        batch.out / saved["execution_receipt"]["logical_ref"],
        saved["execution_receipt"]["sha256"],
        ExecutionReceipt,
    )
    assert receipt.input_record_sha256 == saved["generator_input"]["sha256"]
    assert receipt.facts.output_token_ids == saved["output_token_ids"]
    assert receipt.facts.generation.effective["num_beams"] == 1
    anchor_sha = rg.seal(batch.out)
    anchor = max((batch.out / "anchors").glob("*.json"))
    rg.check_anchor(batch.out, anchor, anchor_sha)
    assert "inputs/" in anchor.read_text()
    assert "receipts/" in anchor.read_text() and "environments/" in anchor.read_text()
    assert rg.run(batch) == 0
    saved["output_token_ids"][0] += 1
    result_path.chmod(0o644)
    result_path.write_text(json.dumps(saved))
    with pytest.raises(rg.Refused, match="token|receipt"):
        rg.run(batch)


def test_receipt_decoding_arguments_cannot_drift(batch):
    package = json.loads(batch.contexts.read_bytes())
    package["header"]["arms"] = ["closed_book"]
    package["records"] = [r for r in package["records"] if r["matrix_arm"] == "closed_book"]
    batch.contexts.write_text(json.dumps(package))
    batch.contexts_sha256 = sha256_bytes(batch.contexts.read_bytes())
    batch.published_context_root = batch.out.parent
    rg.run(batch)
    path = next((batch.out / "results").glob("*.json"))
    saved = json.loads(path.read_bytes())
    saved["generation_arguments"]["do_sample"] = True
    path.chmod(0o644)
    path.write_text(json.dumps(saved))
    with pytest.raises(rg.Refused, match="receipt"):
        rg.run(batch)


def test_official_verification_never_promotes_a_synthetic_batch(batch):
    rg.run(batch)
    batch.anchor_sha256 = rg.seal(batch.out)
    batch.anchor = max((batch.out / "anchors").glob("*.json"))
    with pytest.raises(rg.Refused, match="official|approved"):
        rg.verify_official_batch(batch)


def test_seal_cannot_resolve_a_file_outside_the_batch(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"not batch evidence")
    batch = tmp_path / "batch"
    batch.mkdir()
    anchor = batch / "anchor.json"
    anchor.write_text(
        json.dumps({"files": {"../outside.json": sha256_bytes(outside.read_bytes())}})
    )
    with pytest.raises(rg.Refused, match="path|outside"):
        rg.check_anchor(batch, anchor, sha256_bytes(anchor.read_bytes()))


def test_preflight_refuses_a_partial_roster_without_model_execution(batch):
    with pytest.raises(rg.Refused, match="complete five-arm"):
        rg.preflight(batch)


@pytest.mark.parametrize("damage", ["missing", "malformed", "valid"])
def test_published_completion_refuses_missing_or_malformed_typed_receipt(batch, damage):
    package = json.loads(batch.contexts.read_bytes())
    package["header"]["arms"] = ["closed_book"]
    package["records"] = [r for r in package["records"] if r["matrix_arm"] == "closed_book"]
    batch.contexts.write_text(json.dumps(package))
    batch.contexts_sha256 = sha256_bytes(batch.contexts.read_bytes())
    batch.published_context_root = batch.out.parent
    rg.run(batch)
    last = rg.read_events(batch.out)[-1]
    saved = json.loads((batch.out / "results" / f"{last['observation_key']}.json").read_bytes())
    # Model a crash after the result write but before its completion event.
    event_path = batch.out / "attempts" / f"{last['seq']:06d}.json"
    event_path.chmod(0o644)
    event_path.unlink()
    events_before = rg.read_events(batch.out)
    path = batch.out / saved["execution_receipt"]["logical_ref"]
    path.chmod(0o644)
    if damage == "valid":
        before = {
            p: p.read_bytes()
            for folder in ("results", "receipts", "raw")
            for p in (batch.out / folder).rglob("*")
            if p.is_file()
        }
        assert rg.run(batch) == 0
        assert rg.read_events(batch.out)[-1]["recovered"] is True
        assert all(p.read_bytes() == content for p, content in before.items())
    else:
        if damage == "missing":
            path.unlink()
        else:
            path.write_text('{"schema_version":"1"}')
        with pytest.raises(rg.Refused, match="receipt"):
            rg.run(batch)
        assert rg.read_events(batch.out) == events_before
