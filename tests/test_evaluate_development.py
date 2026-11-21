"""The official postprocessor cannot turn a synthetic batch into scientific evidence."""

import gc
import json
import weakref
from argparse import Namespace
from types import SimpleNamespace

import pytest

from scripts.evaluate_development import evaluate
from scripts.run_generation import Refused


@pytest.mark.parametrize("determination", ["needs_review", "automatic_not_delivered"])
def test_v2_incomplete_repeats_keep_answer_scores_and_require_separate_p027(
    tmp_path, monkeypatch, determination
):
    from scripts import evaluate_development as evaluator

    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps({"protocol": evaluator.ACCOUNTING_V2}))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "view_dir": str(tmp_path),
                "view_sha256": "0" * 64,
                "canonical": {
                    "P027/JATS": {
                        "root": str(tmp_path),
                        "logical_ref": "validation.json",
                        "sha256": "4" * 64,
                    }
                },
            }
        )
    )
    p027_path = tmp_path / "p027.json"
    p027_path.write_text(
        json.dumps(
            {
                "status": "human_judgments_submitted",
                "judgments": {
                    f"P027_item_{i}/{representation}": {
                        "artifact_id": "judgment",
                        "logical_ref": "judgment.json",
                        "sha256": "8" * 64,
                    }
                    for i in range(8)
                    for representation in ("JATS", "PDF_DOCLING")
                },
            }
        )
    )
    items = {"incomplete": SimpleNamespace(item_id="incomplete", paper_id="P001")}
    items.update(
        {
            f"P027_item_{i}": SimpleNamespace(item_id=f"P027_item_{i}", paper_id="P027")
            for i in range(8)
        }
    )
    results = [
        SimpleNamespace(
            item_id="incomplete",
            matrix_arm="confirmatory",
            representation_or_condition="JATS",
            repeat_index=repeat,
            resolved_config_sha256="1" * 64,
            status="completed",
        )
        for repeat in (1, 2)
    ]
    results += [
        SimpleNamespace(
            item_id="P027_item_0",
            matrix_arm="confirmatory",
            representation_or_condition="JATS",
            repeat_index=repeat,
            resolved_config_sha256="1" * 64,
            status="operational_failure",
        )
        for repeat in (1, 2)
    ]
    results.append(
        SimpleNamespace(
            item_id="incomplete",
            matrix_arm="dense_robustness",
            representation_or_condition="JATS",
            repeat_index=1,
            resolved_config_sha256="1" * 64,
            status="completed",
        )
    )
    monkeypatch.setattr(evaluator.generation, "verify_official_batch", lambda args: results)
    monkeypatch.setattr(
        evaluator, "read_development_view", lambda *args: (None, list(items.values()))
    )
    monkeypatch.setattr(
        evaluator.generation,
        "load_inputs",
        lambda args: (
            {
                "records": [
                    {
                        "item_id": item_id,
                        "matrix_arm": "confirmatory",
                        "condition": "JATS",
                        "retrieval_config_sha256": "2" * 64,
                        "context_record": {
                            "artifact_id": "context",
                            "logical_ref": "context.json",
                            "sha256": "5" * 64,
                        },
                    }
                    for item_id in ("incomplete", "P027_item_0")
                ]
                + [
                    {
                        "item_id": "incomplete",
                        "matrix_arm": "dense_robustness",
                        "condition": "JATS",
                        "retrieval_config_sha256": "2" * 64,
                    }
                ]
            },
            [],
            None,
            None,
        ),
    )
    monkeypatch.setattr(
        evaluator,
        "read_accounting_v2",
        lambda *args: {
            "entries": [
                {
                    "item_id": "incomplete",
                    "representation": "JATS",
                    "verification": "provenance_incomplete",
                }
            ]
            + [
                {
                    "item_id": f"unused_{i}",
                    "representation": "JATS",
                    "verification": "provenance_incomplete",
                }
                for i in range(155)
            ]
        },
    )
    seen_rows = []
    chunk_refs = []

    def score(item, result):
        if result.matrix_arm == "dense_robustness":
            gc.collect()
            assert all(ref() is None for ref in seen_rows)
            assert all(ref() is None for ref in chunk_refs)
        return result.repeat_index

    monkeypatch.setattr(evaluator.stage_c, "score", score)
    monkeypatch.setattr(evaluator.stage_c, "dependencies", lambda score: [])
    monkeypatch.setattr(
        evaluator.stage_c,
        "score_ref",
        lambda score: SimpleNamespace(logical_ref=f"score_{score}.json", sha256="3" * 64),
    )
    monkeypatch.setattr(evaluator, "publish_score", lambda *args: None)

    def ref(name):
        return SimpleNamespace(artifact_id=name, logical_ref=name + ".json", sha256="6" * 64)

    monkeypatch.setattr(evaluator, "read_judgment", lambda *args: object())
    monkeypatch.setattr(
        evaluator,
        "read_project_json",
        lambda path, digest, kind: {
            evaluator.Phase4Validation: SimpleNamespace(canonical_artifact=ref("canonical")),
            evaluator.CanonicalDocument: SimpleNamespace(),
            evaluator.ContextRecord: SimpleNamespace(retrieval=ref("ranked")),
            evaluator.RankedRetrieval: SimpleNamespace(index=ref("index"), retriever="bm25"),
            evaluator.Bm25Index: SimpleNamespace(chunk_sets=[ref("chunk_set")]),
            evaluator.ChunkSet: SimpleNamespace(
                input=SimpleNamespace(paper_id="P027"), chunks=ref("chunks")
            ),
        }[kind],
    )

    class ChunkToken:
        pass

    def load_chunks(*args):
        chunk = ChunkToken()
        chunk_refs.append(weakref.ref(chunk))
        return [chunk]

    monkeypatch.setattr(evaluator, "read_jsonl", load_chunks)
    monkeypatch.setattr(evaluator, "record_ref", lambda chunk: SimpleNamespace(artifact_id="chunk"))
    monkeypatch.setattr(evaluator.stage_b, "Inputs", lambda *args: SimpleNamespace(chunks=args[6]))
    monkeypatch.setattr(
        evaluator.stage_b, "evaluate", lambda *args: SimpleNamespace(determination=determination)
    )
    monkeypatch.setattr(
        evaluator.stage_b, "deterministic_judgment", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(
        evaluator.stage_b,
        "stage_b_ref",
        lambda *args: SimpleNamespace(
            logical_ref="stage_b_candidate.json",
            sha256="7" * 64,
            model_dump=lambda: {"logical_ref": "stage_b_candidate.json"},
        ),
    )
    monkeypatch.setattr(evaluator, "publish_candidate", lambda *args: None)
    monkeypatch.setattr(evaluator, "publish_judgment", lambda *args: None)

    def row(observation):
        seen_rows.append(weakref.ref(observation))
        return SimpleNamespace(
            condition="JATS",
            paper_id=observation.item.paper_id,
            evidence_reason=(
                "retrospective_provenance_incomplete"
                if observation.retrospective_provenance_incomplete
                else (
                    "stage_b_not_assessed" if observation.evidence.stage_b is None else "evaluated"
                )
            ),
            answer_correct="yes" if observation.score == 1 else "no",
        ), []

    monkeypatch.setattr(evaluator, "build_row", row)
    args = Namespace(
        analysis_root=tmp_path,
        published_context_root=tmp_path,
        review_root=tmp_path,
        out=tmp_path / "batch",
        selection=selection_path,
        reviewer_inventory=inventory_path,
        p027_inventory=p027_path,
    )
    result = evaluator.evaluate(args)
    assert result["status"] == "development_diagnostics_ready"
    assert result["answer_correctness"] == {"yes": 3, "no": 2}
    assert result["evidence_missingness_by_paper_representation"] == {
        "P001/JATS": {"retrospective_provenance_incomplete": 3},
        "P027/JATS": {
            "stage_b_not_assessed" if determination == "needs_review" else "evaluated": 2
        },
    }
    assert len(seen_rows) == 5
    assert all(ref() is None for ref in seen_rows)
    assert [entry["stage"] for entry in result["pending_stage_b_reviews"]] == (
        ["B"] if determination == "needs_review" else []
    )

    args.p027_inventory = None
    with pytest.raises(Refused, match="separate approved P027 inventory"):
        evaluator.evaluate(args)


def test_synthetic_batch_refused_before_evaluation_writes(tmp_path):
    batch = tmp_path / "runs" / "batch"
    batch.mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"identity": {"phase": "development"}}))
    args = Namespace(
        out=batch, analysis_root=tmp_path, published_context_root=tmp_path, review_root=tmp_path
    )
    before = set(tmp_path.rglob("*"))
    with pytest.raises(Refused, match="official"):
        evaluate(args)
    assert set(tmp_path.rglob("*")) == before


def test_prior_reconstruction_requires_exact_success_and_input_bytes(tmp_path):
    from biorag_repro.core.hashing import sha256_file
    from scripts import run_generation

    package = tmp_path / "package"
    inputs = package / "inputs"
    configs = inputs / "configs"
    configs.mkdir(parents=True)
    files = {
        inputs / "contexts.json": b"contexts",
        configs / "confirmatory.toml": b"config",
        inputs / "prompt.json": b"prompt",
        inputs / "seeds.json": b"seeds",
        package / "environment.toml": b"lock",
        tmp_path / "selection.json": b"selection",
        tmp_path / "anchor.json": b"anchor",
    }
    for path, data in files.items():
        path.write_bytes(data)
    manifest = package / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": {
                    "inputs/contexts.json": {
                        "sha256": sha256_file(inputs / "contexts.json"),
                        "bytes": 8,
                    }
                }
            }
        )
    )
    prefix = tmp_path / "prior"
    stdout = prefix.with_name(prefix.name + ".stdout.log")
    stdout.write_text(
        "verified 1290 approved development observations; final test remains closed\n"
    )
    prefix.with_name(prefix.name + ".stderr.log").write_bytes(b"")
    prefix.with_name(prefix.name + ".exit_status").write_text("0\n")
    prefix.with_name(prefix.name + ".start").write_text(
        "start_utc=2026-11-18T00:00:00Z\ncheckout=revision\n"
    )
    prefix.with_name(prefix.name + ".end").write_text("end_utc=2026-11-18T01:00:00Z\n")
    inventory = prefix.with_name(prefix.name + ".inputs.sha256")
    inventory.write_text("".join(f"{sha256_file(path)}  {path}\n" for path in (*files, manifest)))
    args = Namespace(
        prior_verification_prefix=prefix,
        prior_stdout_sha256=sha256_file(stdout),
        prior_inputs_sha256=sha256_file(inventory),
        contexts=inputs / "contexts.json",
        contexts_sha256=sha256_file(inputs / "contexts.json"),
        configs=configs,
        prompt=inputs / "prompt.json",
        seeds=inputs / "seeds.json",
        environment_lock=package / "environment.toml",
        selection=tmp_path / "selection.json",
        anchor=tmp_path / "anchor.json",
        anchor_sha256=sha256_file(tmp_path / "anchor.json"),
    )
    batch = {"identity": {"git": {"head": "revision"}}}
    run_generation.verify_prior_reconstruction(args, batch)
    (tmp_path / "selection.json").write_bytes(b"changed")
    with pytest.raises(Refused, match="input inventory"):
        run_generation.verify_prior_reconstruction(args, batch)


def test_stage_c_benchmark_dependency_uses_canonical_item_bytes():
    from biorag_repro.core.hashing import sha256_bytes
    from biorag_repro.core.manifest import ArtifactRef
    from scripts.evaluate_development import benchmark_dependency_bytes

    class Item:
        def model_dump(self, mode):
            assert mode == "json"
            return {"item_id": "q_one", "question": "What value?"}

    expected = b'{"item_id":"q_one","question":"What value?"}'
    ref = ArtifactRef(
        artifact_id="q_one", logical_ref="benchmark/item", sha256=sha256_bytes(expected)
    )
    assert benchmark_dependency_bytes(Item(), ref) == expected
    with pytest.raises(Refused, match="scientific hash"):
        benchmark_dependency_bytes(Item(), ref.model_copy(update={"sha256": "0" * 64}))
