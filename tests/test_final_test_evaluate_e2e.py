"""The real `evaluate()` pathway on an isolated synthetic batch.

Only the protected-input stages (custody verification, Stage A/B reads, the roster and the
source runs) are replaced; Stage C adjudication, score publication and read-back, table build,
handoff description and storage run as in an official evaluation. No protected text is read.
"""

import hashlib
import json
from types import SimpleNamespace

import pytest
from test_analysis_export import CFG, result_for
from test_final_test_evaluation import AT, _evidence, final_worlds

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation.analysis_models import PlannedRoster, RosterArm, RosterItem
from biorag_repro.evaluation.stage_a import final_test_scope, item_digest
from biorag_repro.evaluation.stage_c_storage import publish_score as real_publish_score

HEAD = "a" * 40
CONDITIONS = ("JATS", "PDF_DOCLING")
# item -> answer per condition: "1" is correct; "1 genes" is adjudicated by the reviewer
ANSWERS = {"q_001": ("1", "1"), "q_002": ("1", "1 genes")}


def _build_world():
    with final_test_scope():
        evidence, items, results = {}, {}, []
        for item_id, answers in ANSWERS.items():
            worlds = final_worlds(item_id)
            items[item_id] = worlds[0].item
            for world, condition, answer in zip(worlds, CONDITIONS, answers, strict=True):
                evidence[f"{item_id}/confirmatory/{condition}"] = _evidence(
                    world, condition, "delivered"
                )[1]
                for repeat in (1, 2, 3):
                    results.append(result_for(
                        world.item, answer=answer, phase="final_test", matrix_arm="confirmatory",
                        representation_or_condition=condition, repeat_index=repeat,
                    ))  # fmt: skip
    return items, evidence, results


def _roster(items):
    return PlannedRoster(
        scope="confirmatory_only", phase="final_test", benchmark_version="benchmark_v1",
        corpus_version="corpus_v1", declared_by="synthetic", basis="synthetic roster",
        arms=[RosterArm(matrix_arm="confirmatory", conditions=list(CONDITIONS), repeats=[1, 2, 3])],
        items=[RosterItem(item_id=i.item_id, state="included", paper_id=i.paper_id,
                          item_sha256=item_digest(i)) for i in items.values()],
    )  # fmt: skip


def _setup(tmp_path, monkeypatch, name="run"):
    from scripts import evaluate_final_test as evaluator

    items, evidence, results = _build_world()
    base = tmp_path / name
    batch, root, review = base / "batch", base / "analysis_root", base / "stage_c"
    (batch / "generation").mkdir(parents=True)
    (batch / "results").mkdir()
    (batch / "inputs").mkdir()
    raw = results[0].raw_raw if hasattr(results[0], "raw_raw") else results[0].raw_output
    (batch / results[0].output.logical_ref).write_bytes(raw.encode())
    gin = b"synthetic generator input"
    (batch / "inputs" / "g.txt").write_bytes(gin)
    gref = {"artifact_id": "generator_input", "logical_ref": "inputs/g.txt",
            "sha256": sha256_bytes(gin)}  # fmt: skip
    for r in results:
        (batch / "results" / f"{r.observation_key}.json").write_text(
            json.dumps({"generator_input": gref})
        )
    (batch / "batch.json").write_text(json.dumps({"identity": {"git": {"head": HEAD}}}))
    inventory = base / "stage_b_inventory.json"
    inventory.write_text(json.dumps({
        "status": "stage_b_judgments_submitted", "rules": "v2", "scope": "confirmatory_only",
        "stage_b_judgments": {c: {"artifact_id": c, "logical_ref": "x", "sha256": "0" * 64}
                              for c in evidence},
    }))  # fmt: skip
    paths = {"batch": batch, "analysis_root": root, "stage_c_review": review,
             "stage_b_inventory": inventory}  # fmt: skip
    auth = SimpleNamespace(drive_manifest_sha256="9" * 64, custody_exception_sha256="8" * 64,
                           run_id="run_synthetic")  # fmt: skip
    records = [
        {
            "item_id": c.split("/")[0],
            "matrix_arm": "confirmatory",
            "condition": c.split("/")[2],
            "retrieval_config_sha256": CFG,
        }
        for c in evidence
    ]
    ctx = evaluator.Inputs({}, auth, ArtifactRef(artifact_id="a", logical_ref="a", sha256="4" * 64),
                           dict(items), {"records": records}, [], None, None,
                           "confirmatory_only")  # fmt: skip
    ctx.request = {}
    ctx.path = lambda key: paths[key]
    judged = {f"{c.split('/')[0]}/{c.split('/')[2]}": (e.document, e.canonical, e.stage_a)
              for c, e in evidence.items()}  # fmt: skip
    monkeypatch.setattr(evaluator, "verify_batch", lambda c: results)
    monkeypatch.setattr(evaluator, "stage_a", lambda c: judged)
    monkeypatch.setattr(evaluator, "contexts", lambda c, j: [
        (c_, {"condition": c_.split("/")[2]}, e.b_inputs) for c_, e in evidence.items()])  # fmt: skip
    monkeypatch.setattr(evaluator, "read_b_judgment", lambda ref, root, inputs:
                        evidence[ref.artifact_id].stage_b)  # fmt: skip
    monkeypatch.setattr(evaluator, "read_candidate", lambda *a: None)
    monkeypatch.setattr(evaluator, "planned_roster", lambda c: _roster(items))
    run = ah.SourceRun(
        run_id="run_synthetic", completion=ctx.authorization_ref, run_record_sha256="5" * 64,
        resolved_config_sha256=results[0].resolved_config_sha256, raw_config_sha256="6" * 64,
        roster_sha256="7" * 64, matrix_arm="confirmatory", backend_kind="official",
        observations=len(results),
    )  # fmt: skip
    monkeypatch.setattr(evaluator, "source_runs", lambda *a: [run])
    summary = evaluator.stage_c_queue(ctx, review, 5)
    template = json.loads((review / "reviewer" / "decisions.json").read_text())
    assert summary["entries"] == len(template) == 1  # one distinct non-evaluable answer
    for handle in template:
        template[handle] = {"status": "correct", "rationale": "count matches; C1", "at_utc": AT}
    (review / "reviewer" / "decisions.json").write_text(json.dumps(template))
    return evaluator, ctx, root


def _tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }  # fmt: skip


def test_evaluate_publishes_scores_table_and_handoff_and_is_repeatable(tmp_path, monkeypatch):
    evaluator, ctx, root = _setup(tmp_path, monkeypatch)
    with final_test_scope():
        first = evaluator.evaluate(ctx)
    published = _tree(root)
    assert first["status"] == "verified_official_final_test_handoff"
    assert first["observations"] == 12 and any("stage_c_score" in k for k in published)
    with final_test_scope():
        second = evaluator.evaluate(ctx)
    assert second == first and _tree(root) == published  # write-once: nothing rewritten


def test_interrupted_publication_recovers_to_the_same_result(tmp_path, monkeypatch):
    evaluator, ctx, root = _setup(tmp_path, monkeypatch, "clean")
    with final_test_scope():
        clean = evaluator.evaluate(ctx)
    clean_tree = _tree(root)
    evaluator, ctx, root = _setup(tmp_path, monkeypatch, "crash")
    calls = []

    def crashing(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("simulated interruption")
        return real_publish_score(*args, **kwargs)

    monkeypatch.setattr(evaluator, "publish_score", crashing)
    with final_test_scope(), pytest.raises(RuntimeError, match="simulated"):
        evaluator.evaluate(ctx)
    partial = _tree(root)
    assert 0 < len(partial) < len(clean_tree)  # some scores published, no table or handoff yet
    monkeypatch.setattr(evaluator, "publish_score", real_publish_score)
    with final_test_scope():
        resumed = evaluator.evaluate(ctx)
    assert resumed == clean and _tree(root) == clean_tree
    assert all(partial[k] == v for k, v in _tree(root).items() if k in partial)  # untouched


def test_interruption_after_the_table_and_before_the_handoff_recovers(tmp_path, monkeypatch):
    evaluator, ctx, root = _setup(tmp_path, monkeypatch, "clean")
    with final_test_scope():
        clean = evaluator.evaluate(ctx)
    clean_tree = _tree(root)
    evaluator, ctx, root = _setup(tmp_path, monkeypatch, "crash")
    real_store = ah._store

    def crashing(*args, **kwargs):
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(ah, "_store", crashing)
    with final_test_scope(), pytest.raises(RuntimeError, match="simulated"):
        evaluator.evaluate(ctx)
    assert any("analysis_table" in k for k in _tree(root))  # the table is already published
    monkeypatch.setattr(ah, "_store", real_store)
    with final_test_scope():
        resumed = evaluator.evaluate(ctx)
    assert resumed == clean and _tree(root) == clean_tree


def test_a_tampered_published_score_is_refused_not_overwritten(tmp_path, monkeypatch):
    evaluator, ctx, root = _setup(tmp_path, monkeypatch)
    with final_test_scope():
        evaluator.evaluate(ctx)
    victim = next(p for p in root.rglob("*") if p.is_file() and "stage_c_score" in str(p)
                  and p.suffix == ".json")  # fmt: skip
    victim.chmod(0o644)
    original = victim.read_bytes()
    victim.write_bytes(original.replace(b'"correct"', b'"incorrect"', 1) or original + b" ")
    tampered = victim.read_bytes()
    assert tampered != original
    with final_test_scope(), pytest.raises(Exception):  # noqa: B017 - any refusal is correct
        evaluator.evaluate(ctx)
    assert victim.read_bytes() == tampered  # refused, never repaired in place


def test_evaluation_refuses_when_a_stage_c_decision_is_missing(tmp_path, monkeypatch):
    evaluator, ctx, _ = _setup(tmp_path, monkeypatch)
    path = ctx.path("stage_c_review") / "reviewer" / "decisions.json"
    template = json.loads(path.read_text())
    path.write_text(json.dumps({h: {"status": None, "rationale": None, "at_utc": None}
                                for h in template}))  # fmt: skip
    with final_test_scope(), pytest.raises(evaluator.Refused, match="every Stage C queue entry"):
        evaluator.evaluate(ctx)
