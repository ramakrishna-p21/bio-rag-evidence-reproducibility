"""Final-test item source and package contract of the context builder (synthetic items only)."""

import json
import os
from argparse import Namespace
from collections import Counter
from pathlib import Path

import pytest
from test_development_view import DEV, FINAL, SPLIT, make_item

from biorag_repro.benchmark.release import BenchmarkRelease, read_final_test_items
from biorag_repro.benchmark.storage import encode_jsonl
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.readiness import evidence


def synthetic_items(papers=FINAL, count=305, prefix="F"):
    return [
        make_item(f"{prefix}{n:03d}", papers[n % len(papers)], "final_test",
                  question=f"Synthetic question {n}: which method used marker{int(papers[n % len(papers)][1:])}?")
        for n in range(count)
    ]  # fmt: skip


def write_release(data: Path, items) -> None:
    def ref(name, content):
        path = data / "benchmark/releases/benchmark_v1" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return ArtifactRef(
            artifact_id=name.split(".")[0],
            logical_ref=path.relative_to(data).as_posix(),
            sha256=sha256_bytes(content),
        )

    filler = ref("filler.json", b"{}\n")
    release = BenchmarkRelease(
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        corpus_content_sha256="a" * 64,
        corpus_release=filler,
        implementation_git_sha="f" * 40,
        seed_registry=filler,
        relatedness=filler,
        split=ref("split.json", serialize_project_json(SPLIT, type(SPLIT))),
        items=ref("items.jsonl", encode_jsonl(items)),
        decisions=filler,
        dispositions=filler,
        pre_audit_population=filler,
        audit=filler,
        validation_report=filler,
        item_count=len(items),
        development_paper_count=10,
        final_test_paper_count=40,
    )
    (data / evidence.BENCHMARK_RELEASE).write_bytes(
        serialize_project_json(release, BenchmarkRelease)
    )


@pytest.fixture
def data_root(tmp_path):
    dev = [make_item(f"D{n:03d}", DEV[n % 10], "development") for n in range(86)]
    write_release(tmp_path, dev + synthetic_items())
    return tmp_path


def test_final_test_items_are_the_frozen_final_split_only(data_root):
    release_sha256, items = read_final_test_items(data_root)
    assert len(items) == 305 and len({i.item_id for i in items}) == 305
    assert {i.paper_id for i in items} == set(FINAL)
    assert all(i.split == "final_test" for i in items)
    record = data_root / evidence.BENCHMARK_RELEASE
    assert release_sha256 == sha256_bytes(record.read_bytes())


def test_final_test_items_refuse_changed_items_or_wrong_roster(data_root, tmp_path_factory):
    items = data_root / "benchmark/releases/benchmark_v1/items.jsonl"
    items.write_bytes(items.read_bytes().replace(b"question 7:", b"question 8:"))
    with pytest.raises(ValueError, match="differ from the release") as raised:
        read_final_test_items(data_root)
    assert "Synthetic" not in str(raised.value)

    short = tmp_path_factory.mktemp("short")
    write_release(short, synthetic_items(count=304))
    with pytest.raises(ValueError, match="305"):
        read_final_test_items(short)


def test_runner_roster_check_uses_the_same_frozen_item_source(data_root):
    from scripts.run_generation import Refused, check_authorized_final_test_roster

    _, items = read_final_test_items(data_root)
    package = {
        "header": {"item_ids": [i.item_id for i in items]},
        "records": [
            {"item_id": i.item_id, "paper_id": i.paper_id, "question": i.question,
             "benchmark_version": i.benchmark_version}
            for i in items
        ],
    }  # fmt: skip
    check_authorized_final_test_roster(data_root, package)
    package["records"][3]["question"] = "A substituted question?"
    with pytest.raises(Refused, match="differs from the frozen benchmark") as raised:
        check_authorized_final_test_roster(data_root, package)
    assert "substituted" not in str(raised.value)


@pytest.mark.parametrize(
    "change",
    ["304_items", "duplicate_item", "development_item", "missing_arm", "39_papers"],
)
def test_builder_refuses_a_wrong_final_test_roster_before_opening_sources(tmp_path, change):
    from scripts.build_contexts import build_published

    items = synthetic_items()
    arms = list(FROZEN_MATRIX)
    if change == "304_items":
        items = items[:304]
    elif change == "duplicate_item":
        items[1] = items[0]
    elif change == "development_item":
        items[5] = make_item("F005", DEV[0], "development")
    elif change == "missing_arm":
        arms = arms[:-1]
    elif change == "39_papers":
        items = synthetic_items(papers=FINAL[:39])
    selection = {"arms": arms, "pool_papers": [], "canonical": {}}  # no source is ever opened
    with pytest.raises(ValueError, match="final-test"):
        build_published(
            selection, tmp_path / "contexts.json", tmp_path, final_test=("a" * 64, items)
        )
    assert not (tmp_path / "contexts.json").exists()


# ---- end to end: 50 tiny synthetic papers, the real builder, real tokenizers and BGE ----------

REAL_SELECTION = Path("artifacts/phase8/source_index_selection_verified_v1.json")


def synthetic_corpus(base: Path, monkeypatch, selection: dict) -> dict:
    """Phase 4-validated synthetic JATS/PDF pairs for P001-P050 and a corpus release."""
    from types import SimpleNamespace

    from test_corpus import paper as paper_metadata
    from test_documents_ingestion import _source
    from test_documents_table_fidelity import _native_data, _typed_native

    from biorag_repro.corpus.manifest import (
        CorpusManifest,
        PaperRecord,
        ReleaseRecord,
        corpus_content_hash,
        manifest_semantic_hash,
        serialize_csv,
    )
    from biorag_repro.documents import authorization
    from biorag_repro.documents.ingestion import build_arm, publish_validation, validate_build
    from biorag_repro.documents.pdf_ingest import publish_pdf_result

    corpus, artifacts = base / "source", base / "native"
    artifacts.mkdir(parents=True)
    pairs = {}
    for number in range(1, 51):
        pid = f"P{number:03d}"
        words = f"marker{number} method{number % 7} sample{number % 5} common"
        xml = f"<article><body><p>{words} study.</p><p>{words} result.</p></body></article>"
        jr = _source(pid, "JATS", xml.encode(), corpus)
        pr = _source(pid, "PDF", b"%PDF-1.4 synthetic " + pid.encode(), corpus)
        paper = SimpleNamespace(
            paper_id=pid,
            corpus_version="corpus_v1",
            jats=jr,
            pdf=pr,
            pair=SimpleNamespace(pair_status="verified_exact_pair"),
        )
        jats = build_arm(paper, pid, "JATS", corpus, artifacts, "d" * 64)
        data = _native_data()

        class Native:
            def save_as_json(self, path, data=data, **kwargs):
                path.write_text(json.dumps(data))

        native = SimpleNamespace(
            status=SimpleNamespace(value="success"), errors=[], pages=[], document=Native()
        )
        pdf = publish_pdf_result(
            native, corpus / "corpus/raw" / pr.logical_ref, pr, pid, "a" * 64, "b" * 64,
            "c" * 64, artifacts, load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
        )  # fmt: skip
        pairs[pid] = (
            paper,
            publish_validation(validate_build(jats, artifacts, corpus_root=corpus), artifacts),
            publish_validation(validate_build(pdf, artifacts, corpus_root=corpus), artifacts),
        )
    rows = []
    for number in range(1, 51):
        row = paper_metadata(number)
        row["corpus_version"] = "corpus_v1"
        for kind in ("jats", "pdf"):
            row[kind].update(getattr(pairs[f"P{number:03d}"][0], kind).model_dump())
        rows.append(PaperRecord.model_validate(row))
    manifest = CorpusManifest(papers=rows)
    csv = serialize_csv(manifest)
    (corpus / "manifest.csv").write_bytes(csv)
    csvref = ArtifactRef(
        artifact_id="manifest_csv", logical_ref="manifest.csv", sha256=sha256_bytes(csv)
    )
    release = ReleaseRecord(
        record_schema_version="1",
        schema_version="1",
        corpus_version="corpus_v1",
        manifest_revision="synthetic",
        csv_sha256=csvref.sha256,
        corpus_content_sha256=corpus_content_hash(manifest),
        manifest_semantic_sha256=manifest_semantic_hash(manifest),
        manifest_artifact=csvref,
        validation_report_artifact=csvref,
        git_sha="f" * 40,
    )
    content = serialize_project_json(release, ReleaseRecord)
    (corpus / "release.json").write_bytes(content)
    monkeypatch.setattr(authorization, "PHASE2H_SHA256", sha256_bytes(content))

    def entry(ref):
        return {"root": str(artifacts), **ref.model_dump()}

    return selection | {
        "corpus_root": str(corpus),
        "release_record": str(corpus / "release.json"),
        "release_sha256": sha256_bytes(content),
        "pool_papers": [f"P{n:03d}" for n in range(1, 51)],
        "canonical": {
            f"{pid}/{rep}": entry(ref)
            for pid, (_, jats, pdf) in pairs.items()
            for rep, ref in (("JATS", jats), ("PDF_DOCLING", pdf))
        },
    }


@pytest.mark.real_data  # about 40 minutes on CPU: real BGE inference and context packing
def test_synthetic_final_test_package_passes_the_real_runner_input_boundaries(
    tmp_path, monkeypatch
):
    """305 synthetic questions on 40 synthetic papers through the real builder and runner."""
    from scripts import run_generation as rg
    from scripts.build_contexts import build_published

    if not REAL_SELECTION.exists():
        pytest.skip("frozen retrieval configuration and tokenizer selection are not on this host")
    real = json.loads(REAL_SELECTION.read_bytes())
    keep = ("arms", "retrieval_configs", "bge_model_root", "bge_tokenizer_root",
            "qwen_tokenizer_root", "corpus_version")  # fmt: skip
    if not all(Path(real[k]).exists() for k in keep[2:5]):
        pytest.skip("pinned BGE and Qwen assets are not on this host")
    selection = synthetic_corpus(tmp_path, monkeypatch, {k: real[k] for k in keep})
    root = tmp_path / "published"
    items = synthetic_items()
    out = tmp_path / "contexts.json"
    build_published(selection, out, root, resume=True, final_test=("b" * 64, items))

    data = out.read_bytes()
    assert not os.access(out, os.W_OK)
    package = json.loads(data)
    records = package["records"]
    assert len(records) == 2745
    assert package["header"]["split"] == "final_test"
    assert package["header"]["benchmark_release_sha256"] == "b" * 64
    assert not {"evidence_sets", "accepted_answers", "gold"} & {k for r in records for k in r}
    for record in records:
        closed = record["matrix_arm"] == "closed_book"
        assert closed == (record["context"] is None) == ("context_record" not in record)

    args = Namespace(
        contexts=out,
        contexts_sha256=sha256_bytes(data),
        phase="final_test",
        published_context_root=root,
        prompt=Path("artifacts/phase8/colab_development_v5/inputs/prompt.json"),
        seeds=Path("artifacts/phase8/colab_development_v5/inputs/seeds.json"),
        configs=Path("configs/production"),
    )
    frozen = json.loads(Path("docs/reports/g03_production_freeze_2026_11_20.json").read_bytes())
    assert {r["matrix_arm"]: r["retrieval_config_sha256"] for r in records} == frozen[
        "retrieval_sha256"
    ]
    loaded, plan, _, _ = rg._load_inputs(args)  # includes check_published_contexts
    rg.validate_final_test_plan(loaded, plan)
    assert len(plan) == 4575
    arms = Counter(p.record["matrix_arm"] for p in plan)
    assert arms == {"confirmatory": 1830, "closed_book": 915, "dense_robustness": 610,
                    "chunk_robustness": 610, "scope_robustness": 610}  # fmt: skip
    shared = {}
    for p in plan:
        cell = (p.record["item_id"], p.record["matrix_arm"], p.record["condition"])
        shared.setdefault(cell, set()).add(p.record["context_sha256"])
    assert len(shared) == 2745 and all(len(v) == 1 for v in shared.values())

    # Synthetic questions are not the frozen roster: the authorized roster check refuses them.
    def tampered(change):
        copy = json.loads(data)
        change(copy)
        path = tmp_path / f"tampered_{len(list(tmp_path.glob('tampered_*')))}.json"
        path.write_bytes(json.dumps(copy).encode())
        return Namespace(**(vars(args) | {"contexts": path,
                                          "contexts_sha256": sha256_bytes(path.read_bytes())}))  # fmt: skip

    retrieval = next(r for r in records if r["matrix_arm"] == "dense_robustness")
    index = records.index(retrieval)
    attacks = {
        "missing": lambda c: c["records"].pop(index),
        "duplicate": lambda c: c["records"].append(c["records"][index]),
        "context_text": lambda c: c["records"][index].update(
            context=c["records"][index]["context"] + " ",
            context_sha256=sha256_bytes((c["records"][index]["context"] + " ").encode()),
        ),
        "context_record": lambda c: c["records"][index]["context_record"].update(sha256="0" * 64),
        "retrieval_config": lambda c: c["records"][index].update(retrieval_config_sha256="1" * 64),
        "closed_book_evidence": lambda c: next(
            r for r in c["records"] if r["matrix_arm"] == "closed_book"
        ).update(context_record=retrieval["context_record"]),
        "development_split": lambda c: c["records"][index].update(split="development"),
    }
    for name, attack in attacks.items():
        with pytest.raises(SystemExit, match="refused"):
            rg._load_inputs(tampered(attack))
        assert name
    with pytest.raises(SystemExit, match="refused"):
        rg._load_inputs(Namespace(**(vars(args) | {"published_context_root": tmp_path})))
    with pytest.raises(FileExistsError):
        build_published(selection, out, root, resume=True, final_test=("b" * 64, items))
