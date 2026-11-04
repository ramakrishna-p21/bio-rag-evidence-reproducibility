"""Paired Phase 4 build and provenance checks on original synthetic material."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.pdf_ingest import publish_pdf_result
from biorag_repro.documents.phase4 import (
    Phase4GateIndex,
    Phase4Validation,
    build_arm,
    load_frozen_paper,
    publish_gate_index,
    publish_validation,
    validate_build,
    validate_jats_coverage,
    validate_pdf_coverage,
)


def _source(paper_id: str, kind: str, content: bytes, root: Path) -> ArtifactRef:
    digest = sha256_bytes(content)
    ref = ArtifactRef(
        artifact_id=raw_source_artifact_id(paper_id=paper_id, representation=kind, sha256=digest),
        logical_ref=f"{paper_id}/{kind}/source",
        sha256=digest,
    )
    path = root / "corpus/raw" / ref.logical_ref
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return ref


def _paper(root: Path):
    jats = _source(
        "P001", "JATS", b"<article><body><p>A <italic>B</italic> C</p></body></article>", root
    )
    pdf = _source("P001", "PDF", b"%PDF-1.4 synthetic", root)
    return SimpleNamespace(
        paper_id="P001",
        corpus_version="corpus_v1",
        jats=SimpleNamespace(**jats.model_dump(), media_type="application/xml"),
        pdf=SimpleNamespace(**pdf.model_dump(), media_type="application/pdf"),
        pair=SimpleNamespace(pair_status="verified_exact_pair"),
    )


def test_jats_build_rejects_wrong_paper_source_hash_and_publishes_complete_lineage(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    paper = _paper(corpus)
    with pytest.raises(ValueError, match="paper ID"):
        build_arm(paper, "P002", "JATS", corpus, output, "a" * 64)
    built = build_arm(paper, "P001", "JATS", corpus, output, "a" * 64)
    assert built.canonical_artifact is not None and built.build_manifest is not None
    checked = validate_build(built, output, corpus_root=corpus)
    assert checked.status == "pass"
    assert checked.source == built.source
    with pytest.raises(FileExistsError):
        build_arm(paper, "P001", "JATS", corpus, output, "a" * 64)
    (corpus / "corpus/raw" / paper.jats.logical_ref).write_bytes(b"<article><body/></article>")
    with pytest.raises(ValueError, match="SHA-256"):
        build_arm(paper, "P001", "JATS", corpus, output, "a" * 64)


def test_jats_coverage_rejects_a_missing_required_tail():
    from biorag_repro.documents.jats import ingest_jats

    xml = b"<article><body><custom>A<italic>B</italic> tail</custom></body></article>"
    digest = sha256_bytes(xml)
    source_id = raw_source_artifact_id(paper_id="P001", representation="JATS", sha256=digest)
    doc = ingest_jats(
        xml,
        paper_id="P001",
        source_artifact_id=source_id,
        source_sha256=digest,
        source_logical_ref="synthetic.xml",
        configuration_sha256="a" * 64,
    )
    validate_jats_coverage(doc, xml)
    nodes = [node.model_copy(deep=True) for node in doc.nodes]
    for node in nodes:
        node.segments = [
            part
            for part in node.segments
            if not (part.mapping.path or "").endswith("/italic[1]/tail()")
        ]
    with pytest.raises(ValueError, match="source slot"):
        validate_jats_coverage(doc.model_copy(update={"nodes": nodes}), xml)


def test_pdf_coverage_rejects_missing_native_object_and_orig():
    from types import SimpleNamespace as Item

    from biorag_repro.documents.pdf import canonicalize_native

    source = ArtifactRef(artifact_id="raw_pdf", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    body = Item(
        self_ref="#/body",
        parent=None,
        label="unspecified",
        content_layer="body",
        children=[Item(cref="#/texts/0")],
        prov=[],
        captions=[],
    )
    formula = Item(
        self_ref="#/texts/0",
        parent=Item(cref="#/body"),
        label="formula",
        content_layer="body",
        children=[],
        prov=[],
        captions=[],
        text="",
        orig="uncertain",
    )
    typed = Item(
        body=body,
        texts=[formula],
        groups=[],
        pictures=[],
        tables=[],
        form_items=[],
        key_value_items=[],
        iterate_items=lambda with_groups: iter([(body, 0), (formula, 1)]),
    )
    doc = canonicalize_native(typed, "P001", source, native, "c" * 64)
    data = {
        "body": {
            "self_ref": "#/body",
            "children": [{"$ref": "#/texts/0"}],
            "label": "unspecified",
            "content_layer": "body",
        },
        "texts": [
            {
                "self_ref": "#/texts/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "label": "formula",
                "content_layer": "body",
                "text": "",
                "orig": "uncertain",
                "prov": [],
            }
        ],
        "groups": [],
        "pictures": [],
        "tables": [],
        "form_items": [],
        "key_value_items": [],
    }
    validate_pdf_coverage(doc, data)
    with pytest.raises(ValueError, match="native object"):
        validate_pdf_coverage(doc.model_copy(update={"nodes": doc.nodes[:1]}), data)
    altered = [node.model_copy(deep=True) for node in doc.nodes]
    altered[1].native_orig = None
    with pytest.raises(ValueError, match="native orig"):
        validate_pdf_coverage(doc.model_copy(update={"nodes": altered}), data)


def test_pdf_boundary_refuses_partial_or_failed_complete_build(tmp_path, monkeypatch):
    from biorag_repro.documents import phase4

    paper = _paper(tmp_path / "corpus")
    incomplete = SimpleNamespace(
        status="partial_success",
        build_status="incomplete",
        canonical_artifact=None,
        build_manifest=None,
    )
    monkeypatch.setattr(phase4, "convert_pdf", lambda *args: incomplete)
    built = build_arm(
        paper,
        "P001",
        "PDF_DOCLING",
        tmp_path / "corpus",
        tmp_path / "output",
        "a" * 64,
        assets_root=tmp_path,
        lock_path=tmp_path / "lock",
    )
    assert built is incomplete
    with pytest.raises(ValueError, match="complete"):
        validate_build(built, tmp_path / "output")


def test_release_lookup_rejects_wrong_version_and_missing_pair(tmp_path, monkeypatch):
    from biorag_repro.documents import phase4

    paper = _paper(tmp_path / "corpus")
    release = SimpleNamespace(
        corpus_version="corpus_v1",
        csv_sha256="b" * 64,
        corpus_content_sha256="c" * 64,
        manifest_semantic_sha256="d" * 64,
        manifest_artifact=SimpleNamespace(logical_ref="manifest.csv"),
    )
    monkeypatch.setattr(phase4, "read_project_json", lambda *args: release)
    monkeypatch.setattr(phase4, "parse_csv", lambda *args: SimpleNamespace(papers=[paper]))
    monkeypatch.setattr(phase4, "sha256_bytes", lambda *args: "b" * 64)
    monkeypatch.setattr(phase4, "corpus_content_hash", lambda *args: "c" * 64)
    monkeypatch.setattr(phase4, "manifest_semantic_hash", lambda *args: "d" * 64)
    (tmp_path / "corpus" / "manifest.csv").write_bytes(b"synthetic")
    with pytest.raises(ValueError, match="corpus version"):
        load_frozen_paper(tmp_path / "corpus", tmp_path / "release.json", "a" * 64, "wrong", "P001")
    with pytest.raises(ValueError, match="paper ID"):
        load_frozen_paper(
            tmp_path / "corpus", tmp_path / "release.json", "a" * 64, "corpus_v1", "P002"
        )
    paper.pair.pair_status = "unverified"
    with pytest.raises(ValueError, match="pair"):
        load_frozen_paper(
            tmp_path / "corpus", tmp_path / "release.json", "a" * 64, "corpus_v1", "P001"
        )


def test_validation_and_gate_index_are_immutable_and_pair_bound(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    paper = _paper(corpus)
    built = build_arm(paper, "P001", "JATS", corpus, output, "a" * 64)
    report = validate_build(built, output, corpus_root=corpus)
    report_ref = publish_validation(report, output)
    assert (output / report_ref.logical_ref).is_file()
    with pytest.raises(FileExistsError):
        publish_validation(report, output)
    with pytest.raises(ValueError, match="requires complete native"):
        Phase4Validation.model_validate(
            report.model_copy(update={"representation": "PDF_DOCLING"}).model_dump()
        )
    index = Phase4GateIndex(
        corpus_version="corpus_v1",
        paper_id="P001",
        corpus_release_sha256="b" * 64,
        implementation_git_sha="a" * 40,
        jats_validation=report_ref,
        pdf_validation=report_ref,
    )
    with pytest.raises(ValueError, match="independent human authorization"):
        publish_gate_index(index, output, corpus)


def test_original_synthetic_pdf_pair_has_saved_native_lineage(tmp_path, monkeypatch):
    from biorag_repro.documents import phase4

    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    paper = _paper(corpus)

    class Native:
        def save_as_json(self, path, **kwargs):
            Path(path).write_text(
                json.dumps(
                    {
                        "schema_name": "DoclingDocument",
                        "version": "1.10.0",
                        "body": {
                            "self_ref": "#/body",
                            "label": "text",
                            "text": "A",
                            "prov": [
                                {
                                    "page_no": 1,
                                    "bbox": {
                                        "l": 1.25,
                                        "t": 3.5,
                                        "r": 2.75,
                                        "b": 1.5,
                                        "coord_origin": "BOTTOMLEFT",
                                    },
                                    "charspan": [0, 1],
                                }
                            ],
                        },
                    }
                )
            )

    def load_native(path):
        body = json.loads(Path(path).read_text())["body"]
        bbox = SimpleNamespace(**body["prov"][0]["bbox"])
        prov = SimpleNamespace(page_no=1, bbox=bbox, charspan=[0, 1])
        item = SimpleNamespace(
            self_ref="#/body",
            label="text",
            text="A",
            prov=[prov],
            parent=None,
            captions=[],
        )
        return SimpleNamespace(iterate_items=lambda with_groups: iter([(item, 0)]))

    def fake_convert(source_path, source, paper_id, config, assets, lock, data_root):
        result = SimpleNamespace(
            status=SimpleNamespace(value="success"), errors=[], pages=[], document=Native()
        )
        return publish_pdf_result(
            result,
            source_path,
            source,
            paper_id,
            config,
            "e" * 64,
            "f" * 64,
            data_root,
            load_native=load_native,
        )

    monkeypatch.setattr(phase4, "convert_pdf", fake_convert)
    jats = build_arm(paper, "P001", "JATS", corpus, output, "a" * 64)
    pdf = build_arm(
        paper,
        "P001",
        "PDF_DOCLING",
        corpus,
        output,
        "b" * 64,
        assets_root=tmp_path,
        lock_path=tmp_path / "lock",
    )
    jats_report = validate_build(jats, output, corpus_root=corpus)
    pdf_report = validate_build(pdf, output, corpus_root=corpus)
    assert pdf_report.native_artifact is not None
    assert pdf_report.attempt_artifact is not None
    assert pdf_report.warnings == []
    jats_ref = publish_validation(jats_report, output)
    pdf_ref = publish_validation(pdf_report, output)
    index = Phase4GateIndex(
        corpus_version="corpus_v1",
        paper_id="P001",
        corpus_release_sha256="a" * 64,
        implementation_git_sha="b" * 40,
        jats_validation=jats_ref,
        pdf_validation=pdf_ref,
    )
    with pytest.raises(ValueError, match="independent human authorization"):
        publish_gate_index(index, output, corpus)
