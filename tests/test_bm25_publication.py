"""Scientific read-back, multipaper scope and bounded renewed counterfeits."""

import json
import math
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_corpus import paper as paper_metadata
from test_documents_phase4 import _source
from test_documents_table_fidelity import _native_data, _typed_native
from test_retrieval_models import settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest, serialize_project_json
from biorag_repro.corpus.manifest import (
    CorpusManifest,
    PaperRecord,
    ReleaseRecord,
    corpus_content_hash,
    manifest_semantic_hash,
    serialize_csv,
)
from biorag_repro.documents.authorization import ScientificAuthority
from biorag_repro.documents.chunking import chunk_document, publish_chunking
from biorag_repro.documents.pdf_ingest import publish_pdf_result
from biorag_repro.documents.phase4 import (
    Phase4GateIndex,
    build_arm,
    publish_validation,
    validate_build,
)
from biorag_repro.retrieval import bm25
from biorag_repro.retrieval.models import CanonicalInput, ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import CanonicalAuthority, load_canonical_input


def config(root, scope="designated_paper"):
    value = settings().model_copy(update={"scope": scope})
    content = serialize_project_json(value, Phase5Configuration)
    digest = sha256_bytes(canonical_json_bytes(value.model_dump()))
    path = f"{scope}.json"
    (root / path).write_bytes(content)
    return ConfigurationRef(
        artifact=ArtifactRef(
            artifact_id="configuration_" + digest, logical_ref=path, sha256=sha256_bytes(content)
        ),
        resolved_sha256=digest,
    )


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    corpus, artifacts = tmp_path / "source", tmp_path / "native"
    artifacts.mkdir()
    pairs = []
    padding = " filler" * 17
    texts = [
        f"<p>alpha alpha common{padding}</p><p>beta common{padding}</p>",
        f"<p>gamma common{padding}</p><p>delta common{padding}</p>",
    ]
    for number, text in enumerate(texts, 1):
        pid = f"P{number:03d}"
        xml = f"<article><body>{text}</body></article>".encode()
        jr = _source(pid, "JATS", xml, corpus)
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

        native_result = SimpleNamespace(
            status=SimpleNamespace(value="success"), errors=[], pages=[], document=Native()
        )
        pdf = publish_pdf_result(
            native_result,
            corpus / "corpus/raw" / pr.logical_ref,
            pr,
            pid,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            artifacts,
            load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
        )
        pairs.append(
            (
                paper,
                validate_build(jats, artifacts, corpus_root=corpus),
                validate_build(pdf, artifacts, corpus_root=corpus),
            )
        )
    rows = []
    for number in range(1, 51):
        row = paper_metadata(number)
        row["corpus_version"] = "corpus_v1"
        if number <= 2:
            for kind in ("jats", "pdf"):
                row[kind].update(getattr(pairs[number - 1][0], kind).model_dump())
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
    digest = sha256_bytes(content)
    from biorag_repro.documents import authorization

    monkeypatch.setattr(authorization, "PHASE2H_SHA256", digest)
    cfg = config(tmp_path)
    sources = []
    for paper, jr, pr in pairs:
        evidence = Phase4GateIndex(
            corpus_version="corpus_v1",
            paper_id=paper.paper_id,
            corpus_release_sha256=digest,
            implementation_git_sha="f" * 40,
            jats_validation=publish_validation(jr, artifacts),
            pdf_validation=publish_validation(pr, artifacts),
        )
        authority = CanonicalAuthority(
            corpus,
            artifacts,
            ScientificAuthority(
                corpus / "release.json",
                digest,
                pr.attempt_artifact,
                "d" * 64,
                "a" * 64,
                "b" * 64,
                "c" * 64,
            ),
            evidence,
            "f" * 40,
        )
        selected = CanonicalInput(
            corpus_version="corpus_v1",
            paper_id=paper.paper_id,
            representation="JATS",
            corpus_release_sha256=digest,
            canonical=jr.canonical_artifact,
            validation=evidence.jats_validation,
            build_manifest=jr.build_manifest,
        )
        built = chunk_document(
            load_canonical_input(selected, authority), selected, cfg, settings(), len
        )
        refs = publish_chunking(
            built, authority, cfg, tmp_path, tmp_path / "chunks", fixture_counter=len
        )
        sources.append(
            bm25.ChunkSelection(
                *refs, authority, cfg, tmp_path, tmp_path / "chunks", fixture_counter=len
            )
        )
    return sources, cfg


def query(scope="designated_paper", text="alpha", paper="P001"):
    return bm25.Query(
        query_id="synthetic-q1",
        question=text,
        scope=scope,
        designated_paper_id=paper if scope == "designated_paper" else None,
    )


def test_scope_statistics_scores_and_stable_membership(inputs, tmp_path):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    assert index == bm25.build_index(list(reversed(sources)))
    assert index.paper_ids == ["P001", "P002"]
    assert index.ordered_chunk_ids == sorted(index.ordered_chunk_ids)
    assert index.statistics.corpus_size == len(index.ordered_chunk_ids)
    fullcfg = config(tmp_path, "full_corpus")
    full = bm25.retrieve(index, sources, query("full_corpus"), fullcfg, tmp_path)
    designated = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    assert [(h.chunk, h.score) for h in designated.hits] == [
        (h.chunk, h.score) for h in full.hits if h.paper_id == "P001"
    ]
    assert {h.paper_id for h in full.hits} == {"P001", "P002"}
    # Literal fixture: lengths 20,19,19,19; N=4, df(alpha)=1 and tf(alpha)=2.
    expected = math.log(7 / 3) * 5 / (2 + 1.5 * (0.25 + 0.75 * 20 / 19.25))
    assert index.statistics.document_lengths.count(20) == 1
    assert index.statistics.average_document_length == 19.25
    assert [h.score for h in full.hits] == pytest.approx([expected, 0, 0, 0])
    subset = [
        row
        for row, p in zip(index.tokenized_corpus, index.chunk_paper_ids, strict=True)
        if p == "P001"
    ]
    # For the designated two-document subset, df(alpha)=1 gives log(1)=0.
    assert bm25.scorer(subset).idf["alpha"] == 0
    ties = bm25.retrieve(index, sources, query(text="absent"), cfg, tmp_path)
    assert all(h.score == 0 for h in ties.hits)
    assert [h.chunk.artifact_id for h in ties.hits] == sorted(
        h.chunk.artifact_id for h in ties.hits
    )


def test_immutable_repeatable_readback(inputs, tmp_path):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    ranked = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    references = []
    for name in ("one", "two"):
        out = tmp_path / name
        ir = bm25.publish_index(index, sources, out)
        assert bm25.read_index(*ir, sources, out) == index
        rr = bm25.publish_retrieval(ranked, *ir, sources, query(), cfg, tmp_path, out)
        assert bm25.read_retrieval(*rr, *ir, sources, query(), cfg, tmp_path, out) == ranked
        assert read_manifest(out / ir[1].logical_ref, ir[1].sha256).status == "complete"
        references.append((ir, rr))
    assert references[0] == references[1]
    for ref in (*references[0][0], *references[0][1]):
        assert (tmp_path / "one" / ref.logical_ref).read_bytes() == (
            tmp_path / "two" / ref.logical_ref
        ).read_bytes()
    with pytest.raises(FileExistsError):
        bm25.publish_index(index, sources, tmp_path / "one")
    with pytest.raises(FileExistsError):
        bm25.publish_retrieval(
            ranked, *references[0][0], sources, query(), cfg, tmp_path, tmp_path / "one"
        )


@pytest.mark.parametrize("family", range(1, 11))
def test_ten_renewed_counterfeit_families(inputs, tmp_path, family):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    out = tmp_path / "valid"
    ir = bm25.publish_index(index, sources, out)
    ranked = bm25.retrieve(index, sources, query(text="absent"), cfg, tmp_path)
    rr = bm25.publish_retrieval(ranked, *ir, sources, query(text="absent"), cfg, tmp_path, out)
    assert (
        bm25.read_retrieval(*rr, *ir, sources, query(text="absent"), cfg, tmp_path, out) == ranked
    )
    forged = index.model_copy(deep=True)
    if family == 1:
        forged.tokenized_corpus[0] = ["rewritten"]
    elif family == 2:
        forged.ordered_chunk_ids.pop()
        forged.tokenized_corpus.pop()
        forged.chunk_paper_ids.pop()
    elif family == 3:
        forged.representation = "PDF_DOCLING"
    elif family == 4:
        forged.statistics.average_document_length += 1
    elif family == 5:
        # Substitute all statistics with a designated-paper-only calculation.
        rows = [
            r
            for r, p in zip(index.tokenized_corpus, index.chunk_paper_ids, strict=True)
            if p == "P001"
        ]
        forged.statistics = bm25.statistics(bm25.scorer(rows))
    else:
        forged = ranked.model_copy(deep=True)
        if family == 6:
            forged.hits[0].score += 1
        elif family == 7:
            forged.hits.reverse()
            for n, h in enumerate(forged.hits, 1):
                h.rank = n
        elif family == 8:
            forged.question = "substituted question"
            forged.query_id = "substituted-id"
        elif family == 9:
            fullcfg = config(tmp_path, "full_corpus")
            full = bm25.retrieve(
                index, sources, query("full_corpus", text="absent"), fullcfg, tmp_path
            )
            forged.hits = [
                next(h.model_copy(update={"rank": 1}) for h in full.hits if h.paper_id == "P002")
            ]
        else:
            manifest = read_manifest(out / ir[1].logical_ref, ir[1].sha256)
            manifest.status = "failed"
            manifest.failure_reason = "interrupted"
            content = serialize_project_json(manifest, type(manifest))
            (out / "incomplete.json").write_bytes(content)
            bad = ArtifactRef(
                artifact_id="build_" + sha256_bytes(content),
                logical_ref="incomplete.json",
                sha256=sha256_bytes(content),
            )
            with pytest.raises(ValueError, match="manifest"):
                bm25.read_index(ir[0], bad, sources, out)
            return
    # Renew record identity, exact bytes, and complete manifest wherever structurally valid.
    # Shape-invalid attacks are rejected by the same serialization boundary.
    with pytest.raises(ValueError):
        ref = record_ref(forged)
        content = serialize_project_json(forged, type(forged))
        target = out / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if family <= 5:
            mr = bm25.manifest_ref(forged)
            (out / mr.logical_ref).write_bytes(
                serialize_project_json(
                    bm25.build_manifest(forged), type(bm25.build_manifest(forged))
                )
            )
            bm25.read_index(ref, mr, sources, out)
        else:
            mr = bm25.manifest_ref(forged, ir[1])
            (out / mr.logical_ref).write_bytes(
                serialize_project_json(
                    bm25.build_manifest(forged, ir[1]), type(bm25.build_manifest(forged, ir[1]))
                )
            )
            bm25.read_retrieval(ref, mr, *ir, sources, query(text="absent"), cfg, tmp_path, out)


def test_interrupted_publication_no_complete_marker(inputs, tmp_path, monkeypatch):
    from biorag_repro.core import storage

    sources, _ = inputs
    index = bm25.build_index(sources)
    original = storage.os.link

    def interrupt(source, target):
        if str(target).endswith(".bm25.build.json"):
            raise OSError("interrupted")
        return original(source, target)

    monkeypatch.setattr(storage.os, "link", interrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(OSError, match="interrupted"):
        bm25.publish_index(index, sources, out)
    assert not list(out.rglob("*.bm25.build.json"))
    with pytest.raises(FileNotFoundError):
        bm25.read_index(record_ref(index), bm25.manifest_ref(index), sources, out)
    assert not list(out.rglob("*.tmp"))


@pytest.mark.parametrize("text", ["", "  \n"])
def test_empty_query_rejected(text):
    with pytest.raises(ValueError):
        query(text=text)


def test_invalid_inputs_and_query_scope(inputs, tmp_path):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    with pytest.raises(ValueError):
        bm25.build_index([])
    with pytest.raises(ValueError):
        bm25.build_index([sources[0], sources[0]])
    with pytest.raises(ValueError):
        bm25.retrieve(index, sources, query(paper="P050"), cfg, tmp_path)
    with pytest.raises(ValueError):
        bm25.retrieve(index, sources, query("full_corpus"), cfg, tmp_path)
    with pytest.raises(ValueError):
        bm25.validate_index(index, sources[:1])
    with pytest.raises(ValueError):
        bm25.Query(query_id=" ", question="x", scope="designated_paper", designated_paper_id="P001")
    bad = replace(
        sources[0], chunk_set=sources[0].chunk_set.model_copy(update={"sha256": "0" * 64})
    )
    with pytest.raises(ValueError):
        bm25.build_index([bad])


@pytest.mark.parametrize(
    "field,value",
    [("paper_ids", ["P001"]), ("chunk_paper_ids", ["P002"] * 4), ("ordered_chunk_ids", None)],
)
def test_additional_membership_counterfeits(inputs, field, value):
    sources, _ = inputs
    index = bm25.build_index(sources)
    bad = index.model_copy(deep=True)
    setattr(
        bad,
        field,
        value if value is not None else [bad.ordered_chunk_ids[0]] * len(bad.ordered_chunk_ids),
    )
    with pytest.raises(ValueError):
        bm25.validate_index(bad, sources)


def test_query_identity_and_text_checked_separately(inputs, tmp_path):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    ranked = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    for changed in [query().model_copy(update={"query_id": "other"}), query(text="different")]:
        with pytest.raises(ValueError, match="query"):
            bm25.validate_retrieval(ranked, index, sources, changed, cfg, tmp_path)


def test_nonfinite_scores_rejected_at_write_boundary(inputs, tmp_path):
    sources, cfg = inputs
    index = bm25.build_index(sources)
    ranked = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    for score in [float("nan"), float("inf"), float("-inf")]:
        bad = ranked.model_copy(deep=True)
        bad.hits[0].score = score
        with pytest.raises(ValueError):
            bm25.validate_retrieval(bad, index, sources, query(), cfg, tmp_path)


def test_interrupted_retrieval_requires_its_manifest(inputs, tmp_path, monkeypatch):
    from biorag_repro.core import storage

    sources, cfg = inputs
    index = bm25.build_index(sources)
    ir = bm25.publish_index(index, sources, tmp_path / "out")
    ranked = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    original = storage.os.link

    def interrupt(source, target):
        if str(target).endswith(".bm25.build.json"):
            raise OSError("interrupted retrieval")
        return original(source, target)

    monkeypatch.setattr(storage.os, "link", interrupt)
    with pytest.raises(OSError, match="interrupted retrieval"):
        bm25.publish_retrieval(ranked, *ir, sources, query(), cfg, tmp_path, tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        bm25.read_retrieval(
            record_ref(ranked),
            bm25.manifest_ref(ranked, ir[1]),
            *ir,
            sources,
            query(),
            cfg,
            tmp_path,
            tmp_path / "out",
        )


def test_existing_index_links_accept_sorted_verified_membership(inputs):
    from biorag_repro.documents.chunking import read_chunking
    from biorag_repro.retrieval.validation import validate_index_links

    sources, _ = inputs
    index = bm25.build_index(sources)
    sets = [
        read_chunking(
            s.chunk_set,
            s.manifest,
            s.authority,
            s.configuration,
            s.configuration_root,
            s.chunk_root,
            fixture_counter=len,
        ).chunk_set
        for s in reversed(sources)
    ]
    validate_index_links(index, sets)


def test_complete_index_marker_cannot_hide_missing_upstream_publication(inputs, tmp_path):
    sources, _ = inputs
    index = bm25.build_index(sources)
    refs = bm25.publish_index(index, sources, tmp_path / "out")
    assert bm25.read_index(*refs, sources, tmp_path / "out") == index
    # A valid complete downstream manifest cannot certify a missing input manifest.
    (sources[0].chunk_root / sources[0].manifest.logical_ref).unlink()
    assert (
        read_manifest(tmp_path / "out" / refs[1].logical_ref, refs[1].sha256).status == "complete"
    )
    with pytest.raises(FileNotFoundError):
        bm25.read_index(*refs, sources, tmp_path / "out")
