"""One immutable verification may reuse derivations, never mutable submitted results."""

import pytest
from test_bm25_publication import inputs  # noqa: F401
from test_dense_publication import built  # noqa: F401

from biorag_repro.retrieval import bm25, dense
from biorag_repro.retrieval.chunking import verified_once


def test_bm25_scoped_reuse_does_not_trust_mutated_index(inputs, monkeypatch):  # noqa: F811
    sources, _ = inputs
    expected = bm25.build_index(sources)
    original = bm25.statistics
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(bm25, "statistics", counted)
    with verified_once():
        first = bm25.build_index(sources)
        assert bm25.build_index(list(reversed(sources))) == expected
        assert len(calls) == 1
        term = next(iter(first.statistics.idf))
        first.statistics.idf[term] += 1
        with pytest.raises(ValueError, match="BM25 index differs"):
            bm25.validate_index(first, sources)
    assert bm25.build_index(sources) == expected and len(calls) == 2


def test_dense_reuse_keeps_vector_identity_and_explicit_reproduction(built, monkeypatch):  # noqa: F811
    b = built
    original = dense._selected
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(dense, "_selected", counted)
    with verified_once():
        for _ in range(2):
            dense.validate_index(b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder)
        assert len(calls) == 1
        changed = b.vectors.copy()
        changed[:] = changed[:, ::-1]
        with pytest.raises(ValueError, match="vector bytes/manifest"):
            dense.validate_index(b.index, changed, b.sources, b.cfg, b.root, b.encoder)

        def wrong(texts, refs, role):
            return changed, []

        monkeypatch.setattr(b.encoder, "encode", wrong)
        with pytest.raises(ValueError, match="independently reproduced"):
            dense.validate_index(
                b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder, reproduce=True
            )


def test_bm25_read_and_scoring_reuse_keeps_query_results_and_fresh_scope(
    inputs,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from test_bm25_publication import query

    from biorag_repro.retrieval import storage

    sources, cfg = inputs
    index = bm25.build_index(sources)
    root = tmp_path / "published"
    refs = bm25.publish_index(index, sources, root)
    first_query = query()
    second_query = first_query.model_copy(
        update={"query_id": "q_second", "question": "another term"}
    )
    expected = [
        bm25.retrieve(index, sources, q, cfg, tmp_path) for q in (first_query, second_query)
    ]
    original_read, original_scorer = bm25.read_project_json, bm25.scorer
    reads, scorers = [], []

    def read(*args):
        if args[-1] is type(index):
            reads.append(1)
        return original_read(*args)

    def scorer(*args):
        scorers.append(1)
        return original_scorer(*args)

    monkeypatch.setattr(bm25, "read_project_json", read)
    monkeypatch.setattr(bm25, "scorer", scorer)
    with verified_once():
        loaded = bm25.read_index(*refs, sources, root)
        assert bm25.read_index(*refs, sources, root) == index
        assert len(reads) == 1
        loaded.statistics.idf[next(iter(loaded.statistics.idf))] += 1
        assert bm25.read_index(*refs, sources, root) == index
        for q, baseline in zip((first_query, second_query), expected, strict=True):
            assert bm25.retrieve(index, sources, q, cfg, tmp_path) == baseline
        # One independent selected-index derivation and one private query scorer.
        assert len(scorers) == 2
    path = root / storage.record_ref(index).logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with verified_once(), pytest.raises(ValueError):
        bm25.read_index(*refs, sources, root)


def test_selected_index_references_reused_without_trusting_mutated_results(
    inputs,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from test_bm25_publication import query

    sources, cfg = inputs
    index = bm25.build_index(sources)
    baseline = bm25.retrieve(index, sources, query(), cfg, tmp_path)
    original = bm25.record_ref
    calls = []

    def counted(value):
        if isinstance(value, type(index)):
            calls.append(1)
        return original(value)

    monkeypatch.setattr(bm25, "record_ref", counted)
    with verified_once():
        first = bm25.retrieve(index, sources, query(), cfg, tmp_path)
        assert first == baseline
        first.index.sha256 = "f" * 64
        first.index_build_manifest.sha256 = "e" * 64
        before = len(calls)
        assert bm25.retrieve(index, sources, query(), cfg, tmp_path) == baseline
        assert len(calls) == before
        changed = index.model_copy(deep=True)
        changed.statistics.idf[next(iter(changed.statistics.idf))] += 1
        with pytest.raises(ValueError, match="BM25 index differs"):
            bm25.retrieve(changed, sources, query(), cfg, tmp_path)
    bm25.retrieve(index, sources, query(), cfg, tmp_path)
    assert len(calls) > before


def test_private_retrieval_checks_do_not_copy_complete_verified_indexes(
    inputs,  # noqa: F811
    monkeypatch,
    tmp_path,
):
    from test_bm25_publication import query

    sources, cfg = inputs
    index = bm25.build_index(sources)
    refs = bm25.publish_index(index, sources, tmp_path)
    q = query()
    ranked = bm25.retrieve(index, sources, q, cfg, tmp_path)
    ranked_refs = bm25.publish_retrieval(ranked, *refs, sources, q, cfg, tmp_path, tmp_path)
    with verified_once():
        bm25.read_retrieval(*ranked_refs, *refs, sources, q, cfg, tmp_path, tmp_path)
        original = type(index).model_copy
        copies = []

        def counted(self, **kwargs):
            copies.append(1)
            return original(self, **kwargs)

        monkeypatch.setattr(type(index), "model_copy", counted)
        assert bm25.retrieve(index, sources, q, cfg, tmp_path) == ranked
        assert (
            bm25.read_retrieval(*ranked_refs, *refs, sources, q, cfg, tmp_path, tmp_path) == ranked
        )
        assert copies == []
        returned = bm25.read_index(*refs, sources, tmp_path)
        assert len(copies) == 1
        returned.statistics.idf[next(iter(returned.statistics.idf))] += 1
        assert bm25.read_index(*refs, sources, tmp_path) == index
