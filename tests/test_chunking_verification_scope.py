"""Nested scientific reads share work only inside one explicit verification."""

import pytest
from test_bm25_publication import inputs  # noqa: F401

from biorag_repro.retrieval import chunking


def test_nested_verification_does_not_repeat_or_outlive_source_checks(inputs, monkeypatch):  # noqa: F811
    sources, _ = inputs
    source = sources[0]
    original = chunking._read_chunking
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(chunking, "_read_chunking", counted)

    def read():
        return chunking.read_chunking(
            source.chunk_set,
            source.manifest,
            source.authority,
            source.configuration,
            source.configuration_root,
            source.chunk_root,
            fixture_counter=len,
        )

    with chunking.verified_once():
        first = read()
        with chunking.verified_once():
            assert read() == first
        assert len(calls) == 1
    path = source.chunk_root / first.chunk_set.chunks.logical_ref
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash|SHA|sha"):
        read()


def test_chunk_identity_reuse_preserves_content_checks_and_scope(inputs, monkeypatch):  # noqa: F811
    from biorag_repro.retrieval import storage

    source = inputs[0][0]
    chunk = chunking.read_chunking(
        source.chunk_set,
        source.manifest,
        source.authority,
        source.configuration,
        source.configuration_root,
        source.chunk_root,
        fixture_counter=len,
    ).chunks[0]
    expected = storage.record_ref(chunk)
    original = storage.serialize_project_json
    calls = []

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(storage, "serialize_project_json", counted)
    with chunking.verified_once():
        first = storage.record_ref(chunk)
        assert storage.record_ref(chunk) == expected
        assert len(calls) == 1
        # A caller cannot mutate the retained identity through its returned reference.
        first.sha256 = "f" * 64
        assert storage.record_ref(chunk) == expected
        chunk.rendered_text += " changed"
        changed = storage.record_ref(chunk)
        assert changed != expected and len(calls) == 2
    assert storage.record_ref(chunk) == changed
    assert len(calls) == 3


def test_prose_rendering_does_not_rescan_all_nodes_per_block():
    from test_chunking import build

    class CountedNodes(list):
        scans = 0

        def __iter__(self):
            self.scans += 1
            return super().__iter__()

    doc, _, _, _, result = build(
        "<article><body>" + "<p>distinct prose block</p>" * 40 + "</body></article>"
    )
    expected = chunking.render_units(doc, result.chunk_set.occurrences)
    nodes = CountedNodes(doc.nodes)
    doc.nodes = nodes
    assert chunking.render_units(doc, result.chunk_set.occurrences) == expected
    assert nodes.scans < 10


def test_caller_cannot_mutate_verified_chunk_read_cache(inputs):  # noqa: F811
    source = inputs[0][0]

    def read():
        return chunking.read_chunking(
            source.chunk_set,
            source.manifest,
            source.authority,
            source.configuration,
            source.configuration_root,
            source.chunk_root,
            fixture_counter=len,
        )

    with chunking.verified_once():
        first = read()
        text = first.chunks[0].rendered_text
        paper = first.chunk_set.input.paper_id
        first.chunks[0].rendered_text += " altered by caller"
        first.chunk_set.input.paper_id = "P050"
        again = read()
        assert again.chunks[0].rendered_text == text
        assert again.chunk_set.input.paper_id == paper
