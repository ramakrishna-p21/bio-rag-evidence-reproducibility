"""Bounded selection cannot confer publication approval or substitute paper identity."""

import pytest
from test_bm25_publication import inputs  # noqa: F401

from biorag_repro.retrieval.source_selection import select_pair


def selection(source, **changes):
    authority = source.authority
    args = {
        "corpus_root": authority.corpus_root,
        "artifact_root": authority.artifact_root,
        "release_path": authority.scientific.release_path,
        "release_sha256": authority.scientific.release_sha256,
        "paper_id": authority.evidence.paper_id,
        "jats_validation": authority.evidence.jats_validation,
        "pdf_validation": authority.evidence.pdf_validation,
        "verification_revision": authority.reproduction_revision,
    }
    return select_pair(**(args | changes))


def test_selection_revalidates_without_publishing_a_gate(inputs):  # noqa: F811
    sources, _ = inputs
    root = sources[0].authority.artifact_root
    before = set(root.rglob("*"))
    assert selection(sources[0]) == sources[0].authority
    assert set(root.rglob("*")) == before


def test_selection_refuses_other_paper(inputs):  # noqa: F811
    sources, _ = inputs
    with pytest.raises(ValueError, match="mismatch|differs"):
        selection(sources[0], paper_id="P002")
