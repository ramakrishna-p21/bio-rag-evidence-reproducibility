"""Dense scientific read-back and the ten bounded counterfeit families."""

from types import SimpleNamespace

import numpy as np
import pytest
from test_bm25_publication import config, inputs, query  # noqa: F401

from biorag_repro.core.storage import serialize_project_json
from biorag_repro.retrieval import dense
from biorag_repro.retrieval.models import Snapshot


class Encoder:
    """Synthetic numerical oracle, used only with independently validated fixtures."""

    def __init__(self):
        self.tokenizer = dense_test_tokenizer()
        self.execution = dense.Execution(
            model=Snapshot(
                repository="BAAI/bge-base-en-v1.5",
                state="pinned",
                revision=dense.REVISION,
                file_hashes={"fixture": "a" * 64},
            ),
            tokenizer=Snapshot(
                repository="BAAI/bge-base-en-v1.5",
                state="pinned",
                revision=dense.REVISION,
                file_hashes={"fixture": "b" * 64},
            ),
            versions={"python": "synthetic"},
            batch_size=2,
            threads=1,
            interop_threads=1,
        )

    def encode(self, texts, references, role):
        rows = dense.prepare_inputs(self.tokenizer, texts, references, role, 512)
        # Independently define a unit vector from the literal text bytes; distinct rows
        # permit association swaps to be detected through independent reproduction.
        result = np.zeros((len(texts), 768), dtype="float32")
        for n, text in enumerate(texts):
            result[n, sum(text.encode()) % 768] = 1.0
        evidence = dense.input_evidence(rows, references)
        return result, evidence


def dense_test_tokenizer():
    from test_dense import Tokenizer

    return Tokenizer()


@pytest.fixture
def built(inputs, tmp_path):  # noqa: F811
    sources, cfg = inputs
    encoder = Encoder()
    index, vectors = dense.build_index(sources, cfg, tmp_path, encoder)
    return SimpleNamespace(
        sources=sources, cfg=cfg, encoder=encoder, index=index, vectors=vectors, root=tmp_path
    )


def test_publication_scope_and_reproduction(built, tmp_path):
    b = built
    ir = dense.publish_index(
        b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder, tmp_path / "out"
    )
    index, stored = dense.read_index(
        *ir, b.sources, b.cfg, b.root, b.encoder, tmp_path / "out", reproduce=True
    )
    assert index == b.index
    np.testing.assert_array_equal(stored, b.vectors)
    assert not stored.flags.writeable
    assert index.paper_ids == ["P001", "P002"]
    assert index.ordered_chunk_ids == sorted(index.ordered_chunk_ids)
    other, again = dense.build_index(list(reversed(b.sources)), b.cfg, b.root, b.encoder)
    assert other == index
    assert dense.vector_bytes(again) == dense.vector_bytes(stored)
    fullcfg = config(tmp_path, "full_corpus")
    ranked = dense.retrieve(
        index, stored, b.sources, query("full_corpus"), fullcfg, b.root, b.encoder
    )
    designated = dense.retrieve(index, stored, b.sources, query(), b.cfg, b.root, b.encoder)
    assert {h.paper_id for h in ranked.hits} == {"P001", "P002"}
    assert designated.hits == [
        h.model_copy(update={"rank": n})
        for n, h in enumerate([h for h in ranked.hits if h.paper_id == "P001"], 1)
    ]
    rr = dense.publish_retrieval(
        designated, *ir, b.sources, query(), b.cfg, b.root, b.encoder, tmp_path / "out"
    )
    assert (
        dense.read_retrieval(
            *rr, *ir, b.sources, query(), b.cfg, b.root, b.encoder, tmp_path / "out"
        )
        == designated
    )
    with pytest.raises(FileExistsError):
        dense.publish_index(index, stored, b.sources, b.cfg, b.root, b.encoder, tmp_path / "out")


@pytest.mark.parametrize("attack", range(1, 11))
def test_ten_renewed_counterfeit_families(built, tmp_path, monkeypatch, attack):
    b = built
    # The valid control crosses exactly the same boundary as renewed counterfeits.
    dense.validate_index(b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder, reproduce=True)
    bad = b.index.model_copy(deep=True)
    vectors = b.vectors.copy()
    if attack == 1:
        bad.input_token_counts[0] -= 1
        bad.input_id_hashes[0] = "d" * 64
    elif attack == 2:
        bad.execution.query_instruction = ""
    elif attack == 3:
        vectors = np.roll(vectors, 1, axis=1)  # plausible normalized alternate pooling
    elif attack == 4:
        vectors[0] *= 2
    elif attack == 5:
        vectors[[0, 1]] = vectors[[1, 0]]
        assert not np.array_equal(vectors, b.vectors)
    elif attack == 6:
        bad.execution.model.file_hashes["fixture"] = "c" * 64
    elif attack == 7:
        bad.representation = "PDF_DOCLING"
        bad.chunk_paper_ids[0] = "P003"
        bad.paper_ids = sorted(set(bad.chunk_paper_ids))
    elif attack in (8, 9):
        value = dense.retrieve(b.index, b.vectors, b.sources, query(), b.cfg, b.root, b.encoder)
        dense.validate_retrieval(
            value, b.index, b.vectors, b.sources, query(), b.cfg, b.root, b.encoder
        )
        if attack == 8:
            value.hits[0].score += 0.125
        else:
            value.hits[0].paper_id = "P002"
        # Renew identity/bytes, rather than relying on an obsolete checksum.
        serialize_project_json(value, type(value)) if attack == 8 else None
        with pytest.raises(ValueError):
            dense.validate_retrieval(
                value, b.index, b.vectors, b.sources, query(), b.cfg, b.root, b.encoder
            )
        return
    else:

        def interrupted(*args, **kwargs):
            raise OSError("interrupted")

        monkeypatch.setattr(dense, "publish_manifest", interrupted)
        with pytest.raises(OSError, match="interrupted"):
            dense.publish_index(
                b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder, tmp_path / "partial"
            )
        assert not list((tmp_path / "partial").rglob("*.dense.build.json"))
        with pytest.raises(FileNotFoundError):
            dense.read_index(
                dense.record_ref(b.index),
                dense.manifest_ref(b.index),
                b.sources,
                b.cfg,
                b.root,
                b.encoder,
                tmp_path / "partial",
            )
        return
    bad.vectors = dense.vector_ref(vectors, bad)
    # The rehashed index and vector artifact are internally well identified.
    bad.vectors_build_manifest = dense._vector_manifest_ref(bad)
    if attack != 2:
        dense.record_ref(bad)
    with pytest.raises(ValueError):
        dense.validate_index(bad, vectors, b.sources, b.cfg, b.root, b.encoder, reproduce=True)


def test_stored_consistency_does_not_authenticate_model_execution(built):
    b = built
    vectors = np.roll(b.vectors, 1, axis=1)
    bad = b.index.model_copy(deep=True)
    bad.vectors = dense.vector_ref(vectors, bad)
    bad.vectors_build_manifest = dense._vector_manifest_ref(bad)
    dense.validate_index(bad, vectors, b.sources, b.cfg, b.root, b.encoder)
    with pytest.raises(ValueError, match="reproduced"):
        dense.validate_index(bad, vectors, b.sources, b.cfg, b.root, b.encoder, reproduce=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("dimension", 767),
        ("input_token_counts", [1, 2, 3, 4]),
        ("chunk_sets", []),
        ("ordered_chunk_ids", ["substituted"] * 4),
    ],
)
def test_false_scientific_metadata_rejected(built, field, value):
    b = built
    bad = b.index.model_copy(update={field: value})
    with pytest.raises(ValueError):
        dense.validate_index(bad, b.vectors, b.sources, b.cfg, b.root, b.encoder)


def test_execution_identity_and_missing_lineage(built):
    b = built
    changed = b.index.model_copy(deep=True)
    changed.execution.tokenizer.file_hashes["fixture"] = "e" * 64
    assert dense.vector_ref(b.vectors, changed) != b.index.vectors
    with pytest.raises(ValueError, match="execution"):
        dense.validate_index(changed, b.vectors, b.sources, b.cfg, b.root, b.encoder)
    from dataclasses import replace

    missing = [
        replace(
            b.sources[0],
            manifest=b.sources[0].manifest.model_copy(update={"logical_ref": "missing.json"}),
        )
    ] + b.sources[1:]
    with pytest.raises(FileNotFoundError):
        dense.validate_index(b.index, b.vectors, missing, b.cfg, b.root, b.encoder)


def test_interrupted_final_index_manifest(built, tmp_path, monkeypatch):
    b = built
    original = dense.publish_manifest

    def interrupted(manifest, path, *args, **kwargs):
        if str(path).endswith(".dense.build.json"):
            raise OSError("final manifest interrupted")
        return original(manifest, path, *args, **kwargs)

    monkeypatch.setattr(dense, "publish_manifest", interrupted)
    root = tmp_path / "partial-final"
    with pytest.raises(OSError):
        dense.publish_index(b.index, b.vectors, b.sources, b.cfg, b.root, b.encoder, root)
    assert list(root.rglob("*.npy"))
    assert list(root.rglob("*.npy.build.json"))
    assert not list(root.rglob("*.dense.build.json"))
    with pytest.raises(FileNotFoundError):
        dense.read_index(
            dense.record_ref(b.index),
            dense.manifest_ref(b.index),
            b.sources,
            b.cfg,
            b.root,
            b.encoder,
            root,
        )


def test_existing_index_link_boundary_accepts_dense_sorted_membership(built):
    from biorag_repro.documents.chunking import read_chunking
    from biorag_repro.retrieval.validation import validate_index_links

    b = built
    sets = [
        read_chunking(
            s.chunk_set,
            s.manifest,
            s.authority,
            s.configuration,
            s.configuration_root,
            s.chunk_root,
            fixture_counter=s.fixture_counter,
        ).chunk_set
        for s in b.sources
    ]
    validate_index_links(b.index, sets)


def test_bge_result_requires_complete_query_evidence(built):
    b = built
    value = dense.retrieve(b.index, b.vectors, b.sources, query(), b.cfg, b.root, b.encoder)
    assert value.dense_query is not None
    with pytest.raises(ValueError):
        type(value).model_validate(value.model_copy(update={"dense_query": None}).model_dump())
