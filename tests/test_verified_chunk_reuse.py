"""One verified chunk-set read per verification, never shared between verifications."""

from dataclasses import replace

import pytest
from test_bm25_publication import inputs  # noqa: F401
from test_context_publication import prepared  # noqa: F401

from biorag_repro.core.hashing import HashMismatchError
from biorag_repro.documents import chunking
from biorag_repro.retrieval import bm25, context
from biorag_repro.retrieval.storage import record_ref


@pytest.fixture
def validations(monkeypatch):
    """Count full scientific re-validations of chunk sets."""
    calls = []
    real = chunking.validate_chunking

    def counted(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(chunking, "validate_chunking", counted)
    return calls


def chunk_args(source):
    return (
        source.chunk_set,
        source.manifest,
        source.authority,
        source.configuration,
        source.configuration_root,
        source.chunk_root,
    )


def test_assembly_validates_each_chunk_set_once(prepared, validations):  # noqa: F811
    selected, cfg, root = prepared
    context.assemble(selected, cfg, root, None, fixture_counter=len)
    assert len(validations) == len(selected.sources)


def test_reading_a_context_validates_each_chunk_set_once(prepared, validations):  # noqa: F811
    selected, cfg, root = prepared
    value = context.assemble(selected, cfg, root, None, fixture_counter=len)
    refs = context.publish(value, selected, cfg, root, None, root / "out", fixture_counter=len)
    validations.clear()
    assert (
        context.read(*refs, selected, cfg, root, None, root / "out", fixture_counter=len) == value
    )
    assert len(validations) == len(selected.sources)


def test_shared_read_gives_what_independent_reads_give(prepared):  # noqa: F811
    selected, cfg, root = prepared
    shared = context.assemble(selected, cfg, root, None, fixture_counter=len)
    ranked = bm25.read_retrieval(
        selected.retrieval,
        selected.retrieval_manifest,
        selected.index,
        selected.index_manifest,
        selected.sources,
        selected.query,
        selected.retrieval_configuration,
        selected.retrieval_configuration_root,
        selected.retrieval_root,
    )
    by_id = {}
    for source in selected.sources:
        result = chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
        by_id.update((record_ref(c).artifact_id, c) for c in result.chunks)
    decisions, text, tokens, stop = context.pack(
        [by_id[hit.chunk.artifact_id] for hit in ranked.hits], len
    )
    assert (
        shared.decisions,
        shared.exact_context,
        shared.generator_tokens,
        shared.stop_reason,
    ) == (
        decisions,
        text,
        tokens,
        stop,
    )


def test_reuse_does_not_outlive_one_assembly(prepared, validations):  # noqa: F811
    selected, cfg, root = prepared
    context.assemble(selected, cfg, root, None, fixture_counter=len)
    context.assemble(selected, cfg, root, None, fixture_counter=len)
    assert len(validations) == 2 * len(selected.sources)


def test_a_chunk_file_changed_between_assemblies_is_detected(prepared):  # noqa: F811
    selected, cfg, root = prepared
    context.assemble(selected, cfg, root, None, fixture_counter=len)
    source = selected.sources[0]
    path = source.chunk_root / source.chunk_set.logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(HashMismatchError):
        context.assemble(selected, cfg, root, None, fixture_counter=len)


def test_scope_shares_identical_reads_only(prepared, validations):  # noqa: F811
    selected, _cfg, _root = prepared
    source = selected.sources[0]
    other_counter = lambda text: len(text)
    chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
    chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
    assert len(validations) == 2  # no scope, nothing shared
    validations.clear()
    with chunking.verified_once():
        first = chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
        again = chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
        assert len(validations) == 1 and again == first
        chunking.read_chunking(*chunk_args(source), fixture_counter=other_counter)
        assert len(validations) == 2  # another counter is another read
    chunking.read_chunking(*chunk_args(source), fixture_counter=source.fixture_counter)
    assert len(validations) == 3  # the scope ended


def test_scope_never_serves_another_configuration_or_tokenizer(prepared, validations, tmp_path):  # noqa: F811
    selected, _cfg, _root = prepared
    source = selected.sources[0]
    args = chunk_args(source)
    with chunking.verified_once():
        chunking.read_chunking(*args, fixture_counter=source.fixture_counter)
        other = source.configuration.model_copy(update={"resolved_sha256": "0" * 64})
        with pytest.raises(Exception):  # noqa: B017  (any refusal; a cache hit would not raise)
            chunking.read_chunking(
                *args[:3], other, *args[4:], fixture_counter=source.fixture_counter
            )
        validations.clear()
        chunking.read_chunking(
            *args, tokenizer_root=tmp_path, fixture_counter=source.fixture_counter
        )
        assert len(validations) == 1  # another tokenizer root is another read


@pytest.mark.parametrize("position", [0, 1, 5])
def test_scope_never_serves_another_reference_or_root(prepared, position, tmp_path):  # noqa: F811
    selected, _cfg, _root = prepared
    source = selected.sources[0]
    args = list(chunk_args(source))
    with chunking.verified_once():
        chunking.read_chunking(*args, fixture_counter=source.fixture_counter)
        if position == 5:
            args[5] = tmp_path
        else:
            args[position] = args[position].model_copy(update={"sha256": "0" * 64})
        with pytest.raises(Exception):  # noqa: B017  (any refusal; a cache hit would not raise)
            chunking.read_chunking(*args, fixture_counter=source.fixture_counter)


def test_scope_does_not_trust_an_equal_looking_authority(prepared, validations):  # noqa: F811
    selected, _cfg, _root = prepared
    source = selected.sources[0]
    args = list(chunk_args(source))
    with chunking.verified_once():
        chunking.read_chunking(*args, fixture_counter=source.fixture_counter)
        args[2] = replace(args[2])
        validations.clear()
        chunking.read_chunking(*args, fixture_counter=source.fixture_counter)
        assert len(validations) == 1
