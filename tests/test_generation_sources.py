"""Serialized context-source specification: pinned, strict and free of fixture shortcuts.

The chain of Phase 5 artifacts is the one built for the roster tests. The spec only names
those artifacts; the roster still recomputes every context through the Phase 5 reader.
"""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from test_generation_roster import ITEMS
from test_generation_roster import shared as roster_world  # noqa: F401  (fixture)

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation.sources import (
    SourceError,
    load_context_sources,
    spec_from_sources,
)


@pytest.fixture
def world(roster_world):  # noqa: F811
    return roster_world


def describable(sources):
    """Fixture chunks count with a function; give them the tokenizer path a spec requires."""
    return [
        replace(
            s,
            selection=replace(
                s.selection,
                sources=[replace(c, tokenizer_root=Path("tok")) for c in s.selection.sources],
            ),
        )
        for s in sources
    ]


def write_spec(tmp_path, world, **changes):
    directory, record_sha, sources, _renderer, _ = world
    spec = spec_from_sources(describable(sources), retriever="bm25")
    data = {**json.loads(spec.model_dump_json()), **changes}
    path = tmp_path / "sources.json"
    raw = json.dumps(data, indent=1).encode()
    path.write_bytes(raw)
    return path, sha256_bytes(raw), directory, record_sha


def load(world, tmp_path, **changes):
    path, digest, directory, record_sha = write_spec(tmp_path, world, **changes)
    return load_context_sources(path, digest, directory, record_sha)


def test_round_trip_reproduces_the_selection_without_the_fixture_counter(world, tmp_path):
    loaded = load(world, tmp_path)
    original = {s.selection.query.query_id: s for s in world[2]}
    assert {s.selection.query.query_id for s in loaded} == {"q_a", "q_b"}
    for source in loaded:
        expected = original[source.selection.query.query_id]
        stripped = [
            replace(
                c, fixture_counter=None, tokenizer_root=source.selection.sources[0].tokenizer_root
            )
            for c in expected.selection.sources
        ]
        assert source.selection.sources == stripped
        assert source.selection.query == expected.selection.query
        assert source.context == expected.context and source.manifest == expected.manifest
        assert source.configuration == expected.configuration
        assert source.selection.method == "bm25"


def test_questions_come_from_the_development_view_not_the_spec(world, tmp_path):
    questions = {i.item_id: i.question for i in ITEMS}
    for source in load(world, tmp_path):
        query = source.selection.query
        assert query.question == questions[query.query_id]


def test_spec_hash_must_match_the_pinned_hash(world, tmp_path):
    path, _digest, directory, record_sha = write_spec(tmp_path, world)
    with pytest.raises(SourceError, match="hash"):
        load_context_sources(path, "0" * 64, directory, record_sha)


def test_unsupported_retriever_is_refused_not_approximated(world, tmp_path):
    with pytest.raises(SourceError, match="retriever"):
        load(world, tmp_path, retriever="bge")


def test_item_outside_the_development_view_is_refused(world, tmp_path):
    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    spec["entries"][0]["item_id"] = "q_z"  # protected final-test item
    path = tmp_path / "s.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    with pytest.raises(SourceError, match="development view"):
        load_context_sources(path, sha256_bytes(raw), world[0], world[1])


def test_unknown_and_missing_keys_are_refused(world, tmp_path):
    with pytest.raises(SourceError, match="not a valid"):
        load(world, tmp_path, unexpected=1)
    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    del spec["entries"]
    path = tmp_path / "s.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    with pytest.raises(SourceError, match="not a valid"):
        load_context_sources(path, sha256_bytes(raw), world[0], world[1])


def test_duplicate_entries_are_refused(world, tmp_path):
    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    spec["entries"].append(spec["entries"][0])
    path = tmp_path / "s.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    with pytest.raises(SourceError, match="duplicate"):
        load_context_sources(path, sha256_bytes(raw), world[0], world[1])


def test_unreadable_spec_is_refused(world, tmp_path):
    with pytest.raises(SourceError, match="unreadable"):
        load_context_sources(tmp_path / "missing.json", "0" * 64, world[0], world[1])


def with_fixture_counter(sources):
    """Fixture chunks have no tokenizer; restore the counter the spec cannot carry."""
    return [
        replace(
            s,
            selection=replace(
                s.selection, sources=[replace(c, fixture_counter=len) for c in s.selection.sources]
            ),
        )
        for s in sources
    ]


def test_spec_built_roster_equals_the_directly_built_roster(world, tmp_path):
    from test_generation_roster import build

    loaded = with_fixture_counter(load(world, tmp_path))
    assert build(world, sources=loaded).sha256 == build(world).sha256


def test_a_changed_artifact_hash_stops_the_roster(world, tmp_path):
    from test_generation_roster import build

    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    spec["entries"][0]["context"]["sha256"] = "0" * 64
    path = tmp_path / "s.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    loaded = with_fixture_counter(load_context_sources(path, sha256_bytes(raw), world[0], world[1]))
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        build(world, sources=loaded)


def test_entry_representation_must_match_its_chunk_set(world, tmp_path):
    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    spec["entries"][0]["representation"] = "PDF_DOCLING"
    path = tmp_path / "s.json"
    raw = json.dumps(spec).encode()
    path.write_bytes(raw)
    with pytest.raises(SourceError, match="representation"):
        load_context_sources(path, sha256_bytes(raw), world[0], world[1])


def test_one_item_may_have_a_source_per_representation(world, tmp_path):
    spec = json.loads(spec_from_sources(describable(world[2]), retriever="bm25").model_dump_json())
    assert {e["representation"] for e in spec["entries"]} == {"JATS"}
    assert len({(e["item_id"], e["representation"]) for e in spec["entries"]}) == 2
