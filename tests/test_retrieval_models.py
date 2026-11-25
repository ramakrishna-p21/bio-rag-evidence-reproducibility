"""Phase 5 contracts use synthetic values, never qualified model identities."""

import json
from hashlib import sha256

import pytest
from pydantic import ValidationError

from biorag_repro.core.config import resolve_config
from biorag_repro.core.hashing import canonical_json_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.retrieval.models import (
    Bm25Index,
    CanonicalInput,
    Chunk,
    ChunkSet,
    ConfigurationRef,
    ContextRecord,
    DenseExecution,
    DenseIndex,
    Phase5Configuration,
    RankedRetrieval,
    SourceOccurrence,
)


def ref(name="canonical"):
    return ArtifactRef(artifact_id=name, logical_ref=f"synthetic/{name}.json", sha256="a" * 64)


def settings():
    return Phase5Configuration.model_validate(
        {
            "schema_version": "1",
            "purpose": "synthetic_fixture",
            "chunk_tokens": 192,
            "construction_tokenizer": {
                "repository": "BAAI/bge-base-en-v1.5",
                "state": "unresolved",
                "revision": None,
                "file_hashes": {},
            },
            "packing_tokenizer": {
                "repository": "Qwen/Qwen3-8B",
                "state": "unresolved",
                "revision": None,
                "file_hashes": {},
            },
            "bge_model": {
                "repository": "BAAI/bge-base-en-v1.5",
                "state": "unresolved",
                "revision": None,
                "file_hashes": {},
            },
            "lexical_tokenizer": "unicode_word_punctuation_v1",
            "scope": "designated_paper",
        }
    )


def configuration():
    return ConfigurationRef(artifact=ref("configuration"), resolved_sha256="b" * 64)


def selection():
    return CanonicalInput(
        corpus_version="corpus_v1",
        corpus_release_sha256="c" * 64,
        paper_id="P001",
        representation="JATS",
        canonical=ref(),
        validation=ref("validation"),
        build_manifest=ref("build"),
    )


def occurrence(path="/article/body[1]/p[1]/text()", node="p1"):
    from biorag_repro.retrieval.validation import occurrence_id

    data = {
        "canonical": ref(),
        "owner_node_id": node,
        "field": "segment",
        "segment_index": 0,
        "mapping": {
            "kind": "jats",
            "state": "exact_native",
            "source_artifact_id": "raw",
            "path": path,
            "segment_ordinal": 0,
        },
        "text": "−1.2e−3",
        "content_state": "present",
        "interpretation_state": None,
    }
    return SourceOccurrence(occurrence_id=occurrence_id(ref(), data["mapping"], "segment"), **data)


def chunk():
    return Chunk(
        input=selection(),
        configuration=configuration(),
        contributors=[{"occurrence_id": "occ1", "node_id": "p1", "usage": "content"}],
        rendered_text="−1.2e−3",
        construction_tokens=8,
    )


def test_required_configuration_and_frozen_policies():
    value = settings()
    assert (value.chunk_tokens, value.context_candidates, value.evidence_tokens) == (192, 20, 2048)
    assert (value.k1, value.b, value.epsilon) == (1.5, 0.75, 0.25)
    assert value.unit_policy == "complete_units"
    assert value.overlap_tokens == 0
    assert value.oversized_policy == "diagnose_without_truncation"
    assert value.statistics == "representation_wide"
    assert value.stop_policy == "first_nonfitting"
    assert value.duplicate_policy == "text_and_scientific_provenance"
    with pytest.raises(ValueError, match="unresolved"):
        value.require_pinned_assets()


@pytest.mark.parametrize(
    "field",
    [
        "chunk_tokens",
        "construction_tokenizer",
        "packing_tokenizer",
        "bge_model",
        "lexical_tokenizer",
        "scope",
        "purpose",
    ],
)
def test_missing_scientific_settings_rejected(field):
    data = settings().model_dump()
    del data[field]
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "2"),
        ("unknown", True),
        ("chunk_tokens", 193),
        ("chunk_tokens", True),
        ("context_candidates", 21),
        ("evidence_tokens", 2049),
        ("overlap_tokens", 1),
        ("k1", 1.2),
        ("b", 0.5),
        ("epsilon", 0.0),
        ("statistics", "per_paper"),
        ("stop_policy", "skip_ahead"),
        ("representation", "PDF_DOCLING"),
        ("purpose", "qualified_production"),
    ],
)
def test_invalid_or_unapproved_configuration_rejected(field, value):
    data = settings().model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)


@pytest.mark.parametrize("revision", ["main", "v1", "0" * 40, "a" * 39])
def test_snapshot_cannot_claim_floating_or_missing_identity(revision):
    data = settings().model_dump()
    data["construction_tokenizer"].update(
        state="pinned", revision=revision, file_hashes={"tokenizer.json": "a" * 64}
    )
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)


def test_unresolved_snapshot_cannot_hide_revision_or_wrong_repository():
    data = settings().model_dump()
    data["construction_tokenizer"]["revision"] = "a" * 40
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)
    data = settings().model_dump()
    data["packing_tokenizer"]["repository"] = "BAAI/bge-base-en-v1.5"
    with pytest.raises(ValidationError):
        Phase5Configuration.model_validate(data)


def test_configuration_identity_is_semantic_and_policy_sensitive():
    raw = b"""purpose="synthetic_fixture"
chunk_tokens=192
lexical_tokenizer="unicode_word_punctuation_v1"
scope="designated_paper"
[construction_tokenizer]
repository="BAAI/bge-base-en-v1.5"
state="unresolved"
[packing_tokenizer]
repository="Qwen/Qwen3-8B"
state="unresolved"
[bge_model]
repository="BAAI/bge-base-en-v1.5"
state="unresolved"
"""
    first = resolve_config(raw, Phase5Configuration)
    second = resolve_config(b"# formatting only\n" + raw, Phase5Configuration)
    assert first.configuration_sha256 == second.configuration_sha256
    assert (
        first.configuration_sha256
        == sha256(canonical_json_bytes(first.settings.model_dump())).hexdigest()
    )
    changed = raw.replace(b"unicode_word_punctuation_v1", b"synthetic_tokenizer_v2")
    assert (
        resolve_config(changed, Phase5Configuration).configuration_sha256
        != first.configuration_sha256
    )
    assert (
        resolve_config(raw.replace(b"192", b"384"), Phase5Configuration).configuration_sha256
        != first.configuration_sha256
    )


def test_occurrences_are_location_scoped_not_text_or_presentation_scoped():
    from biorag_repro.retrieval.validation import occurrence_id

    first, repeated = occurrence(), occurrence("/article/body[1]/p[2]/text()", "p2")
    assert first.text == repeated.text
    assert first.occurrence_id != repeated.occurrence_id
    containing_mapping = first.mapping.model_copy(update={"segment_ordinal": 7})
    assert occurrence_id(first.canonical, containing_mapping, "segment") == first.occurrence_id
    changed = first.mapping.model_copy(update={"state": "approximate"})
    assert occurrence_id(first.canonical, changed, "segment") == first.occurrence_id


@pytest.mark.parametrize("state", ["ambiguous", "approximate", "unavailable"])
def test_occurrence_uncertainty_is_preserved(state):
    data = occurrence().model_dump()
    data["mapping"]["state"] = state
    value = SourceOccurrence.model_validate(data)
    assert value.mapping.state == state
    assert SourceOccurrence.model_validate_json(value.model_dump_json()) == value


def test_marker_only_and_nontextual_occurrences_remain_explicit():
    data = occurrence().model_dump()
    data.update(field="nontextual", segment_index=None, text=None, content_state="marker_only")
    value = SourceOccurrence.model_validate(data)
    assert value.text is None and value.content_state == "marker_only"
    data["text"] = "invented"
    with pytest.raises(ValidationError):
        SourceOccurrence.model_validate(data)


def test_ordered_contributors_distinguish_content_and_repeated_context():
    value = chunk()
    data = value.model_dump()
    data["contributors"] += [
        {"occurrence_id": "header", "node_id": "h1", "usage": "repeated_context"},
        {"occurrence_id": "occ1", "node_id": "parent", "usage": "containing_reference"},
    ]
    changed = Chunk.model_validate(data)
    assert [c.usage for c in changed.contributors] == [
        "content",
        "repeated_context",
        "containing_reference",
    ]
    encoded = serialize_project_json(changed, Chunk)
    assert Chunk.model_validate_json(encoded) == changed
    assert encoded == serialize_project_json(changed, Chunk)
    assert json.loads(encoded)["contributors"][0]["node_id"] == "p1"


def metadata():
    return ChunkSet(
        input=selection(),
        configuration=configuration(),
        chunks=ref("chunks"),
        chunk_ids=["chunk1"],
        occurrences=[occurrence()],
        accounting=[
            {
                "occurrence_id": occurrence().occurrence_id,
                "status": "included",
                "chunk_ids": ["chunk1"],
                "reason": None,
            }
        ],
    )


@pytest.mark.parametrize(
    "mutation", ["missing_account", "unknown_chunk", "duplicate_occurrence", "unexplained_oversize"]
)
def test_occurrence_accounting_must_be_complete_and_consistent(mutation):
    data = metadata().model_dump()
    if mutation == "missing_account":
        data["accounting"] = []
    elif mutation == "unknown_chunk":
        data["accounting"][0]["chunk_ids"] = ["other"]
    elif mutation == "duplicate_occurrence":
        data["occurrences"] *= 2
    else:
        data["accounting"][0].update(status="oversized", chunk_ids=[], reason=None)
    with pytest.raises(ValidationError):
        ChunkSet.model_validate(data)


def test_explicit_exclusion_does_not_claim_included_evidence():
    data = metadata().model_dump()
    data["chunk_ids"] = []
    data["accounting"][0].update(
        status="oversized", chunk_ids=[], reason="Complete row exceeds 192 tokens"
    )
    assert ChunkSet.model_validate(data).accounting[0].status == "oversized"


def records():
    cfg = configuration()
    return [
        selection(),
        chunk(),
        metadata(),
        Bm25Index(
            representation="JATS",
            chunk_tokens=192,
            paper_ids=["P001"],
            chunk_paper_ids=["P001"],
            chunk_set_manifests=[ref("set_manifest")],
            statistics={
                "corpus_size": 1,
                "document_lengths": [2],
                "term_frequencies": [{"−": 1, "1.2": 1}],
                "document_frequencies": {"−": 1, "1.2": 1},
                "average_document_length": 2.0,
                "average_idf": -1.0986122886681098,
                "idf": {"−": -0.27465307216702745, "1.2": -0.27465307216702745},
            },
            configuration=cfg,
            chunk_sets=[ref("set")],
            ordered_chunk_ids=["c1"],
            tokenized_corpus=[["−", "1.2"]],
        ),
        DenseIndex(
            representation="JATS",
            chunk_tokens=192,
            paper_ids=["P001"],
            chunk_paper_ids=["P001"],
            chunk_set_manifests=[ref("set_manifest")],
            execution=DenseExecution(
                model=settings().bge_model,
                tokenizer=settings().construction_tokenizer,
                versions={"python": "synthetic"},
                batch_size=1,
                threads=1,
                interop_threads=1,
            ),
            vectors_build_manifest=ref("vectors_manifest"),
            input_id_hashes=["d" * 64],
            configuration=cfg,
            chunk_sets=[ref("set")],
            ordered_chunk_ids=["c1"],
            vectors=ref("vectors"),
            dimension=2,
            vector_count=1,
            shape=[1, 2],
            input_token_counts=[7],
        ),
        RankedRetrieval(
            configuration=cfg,
            index=ref("index"),
            retriever="bm25",
            query_id="synthetic-q1",
            question="Which value?",
            representation="JATS",
            scope="designated_paper",
            designated_paper_id="P001",
            hits=[{"rank": 1, "chunk": ref("c1"), "paper_id": "P001", "score": 0.5}],
        ),
        ContextRecord(
            configuration=cfg,
            retrieval=ref("ranked"),
            representation="JATS",
            decisions=[
                {"rank": 1, "chunk_id": "c1", "decision": "included", "duplicate_of_rank": None}
            ],
            included_chunk_ids=["c1"],
            exact_context="−1.2e−3",
            generator_tokens=8,
            stop_reason="exhausted",
        ),
    ]


@pytest.mark.parametrize("index", range(7))
def test_persistent_contract_versions_and_unknown_fields(index):
    value = records()[index]
    encoded = serialize_project_json(value, type(value))
    assert type(value).model_validate_json(encoded) == value
    for field, item in [("schema_version", "999"), ("authority", {"approved": True})]:
        data = value.model_dump()
        data[field] = item
        with pytest.raises(ValidationError):
            type(value).model_validate(data)


@pytest.mark.parametrize(
    "field,value",
    [
        ("canonical", None),
        ("validation", {}),
        ("build_manifest", {"artifact_id": "", "logical_ref": "/local", "sha256": "wrong"}),
    ],
)
def test_malformed_input_references_rejected(field, value):
    data = selection().model_dump()
    data[field] = value
    with pytest.raises(ValidationError):
        CanonicalInput.model_validate(data)


def test_index_and_context_order_contracts_reject_inconsistent_records():
    values = records()
    data = values[3].model_dump()
    data["tokenized_corpus"] = []
    with pytest.raises(ValidationError):
        Bm25Index.model_validate(data)
    data = values[4].model_dump()
    data["input_token_counts"] = []
    with pytest.raises(ValidationError):
        DenseIndex.model_validate(data)
    data = values[5].model_dump()
    data["hits"][0]["paper_id"] = "P002"
    with pytest.raises(ValidationError):
        RankedRetrieval.model_validate(data)
    data = values[6].model_dump()
    data["included_chunk_ids"] = []
    with pytest.raises(ValidationError):
        ContextRecord.model_validate(data)


def test_all_output_identities_use_ordered_inputs_and_exact_stored_bytes():
    from biorag_repro.retrieval.storage import record_ref

    for value in records()[1:]:
        output = record_ref(value)
        assert output == record_ref(type(value).model_validate(value.model_dump()))
        assert output.sha256 == sha256(serialize_project_json(value, type(value))).hexdigest()
        assert output.artifact_id.startswith("drv_")
    first = chunk()
    first.contributors.append(first.contributors[0].model_copy(update={"node_id": "second"}))
    second = first.model_copy(deep=True)
    second.contributors.reverse()
    assert record_ref(first).artifact_id != record_ref(second).artifact_id


def test_jsonl_contract_reuses_existing_project_serialization():
    from biorag_repro.benchmark.storage import decode_jsonl, encode_jsonl

    values = [chunk(), chunk()]
    content = encode_jsonl(values)
    assert content.count(b"\n") == 2
    assert decode_jsonl(content, Chunk) == values
    assert content == encode_jsonl(values)


def test_index_contract_can_reference_multiple_frozen_papers():
    value = Bm25Index(
        representation="JATS",
        chunk_tokens=192,
        paper_ids=["P001", "P002"],
        chunk_paper_ids=["P001", "P002"],
        chunk_set_manifests=[ref("P001manifest"), ref("P002manifest")],
        statistics={
            "corpus_size": 2,
            "document_lengths": [1, 1],
            "term_frequencies": [{"a": 1}, {"b": 1}],
            "document_frequencies": {"a": 1, "b": 1},
            "average_document_length": 1.0,
            "average_idf": 0.0,
            "idf": {"a": 0.0, "b": 0.0},
        },
        configuration=configuration(),
        chunk_sets=[ref("P001set"), ref("P002set")],
        ordered_chunk_ids=["c1", "c2"],
        tokenized_corpus=[["a"], ["b"]],
    )
    assert [r.artifact_id for r in value.chunk_sets] == ["P001set", "P002set"]


def test_duplicate_context_decision_must_reference_first_retained_inclusion():
    data = records()[6].model_dump()
    data["decisions"] += [
        {"rank": 2, "chunk_id": "c2", "decision": "duplicate", "duplicate_of_rank": 1},
        {"rank": 3, "chunk_id": "c3", "decision": "duplicate", "duplicate_of_rank": 2},
    ]
    with pytest.raises(ValidationError, match="retained"):
        ContextRecord.model_validate(data)
