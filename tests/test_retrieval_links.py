"""Cross-record contracts validate references without running retrieval algorithms."""

import pytest
from test_retrieval_models import configuration, metadata, records, ref, settings
from test_retrieval_occurrences import example

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.retrieval.models import Chunk, ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import (
    load_configuration,
    validate_chunk_set_links,
    validate_context_links,
    validate_index_links,
    validate_retrieval_links,
)


def test_configuration_read_checks_bytes_and_resolved_identity(tmp_path):
    value = settings()
    content = serialize_project_json(value, Phase5Configuration)
    path = tmp_path / "configuration.json"
    path.write_bytes(content)
    actual = ConfigurationRef(
        artifact=ref().model_copy(
            update={"logical_ref": "configuration.json", "sha256": sha256_bytes(content)}
        ),
        resolved_sha256=sha256_bytes(canonical_json_bytes(value.model_dump())),
    )
    assert load_configuration(actual, tmp_path) == value
    bad = actual.model_copy(update={"resolved_sha256": "0" * 64})
    with pytest.raises(ValueError, match="configuration identity"):
        load_configuration(bad, tmp_path)
    path.write_bytes(content + b" ")
    with pytest.raises(ValueError, match="SHA-256"):
        load_configuration(actual, tmp_path)


def test_chunk_set_links_check_contributors_counts_and_occurrence_accounting():
    doc, paragraph, occurrence = example()
    value = metadata()
    value.configuration.resolved_sha256 = sha256_bytes(
        canonical_json_bytes(settings().model_dump())
    )
    value.occurrences = [occurrence]
    value.input.paper_id = doc.paper_id
    built = Chunk(
        input=value.input,
        configuration=value.configuration,
        contributors=[
            {
                "occurrence_id": occurrence.occurrence_id,
                "node_id": paragraph.node_id,
                "usage": "content",
            }
        ],
        rendered_text="same",
        construction_tokens=1,
    )
    value.chunk_ids = [record_ref(built).artifact_id]
    value.accounting[0].occurrence_id = occurrence.occurrence_id
    value.accounting[0].chunk_ids = value.chunk_ids
    validate_chunk_set_links(value, [built], doc, settings())
    bad_settings = settings().model_copy(update={"lexical_tokenizer": "another_policy"})
    with pytest.raises(ValueError, match="configuration identity"):
        validate_chunk_set_links(value, [built], doc, bad_settings)
    bad = built.model_copy(update={"construction_tokens": 193})
    value.chunk_ids = [record_ref(bad).artifact_id]
    value.accounting[0].chunk_ids = value.chunk_ids
    with pytest.raises(ValueError, match="budget"):
        validate_chunk_set_links(value, [bad], doc, settings())
    value.chunk_ids = [record_ref(built).artifact_id]
    value.accounting[0].chunk_ids = []
    value.accounting[0].status = "unsupported"
    value.accounting[0].reason = "false exclusion"
    with pytest.raises(ValueError, match="accounting"):
        validate_chunk_set_links(value, [built], doc, settings())


@pytest.mark.parametrize("index_position", [3, 4])
def test_index_links_reject_renewed_but_wrong_chunk_order(index_position):
    index = records()[index_position]
    first, second = metadata(), metadata()
    first.chunk_ids, second.chunk_ids = ["c1"], ["c2"]
    first.accounting[0].chunk_ids = ["c1"]
    second.accounting[0].chunk_ids = ["c2"]
    references = [record_ref(first), record_ref(second)]
    index.chunk_sets = references
    index.ordered_chunk_ids = ["c1", "c2"]
    if index_position == 3:
        index.tokenized_corpus = [["a"], ["b"]]
    else:
        index.input_token_counts = [1, 1]
    validate_index_links(index, [first, second])
    index.ordered_chunk_ids.reverse()
    assert record_ref(index)
    with pytest.raises(ValueError, match="order"):
        validate_index_links(index, [first, second])


def test_ranked_and_context_records_must_reference_the_selected_records():
    values = records()
    index, ranked, context = values[3], values[5], values[6]
    _doc, paragraph, occurrence = example()
    built = Chunk(
        input=values[0],
        configuration=configuration(),
        contributors=[
            {
                "occurrence_id": occurrence.occurrence_id,
                "node_id": paragraph.node_id,
                "usage": "content",
            }
        ],
        rendered_text="same",
        construction_tokens=1,
    )
    chunk_ref = record_ref(built)
    index.ordered_chunk_ids = [chunk_ref.artifact_id]
    ranked.index = record_ref(index)
    ranked.hits[0].chunk = chunk_ref
    validate_retrieval_links(ranked, index, [built])
    context.retrieval = record_ref(ranked)
    context.decisions[0].chunk_id = chunk_ref.artifact_id
    context.included_chunk_ids = [chunk_ref.artifact_id]
    validate_context_links(context, ranked)
    context.decisions[0].chunk_id = "wrong"
    context.included_chunk_ids = ["wrong"]
    assert record_ref(context)
    with pytest.raises(ValueError, match="candidate"):
        validate_context_links(context, ranked)
    ranked.hits[0].paper_id = "P002"
    with pytest.raises(ValueError):
        validate_retrieval_links(ranked, index, [built])
