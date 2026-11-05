"""Occurrence/contributor checks use literal source locations and text."""

import json
from hashlib import sha256

import pytest
from test_documents_jats import parse
from test_retrieval_models import configuration, ref, selection

from biorag_repro.core.hashing import canonical_json_bytes
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.retrieval.models import Chunk, SourceOccurrence
from biorag_repro.retrieval.validation import (
    occurrence_id,
    validate_chunk_contributors,
    validate_occurrences,
)


def example():
    doc = parse(b"<article><body><p>Effect <italic>same</italic>.</p><p>same</p></body></article>")
    paragraph = next(n for n in doc.nodes if n.kind == "paragraph")
    segment = paragraph.segments[1]
    value = SourceOccurrence(
        occurrence_id=occurrence_id(ref(), segment.mapping, "segment"),
        canonical=ref(),
        owner_node_id=paragraph.node_id,
        field="segment",
        segment_index=1,
        mapping=segment.mapping,
        text="same",
        content_state="present",
        interpretation_state=None,
    )
    return doc, paragraph, value


def test_occurrence_identity_matches_independently_specified_payload():
    doc, _paragraph, value = example()
    locator = {
        "source_artifact_id": doc.source.artifact_id,
        "kind": "jats",
        "path": value.mapping.path,
        "native_artifact": None,
        "native_ref": None,
        "field": "segment",
    }
    digest = sha256(canonical_json_bytes(locator)).hexdigest()
    payload = {
        "identity_version": "1",
        "artifact_kind": "source_occurrence",
        "inputs": [{"role": "canonical", "artifact_id": "canonical", "sha256": "a" * 64}],
        "transformation_id": "source_occurrence_v1",
        "transformation_settings_sha256": None,
        "output_sha256": digest,
        "output_format": "json",
        "serialization_version": "1",
    }
    assert value.occurrence_id == "drv_" + sha256(canonical_json_bytes(payload)).hexdigest()
    validate_occurrences([value], doc, ref())


@pytest.mark.parametrize(
    "case", ["text", "owner", "mapping", "state", "identity", "canonical", "content_state"]
)
def test_rehashed_but_incorrect_occurrence_relationships_reject(case):
    doc, _paragraph, value = example()
    data = value.model_dump()
    if case == "text":
        data["text"] = "other"
    elif case == "owner":
        data["owner_node_id"] = doc.nodes[-1].node_id
    elif case == "mapping":
        data["mapping"]["path"] = "/article/body[1]/p[2]/text()"
    elif case == "state":
        data["mapping"]["state"] = "approximate"
    elif case == "identity":
        data["occurrence_id"] = "drv_" + "0" * 64
    elif case == "canonical":
        data["canonical"] = ref("substitute").model_dump()
    else:
        data["content_state"] = "unsupported"
    if case != "identity":
        data["occurrence_id"] = occurrence_id(data["canonical"], data["mapping"], data["field"])
    counterfeit = SourceOccurrence.model_validate(data)
    # Exact serialization and renewed identity are valid; source relationship is not.
    assert (
        SourceOccurrence.model_validate_json(serialize_project_json(counterfeit, SourceOccurrence))
        == counterfeit
    )
    with pytest.raises(ValueError):
        validate_occurrences([counterfeit], doc, ref())


def test_containing_reference_and_repeated_rendering_do_not_create_occurrences():
    doc, paragraph, value = example()
    inline = next(n for n in doc.nodes if n.source_tag == "italic")
    selected = selection()
    selected.paper_id = doc.paper_id
    chunk = Chunk(
        input=selected,
        configuration=configuration(),
        rendered_text="same",
        construction_tokens=1,
        contributors=[
            {
                "occurrence_id": value.occurrence_id,
                "node_id": paragraph.node_id,
                "usage": "content",
            },
            {
                "occurrence_id": value.occurrence_id,
                "node_id": inline.node_id,
                "usage": "containing_reference",
            },
            {
                "occurrence_id": value.occurrence_id,
                "node_id": paragraph.node_id,
                "usage": "repeated_context",
            },
        ],
    )
    validate_chunk_contributors(chunk, doc, [value])
    assert len({c.occurrence_id for c in chunk.contributors}) == 1
    other = chunk.model_copy(deep=True)
    other.contributors[1].node_id = doc.nodes[-1].node_id
    with pytest.raises(ValueError, match="contributor"):
        validate_chunk_contributors(other, doc, [value])


def test_repeated_identical_text_at_distinct_locations_is_not_deduplicated():
    doc, _paragraph, first = example()
    other = doc.nodes[-1]
    second = first.model_copy(
        update={
            "owner_node_id": other.node_id,
            "segment_index": 0,
            "mapping": other.segments[0].mapping,
        }
    )
    second.occurrence_id = occurrence_id(ref(), second.mapping, "segment")
    validate_occurrences([first, second], doc, ref())
    assert first.text == second.text and first.occurrence_id != second.occurrence_id


def test_duplicate_occurrence_claims_and_hidden_text_as_nontextual_reject():
    doc, _paragraph, value = example()
    with pytest.raises(ValueError, match="duplicate"):
        validate_occurrences([value, value], doc, ref())
    hidden = value.model_copy(update={"field": "nontextual", "segment_index": None, "text": None})
    hidden.occurrence_id = occurrence_id(ref(), hidden.mapping, "nontextual")
    with pytest.raises(ValueError, match="nontextual"):
        validate_occurrences([hidden], doc, ref())


def test_contributors_cannot_relabel_paper_or_representation():
    doc, paragraph, value = example()
    record = Chunk(
        input=selection(),
        configuration=configuration(),
        rendered_text="same",
        construction_tokens=1,
        contributors=[
            {"occurrence_id": value.occurrence_id, "node_id": paragraph.node_id, "usage": "content"}
        ],
    )
    record.input.representation = "PDF_DOCLING"
    with pytest.raises(ValueError, match="paper|representation"):
        validate_chunk_contributors(record, doc, [value])


def test_occurrence_input_does_not_mutate_canonical_evidence():
    doc, _paragraph, value = example()
    before = json.loads(serialize_project_json(doc, CanonicalDocument))
    validate_occurrences([value], doc, ref())
    assert doc.model_dump(mode="json") == before


def test_jats_table_aggregate_cannot_be_a_second_source_occurrence():
    doc = parse(b"<article><body><table><tr><td>same</td></tr></table></body></article>")
    node = next(n for n in doc.nodes if n.kind == "table_cell")
    aggregate = SourceOccurrence(
        occurrence_id=occurrence_id(ref(), node.mapping, "table_cell"),
        canonical=ref(),
        owner_node_id=node.node_id,
        field="table_cell",
        segment_index=None,
        mapping=node.mapping,
        text="same",
        content_state=node.content_state,
        interpretation_state=None,
    )
    with pytest.raises(ValueError, match="JATS.*segment"):
        validate_occurrences([aggregate], doc, ref())


def test_pdf_table_cell_segment_and_structured_view_share_source_identity():
    from biorag_repro.documents.models import SourceMapping

    mapping = SourceMapping(
        kind="docling",
        state="approximate",
        source_artifact_id="raw",
        native_artifact=ref("native"),
        native_ref="#/tables/0/data/table_cells/0",
    )
    assert occurrence_id(ref(), mapping, "segment") == occurrence_id(ref(), mapping, "table_cell")


def test_unverified_native_rendition_and_unknown_order_cannot_be_laundered():
    from biorag_repro.documents.models import CanonicalNode, SourceMapping

    mapping = SourceMapping(
        kind="docling",
        state="approximate",
        source_artifact_id="raw",
        native_artifact=ref("native"),
        native_ref="#/texts/0",
    )
    node = CanonicalNode(
        node_id="formula",
        parent_node_id=None,
        order=0,
        kind="formula",
        source_tag="synthetic",
        mapping=mapping,
        content_state="unverified_native_rendition",
        interpretation_state="ambiguous",
        native_order_state="unpositioned",
        native_orig="x²",
        orig_qualification="unverified_formula",
    )
    doc = CanonicalDocument(
        paper_id="P001",
        representation="PDF_DOCLING",
        source=ref("raw"),
        native_artifact=ref("native"),
        configuration_sha256="b" * 64,
        transformation_id="synthetic",
        status="complete",
        nodes=[node],
    )
    item = SourceOccurrence(
        occurrence_id=occurrence_id(ref(), mapping, "native_orig"),
        canonical=ref(),
        owner_node_id="formula",
        field="native_orig",
        segment_index=None,
        mapping=mapping,
        text="x²",
        content_state="unverified_native_rendition",
        interpretation_state="ambiguous",
        native_order_state="unpositioned",
        orig_qualification="unverified_formula",
    )
    validate_occurrences([item], doc, ref())
    for field, value in [
        ("orig_qualification", "same_as_text"),
        ("native_order_state", "linked"),
        ("interpretation_state", "exact_native"),
    ]:
        with pytest.raises(ValueError, match="uncertainty"):
            validate_occurrences([item.model_copy(update={field: value})], doc, ref())


def test_one_occurrence_cannot_be_claimed_as_two_primary_contributions():
    doc, paragraph, occurrence = example()
    contributor = {
        "occurrence_id": occurrence.occurrence_id,
        "node_id": paragraph.node_id,
        "usage": "content",
    }
    value = Chunk(
        input=selection(),
        configuration=configuration(),
        contributors=[contributor, contributor],
        rendered_text="same",
        construction_tokens=1,
    )
    with pytest.raises(ValueError, match="primary"):
        validate_chunk_contributors(value, doc, [occurrence])
