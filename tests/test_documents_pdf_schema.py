"""Versioned canonical PDF provenance contract."""

from copy import deepcopy

import pytest
from pydantic import ValidationError

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import (
    CanonicalNode,
    NativeBBox,
    NativeProvenance,
    SourceMapping,
    TableData,
)

RAW = ArtifactRef(artifact_id="raw_pdf", logical_ref="raw/p.pdf", sha256="a" * 64)
NATIVE = ArtifactRef(artifact_id="drv_native", logical_ref="derived/n.json", sha256="b" * 64)


def native_mapping() -> SourceMapping:
    return SourceMapping(
        kind="docling",
        state="exact_native",
        source_artifact_id=RAW.artifact_id,
        native_artifact=NATIVE,
        native_ref="#/tables/0",
        provenance=[
            NativeProvenance(
                ordinal=0,
                page_no=2,
                bbox=NativeBBox(l=1.00000000001, t=5.1, r=3.2, b=4.2, coord_origin="BOTTOMLEFT"),
                charspan=[0, 0],
                charspan_state="placeholder",
                state="approximate",
            ),
            NativeProvenance(
                ordinal=1,
                page_no=3,
                bbox=NativeBBox(l=2.0, t=8.0, r=4.0, b=6.0, coord_origin="TOPLEFT"),
                charspan=[1, 9],
                charspan_state="meaningful",
                state="approximate",
            ),
        ],
    )


def test_pdf_mapping_round_trip_keeps_native_identity_and_all_raw_provenance():
    mapping = native_mapping()
    assert mapping.source_artifact_id == RAW.artifact_id
    assert mapping.native_artifact == NATIVE
    assert mapping.native_ref == "#/tables/0"
    assert [p.ordinal for p in mapping.provenance] == [0, 1]
    assert [p.page_no for p in mapping.provenance] == [2, 3]
    assert [p.bbox.coord_origin for p in mapping.provenance] == ["BOTTOMLEFT", "TOPLEFT"]
    assert mapping.provenance[0].bbox.l == 1.00000000001
    assert mapping.provenance[0].charspan == [0, 0]
    assert mapping.provenance[0].charspan_state == "placeholder"
    assert SourceMapping.model_validate_json(mapping.model_dump_json()) == mapping


def test_jats_mapping_has_no_pdf_fields():
    mapping = SourceMapping(
        kind="jats",
        state="exact_native",
        source_artifact_id="raw_jats",
        path="/article[1]/p[1]",
        segment_ordinal=0,
    )
    assert mapping.native_artifact is None and mapping.provenance == []


@pytest.mark.parametrize(
    "change",
    [
        {"native_artifact": None},
        {"native_ref": None},
        {"kind": "jats"},
        {"provenance": [{"ordinal": 1}]},
    ],
)
def test_invalid_mapping_combinations_fail(change):
    payload = native_mapping().model_dump(mode="python")
    payload.update(change)
    with pytest.raises(ValidationError):
        SourceMapping.model_validate(payload)


def test_invalid_provenance_combinations_fail():
    payload = native_mapping().provenance[0].model_dump(mode="python")
    payload["charspan_state"] = "meaningful"
    with pytest.raises(ValidationError):
        NativeProvenance.model_validate(payload)
    payload = native_mapping().model_dump(mode="python")
    payload["provenance"][1]["ordinal"] = 0
    with pytest.raises(ValidationError):
        SourceMapping.model_validate(payload)


def test_native_label_and_interpretation_are_independent():
    node = CanonicalNode(
        node_id="node_1",
        parent_node_id=None,
        order=0,
        kind="source_construct",
        source_tag="pdf_text",
        mapping=native_mapping(),
        native_label="section_header",
        interpretation_state="ambiguous",
    )
    assert node.native_label == "section_header"
    assert node.kind == "source_construct"
    assert node.interpretation_state == "ambiguous"


def test_empty_table_relations_distinguish_unavailable_and_absent():
    table = TableData(
        structure_state="exact_native",
        row_count=0,
        column_count=0,
        cells=[],
        caption_relation_state="unavailable",
        footnote_relation_state="unavailable",
    )
    assert table.caption_node_ids == table.footnote_node_ids == []
    assert table.caption_relation_state == "unavailable"
    assert TableData.model_validate_json(table.model_dump_json()) == table
    bad = deepcopy(table.model_dump(mode="python"))
    bad["caption_relation_state"] = "present"
    with pytest.raises(ValidationError):
        TableData.model_validate(bad)
