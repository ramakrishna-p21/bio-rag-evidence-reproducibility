"""Canonical document validation, schema, and immutable publication."""

import pytest
from pydantic import ValidationError

from biorag_repro.core.config import resolve_config
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.storage import read_manifest, read_project_json, serialize_project_json
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.models import CanonicalDocument, JatsConfiguration, TableData
from biorag_repro.documents.storage import publish_canonical_jats


def document() -> CanonicalDocument:
    content = b"<article><body><p>alpha</p><p>beta</p></body></article>"
    digest = sha256_bytes(content)
    return ingest_jats(
        content,
        paper_id="P001",
        source_artifact_id=raw_source_artifact_id(
            paper_id="P001", representation="JATS", sha256=digest
        ),
        source_sha256=digest,
        source_logical_ref="corpus/raw/P001.xml",
        configuration_sha256="a" * 64,
    )


def test_duplicate_node_and_broken_mapping_rejected():
    data = document().model_dump(mode="python")
    data["nodes"][1]["node_id"] = data["nodes"][0]["node_id"]
    with pytest.raises(ValidationError, match="duplicate canonical node"):
        CanonicalDocument.model_validate(data)
    data = document().model_dump(mode="python")
    data["nodes"][0]["mapping"]["source_artifact_id"] = "other"
    with pytest.raises(ValidationError, match="node source differs"):
        CanonicalDocument.model_validate(data)
    data = document().model_dump(mode="python")
    data["source_doctype_declared"] = None
    with pytest.raises(ValidationError, match="DOCTYPE state"):
        CanonicalDocument.model_validate(data)
    data = document().model_dump(mode="python")
    data["nodes"][1]["mapping"]["path"] = ""
    with pytest.raises(ValidationError):
        CanonicalDocument.model_validate(data)


def test_schema_version_and_serialization_round_trip():
    schema = CanonicalDocument.model_json_schema()
    assert schema["properties"]["schema_version"]["enum"] == ["4", "5"]
    assert schema["properties"]["schema_version"]["default"] == "4"
    doc = document()
    encoded = serialize_project_json(doc, CanonicalDocument)
    assert CanonicalDocument.model_validate_json(encoded) == doc
    assert encoded == serialize_project_json(doc, CanonicalDocument)


def test_jats_configuration_is_resolved_not_raw_toml_identity():
    first = resolve_config(b'schema_version = "1"\n', JatsConfiguration)
    second = resolve_config(b'# comment\nschema_version = "1"\n', JatsConfiguration)
    assert first.configuration_sha256 == second.configuration_sha256
    assert first.raw_toml_sha256 != second.raw_toml_sha256


def test_table_model_rejects_overlapping_cells():
    from biorag_repro.documents.models import SourceMapping, TableCell

    mapping = SourceMapping(
        kind="jats", state="exact_native", source_artifact_id="source", path="/cell[1]"
    )
    cell = TableCell(
        node_id="n1",
        row=0,
        column=0,
        row_span=1,
        column_span=1,
        text="x",
        is_header=False,
        mapping=mapping,
    )
    with pytest.raises(ValidationError, match="overlapping table cells"):
        TableData(
            structure_state="exact_native",
            row_count=1,
            column_count=1,
            cells=[cell, cell.model_copy(update={"node_id": "n2"})],
        )


def test_no_clobber_publication_checks_output_before_complete_manifest(tmp_path):
    doc = document()
    source = b"<article><body><p>alpha</p><p>beta</p></body></article>"
    ref, manifest_ref = publish_canonical_jats(doc, tmp_path, source)
    assert read_project_json(tmp_path / ref.logical_ref, ref.sha256, CanonicalDocument) == doc
    manifest = read_manifest(tmp_path / manifest_ref.logical_ref, manifest_ref.sha256)
    assert manifest.status == "complete"
    assert manifest.inputs[0].artifact.artifact_id == doc.source.artifact_id
    assert manifest.outputs[0].artifact == ref
    with pytest.raises(FileExistsError):
        publish_canonical_jats(doc, tmp_path, source)


def test_incomplete_jats_cannot_publish_complete_manifest(tmp_path):
    doc = document()
    source = b"<article><body><p>alpha</p><p>beta</p></body></article>"
    nodes = [node.model_copy(deep=True) for node in doc.nodes]
    nodes[2].segments = []
    with pytest.raises(ValueError, match="source slot"):
        publish_canonical_jats(doc.model_copy(update={"nodes": nodes}), tmp_path, source)
    assert not list(tmp_path.rglob("*.build.json"))
