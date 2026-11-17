"""Only explicit native table-footnote links survive the versioned correction."""

from types import SimpleNamespace as Item

import pytest
from test_documents_table_fidelity import _native_data, _typed_native

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.coverage import validate_pdf_coverage
from biorag_repro.documents.pdf import canonicalize_native


def fixture():
    data = _native_data()
    text = {
        "self_ref": "#/texts/0",
        "label": "footnote",
        "content_layer": "body",
        "parent": {"$ref": "#/body"},
        "children": [],
        "prov": [],
        "text": "Explicit native note",
        "orig": "Explicit native note",
    }
    data["texts"] = [text]
    data["body"]["children"].append({"$ref": "#/texts/0"})
    data["tables"][0]["footnotes"] = [{"$ref": "#/texts/0"}]
    native = _typed_native(data)
    native.texts = [Item(**(text | {"parent": Item(cref="#/body")}))]
    native.body.children.append(Item(cref="#/texts/0"))
    native.tables[0].footnotes = [Item(cref="#/texts/0")]
    return data, native


def build(data, native):
    return canonicalize_native(
        native,
        "P010",
        ArtifactRef(artifact_id="raw", logical_ref="source", sha256="a" * 64),
        ArtifactRef(artifact_id="native", logical_ref="native", sha256="b" * 64),
        "c" * 64,
        data,
        preserve_native_table_footnotes=True,
    )


def test_native_footnote_identity_and_unavailable_states():
    data, native = fixture()
    doc = build(data, native)
    assert doc.transformation_id == "docling_pdf_adapter_v4"
    table = next(n.table for n in doc.nodes if n.table)
    note = next(n for n in doc.nodes if n.mapping.native_ref == "#/texts/0")
    assert table.footnote_node_ids == [note.node_id]
    assert table.footnote_relation_state == "present"
    validate_pdf_coverage(doc, data)
    table.footnote_node_ids = []
    table.footnote_relation_state = "unavailable"
    with pytest.raises(ValueError, match="footnote"):
        validate_pdf_coverage(doc, data)


def test_footnotes_cannot_be_inferred_or_resolved_to_another_object():
    data, native = fixture()
    data["tables"][0]["footnotes"] = [{"$ref": "#/texts/99"}]
    with pytest.raises(ValueError, match="footnote"):
        build(data, native)
    data, native = fixture()
    native.tables[0].footnotes = []
    with pytest.raises(ValueError, match="footnote"):
        build(data, native)


def test_historical_adapter_still_rejects_populated_unsupported_relations():
    data, native = fixture()
    doc = canonicalize_native(
        native,
        "P010",
        ArtifactRef(artifact_id="raw", logical_ref="source", sha256="a" * 64),
        ArtifactRef(artifact_id="native", logical_ref="native", sha256="b" * 64),
        "c" * 64,
        data,
    )
    assert doc.transformation_id == "docling_pdf_adapter_v3"
    with pytest.raises(ValueError, match="unsupported populated native relation"):
        validate_pdf_coverage(doc, data)


def test_table_correction_does_not_authorize_picture_footnote_relations():
    data, native = fixture()
    picture = {
        "self_ref": "#/pictures/0",
        "label": "picture",
        "content_layer": "body",
        "parent": {"$ref": "#/body"},
        "children": [],
        "prov": [],
        "captions": [],
        "footnotes": [{"$ref": "#/texts/0"}],
    }
    data["pictures"] = [picture]
    data["body"]["children"].append({"$ref": "#/pictures/0"})
    native.pictures = [
        Item(**(picture | {"parent": Item(cref="#/body"), "footnotes": [Item(cref="#/texts/0")]}))
    ]
    native.body.children.append(Item(cref="#/pictures/0"))
    doc = build(data, native)
    with pytest.raises(ValueError, match="unsupported populated native relation"):
        validate_pdf_coverage(doc, data)


def figure_fixture():
    data, native = fixture()
    picture = {
        "self_ref": "#/pictures/0",
        "label": "picture",
        "content_layer": "body",
        "parent": {"$ref": "#/body"},
        "children": [],
        "prov": [],
        "captions": [],
        "footnotes": [{"$ref": "#/texts/0"}],
    }
    data["pictures"] = [picture]
    data["body"]["children"].append({"$ref": "#/pictures/0"})
    native.pictures = [
        Item(**(picture | {"parent": Item(cref="#/body"), "footnotes": [Item(cref="#/texts/0")]}))
    ]
    native.body.children.append(Item(cref="#/pictures/0"))
    return data, native


def figure_build(data, native):
    return canonicalize_native(
        native,
        "P031",
        ArtifactRef(artifact_id="raw", logical_ref="source", sha256="a" * 64),
        ArtifactRef(artifact_id="native", logical_ref="native", sha256="b" * 64),
        "c" * 64,
        data,
        preserve_native_figure_footnotes=True,
    )


def test_versioned_figure_footnote_identity_and_independent_loss_detection():
    from biorag_repro.core.storage import serialize_project_json
    from biorag_repro.documents.models import CanonicalDocument

    data, native = figure_fixture()
    doc = figure_build(data, native)
    assert doc.schema_version == "5" and doc.transformation_id == "docling_pdf_adapter_v5"
    figure = next(n for n in doc.nodes if n.kind == "figure")
    note = next(n for n in doc.nodes if n.mapping.native_ref == "#/texts/0")
    assert figure.figure_footnote_node_ids == [note.node_id]
    assert figure.target_node_ids == []  # caption role stays separate
    validate_pdf_coverage(doc, data)
    historical = build(*fixture())
    assert b"figure_footnote_node_ids" not in serialize_project_json(historical, CanonicalDocument)
    figure.figure_footnote_node_ids = []
    with pytest.raises(ValueError, match="figure footnote"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize(
    "damage", ["dangling", "wrong_role", "duplicate", "malformed", "typed_mismatch"]
)
def test_figure_footnotes_reject_native_reference_conflicts(damage):
    data, native = figure_fixture()
    if damage == "dangling":
        data["pictures"][0]["footnotes"] = [{"$ref": "#/texts/99"}]
        native.pictures[0].footnotes = [Item(cref="#/texts/99")]
    elif damage == "wrong_role":
        data["texts"][0]["label"] = "text"
    elif damage == "duplicate":
        data["pictures"][0]["footnotes"] *= 2
        native.pictures[0].footnotes *= 2
    elif damage == "malformed":
        data["pictures"][0]["footnotes"] = [{"target": "#/texts/0"}]
    else:
        native.pictures[0].footnotes = []
    with pytest.raises(ValueError, match="footnote|role"):
        figure_build(data, native)
