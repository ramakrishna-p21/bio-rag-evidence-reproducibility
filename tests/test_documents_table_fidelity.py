"""Native table roles and complete Phase 4 publication boundaries."""

import json
from pathlib import Path
from types import SimpleNamespace as Item

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.documents.pdf import TRANSFORMATION_ID, canonicalize_native
from biorag_repro.documents.pdf_ingest import (
    PdfCanonicalBuild,
    _manifest,
    _ref,
    canonical_build_ref,
    publish_pdf_result,
)
from biorag_repro.documents.phase4 import (
    JatsBuild,
    Phase4GateIndex,
    publish_gate_index,
    publish_validation,
    validate_build,
    validate_pdf_coverage,
)
from biorag_repro.documents.storage import publish_canonical_jats


def _source(root: Path, arm: str, content: bytes) -> tuple[Path, ArtifactRef]:
    digest = sha256_bytes(content)
    ref = ArtifactRef(
        artifact_id=raw_source_artifact_id(paper_id="P001", representation=arm, sha256=digest),
        logical_ref=f"P001/{arm}/source",
        sha256=digest,
    )
    path = root / "corpus/raw" / ref.logical_ref
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path, ref


def _native_cells():
    roles = [
        ("ClusterONE", True, False, False, False),
        ("Score", False, True, False, False),
        ("Section", False, False, True, False),
        ("Entry", True, True, False, True),
    ]
    return [
        {
            "text": text,
            "start_row_offset_idx": row,
            "end_row_offset_idx": row + 1,
            "start_col_offset_idx": 0,
            "end_col_offset_idx": 1,
            "row_span": 1,
            "col_span": 1,
            "row_header": rh,
            "column_header": ch,
            "row_section": rs,
            "fillable": fill,
            "bbox": None,
        }
        for row, (text, rh, ch, rs, fill) in enumerate(roles)
    ]


def _native_data():
    cells = _native_cells()
    return {
        "schema_name": "DoclingDocument",
        "version": "1.10.0",
        "body": {
            "self_ref": "#/body",
            "label": "unspecified",
            "content_layer": "body",
            "children": [{"$ref": "#/tables/0"}],
        },
        "furniture": None,
        "groups": [],
        "texts": [],
        "pictures": [],
        "tables": [
            {
                "self_ref": "#/tables/0",
                "parent": {"$ref": "#/body"},
                "children": [],
                "label": "table",
                "content_layer": "body",
                "prov": [],
                "captions": [],
                "footnotes": [],
                "data": {
                    "num_rows": 4,
                    "num_cols": 1,
                    "orientation": "rot_90",
                    "table_cells": cells,
                    "grid": [[cell] for cell in cells],
                },
            }
        ],
        "form_items": [],
        "key_value_items": [],
    }


def _typed_native(data):
    table_data = data["tables"][0]["data"]
    cells = [Item(**cell) for cell in table_data["table_cells"]]
    body = Item(
        self_ref="#/body",
        label="unspecified",
        content_layer="body",
        parent=None,
        children=[Item(cref="#/tables/0")],
        prov=[],
    )
    table = Item(
        self_ref="#/tables/0",
        label="table",
        content_layer="body",
        parent=Item(cref="#/body"),
        children=[],
        prov=[],
        captions=[],
        data=Item(
            num_rows=table_data["num_rows"],
            num_cols=table_data["num_cols"],
            **(
                {"orientation": Item(value=table_data["orientation"])}
                if "orientation" in table_data
                else {}
            ),
            table_cells=cells,
            grid=[[cell] for cell in cells],
        ),
    )
    return Item(
        body=body,
        furniture=None,
        groups=[],
        texts=[],
        pictures=[],
        tables=[table],
        form_items=[],
        key_value_items=[],
        iterate_items=lambda with_groups: iter([(body, 0), (table, 1)]),
    )


def test_native_table_fields_are_distinct_and_source_faithful():
    data = _native_data()
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(_typed_native(data), "P001", source, native, "c" * 64)
    table = next(node.table for node in doc.nodes if node.kind == "table")
    assert table.native_orientation == "rot_90"
    assert [
        (
            cell.native_row_header,
            cell.native_column_header,
            cell.native_row_section,
            cell.native_fillable,
            cell.is_header,
        )
        for cell in table.cells
    ] == [
        (True, False, False, False, True),
        (False, True, False, False, True),
        (False, False, True, False, False),
        (True, True, False, True, True),
    ]
    assert validate_pdf_coverage(doc, data) == (2, 0, 4)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_row_header", None),
        ("native_row_header", False),
        ("native_column_header", None),
        ("native_column_header", True),
        ("native_row_section", None),
        ("native_row_section", True),
        ("native_fillable", None),
        ("native_fillable", True),
        ("is_header", False),
    ],
)
def test_table_field_corruption_is_rejected(field, value):
    data = _native_data()
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(_typed_native(data), "P001", source, native, "c" * 64)
    cell = next(node.table for node in doc.nodes if node.kind == "table").cells[0]
    setattr(cell, field, value)
    with pytest.raises(ValueError, match="native table cell"):
        validate_pdf_coverage(doc, data)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("orientation", "orientation"),
        ("end_offset", "end offsets"),
        ("span", "table cell"),
        ("grid", "grid"),
        ("bbox", "provenance"),
        ("caption_state", "caption state"),
    ],
)
def test_table_structure_corruption_is_rejected(mutation, message):
    data = _native_data()
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(_typed_native(data), "P001", source, native, "c" * 64)
    table = next(node.table for node in doc.nodes if node.kind == "table")
    if mutation == "orientation":
        table.native_orientation = "rot_180"
    elif mutation == "end_offset":
        data["tables"][0]["data"]["table_cells"][0]["end_row_offset_idx"] = 3
    elif mutation == "span":
        table.cells[0].row_span = 2
    elif mutation == "grid":
        table.grid_cell_node_ids[0][0] = table.cells[1].node_id
    elif mutation == "bbox":
        data["tables"][0]["data"]["table_cells"][0]["bbox"] = {
            "l": 1.0,
            "t": 2.0,
            "r": 3.0,
            "b": 4.0,
            "coord_origin": "TOPLEFT",
        }
    else:
        table.caption_relation_state = "absent"
    with pytest.raises(ValueError, match=message):
        validate_pdf_coverage(doc, data)


def test_absent_native_table_flags_stay_unknown():
    data = _native_data()
    cell = data["tables"][0]["data"]["table_cells"][0]
    del cell["row_header"]
    del cell["row_section"]
    del data["tables"][0]["data"]["orientation"]
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(_typed_native(data), "P001", source, native, "c" * 64)
    table = next(node.table for node in doc.nodes if node.kind == "table")
    assert table.native_orientation is None
    assert table.cells[0].native_row_header is None
    assert table.cells[0].native_row_section is None
    assert table.cells[0].is_header is None
    assert validate_pdf_coverage(doc, data) == (2, 0, 4)


def test_saved_grid_only_fields_survive_typed_grid_cell_loss():
    data = _native_data()
    grid_cell = dict(data["tables"][0]["data"]["table_cells"][0])
    grid_cell["text"] = "Grid only"
    grid_cell["start_row_offset_idx"] = 4
    grid_cell["end_row_offset_idx"] = 5
    grid_cell["row_header"] = False
    grid_cell["column_header"] = True
    grid_cell["row_section"] = True
    grid_cell["fillable"] = True
    data["tables"][0]["data"]["num_rows"] = 5
    data["tables"][0]["data"]["grid"].append([grid_cell])
    typed = _typed_native(data)
    typed.tables[0].data.grid.append(
        [
            Item(
                text="Grid only",
                start_row_offset_idx=4,
                end_row_offset_idx=5,
                start_col_offset_idx=0,
                end_col_offset_idx=1,
                row_span=1,
                col_span=1,
                bbox=None,
            )
        ]
    )
    source = ArtifactRef(artifact_id="raw", logical_ref="source.pdf", sha256="a" * 64)
    native = ArtifactRef(artifact_id="native", logical_ref="native.json", sha256="b" * 64)
    doc = canonicalize_native(typed, "P001", source, native, "c" * 64, data)
    table = next(node.table for node in doc.nodes if node.kind == "table")
    grid_only = next(cell for cell in table.cells if "/grid/" in cell.mapping.native_ref)
    assert (
        grid_only.native_row_header,
        grid_only.native_column_header,
        grid_only.native_row_section,
        grid_only.native_fillable,
        grid_only.is_header,
    ) == (False, True, True, True, True)
    assert validate_pdf_coverage(doc, data) == (2, 0, 5)


def test_false_header_cannot_pass_complete_build_or_paired_gate(tmp_path):
    corpus = tmp_path / "corpus"
    output = tmp_path / "output"
    pdf_path, pdf_source = _source(corpus, "PDF", b"%PDF-1.4 synthetic")
    jats_bytes = b"<article><body><p>example</p></body></article>"
    _, jats_source = _source(corpus, "JATS", jats_bytes)
    data = _native_data()

    class Native:
        def save_as_json(self, path, **kwargs):
            Path(path).write_text(json.dumps(data))

    def load_native(path):
        assert json.loads(Path(path).read_text()) == data
        return _typed_native(data)

    result = Item(status=Item(value="success"), errors=[], pages=[], document=Native())
    attempt = publish_pdf_result(
        result,
        pdf_path,
        pdf_source,
        "P001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        output,
        load_native=load_native,
    )
    assert attempt.build_status == "complete"
    good_pdf = validate_build(attempt, output, corpus_root=corpus)
    jdoc = ingest_jats(
        jats_bytes,
        paper_id="P001",
        source_artifact_id=jats_source.artifact_id,
        source_sha256=jats_source.sha256,
        source_logical_ref=jats_source.logical_ref,
        configuration_sha256="d" * 64,
    )
    jref, jmanifest = publish_canonical_jats(jdoc, output, jats_bytes)
    good_jats = validate_build(
        JatsBuild(
            paper_id="P001",
            source=jats_source,
            configuration_sha256="d" * 64,
            canonical_artifact=jref,
            build_manifest=jmanifest,
        ),
        output,
        corpus_root=corpus,
    )
    jvalidation = publish_validation(good_jats, output)
    pvalidation = publish_validation(good_pdf, output)
    index = Phase4GateIndex(
        corpus_version="corpus_v1",
        paper_id="P001",
        corpus_release_sha256="e" * 64,
        implementation_git_sha="f" * 40,
        jats_validation=jvalidation,
        pdf_validation=pvalidation,
    )
    with pytest.raises(ValueError, match="independent human authorization"):
        publish_gate_index(index, output, corpus)

    doc = CanonicalDocument.model_validate_json(
        (output / attempt.canonical_artifact.logical_ref).read_bytes()
    )
    table = next(node.table for node in doc.nodes if node.kind == "table")
    assert table.cells[0].is_header is True
    table.cells[0].is_header = False
    table.cells[0].native_row_header = False
    content = serialize_project_json(doc, CanonicalDocument)
    settings = json.loads((output / attempt.build_manifest.logical_ref).read_bytes())[
        "transformation_settings_sha256"
    ]
    bad_ref = _ref(
        "canonical_pdf_docling",
        [
            ArtifactInput("raw_pdf", pdf_source.artifact_id, pdf_source.sha256),
            ArtifactInput(
                "docling_native",
                attempt.native_artifact.artifact_id,
                attempt.native_artifact.sha256,
            ),
        ],
        TRANSFORMATION_ID,
        settings,
        content,
        "4",
        "canonical/PDF_DOCLING",
    )
    target = output / bad_ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    bad_manifest = _manifest(
        [
            BuildInput(role="raw_pdf", artifact=pdf_source),
            BuildInput(role="docling_native", artifact=attempt.native_artifact),
        ],
        TRANSFORMATION_ID,
        settings,
        bad_ref,
        "canonical_pdf_docling",
        "4",
        output,
    )
    bad_build = PdfCanonicalBuild(
        paper_id="P001",
        source=pdf_source,
        configuration_sha256="a" * 64,
        environment_sha256="b" * 64,
        asset_inventory_sha256="c" * 64,
        original_attempt_artifact=good_pdf.attempt_artifact,
        native_artifact=attempt.native_artifact,
        native_build_manifest=attempt.native_build_manifest,
        canonical_artifact=bad_ref,
        build_manifest=bad_manifest,
    )
    record = canonical_build_ref(bad_build)
    record_path = output / record.logical_ref
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_bytes(serialize_project_json(bad_build, PdfCanonicalBuild))

    build_rejected = False
    gate_rejected = False
    try:
        validate_build(bad_build, output, corpus_root=corpus)
    except ValueError:
        build_rejected = True
    forged = publish_validation(
        good_pdf.model_copy(
            update={
                "canonical_artifact": bad_ref,
                "build_manifest": bad_manifest,
                "canonical_build_artifact": record,
            }
        ),
        output,
    )
    try:
        publish_gate_index(index.model_copy(update={"pdf_validation": forged}), output, corpus)
    except ValueError:
        gate_rejected = True
    assert build_rejected and gate_rejected
