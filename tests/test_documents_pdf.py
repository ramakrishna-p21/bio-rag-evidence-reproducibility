"""PDF ingestion contract using original small native-shaped objects."""

from types import SimpleNamespace as Item

import pytest

import biorag_repro.documents.pdf as pdf_module
from biorag_repro.core.config import resolve_config
from biorag_repro.core.hashing import HashMismatchError, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.pdf import (
    ASSETS,
    PdfConfiguration,
    canonicalize_native,
    verify_assets,
    verify_effective_options,
    verify_qualified_lock,
)

RAW = ArtifactRef(artifact_id="raw_pdf", logical_ref="raw/p.pdf", sha256="a" * 64)
NATIVE = ArtifactRef(artifact_id="drv_native", logical_ref="native/n.json", sha256="b" * 64)


def ref(value):
    return Item(cref=value)


def native_fixture():
    bbox = Item(l=1.00000000001, t=9.0, r=4.0, b=2.0, coord_origin="BOTTOMLEFT")
    prov = Item(page_no=1, bbox=bbox, charspan=[0, 0])
    cell = Item(
        text="2.3e-5",
        start_row_offset_idx=0,
        start_col_offset_idx=0,
        row_span=1,
        col_span=2,
        column_header=True,
        row_header=False,
        bbox=bbox,
    )
    table = Item(
        self_ref="#/tables/0",
        parent=ref("#/body"),
        label="table",
        prov=[prov, prov],
        captions=[],
        data=Item(num_rows=1, num_cols=2, table_cells=[cell], grid=[[cell, cell]]),
    )
    formula = Item(
        self_ref="#/texts/0",
        parent=ref("#/body"),
        label="formula",
        prov=[prov],
        text="",
        captions=[],
    )
    body = Item(self_ref="#/body", parent=None, label="unspecified", prov=[], captions=[])
    return Item(iterate_items=lambda with_groups: iter([(body, 0), (table, 1), (formula, 1)]))


def test_qualified_settings_are_fixed():
    config = PdfConfiguration()
    assert config.docling_slim_version == "2.145.0"
    assert config.model_threads == config.parser_threads == 1
    assert config.device == "cpu" and config.ocr is False
    with pytest.raises(ValueError):
        PdfConfiguration(model_threads=4)
    resolved = resolve_config(b'schema_version = "1"\n', PdfConfiguration)
    assert resolved.settings == config
    assert len(resolved.configuration_sha256) == 64


def test_native_traversal_preserves_tables_provenance_and_marker_formula():
    document = canonicalize_native(native_fixture(), "P001", RAW, NATIVE, "c" * 64)
    assert [n.mapping.native_ref for n in document.nodes] == [
        "#/body",
        "#/tables/0",
        "#/tables/0/data/table_cells/0",
        "#/texts/0",
    ]
    table = document.nodes[1]
    assert table.kind == "table" and table.native_label == "table"
    assert table.table.row_count == 1 and table.table.column_count == 2
    assert table.table.cells[0].column_span == 2
    assert table.table.cells[0].text == "2.3e-5"
    assert table.table.grid_cell_node_ids == [[table.table.cells[0].node_id] * 2]
    assert table.table.caption_relation_state == "unavailable"
    assert table.table.footnote_relation_state == "unavailable"
    assert [p.ordinal for p in table.mapping.provenance] == [0, 1]
    assert table.mapping.provenance[0].bbox.l == 1.00000000001
    assert table.mapping.provenance[0].bbox.coord_origin == "BOTTOMLEFT"
    assert table.mapping.provenance[0].charspan_state == "placeholder"
    assert document.nodes[3].content_state == "marker_only"
    assert canonicalize_native(native_fixture(), "P001", RAW, NATIVE, "c" * 64) == document


def test_native_picture_caption_link_is_retained_without_proximity_inference():
    body = Item(self_ref="#/body", parent=None, label="unspecified", prov=[], captions=[])
    picture = Item(
        self_ref="#/pictures/0",
        parent=ref("#/body"),
        label="picture",
        prov=[],
        captions=[ref("#/texts/0")],
    )
    caption = Item(
        self_ref="#/texts/0",
        parent=ref("#/body"),
        label="caption",
        prov=[],
        captions=[],
        text="Figure 1",
    )
    native = Item(iterate_items=lambda with_groups: iter([(body, 0), (picture, 1), (caption, 1)]))
    document = canonicalize_native(native, "P001", RAW, NATIVE, "c" * 64)
    assert document.nodes[1].native_label == "picture"
    assert document.nodes[1].target_node_ids == [document.nodes[2].node_id]
    assert document.nodes[1].relation_state == "exact_native"


def test_native_inventory_retains_picture_child_and_furniture_outside_typed_walk():
    body = Item(
        self_ref="#/body",
        parent=None,
        label="unspecified",
        content_layer="body",
        prov=[],
        captions=[],
        children=[ref("#/pictures/0"), ref("#/texts/1")],
    )
    picture = Item(
        self_ref="#/pictures/0",
        parent=ref("#/body"),
        label="picture",
        content_layer="body",
        prov=[],
        captions=[],
        children=[ref("#/texts/0")],
    )
    child = Item(
        self_ref="#/texts/0",
        parent=ref("#/pictures/0"),
        label="text",
        content_layer="body",
        prov=[],
        captions=[],
        children=[],
        text="figure value",
        orig="figure value",
    )
    header = Item(
        self_ref="#/texts/1",
        parent=ref("#/body"),
        label="page_header",
        content_layer="furniture",
        prov=[],
        captions=[],
        children=[],
        text="Page 1",
        orig="Page 1",
    )
    native = Item(
        body=body,
        furniture=Item(
            self_ref="#/furniture",
            parent=None,
            label="unspecified",
            content_layer="furniture",
            prov=[],
            captions=[],
            children=[],
        ),
        texts=[child, header],
        pictures=[picture],
        tables=[],
        groups=[],
        form_items=[],
        key_value_items=[],
        iterate_items=lambda with_groups: iter([(body, 0), (picture, 1)]),
    )
    document = canonicalize_native(native, "P001", RAW, NATIVE, "c" * 64)
    by_ref = {node.mapping.native_ref: node for node in document.nodes}
    assert len(by_ref) == len(document.nodes) == 5
    assert by_ref["#/texts/0"].parent_node_id == by_ref["#/pictures/0"].node_id
    assert by_ref["#/texts/0"].segments[0].text == "figure value"
    assert by_ref["#/texts/1"].kind == "page_furniture"
    assert by_ref["#/texts/1"].native_content_layer == "furniture"


def test_native_formula_list_and_heading_fields_remain_qualified():
    body = Item(
        self_ref="#/body",
        parent=None,
        label="unspecified",
        content_layer="body",
        prov=[],
        captions=[],
        children=[ref("#/texts/0"), ref("#/texts/1"), ref("#/texts/2")],
    )
    formula = Item(
        self_ref="#/texts/0",
        parent=ref("#/body"),
        label="formula",
        content_layer="body",
        prov=[],
        captions=[],
        children=[],
        text="",
        orig="damaged glyph",
    )
    listed = Item(
        self_ref="#/texts/1",
        parent=ref("#/body"),
        label="list_item",
        content_layer="body",
        prov=[],
        captions=[],
        children=[],
        text="value",
        orig="1. value",
        marker="1.",
        enumerated=True,
    )
    heading = Item(
        self_ref="#/texts/2",
        parent=ref("#/body"),
        label="section_header",
        content_layer="body",
        prov=[],
        captions=[],
        children=[],
        text="Methods",
        orig="Methods",
        level=2,
    )
    native = Item(
        body=body,
        texts=[formula, listed, heading],
        pictures=[],
        tables=[],
        groups=[],
        form_items=[],
        key_value_items=[],
        iterate_items=lambda with_groups: iter(
            [(body, 0), (formula, 1), (listed, 1), (heading, 1)]
        ),
    )
    nodes = canonicalize_native(native, "P001", RAW, NATIVE, "c" * 64).nodes
    assert nodes[1].native_orig == "damaged glyph"
    assert nodes[1].orig_qualification == "unverified_formula"
    assert nodes[1].content_state == "unverified_native_rendition"
    assert (nodes[2].native_marker, nodes[2].native_enumerated, nodes[2].native_orig) == (
        "1.",
        True,
        "1. value",
    )
    assert nodes[2].orig_qualification == "marker_inclusive"
    assert nodes[3].native_heading_level == 2


def test_unlinked_native_item_has_explicit_unknown_reading_order():
    body = Item(
        self_ref="#/body", parent=None, label="unspecified", children=[], prov=[], captions=[]
    )
    detached = Item(
        self_ref="#/texts/0",
        parent=None,
        label="text",
        children=[],
        prov=[],
        captions=[],
        text="detached",
        orig="detached",
    )
    native = Item(
        body=body,
        texts=[detached],
        groups=[],
        pictures=[],
        tables=[],
        form_items=[],
        key_value_items=[],
        iterate_items=lambda with_groups: iter([(body, 0)]),
    )
    document = canonicalize_native(native, "P001", RAW, NATIVE, "c" * 64)
    assert document.nodes[1].mapping.native_ref == "#/texts/0"
    assert document.nodes[1].native_order_state == "unpositioned"
    assert document.nodes[1].parent_node_id is None


def test_native_grid_only_empty_cell_is_preserved():
    body = Item(self_ref="#/body", parent=None, label="unspecified", prov=[], captions=[])
    first = Item(
        text="A",
        start_row_offset_idx=0,
        start_col_offset_idx=0,
        row_span=1,
        col_span=1,
        column_header=False,
        row_header=False,
        bbox=None,
    )
    blank = Item(
        text="",
        start_row_offset_idx=0,
        start_col_offset_idx=1,
        row_span=1,
        col_span=1,
        column_header=False,
        row_header=False,
        bbox=None,
    )
    table = Item(
        self_ref="#/tables/0",
        parent=ref("#/body"),
        label="table",
        prov=[],
        captions=[],
        data=Item(num_rows=1, num_cols=2, table_cells=[first], grid=[[first, blank]]),
    )
    native = Item(iterate_items=lambda with_groups: iter([(body, 0), (table, 1)]))
    document = canonicalize_native(native, "P001", RAW, NATIVE, "c" * 64)
    table_data = document.nodes[1].table
    assert table_data.structure_state == "exact_native"
    assert len(table_data.cells) == 2
    assert table_data.cells[1].mapping.native_ref == "#/tables/0/data/grid/0/1"
    assert table_data.cells[1].text == ""
    assert table_data.grid_cell_node_ids == [[c.node_id for c in table_data.cells]]


def test_asset_verification_rejects_changed_bytes(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        verify_assets(tmp_path, PdfConfiguration())
    first = tmp_path / ASSETS[0][0]
    first.parent.mkdir(parents=True)
    first.write_bytes(b"changed")
    with pytest.raises(HashMismatchError):
        verify_assets(tmp_path, PdfConfiguration())


def test_qualified_lock_rejects_wrong_bytes(tmp_path):
    path = tmp_path / "lock.txt"
    path.write_text("unqualified")
    with pytest.raises(ValueError):
        verify_qualified_lock(path, PdfConfiguration())


def test_effective_option_drift_is_rejected():
    with pytest.raises(ValueError, match="qualified pipeline option drift"):
        verify_effective_options({"artifacts_path": "/cache"}, {}, PdfConfiguration())


@pytest.fixture
def qualified_runtime(monkeypatch, tmp_path):
    monkeypatch.setattr(pdf_module, "sys", Item(version_info=(3, 11, 15), prefix=str(tmp_path)))
    packages = dict(pdf_module.VERSIONS)
    rows = sorted(f"{name}=={version}" for name, version in packages.items())
    monkeypatch.setattr(
        pdf_module,
        "QUALIFIED_INVENTORY_CANONICAL_SHA256",
        sha256_bytes(("\n".join(rows) + "\n").encode()),
    )
    installed = packages | {"rfc8785": "0.1.4", "biorag-repro": "0.1.0"}
    monkeypatch.setattr(pdf_module.metadata, "version", lambda name: installed[name])
    monkeypatch.setattr(
        pdf_module.metadata,
        "distributions",
        lambda **_: [
            Item(metadata={"Name": name}, version=version) for name, version in installed.items()
        ],
    )
    return installed


def test_verified_runtime_inventory_retains_installed_project(qualified_runtime):
    assert pdf_module.verify_environment() == qualified_runtime


def test_qualified_runtime_rejects_python_patch_drift(qualified_runtime, monkeypatch):
    monkeypatch.setattr(pdf_module.sys, "version_info", (3, 11, 18))
    with pytest.raises(ValueError, match=r"Python drift: 3\.11\.18"):
        pdf_module.verify_environment()


@pytest.mark.parametrize("version", [None, "0.0.0"])
def test_qualified_runtime_requires_installed_project(qualified_runtime, version):
    if version is None:
        qualified_runtime.pop("biorag-repro")
    else:
        qualified_runtime["biorag-repro"] = version
    with pytest.raises(ValueError, match="self-contained project runtime is required"):
        pdf_module.verify_environment()


def test_qualified_runtime_rejects_package_version_drift(qualified_runtime):
    qualified_runtime["docling-slim"] = "0.0.0"
    with pytest.raises(ValueError, match="qualified package drift"):
        pdf_module.verify_environment()


def test_qualified_runtime_rejects_unqualified_inventory(qualified_runtime):
    qualified_runtime["unqualified-package"] = "1.0"
    with pytest.raises(ValueError, match="qualified installed package inventory drift"):
        pdf_module.verify_environment()


def test_qualified_runtime_rejects_canonicalizer_drift(qualified_runtime):
    qualified_runtime["rfc8785"] = "0.0.0"
    with pytest.raises(ValueError, match="self-contained JSON canonicalizer version drift"):
        pdf_module.verify_environment()
