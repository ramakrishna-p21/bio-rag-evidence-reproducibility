"""Qualified Docling settings and source-faithful native PDF traversal."""

import sys
from copy import deepcopy
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, verify_file_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import (
    CanonicalDocument,
    CanonicalNode,
    CharspanState,
    MappingState,
    NativeBBox,
    NativeProvenance,
    NodeKind,
    Record,
    SourceMapping,
    TableCell,
    TableData,
    TextSegment,
)

TRANSFORMATION_ID = "docling_pdf_adapter_v3"
ASSETS = (
    (
        "docling-project--docling-layout-heron/config.json",
        "fdea30805ce2f5666b147fca941dcdd27ad468e27d6ed21902207d3da056a97d",
    ),
    (
        "docling-project--docling-layout-heron/model.safetensors",
        "00333a43451945aaf89db8ca9c0a17e75d1537c17db60fdb91aa95f4c7929e0c",
    ),
    (
        "docling-project--docling-layout-heron/preprocessor_config.json",
        "cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1",
    ),
    (
        "docling-project--docling-models/model_artifacts/tableformer/accurate/tableformer_accurate.safetensors",
        "2a7d6c924b3cd12fb99a09280ca9c33a89c5d60b93253617d2e088c1a40374d9",
    ),
    (
        "docling-project--docling-models/model_artifacts/tableformer/accurate/tm_config.json",
        "984e122ceb8ccf84d84c9d2882f6f2302a44b4f1e577babd6289892c36f3cffd",
    ),
)
VERSIONS = {
    "docling-slim": "2.145.0",
    "docling-core": "2.105.0",
    "docling-parse": "7.24.2",
    "docling-ibm-models": "4.0.4",
    "torch": "2.15.0",
    "torchvision": "0.29.2",
    "pypdfium2": "5.14.0",
    "opencv-python-headless": "4.14.0.94",
}
QUALIFIED_INVENTORY_CANONICAL_SHA256 = (
    "a39c21bbc5e24830963f54757e65f54e5ee17c5e7d5b4d0e096c4c6a1eef754a"
)
QUALIFIED_LOCK_SHA256 = "8298439ddc230726d65c959e61e0aa0f847cdad65bd2cb44da7003857bcf97de"


class PdfConfiguration(Record):
    schema_version: Literal["1"] = "1"
    python_version: Literal["3.11.15"] = "3.11.15"
    docling_slim_version: Literal["2.145.0"] = "2.145.0"
    docling_core_version: Literal["2.105.0"] = "2.105.0"
    docling_parse_version: Literal["7.24.2"] = "7.24.2"
    qualified_lock_sha256: Literal[
        "8298439ddc230726d65c959e61e0aa0f847cdad65bd2cb44da7003857bcf97de"
    ] = "8298439ddc230726d65c959e61e0aa0f847cdad65bd2cb44da7003857bcf97de"
    project_rfc8785_version: Literal["0.1.4"] = "0.1.4"
    qualified_options_sha256: Literal[
        "ea907fead9260281b219bca5f42ce49f384be5cb967f5ed8030623d0c997ca98"
    ] = "ea907fead9260281b219bca5f42ce49f384be5cb967f5ed8030623d0c997ca98"
    qualified_backend_sha256: Literal[
        "2cbd70bd43815d3f42e1f555612394005ed5b8cfddc19dc562236aa97a398ce4"
    ] = "2cbd70bd43815d3f42e1f555612394005ed5b8cfddc19dc562236aa97a398ce4"
    pipeline: Literal["StandardPdfPipeline"] = "StandardPdfPipeline"
    backend: Literal["ThreadedDoclingParseDocumentBackend"] = "ThreadedDoclingParseDocumentBackend"
    ocr: Literal[False] = False
    table_mode: Literal["ACCURATE"] = "ACCURATE"
    cell_matching: Literal[True] = True
    heading_inference: Literal[False] = False
    formula_enrichment: Literal[False] = False
    image_generation: Literal[False] = False
    remote_services: Literal[False] = False
    external_plugins: Literal[False] = False
    device: Literal["cpu"] = "cpu"
    precision: Literal["float32"] = "float32"
    model_threads: Literal[1] = 1
    parser_threads: Literal[1] = 1
    offline: Literal[True] = True
    failure_policy: Literal["record_no_fallback"] = "record_no_fallback"
    heron_revision: Literal["8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"] = (
        "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8"
    )
    table_revision: Literal["fc0f2d45e2218ea24bce5045f58a389aed16dc23"] = (
        "fc0f2d45e2218ea24bce5045f58a389aed16dc23"
    )
    asset_hashes: list[str] = Field(default_factory=lambda: [x[1] for x in ASSETS])
    adapter_version: Literal["docling_pdf_adapter_v1"] = "docling_pdf_adapter_v1"

    @model_validator(mode="after")
    def qualified_assets(self) -> Self:
        if self.asset_hashes != [x[1] for x in ASSETS]:
            raise ValueError("qualified asset identities differ")
        return self


def verify_assets(root: Path, config: PdfConfiguration) -> str:
    """Hash every active model file before conversion."""
    if config.asset_hashes != [x[1] for x in ASSETS]:
        raise ValueError("asset configuration drift")
    for relative, digest in ASSETS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        verify_file_sha256(path, digest)
    return sha256_bytes(canonical_json_bytes([{"path": p, "sha256": h} for p, h in ASSETS]))


def verify_qualified_lock(path: Path, config: PdfConfiguration) -> None:
    if config.qualified_lock_sha256 != QUALIFIED_LOCK_SHA256:
        raise ValueError("qualified lock identity drift")
    verify_file_sha256(path, QUALIFIED_LOCK_SHA256)


def verify_effective_options(
    pipeline: dict[str, Any], backend: dict[str, Any], config: PdfConfiguration
) -> None:
    """Compare every effective Docling option with the qualified path-neutral dump."""
    actual = deepcopy(pipeline)
    actual["artifacts_path"] = "<assets_root>"
    engines = (
        actual.get("chart_extraction_options", {}).get("model_spec", {}).get("supported_engines")
    )
    if isinstance(engines, list):
        actual["chart_extraction_options"]["model_spec"]["supported_engines"] = sorted(engines)
    if sha256_bytes(canonical_json_bytes(actual)) != config.qualified_options_sha256:
        raise ValueError("qualified pipeline option drift")
    if sha256_bytes(canonical_json_bytes(backend)) != config.qualified_backend_sha256:
        raise ValueError("qualified backend option drift")


def verify_environment() -> dict[str, str]:
    version = ".".join(str(n) for n in sys.version_info[:3])
    if version != "3.11.15":
        raise ValueError(f"Python drift: {version}")
    actual = {name: metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS:
        raise ValueError(f"qualified package drift: {actual}")
    site = Path(sys.prefix) / "lib/python3.11/site-packages"
    packages = {
        distribution.metadata["Name"].lower().replace("_", "-"): distribution.version
        for distribution in metadata.distributions(path=[str(site)])
    }
    canonicalizer_version = packages.pop("rfc8785", None)
    project_version = packages.pop("biorag-repro", None)
    rows = sorted(f"{name}=={value}" for name, value in packages.items())
    if sha256_bytes(("\n".join(rows) + "\n").encode()) != QUALIFIED_INVENTORY_CANONICAL_SHA256:
        raise ValueError("qualified installed package inventory drift")
    if canonicalizer_version != "0.1.4" or metadata.version("rfc8785") != "0.1.4":
        raise ValueError("self-contained JSON canonicalizer version drift")
    inventory = packages | {"rfc8785": "0.1.4"}
    if project_version != "0.1.0":
        raise ValueError("self-contained project runtime is required")
    inventory["biorag-repro"] = project_version
    return inventory


def _label(item: Any) -> str:
    value = getattr(item, "label", None)
    return str(getattr(value, "value", value)) if value is not None else type(item).__name__


def _bbox(value: Any) -> NativeBBox:
    return NativeBBox(
        l=value.l,
        t=value.t,
        r=value.r,
        b=value.b,
        coord_origin=str(getattr(value.coord_origin, "value", value.coord_origin)),
    )


def _provenance(item: Any) -> list[NativeProvenance]:
    result = []
    for ordinal, entry in enumerate(getattr(item, "prov", [])):
        span = getattr(entry, "charspan", None)
        charspan = list(span) if span is not None else None
        state: CharspanState = (
            "unavailable"
            if charspan is None
            else "placeholder"
            if charspan == [0, 0]
            else "meaningful"
        )
        result.append(
            NativeProvenance(
                ordinal=ordinal,
                page_no=getattr(entry, "page_no", None),
                bbox=_bbox(entry.bbox) if getattr(entry, "bbox", None) else None,
                charspan=charspan,
                charspan_state=state,
                state="approximate",
            )
        )
    return result


def _node_id(source: ArtifactRef, ref: str) -> str:
    return "node_" + sha256_bytes(
        canonical_json_bytes(
            {
                "source_artifact_id": source.artifact_id,
                "schema_version": "4",
                "transformation_id": TRANSFORMATION_ID,
                "native_ref": ref,
            }
        )
    )


def native_classification(
    native_ref: str, label: str, content_layer: str | None, *, table_cell: bool = False
) -> tuple[NodeKind, MappingState]:
    """Map native structure without claiming independent PDF interpretation accuracy."""
    if table_cell:
        return "table_cell", "exact_native"
    if content_layer == "furniture" and native_ref != "#/furniture":
        return "page_furniture", "exact_native"
    kinds: dict[str, NodeKind] = {
        "section_header": "title",
        "text": "paragraph",
        "list_item": "list_item",
        "table": "table",
        "picture": "figure",
        "formula": "formula",
        "caption": "caption",
        "list": "list",
    }
    kind = kinds.get(label, "source_construct")
    return kind, "ambiguous" if kind == "source_construct" else "exact_native"


def canonicalize_native(
    native_document: Any,
    paper_id: str,
    source: ArtifactRef,
    native_artifact: ArtifactRef,
    configuration_sha256: str,
    saved_native: dict[str, Any] | None = None,
) -> CanonicalDocument:
    """Reconcile typed order with all saved native objects and child links."""
    walked = [item for item, _ in native_document.iterate_items(with_groups=True)]
    inventory = {}
    for name in ("body", "furniture"):
        item = getattr(native_document, name, None)
        if item is not None:
            inventory[item.self_ref] = item
    for name in ("groups", "texts", "pictures", "tables", "form_items", "key_value_items"):
        for item in getattr(native_document, name, []):
            if item.self_ref in inventory:
                raise ValueError("duplicate native object reference")
            inventory[item.self_ref] = item
    for item in walked:
        inventory.setdefault(item.self_ref, item)
    items = []
    linked: set[str] = set()

    def visit(ref: str) -> None:
        if ref in linked:
            raise ValueError("repeated or cyclic native child reference")
        item = inventory.get(ref)
        if item is None:
            raise ValueError("unresolved native child reference")
        linked.add(ref)
        items.append(item)
        for child in getattr(item, "children", []):
            visit(child.cref)

    for root in ("#/body", "#/furniture"):
        if root in inventory:
            visit(root)
    positioned = set(linked)
    for ref in inventory:
        if ref not in linked:
            visit(ref)
    refs = {item.self_ref: _node_id(source, item.self_ref) for item in items}
    if len(refs) != len(items):
        raise ValueError("duplicate native object reference")
    nodes: list[CanonicalNode] = []

    def native_field(item: Any, name: str) -> Any:
        supplied = getattr(item, "model_fields_set", None)
        if supplied is not None and name not in supplied:
            return None
        if supplied is None and name not in vars(item):
            return None
        value = getattr(item, name)
        return getattr(value, "value", value)

    for item in items:
        native_ref = item.self_ref
        label = _label(item)
        layer = getattr(item, "content_layer", None)
        layer = str(getattr(layer, "value", layer)) if layer is not None else None
        kind, interpretation = native_classification(native_ref, label, layer)
        parent_ref = getattr(getattr(item, "parent", None), "cref", None)
        if parent_ref is not None and parent_ref not in refs:
            raise ValueError("unresolved native parent link")
        caption_refs = [link.cref for link in getattr(item, "captions", [])]
        if any(ref not in refs for ref in caption_refs):
            raise ValueError("unresolved native caption link")
        caption_ids = [refs[ref] for ref in caption_refs]
        mapping = SourceMapping(
            kind="docling",
            state="exact_native",
            source_artifact_id=source.artifact_id,
            native_artifact=native_artifact,
            native_ref=native_ref,
            provenance=_provenance(item),
        )
        content = getattr(item, "text", None)
        text = content if isinstance(content, str) else None
        original = getattr(item, "orig", None)
        orig = original if isinstance(original, str) else None
        orig_qualification: (
            Literal[
                "same_as_text", "unverified_formula", "marker_inclusive", "unclassified_variant"
            ]
            | None
        ) = (
            (
                "unverified_formula"
                if kind == "formula" and orig != text
                else "marker_inclusive"
                if kind == "list_item" and orig != text
                else "same_as_text"
                if orig == text
                else "unclassified_variant"
            )
            if orig is not None
            else None
        )
        segments = [TextSegment(text=text, mapping=mapping)] if text is not None else []
        table_data = None
        if kind == "table":
            data = item.data
            native_grid = data.grid
            raw_table = (
                next(
                    table["data"]
                    for table in saved_native["tables"]
                    if table["self_ref"] == native_ref
                )
                if saved_native is not None
                else None
            )
            orientation = (
                raw_table.get("orientation")
                if raw_table is not None
                else native_field(data, "orientation")
            )
            cells: list[TableCell] = []
            grid_identity: dict[int, str] = {}

            def add_cell(
                cell: Any,
                cell_ref: str,
                cells: list[TableCell],
                grid_identity: dict[int, str],
                raw_cell: dict[str, Any] | None = None,
            ) -> None:
                cell_mapping = SourceMapping(
                    kind="docling",
                    state="exact_native",
                    source_artifact_id=source.artifact_id,
                    native_artifact=native_artifact,
                    native_ref=cell_ref,
                    provenance=[
                        NativeProvenance(
                            ordinal=0,
                            bbox=_bbox(cell.bbox),
                            page_no=None,
                            charspan=None,
                            charspan_state="unavailable",
                            state="ambiguous",
                        )
                    ]
                    if getattr(cell, "bbox", None)
                    else [],
                )
                cell_id = _node_id(source, cell_ref)
                grid_identity[id(cell)] = cell_id
                row_header = (
                    raw_cell.get("row_header")
                    if raw_cell is not None
                    else native_field(cell, "row_header")
                )
                column_header = (
                    raw_cell.get("column_header")
                    if raw_cell is not None
                    else native_field(cell, "column_header")
                )
                cells.append(
                    TableCell(
                        node_id=cell_id,
                        row=cell.start_row_offset_idx,
                        column=cell.start_col_offset_idx,
                        row_span=cell.row_span,
                        column_span=cell.col_span,
                        text=cell.text,
                        is_header=(
                            bool(row_header or column_header)
                            if row_header is not None and column_header is not None
                            else None
                        ),
                        native_row_header=row_header,
                        native_column_header=column_header,
                        native_row_section=(
                            raw_cell.get("row_section")
                            if raw_cell is not None
                            else native_field(cell, "row_section")
                        ),
                        native_fillable=(
                            raw_cell.get("fillable")
                            if raw_cell is not None
                            else native_field(cell, "fillable")
                        ),
                        mapping=cell_mapping,
                    )
                )

            for index, cell in enumerate(data.table_cells):
                add_cell(
                    cell,
                    f"{native_ref}/data/table_cells/{index}",
                    cells,
                    grid_identity,
                    raw_table["table_cells"][index] if raw_table is not None else None,
                )
            for row_index, row in enumerate(native_grid):
                for column_index, cell in enumerate(row):
                    if cell is not None and id(cell) not in grid_identity:
                        add_cell(
                            cell,
                            f"{native_ref}/data/grid/{row_index}/{column_index}",
                            cells,
                            grid_identity,
                            raw_table["grid"][row_index][column_index]
                            if raw_table is not None
                            else None,
                        )
            grid = [
                [grid_identity.get(id(cell)) if cell is not None else None for cell in row]
                for row in native_grid
            ]
            grid_complete = len(grid) == data.num_rows and all(
                len(row) == data.num_cols and all(cell_id is not None for cell_id in row)
                for row in grid
            )
            table_data = TableData(
                structure_state="exact_native" if grid_complete else "parser_damaged",
                row_count=data.num_rows,
                column_count=data.num_cols,
                native_orientation=orientation,
                cells=cells,
                grid_cell_node_ids=grid,
                caption_node_ids=caption_ids,
                caption_relation_state="present" if caption_ids else "unavailable",
                footnote_relation_state="unavailable",
            )
        node = CanonicalNode(
            node_id=refs[native_ref],
            parent_node_id=refs.get(parent_ref),
            order=len(nodes),
            kind=kind,
            source_tag=type(item).__name__,
            native_label=label,
            native_content_layer=layer,
            native_child_node_ids=[refs[child.cref] for child in getattr(item, "children", [])],
            native_order_state="linked" if native_ref in positioned else "unpositioned",
            native_orig=orig,
            orig_qualification=orig_qualification,
            native_marker=getattr(item, "marker", None),
            native_enumerated=getattr(item, "enumerated", None),
            native_heading_level=getattr(item, "level", None),
            native_group_name=getattr(item, "name", None),
            interpretation_state=interpretation,
            mapping=mapping,
            segments=segments,
            target_node_ids=caption_ids if kind == "figure" else [],
            relation_state=("exact_native" if caption_ids else "unavailable")
            if kind == "figure"
            else None,
            content_state=("unverified_native_rendition" if orig else "marker_only")
            if kind == "formula" and not text
            else "present",
            table=table_data,
        )
        nodes.append(node)
        if table_data:
            for cell in table_data.cells:
                cell_kind, cell_interpretation = native_classification(
                    cell.mapping.native_ref or "", "table_cell", None, table_cell=True
                )
                nodes.append(
                    CanonicalNode(
                        node_id=cell.node_id,
                        parent_node_id=node.node_id,
                        order=len(nodes),
                        kind=cell_kind,
                        source_tag="TableCell",
                        native_label="table_cell",
                        interpretation_state=cell_interpretation,
                        mapping=cell.mapping,
                        segments=[TextSegment(text=cell.text, mapping=cell.mapping)],
                    )
                )
    return CanonicalDocument(
        paper_id=paper_id,
        representation="PDF_DOCLING",
        source=source,
        native_artifact=native_artifact,
        transformation_id=TRANSFORMATION_ID,
        configuration_sha256=configuration_sha256,
        status="complete",
        nodes=nodes,
    )
