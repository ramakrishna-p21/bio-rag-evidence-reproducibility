"""Independent coverage checks: every source unit appears once in the canonical document."""

from typing import Any
from xml.etree import ElementTree as ET

from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.documents.pdf import native_classification


def jats_paths(content: bytes) -> dict[str, ET.Element]:
    root = ET.fromstring(content)
    paths: dict[str, ET.Element] = {}

    def visit(element: ET.Element, path: str) -> None:
        paths[path] = element
        counts: dict[str, int] = {}
        for child in element:
            counts[child.tag] = counts.get(child.tag, 0) + 1
            visit(child, f"{path}/{child.tag}[{counts[child.tag]}]")

    visit(root, f"/{root.tag}[1]")
    return paths


def resolve_native_ref(data: object, reference: str) -> object:
    if not reference.startswith("#/"):
        raise ValueError("invalid native object reference")
    current = data
    for part in reference[2:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            current = current[int(key)]
        elif isinstance(current, dict):
            current = current[key]
        else:
            raise TypeError("unresolved native object reference")
    return current


def validate_jats_coverage(doc: CanonicalDocument, content: bytes) -> tuple[int, int, int]:
    """Compare canonical XML paths and slots with the original XML inventory."""
    paths = jats_paths(content)
    nodes = {node.mapping.path: node for node in doc.nodes if node.mapping.path is not None}
    if set(nodes) != set(paths) or len(nodes) != len(doc.nodes):
        raise ValueError("JATS source element coverage differs")
    source_slots = {}
    for path, element in paths.items():
        for name, value in (("text()", element.text), ("tail()", element.tail)):
            if value is not None:
                source_slots[f"{path}/{name}"] = value
    represented: dict[str, list[str]] = {}
    for path, node in nodes.items():
        element = paths[path]
        if (
            node.source_tag != element.tag
            or node.source_id != element.attrib.get("id")
            or node.attributes != element.attrib
            or node.mapping.state != "exact_native"
        ):
            raise ValueError("JATS source element differs")
        for ordinal, part in enumerate(node.segments):
            slot = part.mapping.path
            if slot is None:
                raise ValueError("JATS source slot missing path")
            if (
                slot not in source_slots
                or part.text != source_slots[slot]
                or part.mapping.segment_ordinal != ordinal
            ):
                raise ValueError("JATS source slot differs")
            represented.setdefault(slot, []).append(node.node_id)
    if set(represented) != set(source_slots):
        raise ValueError("JATS source slot coverage differs")
    ordered = list(paths)
    if [node.mapping.path for node in doc.nodes] != ordered:
        raise ValueError("JATS source element order differs")
    path_by_element = {element: path for path, element in paths.items()}
    parent_by_element = {child: parent for parent in paths.values() for child in parent}

    def subtree_slots(element: ET.Element) -> list[str]:
        result = []
        own = f"{path_by_element[element]}/text()"
        if own in source_slots:
            result.append(own)
        for child in element:
            result.extend(subtree_slots(child))
            tail = f"{path_by_element[child]}/tail()"
            if tail in source_slots:
                result.append(tail)
        return result

    xml_id_nodes = {
        element.attrib["id"]: nodes[path].node_id
        for path, element in paths.items()
        if "id" in element.attrib
    }
    for path, element in paths.items():
        parent = parent_by_element.get(element)
        expected_parent = nodes[path_by_element[parent]].node_id if parent is not None else None
        if nodes[path].parent_node_id != expected_parent:
            raise ValueError("JATS source parent differs")
        direct = [f"{path}/text()"] + [f"{path_by_element[child]}/tail()" for child in element]
        direct = [slot for slot in direct if slot in source_slots]
        actual = [part.mapping.path for part in nodes[path].segments if part.mapping.path in direct]
        if actual != direct:
            # Semantic nodes may also contain descendant slots, but their direct
            # slots still appear in their original mixed-content order.
            raise ValueError("JATS mixed content source order differs")
        if nodes[path].kind in {
            "title",
            "paragraph",
            "table_cell",
            "caption",
            "formula",
            "xref",
            "reference",
        } and [part.mapping.path for part in nodes[path].segments] != subtree_slots(element):
            raise ValueError("JATS subtree mixed content order differs")
        tag = element.tag.rsplit("}", 1)[-1]
        if tag in {"disp-formula", "inline-formula"} and nodes[path].kind != "formula":
            raise ValueError("JATS formula structure differs")
        if tag == "xref" and (
            nodes[path].rid != element.attrib.get("rid")
            or nodes[path].ref_type != element.attrib.get("ref-type")
        ):
            raise ValueError("JATS xref fields differ")
        if tag == "xref":
            rid = element.attrib.get("rid", "")
            targets = [xml_id_nodes[target] for target in rid.split() if target in xml_id_nodes]
            if nodes[path].target_node_ids != targets:
                raise ValueError("JATS xref targets differ")
        if tag == "table":
            table = nodes[path].table
            if table is None:
                raise ValueError("JATS table omitted")
            rows = [child for child in element.iter() if child.tag.rsplit("}", 1)[-1] == "tr"]
            source_cells = [
                cell for row in rows for cell in row if cell.tag.rsplit("}", 1)[-1] in {"td", "th"}
            ]
            by_id = {node.node_id: node for node in doc.nodes}
            if table.row_count != len(rows) or len(table.cells) != len(source_cells):
                raise ValueError("JATS table cell coverage differs")
            wrapper = parent_by_element.get(element)
            wrapper = (
                wrapper
                if wrapper is not None and wrapper.tag.rsplit("}", 1)[-1] == "table-wrap"
                else element
            )
            captions = [
                nodes[path_by_element[child]].node_id
                for child in wrapper
                if child.tag.rsplit("}", 1)[-1] == "caption"
            ]
            footnotes = [
                nodes[path_by_element[child]].node_id
                for child in wrapper
                if child.tag.rsplit("}", 1)[-1] == "table-wrap-foot"
            ]
            if table.caption_node_ids != captions or table.footnote_node_ids != footnotes:
                raise ValueError("JATS table relations differ")
            for source_cell, cell in zip(source_cells, table.cells, strict=True):
                if (
                    by_id[cell.node_id].mapping.path != path_by_element[source_cell]
                    or cell.row_span != int(source_cell.attrib.get("rowspan", "1"))
                    or cell.column_span != int(source_cell.attrib.get("colspan", "1"))
                    or cell.text != "".join(source_cell.itertext())
                ):
                    raise ValueError("JATS table cell differs")
    return len(paths), len(source_slots), sum(len(refs) > 1 for refs in represented.values())


def pdf_objects(data: dict[str, Any]) -> dict[str, Any]:
    """Enumerate objects from saved native roots and arrays, not canonical kinds."""
    source: dict[str, Any] = {}
    for root in ("body", "furniture"):
        if isinstance(data.get(root), dict):
            source[data[root]["self_ref"]] = data[root]
    for array in ("groups", "texts", "pictures", "tables", "form_items", "key_value_items"):
        for item in data.get(array, []):
            ref = item["self_ref"]
            if ref in source:
                raise ValueError("duplicate native object")
            source[ref] = item
    return source


def validate_pdf_coverage(doc: CanonicalDocument, data: dict[str, Any]) -> tuple[int, int, int]:
    """Compare saved native objects and required fields against canonical nodes."""
    if data.get("form_items") or data.get("key_value_items"):
        raise ValueError("unsupported populated native structured item")
    source = pdf_objects(data)
    all_by_ref = {node.mapping.native_ref: node for node in doc.nodes}
    if len(all_by_ref) != len(doc.nodes) or not set(source) <= set(all_by_ref):
        raise ValueError("native object coverage differs")

    def check_classification(ref: str, item: dict[str, Any], *, table_cell: bool = False) -> None:
        node = all_by_ref.get(ref)
        expected = native_classification(
            ref,
            "table_cell" if table_cell else item["label"],
            item.get("content_layer"),
            table_cell=table_cell,
        )
        actual = (node.kind, node.interpretation_state) if node else None
        if actual != expected:
            raise ValueError(
                f"native classification differs at {ref}: expected {expected}, got {actual}"
            )
        assert node is not None
        if (
            node.mapping.source_artifact_id != doc.source.artifact_id
            or node.mapping.native_artifact != doc.native_artifact
        ):
            raise ValueError(f"native classification source identity differs at {ref}")

    for ref, item in source.items():
        check_classification(ref, item)
    by_ref = {ref: all_by_ref[ref] for ref in source}
    positioned = []
    seen = set()

    def visit(ref: str) -> None:
        if ref in seen or ref not in source:
            raise ValueError("native child relation invalid")
        seen.add(ref)
        positioned.append(ref)
        for child in source[ref].get("children", []):
            visit(child["$ref"])

    for root in ("#/body", "#/furniture"):
        if root in source:
            visit(root)
    expected_order = positioned + [ref for ref in source if ref not in seen]
    actual_order = [
        node.mapping.native_ref for node in doc.nodes if node.mapping.native_ref in source
    ]
    if actual_order != expected_order:
        raise ValueError("native reading order differs")
    table_cell_count = 0
    expected_refs = set(source)
    for ref, item in source.items():
        node = by_ref[ref]
        parent_ref = (item.get("parent") or {}).get("$ref")
        if parent_ref is not None and parent_ref not in by_ref:
            raise ValueError("native parent unresolved")
        if node.parent_node_id != (by_ref[parent_ref].node_id if parent_ref else None):
            raise ValueError("native parent differs")
        children = [link["$ref"] for link in item.get("children", [])]
        if node.native_child_node_ids != [by_ref[child].node_id for child in children]:
            raise ValueError("native children differ")
        if node.native_order_state != ("linked" if ref in seen else "unpositioned"):
            raise ValueError("native order state differs")
        if node.native_label != item.get("label") or node.native_content_layer != item.get(
            "content_layer"
        ):
            raise ValueError("native label or layer differs")
        if node.native_orig != item.get("orig"):
            raise ValueError("native orig differs")
        original = item.get("orig")
        original_text = item.get("text")
        expected_qualification = (
            (
                "unverified_formula"
                if item.get("label") == "formula" and original != original_text
                else "marker_inclusive"
                if item.get("label") == "list_item" and original != original_text
                else "same_as_text"
                if original == original_text
                else "unclassified_variant"
            )
            if original is not None
            else None
        )
        if node.orig_qualification != expected_qualification:
            raise ValueError("native orig qualification differs")
        if item.get("label") == "formula" and not original_text:
            state = "unverified_native_rendition" if original else "marker_only"
            if node.content_state != state:
                raise ValueError("native formula state differs")
        if node.native_marker != item.get("marker") or node.native_enumerated != item.get(
            "enumerated"
        ):
            raise ValueError("native list fields differ")
        if node.native_heading_level != item.get("level") or node.native_group_name != item.get(
            "name"
        ):
            raise ValueError("native heading or group field differs")
        if "text" in item and (len(node.segments) != 1 or node.segments[0].text != item["text"]):
            raise ValueError("native text differs")
        if "text" not in item and node.segments:
            raise ValueError("invented native text")
        if len(node.mapping.provenance) != len(item.get("prov", [])):
            raise ValueError("native provenance coverage differs")
        for index, native_prov in enumerate(item.get("prov", [])):
            record = node.mapping.provenance[index]
            span = native_prov.get("charspan")
            expected_state = (
                "unavailable" if span is None else "placeholder" if span == [0, 0] else "meaningful"
            )
            if (
                record.page_no != native_prov.get("page_no")
                or record.charspan != native_prov.get("charspan")
                or record.charspan_state != expected_state
                or record.ordinal != index
                or record.state != "approximate"
                or (record.bbox.model_dump() if record.bbox else None) != native_prov.get("bbox")
            ):
                raise ValueError("native provenance differs")
        if item.get("annotations") or item.get("references") or item.get("footnotes"):
            raise ValueError("unsupported populated native relation")
        captions = [link["$ref"] for link in item.get("captions", [])]
        if node.kind == "figure" and node.target_node_ids != [
            by_ref[target].node_id for target in captions
        ]:
            raise ValueError("native figure captions differ")
        if node.kind == "table":
            if node.table is None:
                raise ValueError("native table structure omitted")
            table = item["data"]
            if (node.table.row_count, node.table.column_count) != (
                table["num_rows"],
                table["num_cols"],
            ):
                raise ValueError("native table dimensions differ")
            if node.table.caption_node_ids != [by_ref[target].node_id for target in captions]:
                raise ValueError("native table captions differ")
            if node.table.caption_relation_state != ("present" if captions else "unavailable"):
                raise ValueError("native table caption state differs")
            if node.table.footnote_node_ids or node.table.footnote_relation_state != "unavailable":
                raise ValueError("native table footnote relation was inferred")
            if node.table.native_orientation != table.get("orientation"):
                raise ValueError("native table orientation differs")
            expected_cells = {
                f"{ref}/data/table_cells/{i}": cell
                for i, cell in enumerate(table.get("table_cells", []))
            }
            for row_index, row in enumerate(table.get("grid", [])):
                for column_index, cell in enumerate(row):
                    if cell is not None and cell not in expected_cells.values():
                        expected_cells[f"{ref}/data/grid/{row_index}/{column_index}"] = cell
            actual_cells = {cell.mapping.native_ref: cell for cell in node.table.cells}
            table_cell_count += len(expected_cells)
            expected_refs.update(expected_cells)
            if set(actual_cells) != set(expected_cells):
                raise ValueError("native table cell coverage differs")
            for cell_ref, native_cell in expected_cells.items():
                check_classification(cell_ref, native_cell, table_cell=True)
                cell = actual_cells[cell_ref]
                row_header = native_cell.get("row_header")
                column_header = native_cell.get("column_header")
                expected_header = (
                    bool(row_header or column_header)
                    if row_header is not None and column_header is not None
                    else None
                )
                if (
                    cell.text != native_cell.get("text")
                    or cell.row != native_cell.get("start_row_offset_idx")
                    or cell.column != native_cell.get("start_col_offset_idx")
                    or cell.row_span != native_cell.get("row_span")
                    or cell.column_span != native_cell.get("col_span")
                    or cell.native_row_header != row_header
                    or cell.native_column_header != column_header
                    or cell.native_row_section != native_cell.get("row_section")
                    or cell.native_fillable != native_cell.get("fillable")
                    or cell.is_header != expected_header
                ):
                    raise ValueError("native table cell differs")
                if (
                    native_cell.get("end_row_offset_idx") != cell.row + cell.row_span
                    or native_cell.get("end_col_offset_idx") != cell.column + cell.column_span
                ):
                    raise ValueError("native table end offsets differ")
                expected_bbox = native_cell.get("bbox")
                provenance = cell.mapping.provenance
                if expected_bbox is None:
                    if provenance:
                        raise ValueError("invented native table cell provenance")
                elif (
                    len(provenance) != 1
                    or provenance[0].bbox is None
                    or provenance[0].bbox.model_dump() != expected_bbox
                    or provenance[0].page_no is not None
                    or provenance[0].charspan is not None
                    or provenance[0].state != "ambiguous"
                ):
                    raise ValueError("native table cell provenance differs")
            grid = table.get("grid", [])
            if len(grid) != len(node.table.grid_cell_node_ids):
                raise ValueError("native table grid differs")
            for row, canonical_row in zip(grid, node.table.grid_cell_node_ids, strict=True):
                if len(row) != len(canonical_row):
                    raise ValueError("native table grid differs")
                for native_cell, cell_id in zip(row, canonical_row, strict=True):
                    if native_cell is None:
                        if cell_id is not None:
                            raise ValueError("invented native grid cell")
                        continue
                    matches = [
                        cell_ref
                        for cell_ref, value in expected_cells.items()
                        if value == native_cell
                    ]
                    if len(matches) != 1 or cell_id != actual_cells[matches[0]].node_id:
                        raise ValueError("native table grid differs or is ambiguous")
    if set(all_by_ref) != expected_refs:
        raise ValueError("native object or cell coverage differs")
    for node in doc.nodes:
        native_ref = node.mapping.native_ref
        if native_ref is None or resolve_native_ref(data, native_ref) is None:
            raise ValueError("invented native reference")
    return len(source), len(data.get("texts", [])), table_cell_count
