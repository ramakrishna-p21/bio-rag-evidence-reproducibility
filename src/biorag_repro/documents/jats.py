"""Deterministic, source-faithful JATS traversal."""

import re
from collections import Counter
from xml.etree import ElementTree as ET

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import (
    CanonicalDocument,
    CanonicalNode,
    ContentState,
    MappingState,
    NodeKind,
    SourceMapping,
    TableCell,
    TableData,
    TextSegment,
)

TRANSFORMATION_ID = "jats_adapter_v2"
MAX_XML_BYTES = 64 * 1024 * 1024
_ENTITY = re.compile(rb"<!\s*ENTITY\b", re.IGNORECASE)
_DOCTYPE = re.compile(rb"<!\s*DOCTYPE\b", re.IGNORECASE)
_KINDS: dict[str, NodeKind] = {
    "article": "article",
    "front": "front",
    "body": "body",
    "back": "back",
    "sec": "section",
    "title": "title",
    "article-title": "title",
    "p": "paragraph",
    "list": "list",
    "list-item": "list_item",
    "table": "table",
    "tr": "table_row",
    "td": "table_cell",
    "th": "table_cell",
    "caption": "caption",
    "fig": "figure",
    "disp-formula": "formula",
    "inline-formula": "formula",
    "ref": "reference",
    "xref": "xref",
    "supplementary-material": "supplement",
}
_TEXT_KINDS = {"title", "paragraph", "table_cell", "caption", "formula", "xref", "reference"}


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _node_id(source_id: str, path: str) -> str:
    payload = {
        "source_artifact_id": source_id,
        "transformation_id": TRANSFORMATION_ID,
        "schema_version": "2",
        "path": path,
    }
    return "node_" + sha256_bytes(canonical_json_bytes(payload))


def _paths(root: ET.Element) -> dict[ET.Element, str]:
    result: dict[ET.Element, str] = {}

    def visit(element: ET.Element, path: str) -> None:
        result[element] = path
        counts: Counter[str] = Counter()
        for child in element:
            counts[child.tag] += 1
            visit(child, f"{path}/{child.tag}[{counts[child.tag]}]")

    visit(root, f"/{root.tag}[1]")
    return result


def _segments(
    element: ET.Element, paths: dict[ET.Element, str], source_id: str
) -> list[TextSegment]:
    result: list[TextSegment] = []

    def add(text: str | None, owner: ET.Element, slot: str) -> None:
        if text is not None:
            result.append(
                TextSegment(
                    text=text,
                    mapping=SourceMapping(
                        kind="jats",
                        state="exact_native",
                        source_artifact_id=source_id,
                        path=f"{paths[owner]}/{slot}",
                        segment_ordinal=len(result),
                    ),
                )
            )

    def walk(current: ET.Element) -> None:
        add(current.text, current, "text()")
        for child in current:
            walk(child)
            add(child.tail, child, "tail()")

    walk(element)
    return result


def _table(
    element: ET.Element,
    paths: dict[ET.Element, str],
    source_id: str,
    node_ids: dict[ET.Element, str],
    parent: ET.Element | None,
) -> TableData:
    rows = [item for item in element.iter() if _local(item.tag) == "tr"]
    occupied: set[tuple[int, int]] = set()
    cells: list[TableCell] = []
    width = 0
    for row_no, row in enumerate(rows):
        column = 0
        for child in row:
            if _local(child.tag) not in {"td", "th"}:
                continue
            while (row_no, column) in occupied:
                column += 1
            try:
                row_span = int(child.attrib.get("rowspan", "1"))
                column_span = int(child.attrib.get("colspan", "1"))
            except ValueError as exc:
                raise ValueError("noninteger table span") from exc
            if row_span < 1 or column_span < 1:
                raise ValueError("nonpositive table span")
            if row_no + row_span > len(rows):
                raise ValueError("table row span exceeds rows")
            slots = {
                (r, c)
                for r in range(row_no, row_no + row_span)
                for c in range(column, column + column_span)
            }
            if occupied & slots:
                raise ValueError("overlapping table cells")
            occupied.update(slots)
            cells.append(
                TableCell(
                    node_id=node_ids[child],
                    row=row_no,
                    column=column,
                    row_span=row_span,
                    column_span=column_span,
                    text="".join(part.text for part in _segments(child, paths, source_id)),
                    is_header=_local(child.tag) == "th",
                    mapping=SourceMapping(
                        kind="jats",
                        state="exact_native",
                        source_artifact_id=source_id,
                        path=paths[child],
                    ),
                )
            )
            column += column_span
            width = max(width, column)
    wrapper = parent if parent is not None and _local(parent.tag) == "table-wrap" else element
    captions = [node_ids[item] for item in wrapper if _local(item.tag) == "caption"]
    footnotes = [node_ids[item] for item in wrapper if _local(item.tag) == "table-wrap-foot"]
    return TableData(
        structure_state="exact_native" if rows and cells else "unavailable",
        row_count=len(rows),
        column_count=width,
        cells=cells,
        caption_node_ids=captions,
        footnote_node_ids=footnotes,
        caption_relation_state="present" if captions else "absent",
        footnote_relation_state="present" if footnotes else "absent",
    )


def ingest_jats(
    content: bytes,
    *,
    paper_id: str,
    source_artifact_id: str,
    source_sha256: str,
    source_logical_ref: str,
    configuration_sha256: str,
) -> CanonicalDocument:
    """Map verified original XML to canonical nodes without scientific cleanup."""
    if sha256_bytes(content) != source_sha256:
        raise ValueError("source hash mismatch")
    if source_artifact_id != raw_source_artifact_id(
        paper_id=paper_id, representation="JATS", sha256=source_sha256
    ):
        raise ValueError("source artifact ID mismatch")
    if len(content) > MAX_XML_BYTES:
        raise ValueError("JATS source exceeds bounded parser size")
    # ElementTree does not fetch an external DTD. Reject entity declarations so
    # source content cannot depend on internal expansion or external resolution.
    if _ENTITY.search(content):
        raise ValueError("entity declarations are unsupported")
    try:
        root = ET.fromstring(content)
    except ET.ParseError as exc:
        raise ValueError("malformed or unresolved JATS XML") from exc
    if _local(root.tag) != "article" or not any(_local(c.tag) == "body" for c in root):
        raise ValueError("JATS article body is required")

    paths = _paths(root)
    node_ids = {element: _node_id(source_artifact_id, path) for element, path in paths.items()}
    by_xml_id = {
        element.attrib["id"]: node_ids[element] for element in paths if "id" in element.attrib
    }
    if len(by_xml_id) != sum("id" in element.attrib for element in paths):
        raise ValueError("duplicate XML ID")
    parents = {child: parent for parent in paths for child in parent}
    nodes: list[CanonicalNode] = []
    for element, path in paths.items():
        tag = _local(element.tag)
        kind = _KINDS.get(tag, "source_construct")
        content_state: ContentState
        if kind == "formula" and not "".join(element.itertext()).strip():
            content_state = "marker_only" if len(element) else "missing"
        else:
            content_state = "unsupported" if kind == "source_construct" else "present"
        rid = element.attrib.get("rid") if tag == "xref" else None
        target_ids = (
            [by_xml_id[target] for target in rid.split() if target in by_xml_id] if rid else []
        )
        relation: MappingState | None = (
            ("exact_native" if len(target_ids) == len(rid.split()) else "unavailable")
            if rid
            else None
        )
        nodes.append(
            CanonicalNode(
                node_id=node_ids[element],
                parent_node_id=node_ids[parents[element]] if element in parents else None,
                order=len(nodes),
                kind=kind,
                source_tag=element.tag,
                source_id=element.attrib.get("id"),
                attributes=dict(element.attrib),
                segments=(
                    _segments(element, paths, source_artifact_id)
                    if kind in _TEXT_KINDS or not len(element)
                    else []
                ),
                mapping=SourceMapping(
                    kind="jats",
                    state="exact_native",
                    source_artifact_id=source_artifact_id,
                    path=path,
                ),
                content_state=content_state,
                rid=rid,
                ref_type=element.attrib.get("ref-type") if tag == "xref" else None,
                target_node_ids=target_ids,
                relation_state=relation,
                table=_table(element, paths, source_artifact_id, node_ids, parents.get(element))
                if tag == "table"
                else None,
            )
        )
    return CanonicalDocument(
        paper_id=paper_id,
        representation="JATS",
        source=ArtifactRef(
            artifact_id=source_artifact_id, logical_ref=source_logical_ref, sha256=source_sha256
        ),
        transformation_id=TRANSFORMATION_ID,
        configuration_sha256=configuration_sha256,
        source_doctype_declared=bool(_DOCTYPE.search(content)),
        status="complete",
        nodes=nodes,
    )
