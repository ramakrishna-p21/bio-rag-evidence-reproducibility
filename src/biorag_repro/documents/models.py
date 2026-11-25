"""Typed source-faithful canonical document contract."""

from typing import Literal, Self

from pydantic import ConfigDict, Field, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record as CoreRecord

MappingState = Literal[
    "exact_native", "derived", "approximate", "unavailable", "ambiguous", "parser_damaged", "failed"
]
RelationState = Literal["present", "absent", "unavailable", "ambiguous", "parser_damaged"]
CharspanState = Literal["meaningful", "placeholder", "unavailable", "ambiguous", "parser_damaged"]
ContentState = Literal[
    "present", "marker_only", "missing", "unsupported", "unverified_native_rendition"
]
NodeKind = Literal[
    "article",
    "front",
    "body",
    "back",
    "section",
    "title",
    "paragraph",
    "list",
    "list_item",
    "table",
    "table_row",
    "table_cell",
    "caption",
    "figure",
    "formula",
    "reference",
    "xref",
    "supplement",
    "source_construct",
    "page_furniture",
]


class Record(CoreRecord):
    """The shared strict base, and no NaN or infinity anywhere in a canonical document."""

    model_config = ConfigDict(allow_inf_nan=False)


class JatsConfiguration(Record):
    schema_version: Literal["1"] = "1"
    entity_policy: Literal["reject_declarations"] = "reject_declarations"
    source_faithful: Literal[True] = True


class NativeBBox(Record):
    l: float
    t: float
    r: float
    b: float
    coord_origin: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_coordinates(self) -> Self:
        if self.l > self.r:
            raise ValueError("bbox left exceeds right")
        return self


class NativeProvenance(Record):
    ordinal: int = Field(ge=0)
    page_no: int | None = Field(default=None, ge=1)
    bbox: NativeBBox | None = None
    charspan: list[int] | None = Field(default=None, min_length=2, max_length=2)
    charspan_state: CharspanState
    state: MappingState

    @model_validator(mode="after")
    def valid_location(self) -> Self:
        if self.charspan is None and self.charspan_state not in {"unavailable", "parser_damaged"}:
            raise ValueError("missing charspan requires unavailable or damaged state")
        if self.charspan is not None:
            start, end = self.charspan
            if start < 0 or end < start:
                raise ValueError("invalid native charspan")
            if self.charspan == [0, 0] and self.charspan_state == "meaningful":
                raise ValueError("placeholder charspan cannot be exact")
        return self


class SourceMapping(Record):
    kind: Literal["jats", "docling"]
    state: MappingState
    source_artifact_id: str = Field(min_length=1)
    path: str | None = None
    segment_ordinal: int | None = Field(default=None, ge=0)
    native_artifact: ArtifactRef | None = None
    native_ref: str | None = None
    provenance: list[NativeProvenance] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_arm(self) -> Self:
        if self.kind == "jats":
            if not self.path or self.native_artifact or self.native_ref or self.provenance:
                raise ValueError("JATS mapping requires XML path and no PDF provenance")
        elif (
            self.path is not None
            or self.segment_ordinal is not None
            or self.native_artifact is None
            or not self.native_ref
        ):
            raise ValueError("Docling mapping requires saved native artifact and object ref")
        if [p.ordinal for p in self.provenance] != list(range(len(self.provenance))):
            raise ValueError("native provenance ordinals must follow native order")
        return self


class TextSegment(Record):
    text: str
    mapping: SourceMapping


class TableCell(Record):
    node_id: str = Field(min_length=1)
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(ge=1)
    column_span: int = Field(ge=1)
    text: str
    is_header: bool | None
    native_row_header: bool | None = None
    native_column_header: bool | None = None
    native_row_section: bool | None = None
    native_fillable: bool | None = None
    mapping: SourceMapping

    @model_validator(mode="after")
    def valid_header(self) -> Self:
        roles = (self.native_row_header, self.native_column_header)
        if self.mapping.kind == "jats":
            if type(self.is_header) is not bool or any(
                value is not None
                for value in (*roles, self.native_row_section, self.native_fillable)
            ):
                raise ValueError("JATS cell cannot claim Docling table fields")
        elif self.is_header != (any(roles) if all(value is not None for value in roles) else None):
            raise ValueError("derived header differs from native roles")
        return self


class TableData(Record):
    structure_state: MappingState
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    native_orientation: Literal["rot_0", "rot_90", "rot_180", "rot_270"] | None = None
    cells: list[TableCell]
    grid_cell_node_ids: list[list[str | None]] = Field(default_factory=list)
    caption_node_ids: list[str] = Field(default_factory=list)
    footnote_node_ids: list[str] = Field(default_factory=list)
    caption_relation_state: RelationState = "absent"
    footnote_relation_state: RelationState = "absent"

    @model_validator(mode="after")
    def valid_grid(self) -> Self:
        cell_ids = {cell.node_id for cell in self.cells}
        if any(
            cell_id is not None and cell_id not in cell_ids
            for row in self.grid_cell_node_ids
            for cell_id in row
        ):
            raise ValueError("grid references unknown table cell")
        if (
            self.grid_cell_node_ids
            and self.structure_state == "exact_native"
            and (
                len(self.grid_cell_node_ids) != self.row_count
                or any(
                    len(row) != self.column_count or any(cell is None for cell in row)
                    for row in self.grid_cell_node_ids
                )
            )
        ):
            raise ValueError("exact native grid is incomplete")
        for ids, state in (
            (self.caption_node_ids, self.caption_relation_state),
            (self.footnote_node_ids, self.footnote_relation_state),
        ):
            if (state == "present") != bool(ids):
                raise ValueError("relation IDs inconsistent with state")
        occupied: set[tuple[int, int]] = set()
        for cell in self.cells:
            if (
                cell.row < 0
                or cell.column < 0
                or cell.row_span < 1
                or cell.column_span < 1
                or cell.row + cell.row_span > self.row_count
                or cell.column + cell.column_span > self.column_count
            ):
                raise ValueError("invalid table cell span")
            slots = {
                (r, c)
                for r in range(cell.row, cell.row + cell.row_span)
                for c in range(cell.column, cell.column + cell.column_span)
            }
            if occupied & slots:
                raise ValueError("overlapping table cells")
            occupied.update(slots)
        return self


class CanonicalNode(Record):
    node_id: str = Field(min_length=1)
    parent_node_id: str | None
    order: int = Field(ge=0)
    kind: NodeKind
    source_tag: str
    native_label: str | None = None
    native_content_layer: str | None = None
    native_child_node_ids: list[str] = Field(default_factory=list)
    native_order_state: Literal["linked", "unpositioned"] | None = None
    native_orig: str | None = None
    orig_qualification: (
        Literal["same_as_text", "unverified_formula", "marker_inclusive", "unclassified_variant"]
        | None
    ) = None
    native_marker: str | None = None
    native_enumerated: bool | None = None
    native_heading_level: int | None = None
    native_group_name: str | None = None
    interpretation_state: MappingState | None = None
    source_id: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    segments: list[TextSegment] = Field(default_factory=list)
    mapping: SourceMapping
    content_state: ContentState = "present"
    rid: str | None = None
    ref_type: str | None = None
    target_node_ids: list[str] = Field(default_factory=list)
    relation_state: MappingState | None = None
    table: TableData | None = None
    figure_footnote_node_ids: list[str] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class CanonicalDocument(Record):
    schema_version: Literal["4", "5"] = "4"
    paper_id: str
    representation: Literal["JATS", "PDF_DOCLING"]
    source: ArtifactRef
    native_artifact: ArtifactRef | None = None
    source_doctype_declared: bool | None = None
    transformation_id: str
    configuration_sha256: str
    status: Literal["complete", "incomplete"]
    nodes: list[CanonicalNode]

    @model_validator(mode="after")
    def valid_document(self) -> Self:
        validate_sha256(self.configuration_sha256)
        if self.representation == "JATS" and (
            self.native_artifact is not None or self.source_doctype_declared is None
        ):
            raise ValueError("JATS document requires DOCTYPE state and no native artifact")
        if (
            self.representation == "PDF_DOCLING"
            and self.status == "complete"
            and self.native_artifact is None
        ):
            raise ValueError("complete PDF document requires saved native artifact")
        if not self.nodes:
            raise ValueError("empty canonical document")
        if self.schema_version == "5" and (
            self.representation != "PDF_DOCLING"
            or self.transformation_id != "docling_pdf_adapter_v5"
        ):
            raise ValueError("canonical schema 5 requires the versioned figure-footnote adapter")
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate canonical node ID")
        if [node.order for node in self.nodes] != list(range(len(self.nodes))):
            raise ValueError("nonsequential canonical order")
        for node in self.nodes:
            if node.mapping.source_artifact_id != self.source.artifact_id:
                raise ValueError("node source differs from document source")
            if self.representation == "JATS" and node.mapping.kind != "jats":
                raise ValueError("JATS node requires XML mapping")
            if self.representation == "PDF_DOCLING" and (
                node.mapping.kind != "docling"
                or node.mapping.native_artifact != self.native_artifact
            ):
                raise ValueError("PDF node requires this saved native artifact")
            if node.parent_node_id is not None and node.parent_node_id not in ids:
                raise ValueError("unresolved canonical parent")
            if node.figure_footnote_node_ids is not None:
                notes = node.figure_footnote_node_ids
                if self.schema_version != "5" or node.kind != "figure":
                    raise ValueError("figure footnotes require a schema 5 figure")
                if len(set(notes)) != len(notes) or any(target not in ids for target in notes):
                    raise ValueError("duplicate or unresolved figure footnote")
                targets = {n.node_id: n for n in self.nodes}
                if any(targets[target].native_label != "footnote" for target in notes):
                    raise ValueError("figure footnote target has a contradictory native role")
            if any(target not in ids for target in node.target_node_ids):
                raise ValueError("unresolved canonical target")
            if any(target not in ids for target in node.native_child_node_ids):
                raise ValueError("unresolved canonical native child")
            for segment in node.segments:
                if segment.mapping.source_artifact_id != self.source.artifact_id:
                    raise ValueError("segment source differs from document source")
                if self.representation == "PDF_DOCLING" and (
                    segment.mapping.kind != "docling"
                    or segment.mapping.native_artifact != self.native_artifact
                ):
                    raise ValueError("PDF segment requires this native artifact")
            if node.table:
                for cell in node.table.cells:
                    if (
                        cell.node_id not in ids
                        or cell.mapping.source_artifact_id != self.source.artifact_id
                    ):
                        raise ValueError("unresolved table cell")
                    if self.representation == "PDF_DOCLING" and (
                        cell.mapping.kind != "docling"
                        or cell.mapping.native_artifact != self.native_artifact
                    ):
                        raise ValueError("PDF table cell requires this native artifact")
                if not set(node.table.caption_node_ids + node.table.footnote_node_ids) <= ids:
                    raise ValueError("unresolved table relation")
        return self
