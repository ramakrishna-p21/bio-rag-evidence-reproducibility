"""Typed source-faithful canonical document contract."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef

MappingState = Literal[
    "exact_native", "derived", "approximate", "unavailable", "ambiguous", "parser_damaged", "failed"
]
ContentState = Literal["present", "marker_only", "missing", "unsupported"]
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
]


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class JatsConfiguration(Record):
    schema_version: Literal["1"] = "1"
    entity_policy: Literal["reject_declarations"] = "reject_declarations"
    source_faithful: Literal[True] = True


class SourceMapping(Record):
    state: MappingState
    source_artifact_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    segment_ordinal: int | None = Field(default=None, ge=0)


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
    is_header: bool
    mapping: SourceMapping


class TableData(Record):
    structure_state: MappingState
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    cells: list[TableCell]
    caption_node_ids: list[str] = Field(default_factory=list)
    footnote_node_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_grid(self) -> Self:
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


class CanonicalDocument(Record):
    schema_version: Literal["1"] = "1"
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
        ids = {node.node_id for node in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate canonical node ID")
        if [node.order for node in self.nodes] != list(range(len(self.nodes))):
            raise ValueError("nonsequential canonical order")
        for node in self.nodes:
            if node.mapping.source_artifact_id != self.source.artifact_id:
                raise ValueError("node source differs from document source")
            if node.parent_node_id is not None and node.parent_node_id not in ids:
                raise ValueError("unresolved canonical parent")
            if any(target not in ids for target in node.target_node_ids):
                raise ValueError("unresolved canonical target")
            for segment in node.segments:
                if segment.mapping.source_artifact_id != self.source.artifact_id:
                    raise ValueError("segment source differs from document source")
            if node.table:
                for cell in node.table.cells:
                    if (
                        cell.node_id not in ids
                        or cell.mapping.source_artifact_id != self.source.artifact_id
                    ):
                        raise ValueError("unresolved table cell")
                if not set(node.table.caption_node_ids + node.table.footnote_node_ids) <= ids:
                    raise ValueError("unresolved table relation")
        return self
