"""Canonical-only occurrence discovery, whole-unit rendering, and chunk construction."""

import re
from collections import Counter as Tally
from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biorag_repro.benchmark.storage import encode_jsonl, read_jsonl
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, verify_file_sha256
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_manifest,
    serialize_project_json,
    verify_manifest_outputs,
)
from biorag_repro.documents.models import CanonicalDocument, CanonicalNode, TableCell
from biorag_repro.retrieval.models import (
    AccountingStatus,
    CanonicalInput,
    Chunk,
    ChunkingDiagnostic,
    ChunkSet,
    ConfigurationRef,
    ContributorUsage,
    OccurrenceAccounting,
    OccurrenceField,
    Phase5Configuration,
    SourceContributor,
    SourceOccurrence,
)
from biorag_repro.retrieval.storage import _record_refs, record_ref
from biorag_repro.retrieval.validation import (
    CanonicalAuthority,
    load_canonical_input,
    load_configuration,
    occurrence_id,
    validate_occurrences,
)

Counter = Callable[[str], int]
V2_LEGEND = "Omitted cell attributes inherit cell_defaults."
V3_LEGEND = (
    "Omitted cell attributes inherit the [headers] tag for header cells, then cell_defaults. "
    "A cell written cC is in the row given by [row N]."
)
_BLOCKS = {"paragraph", "caption", "title", "formula", "reference", "list_item"}
_INLINE = {
    "italic",
    "bold",
    "sup",
    "sub",
    "underline",
    "monospace",
    "named-content",
    "ext-link",
    "xref",
}


@dataclass
class RenderedUnit:
    unit_id: str
    kind: str
    text: str
    context: str
    scope: str
    contributors: list[SourceContributor]
    source_node_ids: list[str]
    uncertainty: list[str]
    unsupported_reason: str | None = None


@dataclass
class ChunkingResult:
    chunks: list[Chunk]
    chunk_set: ChunkSet


def _tree(document: CanonicalDocument) -> tuple[dict[str, CanonicalNode], dict[str, list[str]]]:
    nodes = {node.node_id: node for node in document.nodes}
    children: dict[str, list[str]] = defaultdict(list)
    for node in document.nodes:
        if node.parent_node_id:
            children[node.parent_node_id].append(node.node_id)
        seen = {node.node_id}
        parent = node.parent_node_id
        while parent is not None:
            if parent not in nodes or parent in seen:
                raise ValueError("invalid canonical hierarchy or cycle")
            seen.add(parent)
            parent = nodes[parent].parent_node_id
    return nodes, children


def _ancestors(node_id: str, nodes: dict[str, CanonicalNode]) -> list[str]:
    result = [node_id]
    while nodes[result[-1]].parent_node_id is not None:
        parent = nodes[result[-1]].parent_node_id
        assert parent is not None
        result.append(parent)
    return list(reversed(result))


def _occurrence(
    canonical: ArtifactRef,
    node: CanonicalNode,
    field: OccurrenceField,
    text: str | None,
    index: int | None = None,
) -> SourceOccurrence:
    mapping = node.mapping if index is None else node.segments[index].mapping
    return SourceOccurrence(
        occurrence_id=occurrence_id(canonical, mapping, field),
        canonical=canonical,
        owner_node_id=node.node_id,
        field=field,
        segment_index=index,
        mapping=mapping,
        text=text,
        content_state=node.content_state,
        interpretation_state=node.interpretation_state,
        native_order_state=node.native_order_state,
        orig_qualification=node.orig_qualification,
    )


def discover_occurrences(
    document: CanonicalDocument, canonical: ArtifactRef
) -> list[SourceOccurrence]:
    """Enumerate mapped source fields, deduplicating locations rather than strings."""
    document = CanonicalDocument.model_validate(document.model_dump())
    nodes, children = _tree(document)
    result: list[SourceOccurrence] = []
    if document.representation == "JATS":
        slots: dict[str, tuple[CanonicalNode, int]] = {}
        for node in document.nodes:
            for index, segment in enumerate(node.segments):
                path = segment.mapping.path
                if path is None:
                    raise ValueError("JATS source slot has no path")
                if path in slots:
                    previous, old_index = slots[path]
                    if previous.segments[old_index].text != segment.text:
                        raise ValueError("one JATS source slot has conflicting text")
                slots[path] = (node, index)
        visited: set[str] = set()

        def emit(path: str) -> None:
            if path in slots:
                node, index = slots[path]
                result.append(
                    _occurrence(canonical, node, "segment", node.segments[index].text, index)
                )
                visited.add(path)

        def walk(node_id: str) -> None:
            node = nodes[node_id]
            if not node.segments:
                result.append(_occurrence(canonical, node, "nontextual", None))
            emit(f"{node.mapping.path}/text()")
            for child in children[node_id]:
                walk(child)
                emit(f"{nodes[child].mapping.path}/tail()")

        for node in document.nodes:
            if node.parent_node_id is None:
                walk(node.node_id)
        if visited != set(slots):
            raise ValueError("JATS slots cannot be ordered from canonical hierarchy")
    else:
        for node in document.nodes:
            for index, segment in enumerate(node.segments):
                result.append(_occurrence(canonical, node, "segment", segment.text, index))
            if node.native_orig is not None and node.orig_qualification != "same_as_text":
                result.append(_occurrence(canonical, node, "native_orig", node.native_orig))
            if node.native_marker is not None:
                result.append(_occurrence(canonical, node, "native_marker", node.native_marker))
            if not node.segments and node.native_orig is None and node.native_marker is None:
                result.append(_occurrence(canonical, node, "nontextual", None))
    validate_occurrences(result, document, canonical)
    return result


def _qualifications(items: list[SourceOccurrence]) -> list[str]:
    values: set[str] = set()
    for item in items:
        for name, value in (
            ("mapping", item.mapping.state),
            ("content", item.content_state),
            ("interpretation", item.interpretation_state),
            ("order", item.native_order_state),
            ("orig", item.orig_qualification),
        ):
            if value is not None and value not in {
                "exact_native",
                "present",
                "linked",
                "same_as_text",
            }:
                values.add(f"{name}={value}")
        for record in item.mapping.provenance:
            if record.charspan_state != "meaningful":
                values.add(f"charspan={record.charspan_state}")
    return sorted(values)


def render_units(
    document: CanonicalDocument,
    occurrences: list[SourceOccurrence],
    algorithm: str = "canonical_whole_units_v1",
) -> list[RenderedUnit]:
    """Render only recorded structure; retain source strings inside derived wrappers.

    v1 prints every cell attribute on every cell. v2 declares the most frequent value of each
    attribute once per table and prints a cell attribute only where it differs.
    """
    nodes, children = _tree(document)
    by_owner: dict[str, list[SourceOccurrence]] = defaultdict(list)
    by_slot = {o.mapping.path: o for o in occurrences if o.field == "segment"}
    for item in occurrences:
        by_owner[item.owner_node_id].append(item)

    def below(node_id: str) -> list[SourceOccurrence]:
        ids = {node_id}
        pending = list(children[node_id])
        while pending:
            child = pending.pop()
            ids.add(child)
            pending.extend(children[child])
        return [o for o in occurrences if o.owner_node_id in ids]

    def jats_text(node_id: str) -> str:
        node = nodes[node_id]
        if node.kind == "table" or node.source_tag.rsplit("}", 1)[-1] == "table-wrap":
            return ""  # Structured table children are rendered exclusively as table units.
        start = by_slot.get(f"{node.mapping.path}/text()")
        parts = [start.text or ""] if start else []
        for child_id in children[node_id]:
            child = nodes[child_id]
            text = jats_text(child_id)
            if (
                parts
                and text
                and child.kind in {"paragraph", "list_item"}
                and parts[-1]
                and not parts[-1][-1].isspace()
            ):
                parts.append("\n")
            parts.append(text)
            tail = by_slot.get(f"{child.mapping.path}/tail()")
            if tail:
                parts.append(tail.text or "")
        text = "".join(parts)
        local = node.source_tag.rsplit("}", 1)[-1]
        if local in {"sup", "sub", "alternatives", "tex-math"} or node.source_tag.startswith("{"):
            return f"<{node.source_tag}>{text}</{node.source_tag}>"
        return text

    def text(node_id: str) -> str:
        if document.representation == "JATS":
            return jats_text(node_id)
        parts = []
        for item in by_owner[node_id]:
            if item.text is None:
                continue
            if item.field == "segment":
                parts.append(item.text)
            else:
                parts.append(f"[{item.field}]\n{item.text}\n[/{item.field}]")
        rendered = "\n".join(parts)
        qualifications = _qualifications(by_owner[node_id])
        if qualifications and rendered:
            rendered = "[qualification " + " ".join(qualifications) + "]\n" + rendered
        return rendered

    def contributors(
        items: list[SourceOccurrence], usage: ContributorUsage
    ) -> list[SourceContributor]:
        return [
            SourceContributor(occurrence_id=o.occurrence_id, node_id=o.owner_node_id, usage=usage)
            for o in items
        ]

    heading_context: dict[str, list[str]] = {}
    heading_stack: list[str] = []
    if document.representation == "PDF_DOCLING":
        for node in document.nodes:
            if node.kind == "page_furniture" or node.native_content_layer == "furniture":
                heading_stack.clear()
            linked = node.native_order_state == "linked"
            if node.kind == "title":
                level = node.native_heading_level
                if not linked or level is None:
                    heading_stack.clear()
                else:
                    while (
                        heading_stack
                        and (nodes[heading_stack[-1]].native_heading_level or 0) >= level
                    ):
                        heading_stack.pop()
                heading_context[node.node_id] = list(heading_stack) if linked else []
                if linked and level is not None:
                    heading_stack.append(node.node_id)
            else:
                heading_context[node.node_id] = list(heading_stack) if linked else []

    def section(node_id: str) -> tuple[str, str, list[SourceContributor]]:
        ancestry = _ancestors(node_id, nodes)
        sections = [i for i in ancestry if nodes[i].kind == "section"]
        context = []
        links: list[SourceContributor] = []
        if document.representation == "PDF_DOCLING":
            titles = heading_context[node_id]
            for title in titles:
                context.append(
                    f"[section context=derived_heading_levels]\n{text(title)}\n[/section]\n"
                )
                links.extend(
                    contributors(
                        [o for o in below(title) if o.text is not None], "repeated_context"
                    )
                )
            scope = "headings:" + ":".join(titles) if titles else node_id
            return scope, "".join(context), links
        for section_id in sections:
            titles = [i for i in children[section_id] if nodes[i].kind == "title"]
            if len(titles) == 1 and titles[0] != node_id:
                title = titles[0]
                context.append(f"[section]\n{text(title)}\n[/section]\n")
                links.extend(
                    contributors(
                        [o for o in below(title) if o.text is not None], "repeated_context"
                    )
                )
            elif len(titles) > 1:
                context.append("[section context=ambiguous]\n")
        if sections:
            scope = sections[-1]
        else:
            scope = next(
                (i for i in ancestry if nodes[i].kind in {"body", "front", "back"}), ancestry[0]
            )
        return scope, "".join(context), links

    units: list[tuple[int, int, RenderedUnit]] = []
    assigned: dict[str, list[SourceOccurrence]] = defaultdict(list)
    table_cells = {
        cell.node_id for node in document.nodes if node.table for cell in node.table.cells
    }
    for item in occurrences:
        if item.text is None or item.owner_node_id in table_cells:
            continue
        ancestry = _ancestors(item.owner_node_id, nodes)
        if any(i in table_cells for i in ancestry):
            continue
        boundaries = [
            index
            for index, i in enumerate(ancestry)
            if nodes[i].kind == "table" or nodes[i].source_tag.rsplit("}", 1)[-1] == "table-wrap"
        ]
        if boundaries:
            ancestry = ancestry[boundaries[-1] + 1 :]
        if not ancestry:
            continue
        owner = item.owner_node_id
        for candidate in ancestry:
            node = nodes[candidate]
            own_text = any(o.text and o.text.strip() for o in by_owner[candidate])
            local = node.source_tag.rsplit("}", 1)[-1]
            if node.kind in _BLOCKS or (
                own_text
                and (
                    (node.kind == "source_construct" and local not in _INLINE)
                    or node.kind in {"body", "front", "back", "section", "article"}
                )
            ):
                owner = candidate
                break
        assigned[owner].append(item)
    # Ownership is exclusive even when a paragraph contains a table or caption.
    position = {item.occurrence_id: index for index, item in enumerate(occurrences)}
    table_nodes = [
        node
        for node in document.nodes
        if node.kind == "table" or node.source_tag.rsplit("}", 1)[-1] == "table-wrap"
    ]
    for owner in list(assigned):
        if not any(o.text and o.text.strip() for o in assigned[owner]):
            continue
        scope, context, links = section(owner)
        source = assigned[owner]
        unit = RenderedUnit(
            owner,
            nodes[owner].kind,
            text(owner),
            context,
            scope,
            links + contributors(source, "content"),
            [owner],
            _qualifications(source),
        )
        embedded = [
            node
            for node in table_nodes
            if owner in _ancestors(node.node_id, nodes) and node.node_id != owner
        ]
        for embedded_node in embedded:
            table_positions = [position[o.occurrence_id] for o in below(embedded_node.node_id)]
            if table_positions and any(
                o.text and o.text.strip() and position[o.occurrence_id] > min(table_positions)
                for o in source
            ):
                unit.unsupported_reason = "Whole prose block crosses an embedded table; source order cannot be preserved without splitting the block."
        if unit.text.strip():
            units.append((nodes[owner].order, -1, unit))

    def cell_text(cell: TableCell) -> str:
        return text(cell.node_id) if document.representation == "JATS" else cell.text

    def bool_text(value: bool | None) -> str:
        return "?" if value is None else str(value).lower()

    cell_attributes = (
        ("header", "is_header"),
        ("row_header", "native_row_header"),
        ("column_header", "native_column_header"),
        ("row_section", "native_row_section"),
        ("fillable", "native_fillable"),
    )
    compact = algorithm in {"canonical_whole_units_v2", "canonical_whole_units_v3"}

    def majority(cells: list[TableCell]) -> dict[str, bool | None]:
        # Ties resolve false, then true, then unknown, so the declaration is deterministic.
        rank = {False: 0, True: 1, None: 2}
        values = {}
        for name, attribute in cell_attributes:
            counts = Tally(getattr(cell, attribute) for cell in cells)
            values[name] = min(counts, key=lambda value: (-counts[value], rank[value]))
        return values

    defaults: dict[str, bool | None] = {}

    def cell_line(
        cell: TableCell, base: dict[str, bool | None] | None = None, row: int = -1
    ) -> str:
        if compact:
            # v3 only: a cell that starts in the unit's own row drops the row index, and header
            # cells print deviations from the [headers] tag rather than from cell_defaults.
            same_row = algorithm == "canonical_whole_units_v3" and cell.row == row
            parts = [f"c{cell.column}" if same_row else f"r{cell.row}c{cell.column}"]
            if (cell.row_span, cell.column_span) != (1, 1):
                parts.append(f"span={cell.row_span}x{cell.column_span}")
            parts += [
                f"{name}={bool_text(getattr(cell, attribute))}"
                for name, attribute in cell_attributes
                if getattr(cell, attribute) != (base or defaults)[name]
            ]
            return f"[{' '.join(parts)}] {cell_text(cell)}"
        return (
            f"[r{cell.row}c{cell.column} span={cell.row_span}x{cell.column_span} "
            f"header={bool_text(cell.is_header)} row_header={bool_text(cell.native_row_header)} "
            f"column_header={bool_text(cell.native_column_header)} row_section={bool_text(cell.native_row_section)} "
            f"fillable={bool_text(cell.native_fillable)}] {cell_text(cell)}"
        )

    for node in document.nodes:
        table = node.table
        if table is None:
            continue
        scope, section_text, section_links = section(node.node_id)
        relations = (
            f"[table structure={table.structure_state} captions={table.caption_relation_state} "
            f"footnotes={table.footnote_relation_state} orientation={table.native_orientation or '?'}]\n"
        )
        relation_links: list[SourceContributor] = []
        for label, ids in (
            ("caption", table.caption_node_ids),
            ("footnote", table.footnote_node_ids),
        ):
            for related in ids:
                relations += f"[{label}]\n{text(related)}\n[/{label}]\n"
                relation_links.extend(
                    contributors(
                        [o for o in below(related) if o.text is not None],
                        "repeated_context",
                    )
                )
        if compact and table.cells:
            defaults = majority(table.cells)
            declared = " ".join(f"{name}={bool_text(value)}" for name, value in defaults.items())
            legend = V3_LEGEND if algorithm == "canonical_whole_units_v3" else V2_LEGEND
            relations += f"[cell_defaults span=1x1 {declared}]\n{legend}\n"
        for row in range(table.row_count):
            cells = sorted(
                [cell for cell in table.cells if cell.row <= row < cell.row + cell.row_span],
                key=lambda cell: cell.column,
            )
            if not cells:
                continue
            headers = []
            for cell in table.cells:
                ancestry = _ancestors(cell.node_id, nodes)
                in_thead = any(nodes[i].source_tag.rsplit("}", 1)[-1] == "thead" for i in ancestry)
                if cell not in cells and (
                    cell.native_column_header is True or (cell.is_header and in_thead)
                ):
                    headers.append(cell)
            headers.sort(key=lambda cell: (cell.row, cell.column))
            header_text = ""
            if headers:
                tag, base = "[headers]", None
                if algorithm == "canonical_whole_units_v3":
                    base = majority(headers)
                    differing = [
                        f"{name}={bool_text(value)}"
                        for name, value in base.items()
                        if value != defaults[name]
                    ]
                    tag = f"[headers {' '.join(differing)}]" if differing else tag
                lines = "\n".join(cell_line(c, base) for c in headers)
                header_text = f"{tag}\n{lines}\n[/headers]\n"
            body = f"[row {row}]\n" + "\n".join(cell_line(c, row=row) for c in cells)
            links = section_links + relation_links
            for cell in headers + cells:
                usage: ContributorUsage = (
                    "content" if cell in cells and cell.row == row else "repeated_context"
                )
                links += contributors(below(cell.node_id), usage)
            source_ids = [node.node_id, *[c.node_id for c in cells]]
            uncertainty = sorted(
                set(
                    _qualifications([o for o in occurrences if o.owner_node_id in source_ids])
                    + [
                        f"table_structure={table.structure_state}",
                        f"captions={table.caption_relation_state}",
                        f"footnotes={table.footnote_relation_state}",
                    ]
                )
            )
            units.append(
                (
                    node.order,
                    row,
                    RenderedUnit(
                        f"{node.node_id}:row:{row}",
                        "table_row",
                        body,
                        section_text + relations + header_text,
                        f"table:{node.node_id}:{scope}",
                        links,
                        source_ids,
                        uncertainty,
                    ),
                )
            )
    return [unit for _, _, unit in sorted(units, key=lambda item: (item[0], item[1]))]


def chunks_ref(
    chunks: list[Chunk], selected: CanonicalInput, configuration: ConfigurationRef, algorithm: str
) -> ArtifactRef:
    content = encode_jsonl(chunks)
    digest = sha256_bytes(content)
    artifact_id = derived_artifact_id(
        artifact_kind="chunks_jsonl",
        inputs=[
            ArtifactInput("canonical", selected.canonical.artifact_id, selected.canonical.sha256),
            ArtifactInput(
                "configuration", configuration.artifact.artifact_id, configuration.artifact.sha256
            ),
        ],
        transformation_id=algorithm,
        transformation_settings_sha256=configuration.resolved_sha256,
        output_sha256=digest,
        output_format="jsonl",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"corpus/derived/chunks/{artifact_id}.jsonl",
        sha256=digest,
    )


def _chunks_manifest(
    ref: ArtifactRef, selected: CanonicalInput, cfg: ConfigurationRef, algorithm: str
) -> BuildManifest:
    return BuildManifest(
        schema_version="1",
        inputs=[
            BuildInput(role="canonical", artifact=selected.canonical),
            BuildInput(role="configuration", artifact=cfg.artifact),
        ],
        transformation_id=algorithm,
        transformation_settings_sha256=cfg.resolved_sha256,
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="chunks_jsonl",
                output_format="jsonl",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )


def _chunks_manifest_ref(
    ref: ArtifactRef, selected: CanonicalInput, cfg: ConfigurationRef, algorithm: str
) -> ArtifactRef:
    digest = sha256_bytes(serialize_manifest(_chunks_manifest(ref, selected, cfg, algorithm)))
    return ArtifactRef(
        artifact_id="build_" + digest, logical_ref=ref.logical_ref + ".build.json", sha256=digest
    )


SENTENCE_SEGMENTS = "complete_units_or_sentence_segments"
OVERSIZED_REASON = "Complete source unit and required context exceed the construction limit."
# A sentence ends at . ! or ? (optionally closed by a bracket or quote), then whitespace, then a
# capital, digit, opening bracket or quote. Decimals such as 0.001 have no whitespace and never match.
_SENTENCE_END = re.compile(r"[.!?][)\]\"”’]*(\s+)(?=[A-Z0-9(\[\"“‘])")
_ABBREVIATION = re.compile(
    r"(?:\b(?:e\.g|i\.e|et al|Fig|Figs|Eq|Eqs|Ref|Refs|Suppl|Tab|Vol|No|vs|cf|ca|approx|resp|Dr|St)"
    r"|\b[A-Z])\.[)\]\"”’]*$"
)
# Derived markup between occurrence texts: inline JATS tags and PDF field wrappers.
_MARKUP = re.compile(r"<(/?)[^<>\s]+>|\[(/?)[a-z_]+\]")


def _occurrence_spans(
    unit: RenderedUnit, occurrences: dict[str, SourceOccurrence], body: str
) -> list[tuple[SourceContributor, int, int]] | None:
    """Locate each content occurrence in rendering order; None if any cannot be found."""
    spans, cursor = [], 0
    for link in unit.contributors:
        if link.usage != "content":
            continue
        text = occurrences[link.occurrence_id].text
        if not text or not text.strip():
            spans.append((link, cursor, cursor))
            continue
        start = body.find(text, cursor)
        if start < 0:
            return None
        cursor = start + len(text)
        spans.append((link, start, cursor))
    return spans


def _sentence_cuts(body: str, covered: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Whitespace runs that end a sentence outside markup, brackets and abbreviations."""
    source = bytearray(len(body))
    for start, end in covered:
        source[start:end] = b"\x01" * (end - start)
    depth = [0] * (len(body) + 1)  # markup plus bracket depth before each position
    level = markup = 0
    for i, char in enumerate(body):
        if not source[i] and (tag := _MARKUP.match(body, i)):
            markup += -1 if (tag.group(1) or tag.group(2)) else 1
        elif source[i] and char in "([":
            level += 1
        elif source[i] and char in ")]":
            level = max(level - 1, 0)
        depth[i + 1] = markup + level
    cuts = []
    for match in _SENTENCE_END.finditer(body):
        start, end = match.span(1)
        if depth[start] == 0 and not _ABBREVIATION.search(body[max(start - 16, 0) : start]):
            cuts.append((start, end))
    return cuts


def segment_oversized_prose(
    unit: RenderedUnit,
    occurrences: dict[str, SourceOccurrence],
    count_tokens: Counter,
    limit: int,
) -> tuple[list[RenderedUnit], str | None]:
    """Split an oversized prose unit between sentences without changing any character.

    Uses only the unit, its occurrence texts, the construction counter and the limit. Returns
    the segments in source order, or no segments and the reason the unit stays whole. A unit
    level qualification line is repeated on every segment.
    """
    if unit.kind in {"table_row", "formula"}:
        return [], f"{unit.kind} units are not segmented"
    prefix = ""
    if unit.text.startswith("[qualification "):
        prefix = unit.text[: unit.text.index("\n") + 1]
    body = unit.text[len(prefix) :]
    spans = _occurrence_spans(unit, occurrences, body)
    if spans is None:
        return [], "occurrence text could not be located in order"
    cuts = _sentence_cuts(body, [(s, e) for _, s, e in spans])
    if not cuts:
        return [], "no permitted sentence boundary"
    edges = [0] + [p for cut in cuts for p in cut] + [len(body)]
    sentences = list(zip(edges[::2], edges[1::2], strict=True))

    def fits(start: int, end: int) -> bool:
        return count_tokens(unit.context + prefix + body[start:end]) <= limit

    if not all(fits(start, end) for start, end in sentences):
        return [], "a single sentence exceeds the limit"
    groups = [sentences[0]]
    for start, end in sentences[1:]:
        if fits(groups[-1][0], end):
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    rebuilt = "".join(
        body[start:end] + body[end : nxt[0]]
        for (start, end), nxt in zip(groups, groups[1:] + [(len(body), len(body))], strict=True)
    )
    if rebuilt != body:
        raise ValueError("sentence segments do not reconstruct the unit")

    # The segments each occurrence reaches; a zero-length one stays where its position falls.
    reach: dict[int, list[tuple[int, int]]] = {}
    for link, s, e in spans:
        if e > s:
            reach[id(link)] = [g for g in groups if s < g[1] and e > g[0]]
        else:
            reach[id(link)] = [next(g for g in groups if s < g[1] or g == groups[-1])]
    pieces = []
    for group in groups:
        start, end = group
        links = []
        for link in unit.contributors:
            if link.usage != "content":
                links.append(link)
            elif group in reach[id(link)]:
                usage: ContributorUsage = (
                    "content" if group == reach[id(link)][0] else "content_continuation"
                )
                links.append(link.model_copy(update={"usage": usage}))
        pieces.append(
            RenderedUnit(
                f"{unit.unit_id}:s{len(prefix) + start}-{len(prefix) + end}",
                unit.kind,
                prefix + body[start:end],
                unit.context,
                unit.scope,
                links,
                unit.source_node_ids,
                unit.uncertainty,
            )
        )
    return pieces, None


def chunk_document(
    document: CanonicalDocument,
    selected: CanonicalInput,
    configuration: ConfigurationRef,
    settings: Phase5Configuration,
    count_tokens: Counter,
) -> ChunkingResult:
    """Pack complete adjacent units; an oversized unit is diagnosed, never shortened."""
    if (document.paper_id, document.representation) != (selected.paper_id, selected.representation):
        raise ValueError("chunk input paper/representation differs from selected canonical")
    if configuration.resolved_sha256 != sha256_bytes(canonical_json_bytes(settings.model_dump())):
        raise ValueError("chunk configuration identity mismatch")
    occurrences = discover_occurrences(document, selected.canonical)
    units = render_units(document, occurrences, settings.chunking_algorithm)
    chunks: list[Chunk] = []
    diagnostics: list[ChunkingDiagnostic] = []
    pending: list[RenderedUnit] = []

    def rendered(batch: list[RenderedUnit]) -> str:
        return batch[0].context + "\n\n".join(unit.text for unit in batch)

    def count(text: str) -> int:
        value = count_tokens(text)
        if type(value) is not int or value < 0:
            raise ValueError("invalid construction token count")
        return value

    # Opt-in development candidate: oversized prose is split between sentences, losslessly.
    not_segmented: dict[str, str] = {}
    if settings.unit_policy == SENTENCE_SEGMENTS:
        by_id = {item.occurrence_id: item for item in occurrences}
        expanded = []
        for unit in units:
            if unit.unsupported_reason is None and count(rendered([unit])) > settings.chunk_tokens:
                pieces, why = segment_oversized_prose(unit, by_id, count, settings.chunk_tokens)
                if pieces:
                    expanded.extend(pieces)
                    continue
                not_segmented[unit.unit_id] = f" Not segmented: {why}."
            expanded.append(unit)
        units = expanded

    def flush() -> None:
        if not pending:
            return
        content = rendered(pending)
        links: list[SourceContributor] = []
        seen: set[tuple[str, str, str]] = set()
        for unit in pending:
            for link in unit.contributors:
                key = (link.occurrence_id, link.node_id, link.usage)
                if key not in seen:
                    links.append(link)
                    seen.add(key)
        chunks.append(
            Chunk(
                input=selected,
                configuration=configuration,
                contributors=links,
                rendered_text=content,
                construction_tokens=count(content),
            )
        )
        pending.clear()

    for unit in units:
        content = rendered([unit])
        actual = count(content)
        if unit.unsupported_reason is not None or actual > settings.chunk_tokens:
            flush()
            diagnostics.append(
                ChunkingDiagnostic(
                    unit_id=unit.unit_id,
                    kind=unit.kind,
                    code="unsupported" if unit.unsupported_reason else "oversized",
                    source_node_ids=unit.source_node_ids,
                    contributors=unit.contributors,
                    actual_tokens=actual,
                    token_limit=settings.chunk_tokens,
                    rendered_text=content,
                    reason=unit.unsupported_reason
                    or OVERSIZED_REASON + not_segmented.get(unit.unit_id, ""),
                    uncertainty=unit.uncertainty,
                )
            )
            continue
        if pending and (
            pending[0].scope != unit.scope
            or pending[0].context != unit.context
            or count(rendered(pending + [unit])) > settings.chunk_tokens
        ):
            flush()
        pending.append(unit)
    flush()
    membership: dict[str, list[str]] = defaultdict(list)
    ids = [record_ref(chunk).artifact_id for chunk in chunks]
    for chunk, chunk_id in zip(chunks, ids, strict=True):
        for occurrence in dict.fromkeys(
            c.occurrence_id for c in chunk.contributors if c.usage != "containing_reference"
        ):
            membership[occurrence].append(chunk_id)
    primary = [
        c.occurrence_id for chunk in chunks for c in chunk.contributors if c.usage == "content"
    ]
    if len(primary) != len(set(primary)):
        raise ValueError("source occurrence has duplicate primary rendering")
    excluded = {
        c.occurrence_id: d for d in diagnostics for c in d.contributors if c.usage == "content"
    }
    accounting = []
    for item in occurrences:
        status: AccountingStatus
        if membership[item.occurrence_id]:
            status, reason = "included", None
        elif item.occurrence_id in excluded:
            diagnostic = excluded[item.occurrence_id]
            status = "oversized" if diagnostic.code == "oversized" else "unsupported"
            reason = diagnostic.reason
        elif item.text is None:
            status, reason = (
                "nontextual",
                "Nontextual canonical structure; no scientific text was decoded or invented.",
            )
        elif not item.text.strip():
            status, reason = (
                "unsupported",
                "Whitespace-only source layout retained in the occurrence catalog.",
            )
        else:
            raise ValueError(f"unaccounted textual source occurrence: {item.mapping}")
        accounting.append(
            OccurrenceAccounting(
                occurrence_id=item.occurrence_id,
                status=status,
                chunk_ids=membership[item.occurrence_id],
                reason=reason,
            )
        )
        if reason and item.occurrence_id not in excluded:
            diagnostics.append(
                ChunkingDiagnostic(
                    unit_id=item.occurrence_id,
                    kind="source_field",
                    code="nontextual" if item.text is None else "layout_only",
                    source_node_ids=[item.owner_node_id],
                    contributors=[
                        SourceContributor(
                            occurrence_id=item.occurrence_id,
                            node_id=item.owner_node_id,
                            usage="containing_reference",
                        )
                    ],
                    actual_tokens=None,
                    token_limit=settings.chunk_tokens,
                    rendered_text=None,
                    reason=reason,
                    uncertainty=_qualifications([item]),
                )
            )
    metadata = ChunkSet(
        input=selected,
        configuration=configuration,
        chunks=chunks_ref(chunks, selected, configuration, settings.chunking_algorithm),
        chunk_ids=ids,
        occurrences=occurrences,
        accounting=accounting,
        diagnostics=diagnostics,
    )
    metadata.chunks_build_manifest = _chunks_manifest_ref(
        metadata.chunks, selected, configuration, settings.chunking_algorithm
    )
    return ChunkingResult(chunks, metadata)


def validate_chunking(
    result: ChunkingResult,
    document: CanonicalDocument,
    selected: CanonicalInput,
    configuration: ConfigurationRef,
    settings: Phase5Configuration,
    count_tokens: Counter,
) -> None:
    """Reconcile the complete submitted artifact against its canonical scientific input."""
    expected = chunk_document(document, selected, configuration, settings, count_tokens)
    if result != expected:
        raise ValueError(
            "chunking content, occurrence accounting, tokens or diagnostics differ from canonical reproduction"
        )


def load_construction_counter(settings: Phase5Configuration, root: Path) -> Counter:
    """Verify the pinned tokenizer files; tokenize complete inputs locally without weights."""
    import json
    from importlib.metadata import version

    snapshot = settings.construction_tokenizer
    names = {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"}
    if snapshot.state != "pinned" or set(snapshot.file_hashes) != names:
        raise ValueError("construction tokenizer requires the four verified pinned assets")
    for name, digest in snapshot.file_hashes.items():
        verify_file_sha256(root / name, digest)
    if version("tokenizers") != "0.22.2":
        raise ValueError("construction tokenizer backend version differs from configuration")
    config = json.loads((root / "tokenizer_config.json").read_bytes())
    if config.get("tokenizer_class") != "BertTokenizer" or config.get("do_lower_case") is not True:
        raise ValueError("unexpected BGE tokenizer configuration")
    from tokenizers import Tokenizer  # type: ignore[import-untyped]

    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    vocabulary = (root / "vocab.txt").read_text(encoding="utf-8").splitlines()
    if tokenizer.get_vocab() != {token: index for index, token in enumerate(vocabulary)}:
        raise ValueError("tokenizer JSON and vocabulary identities disagree")
    tokenizer.no_padding()
    tokenizer.no_truncation()

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def _publication_counter(
    settings: Phase5Configuration, root: Path | None, fixture: Counter | None
) -> Counter:
    if fixture is not None:
        if settings.purpose != "synthetic_fixture":
            raise ValueError("a fixture counter cannot certify development scientific chunks")
        return fixture
    if root is None:
        raise ValueError("verified construction tokenizer assets are required")
    return load_construction_counter(settings, root)


def _set_manifest(value: ChunkSet) -> BuildManifest:
    if value.chunks_build_manifest is None:
        raise ValueError("chunk set requires its complete chunks manifest")
    inputs = [
        BuildInput(role=role, artifact=ref)
        for role, ref in (
            ("canonical", value.input.canonical),
            ("phase4_validation", value.input.validation),
            ("canonical_build_manifest", value.input.build_manifest),
            ("chunks_jsonl", value.chunks),
            ("chunks_build_manifest", value.chunks_build_manifest),
            ("configuration", value.configuration.artifact),
        )
    ]
    return BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id="chunk_set_record_v1",
        transformation_settings_sha256=value.configuration.resolved_sha256,
        outputs=[
            DerivedOutput(
                artifact=record_ref(value),
                artifact_kind="chunk_set",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )


def chunk_set_manifest_ref(value: ChunkSet) -> ArtifactRef:
    digest = sha256_bytes(serialize_project_json(_set_manifest(value), BuildManifest))
    return ArtifactRef(
        artifact_id="build_" + digest,
        logical_ref=record_ref(value).logical_ref + ".chunk-set.build.json",
        sha256=digest,
    )


def publish_chunking(
    result: ChunkingResult,
    authority: CanonicalAuthority,
    configuration: ConfigurationRef,
    configuration_root: Path,
    output_root: Path,
    *,
    tokenizer_root: Path | None = None,
    fixture_counter: Counter | None = None,
    reuse_identical: bool = False,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Validate scientific content before writes, and publish the complete set manifest last."""
    settings = load_configuration(configuration, configuration_root)
    document = load_canonical_input(result.chunk_set.input, authority)
    counter = _publication_counter(settings, tokenizer_root, fixture_counter)
    validate_chunking(result, document, result.chunk_set.input, configuration, settings, counter)
    metadata = result.chunk_set
    chunks = metadata.chunks
    target = output_root / chunks.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        encode_jsonl(result.chunks),
        chunks.sha256,
        validate_temp=lambda p: read_jsonl(p, chunks.sha256, Chunk),
        reuse_identical=reuse_identical,
    )
    manifest_ref = metadata.chunks_build_manifest
    assert manifest_ref is not None
    publish_manifest(
        _chunks_manifest(chunks, metadata.input, configuration, settings.chunking_algorithm),
        output_root / manifest_ref.logical_ref,
        {chunks.logical_ref: target},
        reuse_identical=reuse_identical,
    )
    ref = record_ref(metadata)
    meta_path = output_root / ref.logical_ref
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        meta_path,
        serialize_project_json(metadata, ChunkSet),
        ref.sha256,
        validate_temp=lambda p: read_project_json(p, ref.sha256, ChunkSet),
        reuse_identical=reuse_identical,
    )
    logical_ref = ref.logical_ref + ".chunk-set.build.json"
    digest = publish_manifest(
        _set_manifest(metadata),
        output_root / logical_ref,
        {ref.logical_ref: meta_path},
        reuse_identical=reuse_identical,
    )
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


_verified_reads: ContextVar[
    dict[tuple[Any, ...], tuple[ChunkingResult, tuple[Any, ...]]] | None
] = ContextVar("verified_chunk_reads", default=None)


@contextmanager
def verified_once() -> Iterator[None]:
    """Let one verification read each distinct chunk set once instead of once per consumer.

    A context is checked by recomputing its index, its ranking and its packing, and each of
    those used to reopen and re-chunk the same chunk set. Inside this scope, a second read
    with identical inputs returns the first, fully verified result. The scope belongs to a
    single verification: it is never kept between calls, so an artifact that changes
    afterwards is read and checked again.
    """
    if _verified_reads.get() is not None:
        yield
        return
    from biorag_repro.retrieval import bm25, dense

    token = _verified_reads.set({})
    ref_token = _record_refs.set({})
    bm25_token = bm25._verified_selections.set({})
    dense_token = dense._verified_indexes.set({})
    try:
        yield
    finally:
        dense._verified_indexes.reset(dense_token)
        bm25._verified_selections.reset(bm25_token)
        _record_refs.reset(ref_token)
        _verified_reads.reset(token)


def read_chunking(
    ref: ArtifactRef,
    manifest_ref: ArtifactRef,
    authority: CanonicalAuthority,
    configuration: ConfigurationRef,
    configuration_root: Path,
    output_root: Path,
    *,
    tokenizer_root: Path | None = None,
    fixture_counter: Counter | None = None,
) -> ChunkingResult:
    """Read exact bytes and independently selected upstream evidence before scientific reuse."""
    reads = _verified_reads.get()
    key = (
        ref.model_dump_json(),
        manifest_ref.model_dump_json(),
        configuration.model_dump_json(),
        str(configuration_root),
        str(output_root),
        str(tokenizer_root),
        id(authority),
        id(fixture_counter),
    )
    if reads is not None and key in reads:
        shared = reads[key][0]
        return ChunkingResult(
            [chunk.model_copy(deep=True) for chunk in shared.chunks],
            shared.chunk_set.model_copy(deep=True),
        )
    result = _read_chunking(
        ref,
        manifest_ref,
        authority,
        configuration,
        configuration_root,
        output_root,
        tokenizer_root=tokenizer_root,
        fixture_counter=fixture_counter,
    )
    if reads is not None:
        # The authority and counter are kept so that their ids cannot be reused in this scope.
        reads[key] = (result, (authority, fixture_counter))
    # Cached scientific records remain private; callers may mutate their own copies.
    return ChunkingResult(
        [chunk.model_copy(deep=True) for chunk in result.chunks],
        result.chunk_set.model_copy(deep=True),
    )


def _read_chunking(
    ref: ArtifactRef,
    manifest_ref: ArtifactRef,
    authority: CanonicalAuthority,
    configuration: ConfigurationRef,
    configuration_root: Path,
    output_root: Path,
    *,
    tokenizer_root: Path | None,
    fixture_counter: Counter | None,
) -> ChunkingResult:
    metadata = read_project_json(output_root / ref.logical_ref, ref.sha256, ChunkSet)
    if ref != record_ref(metadata):
        raise ValueError("chunk-set reference identity mismatch")
    manifest = read_manifest(output_root / manifest_ref.logical_ref, manifest_ref.sha256)
    if manifest_ref.artifact_id != "build_" + manifest_ref.sha256 or manifest != _set_manifest(
        metadata
    ):
        raise ValueError("chunk-set manifest contract mismatch")
    verify_manifest_outputs(manifest, {ref.logical_ref: output_root / ref.logical_ref})
    assert metadata.chunks_build_manifest is not None
    settings = load_configuration(configuration, configuration_root)
    chunk_manifest = read_manifest(
        output_root / metadata.chunks_build_manifest.logical_ref,
        metadata.chunks_build_manifest.sha256,
    )
    algorithm = settings.chunking_algorithm
    if metadata.chunks_build_manifest != _chunks_manifest_ref(
        metadata.chunks, metadata.input, configuration, algorithm
    ) or chunk_manifest != _chunks_manifest(
        metadata.chunks, metadata.input, configuration, algorithm
    ):
        raise ValueError("chunks manifest contract mismatch")
    verify_manifest_outputs(
        chunk_manifest, {metadata.chunks.logical_ref: output_root / metadata.chunks.logical_ref}
    )
    chunks = read_jsonl(output_root / metadata.chunks.logical_ref, metadata.chunks.sha256, Chunk)
    result = ChunkingResult(chunks, metadata)
    document = load_canonical_input(metadata.input, authority)
    validate_chunking(
        result,
        document,
        metadata.input,
        configuration,
        settings,
        _publication_counter(settings, tokenizer_root, fixture_counter),
    )
    return result
