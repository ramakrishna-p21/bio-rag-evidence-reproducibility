"""Deterministic JATS candidate leads; no gold or eligibility decisions."""

import re
from xml.etree import ElementTree as ET

from pydantic import Field

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

_NUMBER = re.compile(
    r"(?<![\w])(?:[<>≤≥]\s*)?[+−-]?(?:\d+(?:\.\d*)?|\.\d+)(?:\s*(?:[eE][+−-]?\d+|[×x]\s*10[⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹+-]+))?%?"
)
_ENTITY = re.compile(r"\b[A-Z][A-Z0-9-]{1,11}\b")
_RELATION = re.compile(
    r"\b(?:increas\w*|decreas\w*|associat\w*|predict\w*|compar\w*|enrich\w*|higher|lower)\b",
    re.IGNORECASE,
)
_COHORT = re.compile(r"\b(?:cohort|dataset|biobank|population)\b", re.IGNORECASE)


def _tag(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _text(element: ET.Element) -> str:
    return " ".join(" ".join(element.itertext()).split())


class CandidateLead(Record):
    schema_version: str = "1"
    lead_id: str
    paper_id: str
    kind: str
    section: str
    xml_path: str
    text: str
    numeric_tokens: list[str] = Field(default_factory=list)
    entity_tokens: list[str] = Field(default_factory=list)
    relation_tokens: list[str] = Field(default_factory=list)
    cohort_cues: list[str] = Field(default_factory=list)
    table_id: str | None = None
    row_label: str | None = None
    column_label: str | None = None
    status: str = "lead_only"


def _lead(
    paper_id: str,
    kind: str,
    section: str,
    path: str,
    text: str,
    table_id: str | None = None,
    row_label: str | None = None,
    column_label: str | None = None,
) -> CandidateLead:
    identity = {"paper_id": paper_id, "kind": kind, "path": path, "text": text}
    return CandidateLead(
        lead_id="lead_" + sha256_bytes(canonical_json_bytes(identity)),
        paper_id=paper_id,
        kind=kind,
        section=section,
        xml_path=path,
        text=text,
        numeric_tokens=_NUMBER.findall(text),
        entity_tokens=_ENTITY.findall(text),
        relation_tokens=_RELATION.findall(text),
        cohort_cues=_COHORT.findall(text),
        table_id=table_id,
        row_label=row_label,
        column_label=column_label,
    )


def discover_jats(content: bytes, paper_id: str) -> list[CandidateLead]:
    """Extract source-located prose and table-cell leads from original JATS bytes."""
    root = ET.fromstring(content)
    if _tag(root) != "article":
        raise ValueError("JATS root must be article")
    leads: list[CandidateLead] = []

    def visit(node: ET.Element, path: str, section: str, table_id: str | None = None) -> None:
        tag = _tag(node)
        if tag == "abstract":
            section = "abstract"
        elif tag == "sec":
            title = next((child for child in node if _tag(child) == "title"), None)
            if title is not None and _text(title):
                section = _text(title)
        if tag == "table-wrap":
            table_id = node.get("id") or path
        if tag in {"p", "caption"} and table_id is None:
            content_text = _text(node)
            if content_text and (
                _NUMBER.search(content_text)
                or _ENTITY.search(content_text)
                or _RELATION.search(content_text)
                or _COHORT.search(content_text)
            ):
                leads.append(
                    _lead(
                        paper_id, "prose" if tag == "p" else "caption", section, path, content_text
                    )
                )
        if tag == "table":
            paths: dict[int, str] = {}

            def map_paths(element: ET.Element, element_path: str) -> None:
                paths[id(element)] = element_path
                child_counts: dict[str, int] = {}
                for descendant in element:
                    name = _tag(descendant)
                    child_counts[name] = child_counts.get(name, 0) + 1
                    map_paths(descendant, f"{element_path}/{name}[{child_counts[name]}]")

            map_paths(node, path)
            rows = [descendant for descendant in node.iter() if _tag(descendant) == "tr"]
            grid: dict[tuple[int, int], ET.Element] = {}
            starts: list[tuple[int, int, ET.Element]] = []
            for row_index, row in enumerate(rows):
                column = 0
                for cell in row:
                    if _tag(cell) not in {"th", "td"}:
                        continue
                    while (row_index, column) in grid:
                        column += 1
                    try:
                        row_span = int(cell.get("rowspan", "1"))
                        column_span = int(cell.get("colspan", "1"))
                    except ValueError as exc:
                        raise ValueError("invalid JATS table span") from exc
                    if row_span < 1 or column_span < 1:
                        raise ValueError("invalid JATS table span")
                    starts.append((row_index, column, cell))
                    for down in range(row_span):
                        for across in range(column_span):
                            key = (row_index + down, column + across)
                            if key in grid:
                                raise ValueError("overlapping JATS table spans")
                            grid[key] = cell
                    column += column_span
            header_count = 0
            for row in rows:
                if any(_tag(cell) == "th" for cell in row):
                    header_count += 1
                else:
                    break
            for row_index, column, cell in starts:
                if row_index < header_count or column == 0:
                    continue
                cell_text = _text(cell)
                if not cell_text:
                    continue
                column_labels = list(
                    dict.fromkeys(
                        _text(grid[(header_row, column)])
                        for header_row in range(header_count)
                        if (header_row, column) in grid and _text(grid[(header_row, column)])
                    )
                )
                row_labels = list(
                    dict.fromkeys(
                        _text(grid[(row_index, prior)])
                        for prior in range(column)
                        if (row_index, prior) in grid and _text(grid[(row_index, prior)])
                    )
                )
                leads.append(
                    _lead(
                        paper_id,
                        "table_cell",
                        section,
                        paths[id(cell)],
                        cell_text,
                        table_id,
                        " | ".join(row_labels) or None,
                        " | ".join(column_labels) or f"column {column + 1}",
                    )
                )
            return
        counts: dict[str, int] = {}
        for child in node:
            child_tag = _tag(child)
            counts[child_tag] = counts.get(child_tag, 0) + 1
            visit(child, f"{path}/{child_tag}[{counts[child_tag]}]", section, table_id)

    visit(root, "/article", "unlabeled")
    return leads
