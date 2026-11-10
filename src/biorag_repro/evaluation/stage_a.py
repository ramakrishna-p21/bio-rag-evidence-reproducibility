"""Conservative Stage A locator and judgment validation.

Text matching here only locates candidate evidence. A failed match is never
evidence of absence or corruption: those labels need a reviewer's judgment.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from itertools import pairwise

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.chunking import discover_occurrences
from biorag_repro.documents.models import (
    CanonicalDocument,
    CanonicalNode,
    TableCell,
    TableData,
)
from biorag_repro.evaluation.models import (
    ComponentLocus,
    Placement,
    SetLocus,
    StageACandidate,
    StageAJudgment,
    Suggestion,
)
from biorag_repro.evaluation.surface import surface_problem
from biorag_repro.retrieval.models import SourceOccurrence

BLOCKS = {"paragraph", "list_item", "caption", "title", "formula", "figure", "reference"}
TABLE_ROLES = {"answer", "row", "column"}
_BEFORE = set(".,+-−_")
_AFTER = set("_%^×⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_NUMBER = re.compile(r"\d[\d.,]*")
_SD_TAIL = re.compile(r"\s*±\s*\d+(?:\.\d+)?\s*")
_MEAN_STATEMENT = re.compile(r"\bmean\s*±", re.IGNORECASE)
_SUGGESTION_LIMIT = 5
# Dashes and quotation marks differ between sources and Docling; minus signs, plus signs,
# comparison symbols, Greek letters and super/subscripts carry scientific meaning and stay.
_TYPOGRAPHY = {
    **dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015"), "-"),
    **dict.fromkeys(map(ord, "\u2018\u2019"), "'"),
    **dict.fromkeys(map(ord, "\u201c\u201d"), '"'),
    0xAD: None,  # soft hyphen
}
_Box = tuple[int, float, float, float, float]  # page, left, top, right, bottom
_Layouts = dict[int, tuple[float, float, bool]]
_TEXT_FLOW = {"title", "paragraph", "list_item"}
_NOT_A_HEADING_PARENT = {"figure", "table", "table_cell", "caption"}
_WIDE = 0.6  # box at least this fraction of the page text width spans the columns
_COLUMN_SLACK = 0.08  # fraction of the text width allowed across the column midline
_VERTICAL_TOLERANCE = 2.0  # points; Docling boxes are approximate


class IneligibleItem(ValueError):
    """The item cannot be evaluated at Stage A; this is not a Stage A label."""


def item_digest(item: BenchmarkItem) -> str:
    return sha256_bytes(canonical_json_bytes(item.model_dump(mode="json")))


def check_item(item: BenchmarkItem, document: CanonicalDocument, canonical: ArtifactRef) -> None:
    """Require an accepted development item whose source is this canonical document's source."""
    if item.split == "final_test":
        raise IneligibleItem("final-test items are protected from development evaluation")
    if (
        item.candidate_status != "accepted"
        or item.split != "development"
        or item.jats_status != "present_equivalent"
        or item.pdf_status != "present_equivalent"
    ):
        raise IneligibleItem("item is not an accepted paired-equivalent development item")
    if item.paper_id != document.paper_id:
        raise IneligibleItem("item and canonical document name different papers")
    refs = item.jats_refs if document.representation == "JATS" else item.pdf_refs
    if any(
        r.artifact_id != document.source.artifact_id or r.source_sha256 != document.source.sha256
        for r in refs
    ):
        raise IneligibleItem("item source references differ from the document's original source")


def _bounded(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if before and (before.isalnum() or before in _BEFORE):
        return False
    if after:
        if after.isalnum() or after in _AFTER:
            return False
        following = text[end + 1] if end + 1 < len(text) else ""
        if after in ".," and following.isdigit():
            return False
        if after in "-−" and following.isalnum():
            return False
    return True


def _spans(text: str, needle: str, *, bounded: bool = True) -> list[tuple[int, int]]:
    found = []
    start = text.find(needle)
    while needle and start >= 0:
        end = start + len(needle)
        if not bounded or _bounded(text, start, end):
            found.append((start, end))
        start = text.find(needle, start + 1)
    return found


def _digits(token: str) -> str:
    return re.sub(r"\D", "", token).strip("0")


@dataclass
class Unit:
    """Source text evaluated as one scientific context: a block, linked text, or table cell."""

    container: str
    text: str
    spans: list[tuple[int, int, SourceOccurrence]]
    cell: TableCell | None = None

    def ids(self, matches: list[tuple[int, int]]) -> list[str]:
        return [
            o.occurrence_id
            for s, e, o in self.spans
            if any(s < me and ms < e for ms, me in matches)
        ]


@dataclass
class Index:
    document: CanonicalDocument
    units: dict[str, list[Unit]] = field(default_factory=dict)
    where: dict[str, str] = field(default_factory=dict)
    order: dict[str, int] = field(default_factory=dict)


def _index(document: CanonicalDocument, canonical: ArtifactRef) -> Index:
    nodes = {n.node_id: n for n in document.nodes}
    cell_of: dict[str, tuple[str, TableCell]] = {}
    linked: dict[str, str] = {}
    for node in document.nodes:
        if node.table is not None:
            for table_cell in node.table.cells:
                cell_of[table_cell.node_id] = (node.node_id, table_cell)
            if node.table.caption_relation_state == "present":
                linked.update({i: node.node_id for i in node.table.caption_node_ids})
            if node.table.footnote_relation_state == "present":
                linked.update({i: node.node_id for i in node.table.footnote_node_ids})

    def locate_unit(node_id: str) -> tuple[str, str, TableCell | None]:
        chain: list[str] = []
        current: str | None = node_id
        while current is not None:
            chain.append(current)
            current = nodes[current].parent_node_id
        for item in chain:
            if item in cell_of:
                return cell_of[item][0], item, cell_of[item][1]
        for item in chain:
            if item in linked:
                return linked[item], item, None
        for item in chain:
            if nodes[item].kind in BLOCKS:
                return item, item, None
        return node_id, node_id, None

    index = Index(document)
    grouped: dict[tuple[str, str], Unit] = {}
    for position, occ in enumerate(discover_occurrences(document, canonical)):
        index.order[occ.occurrence_id] = position
        if occ.field != "segment" or occ.text is None:
            continue
        container, key, owner_cell = locate_unit(occ.owner_node_id)
        unit = grouped.get((container, key))
        if unit is None:
            unit = grouped[(container, key)] = Unit(container, "", [], owner_cell)
            index.units.setdefault(container, []).append(unit)
        start = len(unit.text)
        unit.text += occ.text
        unit.spans.append((start, len(unit.text), occ))
        index.where[occ.occurrence_id] = container
    return index


def _blocking(occ: SourceOccurrence) -> list[str]:
    values = []
    for name, value, ok in (
        ("mapping", occ.mapping.state, {"exact_native"}),
        ("content", occ.content_state, {"present"}),
        ("interpretation", occ.interpretation_state, {None, "exact_native"}),
        ("order", occ.native_order_state, {None, "linked"}),
        ("orig", occ.orig_qualification, {None, "same_as_text"}),
    ):
        if value not in ok:
            values.append(f"{name}={value}")
    return values


def _component_matches(
    index: Index, container: str, role: str, text: str, table: bool
) -> list[tuple[Unit, list[tuple[int, int]]]]:
    result = []
    for unit in index.units[container]:
        if table and role in TABLE_ROLES:
            spans = [(0, len(text))] if unit.cell is not None and unit.text == text else []
        else:
            spans = _spans(unit.text, text)
        if spans:
            result.append((unit, spans))
    return result


def _associated(
    table: TableData, answer: TableCell, rows: list[str], columns: list[str]
) -> list[TableCell] | None:
    if (
        table.structure_state != "exact_native"
        or not rows
        or not columns
        or answer.is_header is not False
    ):
        return None
    used = [answer]
    for texts, axis in ((rows, "row"), (columns, "column")):
        for text in texts:
            hits = []
            for h in table.cells:
                if h.is_header is not True or h.text != text:
                    continue
                if h.mapping.kind == "docling" and not (
                    h.native_row_header if axis == "row" else h.native_column_header
                ):
                    continue
                if axis == "row":
                    fits = (
                        h.row <= answer.row < h.row + h.row_span
                        and h.column + h.column_span <= answer.column
                    )
                else:
                    fits = (
                        h.column <= answer.column < h.column + h.column_span
                        and h.row + h.row_span <= answer.row
                    )
                if fits:
                    hits.append(h)
            if not hits:
                return None
            used += hits
    return used


def _place(
    index: Index,
    container: str,
    components: list[tuple[str, str]],
    suggestions: list[Suggestion],
    set_index: int,
    qualifications: set[str],
) -> Placement | None:
    node = next(n for n in index.document.nodes if n.node_id == container)
    table = node.table
    rows = [t for r, t in components if r == "row"]
    columns = [t for r, t in components if r == "column"]
    if table is None and (rows or columns):
        return None
    matched: list[tuple[str, str, list[tuple[Unit, list[tuple[int, int]]]]]] = []
    for role, text in components:
        hits = _component_matches(index, container, role, text, table is not None)
        if not hits:
            return None
        matched.append((role, text, hits))
    keep: dict[int, set[str]] = {}
    if table is not None:
        for k, (role, text, hits) in enumerate(matched):
            if role == "answer":
                allowed: set[str] = set()
                for unit, _ in hits:
                    assert unit.cell is not None  # table answers match whole cells only
                    used = _associated(table, unit.cell, rows, columns)
                    if used:
                        allowed |= {c.node_id for c in used}
                if not allowed:
                    suggestions.append(
                        Suggestion(
                            kind="association_mismatch",
                            set_index=set_index,
                            role=role,
                            detail=f"{text!r} is not under the required row and column labels",
                            occurrence_ids=[
                                o.occurrence_id for u, _ in hits for _, _, o in u.spans
                            ],
                        )
                    )
                    return None
                keep[k] = allowed
        for k, (role, text, hits) in enumerate(matched):
            if role in {"row", "column"}:
                keep[k] = {u.cell.node_id for u, _ in hits if u.cell is not None}
    loci = []
    for k, (role, text, hits) in enumerate(matched):
        ids = []
        for unit, spans in hits:
            if k in keep and (unit.cell is None or unit.cell.node_id not in keep[k]):
                continue
            ids += unit.ids(spans)
        ids = list(dict.fromkeys(ids))
        if not ids:
            return None
        loci.append(ComponentLocus(role=role, text=text, occurrence_ids=ids))
    occurrences = {o.occurrence_id: o for unit in index.units[container] for _, _, o in unit.spans}
    problems = sorted(
        {p for locus in loci for i in locus.occurrence_ids for p in _blocking(occurrences[i])}
    )
    if problems:
        qualifications.update(problems)
        suggestions.append(
            Suggestion(
                kind="qualified_mapping",
                set_index=set_index,
                detail="matched text has uncertain mapping or content: " + ", ".join(problems),
                occurrence_ids=[i for locus in loci for i in locus.occurrence_ids],
            )
        )
        return None
    return Placement(container_node_id=container, components=loci)


def _diagnose(
    index: Index,
    set_index: int,
    components: list[tuple[str, str]],
    suggestions: list[Suggestion],
) -> bool:
    everywhere = True
    for role, text in components:
        units = [u for units in index.units.values() for u in units]
        bounded = any(_spans(u.text, text) for u in units)
        if bounded:
            continue
        everywhere = False
        suggestions.append(
            Suggestion(
                kind="component_not_found",
                set_index=set_index,
                role=role,
                detail=f"no verbatim, boundary-checked match for {text!r}; absence is not inferred",
            )
        )
        longer = [u for u in units if _spans(u.text, text, bounded=False)]
        for unit in longer[:_SUGGESTION_LIMIT]:
            suggestions.append(
                Suggestion(
                    kind="contained_in_longer_token",
                    set_index=set_index,
                    role=role,
                    detail=f"{text!r} occurs only inside a longer token",
                    occurrence_ids=[o.occurrence_id for _, _, o in unit.spans],
                )
            )
        stripped = text.lstrip("+-−")
        if stripped != text:
            for unit in units:
                if _spans(unit.text, stripped):
                    suggestions.append(
                        Suggestion(
                            kind="sign_difference",
                            set_index=set_index,
                            role=role,
                            detail=f"{stripped!r} occurs without the sign of {text!r}",
                            occurrence_ids=[o.occurrence_id for _, _, o in unit.spans],
                        )
                    )
                    break
        target = _digits(text)
        if role == "answer" and target and any(ch.isdigit() for ch in text):
            seen = 0
            for unit in units:
                for token in _NUMBER.findall(unit.text):
                    if _digits(token) == target and token != text and seen < _SUGGESTION_LIMIT:
                        seen += 1
                        suggestions.append(
                            Suggestion(
                                kind="numeric_digit_sequence_match",
                                set_index=set_index,
                                role=role,
                                detail=f"{token!r} has the digits of {text!r} but a different value",
                                occurrence_ids=[o.occurrence_id for _, _, o in unit.spans],
                            )
                        )
    return everywhere


def locate(
    item: BenchmarkItem, document: CanonicalDocument, canonical: ArtifactRef
) -> StageACandidate:
    """Propose loci; only a complete verbatim, single-context, exact-mapped fact is preserved."""
    check_item(item, document, canonical)
    document = CanonicalDocument.model_validate(document.model_dump())
    index = _index(document, canonical)
    suggestions: list[Suggestion] = []
    qualifications: set[str] = set()
    sets: list[SetLocus] = []
    if document.status != "complete":
        suggestions.append(
            Suggestion(kind="incomplete_canonical", detail="canonical document is incomplete")
        )
    else:
        for set_index, evidence in enumerate(item.evidence_sets):
            components: list[tuple[str, str]] = [(c.role, c.text) for c in evidence.components]
            placements = []
            for container in index.units:
                placement = _place(
                    index, container, components, suggestions, set_index, qualifications
                )
                if placement is not None:
                    placements.append(placement)
            if placements:
                sets.append(SetLocus(set_index=set_index, placements=placements))
                continue
            if _diagnose(index, set_index, components, suggestions) and not any(
                s.set_index == set_index for s in suggestions
            ):
                suggestions.append(
                    Suggestion(
                        kind="split_across_contexts",
                        set_index=set_index,
                        detail="every component occurs, but no single context holds them all",
                    )
                )
    for locus in sets:
        for placement in locus.placements:
            for component in placement.components:
                for occ_id in component.occurrence_ids:
                    occ = next(
                        o
                        for u in index.units[placement.container_node_id]
                        for _, _, o in u.spans
                        if o.occurrence_id == occ_id
                    )
                    for record in occ.mapping.provenance:
                        if record.charspan_state != "meaningful":
                            qualifications.add(f"charspan={record.charspan_state}")
    return StageACandidate(
        benchmark_version=item.benchmark_version,
        corpus_version=item.corpus_version,
        item_id=item.item_id,
        item_sha256=item_digest(item),
        paper_id=item.paper_id,
        representation=document.representation,
        canonical=canonical,
        determination="automatic_preserved" if sets else "needs_review",
        sets=sets,
        suggestions=suggestions,
        qualifications=sorted(qualifications),
    )


def stage_a_dependencies(value: StageACandidate | StageAJudgment) -> list[tuple[str, ArtifactRef]]:
    dependencies = [
        ("canonical", value.canonical),
        (
            "benchmark_item",
            ArtifactRef(
                artifact_id=value.item_id,
                logical_ref=f"benchmark/{value.benchmark_version}/items/{value.item_id}",
                sha256=value.item_sha256,
            ),
        ),
    ]
    if isinstance(value, StageAJudgment) and value.candidate is not None:
        dependencies.append(("stage_a_candidate", value.candidate))
    return dependencies


def stage_a_ref(value: StageACandidate | StageAJudgment) -> ArtifactRef:
    """Identify a record by its exact bytes and its canonical, item and candidate lineage."""
    kind = "stage_a_candidate" if isinstance(value, StageACandidate) else "stage_a_judgment"
    dependencies = stage_a_dependencies(value)
    digest = sha256_bytes(serialize_project_json(value, type(value)))
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[ArtifactInput(role, ref.artifact_id, ref.sha256) for role, ref in dependencies],
        transformation_id=f"{kind}_v1",
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"evaluation/stage_a/{kind}/{artifact_id}.json",
        sha256=digest,
    )


def automatic_judgment(candidate: StageACandidate, candidate_ref: ArtifactRef) -> StageAJudgment:
    """Record the only deterministic label; it carries no reviewer fields."""
    if candidate.determination != "automatic_preserved":
        raise ValueError("only an automatic_preserved candidate has a deterministic label")
    return StageAJudgment(
        benchmark_version=candidate.benchmark_version,
        corpus_version=candidate.corpus_version,
        item_id=candidate.item_id,
        item_sha256=candidate.item_sha256,
        paper_id=candidate.paper_id,
        representation=candidate.representation,
        canonical=candidate.canonical,
        label="preserved",
        basis="deterministic",
        candidate=candidate_ref,
        sets=candidate.sets,
    )


def validate_candidate(
    candidate: StageACandidate,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> None:
    if locate(item, document, canonical) != candidate:
        raise ValueError("candidate differs from the locator output for this item and document")


def _mean_sd_matches(
    index: Index, container: str, text: str
) -> list[tuple[Unit, list[tuple[int, int]]]]:
    """Cells that read exactly `<text> ± <SD>`; reviewer path only, never automatic."""
    return [
        (unit, [(0, len(text))])
        for unit in index.units[container]
        if unit.cell is not None
        and unit.cell.is_header is False
        and unit.text.startswith(text)
        and _SD_TAIL.fullmatch(unit.text[len(text) :])
    ]


def _row_anchor(cell: TableCell, answer: TableCell) -> bool:
    if cell.mapping.kind == "docling" and cell.is_header and not cell.native_row_header:
        return False
    # The answer cell may span several rows; any covered row links a row label to it.
    return (
        cell.row < answer.row + answer.row_span
        and answer.row < cell.row + cell.row_span
        and (cell.column + cell.column_span <= answer.column)
    )


def _column_anchor(cell: TableCell, answer: TableCell) -> bool:
    if cell.is_header is not True:
        return False
    if cell.mapping.kind == "docling" and not cell.native_column_header:
        return False
    return (
        cell.column < answer.column + answer.column_span
        and answer.column < cell.column + cell.column_span
        and cell.row + cell.row_span <= answer.row
    )


def _check_table_links(index: Index, placement: Placement) -> None:
    """Reviewer anchors must be linked text or lie on the answer cell's own row and column."""
    units = {
        o.occurrence_id: u for u in index.units[placement.container_node_id] for _, _, o in u.spans
    }
    answers = [c for c in placement.components if c.role == "answer"]
    cells: dict[str, TableCell] = {}
    for c in answers:
        for i in c.occurrence_ids:
            owner = units[i].cell
            if owner is None:
                raise ValueError("a table answer must be carried by exactly one table cell")
            cells[owner.node_id] = owner
    if len(cells) != 1:
        raise ValueError("a table answer must be carried by exactly one table cell")
    answer = next(iter(cells.values()))
    if answer.is_header is not False:
        raise ValueError("the answer cell is not a data cell")
    has_row = has_column = False
    for component in placement.components:
        if component.role == "answer":
            continue
        for occ_id in component.occurrence_ids:
            if occ_id not in units:
                continue  # native ancestor section title
            cell = units[occ_id].cell
            if cell is None:
                continue  # linked caption or footnote text of this table
            row, column = _row_anchor(cell, answer), _column_anchor(cell, answer)
            if cell.node_id == answer.node_id or not (row or column):
                raise ValueError("anchor is not linked to the answer cell by row or column")
            has_row, has_column = has_row or row, has_column or column
    if not (has_row and has_column):
        raise ValueError("table evidence needs both a row and column anchor for the answer cell")
    if any(c.form == "mean_sd" for c in answers):
        stated = any(
            _MEAN_STATEMENT.search(u.text)
            for u in units.values()
            if u.cell is None or _column_anchor(u.cell, answer)
        )
        if not stated:
            raise ValueError("a linked caption, note or column header must state mean ± ")


def _scope_titles(index: Index, container: str) -> set[str]:
    """Titles that are native children of a section containing `container`.

    Only the parent chain recorded by the canonical document counts; a heading that
    merely precedes the container in reading order, or sits in a generic group, does not.
    """
    nodes = {n.node_id: n for n in index.document.nodes}
    titles: set[str] = set()
    current = nodes[container].parent_node_id
    while current is not None:
        node = nodes[current]
        if node.kind == "section":
            titles |= {
                n.node_id
                for n in index.document.nodes
                if n.parent_node_id == node.node_id
                and n.kind == "title"
                and n.node_id in index.units
            }
        current = node.parent_node_id
    return titles


def _heading_key(text: str) -> str:
    """Comparison key for a heading: typography is ignored, scientific signs and symbols are not."""
    text = "".join(
        unicodedata.normalize("NFKC", c) if "\ufb00" <= c <= "\ufb06" else c for c in text
    )
    text = unicodedata.normalize("NFC", text).translate(_TYPOGRAPHY)
    return " ".join(text.split()).casefold()


def _boxes(node: CanonicalNode) -> list[_Box] | None:
    """(page, left, top, right, bottom) per provenance entry; None if any entry is unusable."""
    boxes = []
    for entry in node.mapping.provenance:
        box = entry.bbox
        if (
            entry.page_no is None
            or box is None
            or box.coord_origin != "BOTTOMLEFT"
            or box.b > box.t
        ):
            return None
        boxes.append((entry.page_no, box.l, box.t, box.r, box.b))
    return boxes or None


def _layouts(nodes: list[CanonicalNode]) -> _Layouts:
    """Per page: text extent and whether the page's own text boxes demonstrate two columns."""
    boxes: dict[int, list[tuple[float, float]]] = {}
    for n in nodes:
        if n.kind in _TEXT_FLOW and n.native_content_layer == "body":
            for page, left, _, right, _ in _boxes(n) or []:
                boxes.setdefault(page, []).append((left, right))
    layouts = {}
    for page, spans in boxes.items():
        low, high = min(s[0] for s in spans), max(s[1] for s in spans)
        middle, slack = (low + high) / 2, _COLUMN_SLACK * (high - low)
        narrow = [s for s in spans if s[1] - s[0] < _WIDE * (high - low)]
        two = any(s[1] <= middle + slack for s in narrow) and any(
            s[0] >= middle - slack for s in narrow
        )
        layouts[page] = (low, high, two)
    return layouts


def _column(box: _Box, layouts: _Layouts) -> str | None:
    """'full', 'left' or 'right'; a page without demonstrated columns is read as one column.

    None when a box on a two-column page cannot be placed in a column.
    """
    low, high, two = layouts[box[0]]
    width, middle = high - low, (low + high) / 2
    if not two or box[3] - box[1] >= _WIDE * width:
        return "full"
    if box[3] <= middle + _COLUMN_SLACK * width:
        return "left"
    if box[1] >= middle - _COLUMN_SLACK * width:
        return "right"
    return None


def _reads_forward(before: _Box, after: _Box, layouts: _Layouts) -> bool:
    """Whether `after` can follow `before` in reading order on this page layout.

    Only layouts this check can establish are accepted: a later page, the same column or a
    full-width box lower on the page, or the left column followed by the right column.
    """
    if after[0] != before[0]:
        return after[0] > before[0]
    first, second = _column(before, layouts), _column(after, layouts)
    if first is None or second is None:
        return False
    if first == second or "full" in (first, second):
        return after[2] <= before[4] + _VERTICAL_TOLERANCE
    return first == "left"


def _flat_heading(
    index: Index, container: str, item: BenchmarkItem, set_index: int | None = None
) -> str | None:
    """The flat PDF heading that may anchor scope for `container`, or None (fail closed).

    The canonical PDF has no section ancestry, so nothing is inferred from a nearby title
    alone. The heading must be the nearest preceding title in the same document, equal one
    of the item's frozen PDF section labels (a check on existing canonical text, never a
    source of text), and the page and column geometry along the reading chain from the
    heading to the container must contradict nothing. Tables, tied or competing headings,
    unpositioned or unmeasurable nodes, and a complete verbatim placement of the same
    evidence elsewhere in the heading's extent all reject. Structural identity is not
    scientific sufficiency; that stays with the reviewer.
    """
    document = index.document
    if document.representation != "PDF_DOCLING":
        return None
    nodes = document.nodes
    block = next(n for n in nodes if n.node_id == container)
    if block.kind not in {"paragraph", "list_item"} or block.native_content_layer != "body":
        return None
    before = [
        t
        for t in nodes[: block.order]
        if t.kind == "title" and t.native_content_layer != "furniture"
    ]
    if not before:
        return None
    heading = before[-1]
    parent = next((n for n in nodes if n.node_id == heading.parent_node_id), None)
    if (
        heading.native_content_layer != "body"
        or heading.native_label != "section_header"
        or heading.node_id not in index.units
        or (parent is not None and parent.kind in _NOT_A_HEADING_PARENT)
    ):
        return None
    labels = {_heading_key(r.section) for r in item.pdf_refs if r.section}
    if _heading_key(" ".join(u.text for u in index.units[heading.node_id])) not in labels:
        return None
    chain = [
        n
        for n in nodes[heading.order : block.order + 1]
        if n.kind in _TEXT_FLOW and n.native_content_layer == "body"
    ]
    measured = [_boxes(n) for n in chain]
    if any(m is None for m in measured) or any(
        n.native_order_state == "unpositioned" for n in chain
    ):
        return None
    entries = [e for m in measured if m for e in m]
    layouts = _layouts(nodes)
    if any(not _reads_forward(a, b, layouts) for a, b in pairwise(entries)):
        return None
    # Body text outside the chain that sits on a page the chain skips, before the heading on
    # a later page, or after the block on an earlier page means the order is not established.
    inside = {n.node_id for n in chain}
    first_page, last_page = entries[0][0], entries[-1][0]
    for n in nodes:
        if n.node_id in inside or n.kind not in _TEXT_FLOW or n.native_content_layer != "body":
            continue
        pages = [e[0] for e in _boxes(n) or []]
        if pages and (
            any(first_page < page < last_page for page in pages)
            or (n.order < heading.order and max(pages) > first_page)
            or (n.order > block.order and min(pages) < last_page)
        ):
            return None
    end = next(
        (
            n.order
            for n in nodes[block.order + 1 :]
            if n.kind == "title" and n.native_content_layer != "furniture"
        ),
        len(nodes),
    )
    for other in index.units:
        position = next(n.order for n in nodes if n.node_id == other)
        if other == container or not heading.order < position < end:
            continue
        for number in range(len(item.evidence_sets)) if set_index is None else [set_index]:
            components: list[tuple[str, str]] = [
                (c.role, c.text) for c in item.evidence_sets[number].components
            ]
            if _place(index, other, components, [], number, set()) is not None:
                return None
    return heading.node_id


def _scope_titles_for(
    index: Index, container: str, item: BenchmarkItem, set_index: int | None = None
) -> set[str]:
    """Native ancestor titles, plus the accepted flat PDF heading when there is one."""
    titles = _scope_titles(index, container)
    heading = _flat_heading(index, container, item, set_index)
    return titles | {heading} if heading is not None else titles


def _check_loci(judgment: StageAJudgment, item: BenchmarkItem, index: Index) -> None:
    for locus in judgment.sets:
        if locus.set_index >= len(item.evidence_sets):
            raise ValueError("locus names an unknown evidence set")
        expected = [(c.role, c.text) for c in item.evidence_sets[locus.set_index].components]
        for placement in locus.placements:
            if placement.container_node_id not in index.units:
                raise ValueError("placement names an unknown text container")
            if [(c.role, c.text) for c in placement.components] != expected:
                raise ValueError("placement must cover exactly the item's evidence components")
            table = any(
                n.table is not None
                for n in index.document.nodes
                if n.node_id == placement.container_node_id
            )
            scope = _scope_titles_for(index, placement.container_node_id, item, locus.set_index)
            for component in placement.components:
                for occ_id in component.occurrence_ids:
                    owner = index.where.get(occ_id)
                    if owner == placement.container_node_id:
                        continue
                    # An attested descriptor may cite the title of a native ancestor section.
                    if component.equivalence_note is None or owner not in scope:
                        raise ValueError("occurrence is not text in the named container")
                if component.equivalence_note is not None:
                    continue  # attested; the anchors are checked structurally below
                surface = component.answer_surface
                if surface is not None:
                    if table:
                        raise ValueError("an answer surface applies to prose containers only")
                    others = [c.text for c in placement.components if c is not component]
                    if problem := surface_problem(component.text, surface, others):
                        raise ValueError(f"answer surface rejected: {problem}")
                carried = surface.canonical_text if surface else component.observed_text
                if component.form == "mean_sd":
                    if not table:
                        raise ValueError("mean ± SD applies to table cells only")
                    found = _mean_sd_matches(index, placement.container_node_id, component.text)
                elif carried is None:
                    found = _component_matches(
                        index, placement.container_node_id, component.role, component.text, table
                    )
                else:
                    found = [
                        (u, s)
                        for u in index.units[placement.container_node_id]
                        if (s := _spans(u.text, carried, bounded=False))
                    ]
                allowed = {i for unit, spans in found for i in unit.ids(spans)}
                if not allowed or not set(component.occurrence_ids) <= allowed:
                    raise ValueError("component occurrences do not carry the recorded text")
            if table and any(
                not c.verbatim and c.observed_text is None for c in placement.components
            ):
                _check_table_links(index, placement)


def validate_judgment(
    judgment: StageAJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> None:
    """Scientific read-back of a judgment against its item and canonical document."""
    check_item(item, document, canonical)
    if (
        judgment.item_sha256 != item_digest(item)
        or (
            judgment.benchmark_version,
            judgment.corpus_version,
            judgment.item_id,
            judgment.paper_id,
        )
        != (item.benchmark_version, item.corpus_version, item.item_id, item.paper_id)
        or judgment.representation != document.representation
        or judgment.canonical != canonical
    ):
        raise ValueError("judgment identity differs from the supplied item and canonical artifact")
    candidate = locate(item, document, canonical)
    candidate_ref = stage_a_ref(candidate)
    if judgment.candidate is not None and judgment.candidate != candidate_ref:
        raise ValueError("judgment cites a candidate that is not the locator output")
    if judgment.basis == "deterministic" and (
        candidate.determination != "automatic_preserved" or judgment.sets != candidate.sets
    ):
        raise ValueError("deterministic judgment is not reproduced by the locator")
    if candidate.determination == "automatic_preserved" and judgment.label in {
        "absent",
        "scientifically_corrupted",
    }:
        raise ValueError(f"{judgment.label} contradicts a complete verbatim located fact")
    _check_loci(
        judgment, item, _index(CanonicalDocument.model_validate(document.model_dump()), canonical)
    )
    observed = [
        c.observed_text is not None
        for s in judgment.sets
        for p in s.placements
        for c in p.components
    ]
    if judgment.label == "preserved" and any(observed):
        raise ValueError("preserved evidence must be verbatim; use altered_but_sufficient")
    if judgment.label == "altered_but_sufficient" and not any(observed):
        raise ValueError("altered evidence must record an observed text that differs")
