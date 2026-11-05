"""Conservative Stage A locator and judgment validation.

Text matching here only locates candidate evidence. A failed match is never
evidence of absence or corruption: those labels need a reviewer's judgment.
"""

import re
from dataclasses import dataclass, field

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.chunking import discover_occurrences
from biorag_repro.documents.models import CanonicalDocument, TableCell, TableData
from biorag_repro.evaluation.models import (
    ComponentLocus,
    Placement,
    SetLocus,
    StageACandidate,
    StageAJudgment,
    Suggestion,
)
from biorag_repro.retrieval.models import SourceOccurrence

BLOCKS = {"paragraph", "list_item", "caption", "title", "formula", "figure", "reference"}
TABLE_ROLES = {"answer", "row", "column"}
_BEFORE = set(".,+-−_")
_AFTER = set("_%^×⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_NUMBER = re.compile(r"\d[\d.,]*")
_SD_TAIL = re.compile(r"\s*±\s*\d+(?:\.\d+)?\s*")
_MEAN_STATEMENT = re.compile(r"\bmean\s*±", re.IGNORECASE)
_SUGGESTION_LIMIT = 5


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
    return cell.row <= answer.row < cell.row + cell.row_span and (
        cell.column + cell.column_span <= answer.column
    )


def _column_anchor(cell: TableCell, answer: TableCell) -> bool:
    if cell.is_header is not True:
        return False
    if cell.mapping.kind == "docling" and not cell.native_column_header:
        return False
    return cell.column <= answer.column < cell.column + cell.column_span and (
        cell.row + cell.row_span <= answer.row
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
            for component in placement.components:
                for occ_id in component.occurrence_ids:
                    if index.where.get(occ_id) != placement.container_node_id:
                        raise ValueError("occurrence is not text in the named container")
                if component.equivalence_note is not None:
                    continue  # attested; the anchors are checked structurally below
                if component.form == "mean_sd":
                    if not table:
                        raise ValueError("mean ± SD applies to table cells only")
                    found = _mean_sd_matches(index, placement.container_node_id, component.text)
                elif component.observed_text is None:
                    found = _component_matches(
                        index, placement.container_node_id, component.role, component.text, table
                    )
                else:
                    found = [
                        (u, s)
                        for u in index.units[placement.container_node_id]
                        if (s := _spans(u.text, component.observed_text, bounded=False))
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
