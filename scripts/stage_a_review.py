"""Seeded, representation-masked Stage A review packets.

Development rehearsals draw a sample of development items. `final-queue` builds the packet of
all 610 final-test item x representation units, only after the researcher's evaluation
authorization is verified, inside `stage_a.final_test_scope`.

The reviewer-facing packet shows canonical source text exactly as stored. Only identifiers,
file names, representation names and presentation wrappers are withheld. Unmasked provenance
is written to a separate directory and its hash is fixed in the session manifest before any
review. Judgments use the existing Stage A validator. An explicit publication command freezes
only supplied human judgments from a development review queue.
"""

import argparse
import json
import math
import random
import re
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, get_args

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.ingestion import Phase4Validation
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import (
    ComponentLocus,
    Label,
    Placement,
    SetLocus,
    StageAJudgment,
)
from biorag_repro.evaluation.stage_a import (
    Index,
    _index,
    _scope_titles_for,
    _spans,
    item_digest,
    locate,
    stage_a_ref,
    validate_judgment,
)

REHEARSAL_PAPERS = ("P006", "P007", "P022", "P042", "P043")
EXCLUDED_PAPERS = frozenset({"P027"})  # judged unmasked on 2026-11-05/06
REPRESENTATIONS = ("JATS", "PDF_DOCLING")
GUESSES = (*REPRESENTATIONS, "cannot_tell")
LABELS: tuple[str, ...] = get_args(Label)
GAP_FRACTION = 0.3
ROLES = ("first_rater", "second_rater", "intra_rater")
STATUS = "development_rehearsal_not_for_inventory"
FINAL_STATUS = "final_test_pending_human_review"
PUBLISHABLE = ("development_pending_human_review", FINAL_STATUS)
VIEW_SHA256 = "5ff34235998b1aa8281e986e5a720a0ef23f62a6c2158f789bd145b17fe99f4d"
FULL_BENCHMARK_UNITS = 610  # 305 final-test items x 2 representations (planning figure)
FORBIDDEN_WORDS = re.compile(r"jats|pdf|docling|xml", re.IGNORECASE)


class ReviewError(ValueError):
    """The packet or a review record cannot be accepted."""


# ---- sources and selection ------------------------------------------------------------


@dataclass
class Source:
    item: BenchmarkItem
    document: CanonicalDocument
    root: Path
    canonical: ArtifactRef

    @property
    def representation(self) -> str:
        return self.document.representation


def structure(item: BenchmarkItem) -> str:
    roles = {c.role for s in item.evidence_sets for c in s.components}
    return "table" if roles & {"row", "column"} else "prose"


def select_items(
    items: list[BenchmarkItem],
    papers: tuple[str, ...],
    pairs: int,
    seed: int,
    *,
    include_previously_reviewed: bool = False,
) -> list[BenchmarkItem]:
    """Seeded draw: papers balanced first, then uncovered and rarer class x structure strata."""
    if set(papers) & EXCLUDED_PAPERS and not include_previously_reviewed:
        raise ReviewError("an excluded paper was requested")
    pool = sorted(
        (i for i in items if i.paper_id in papers and i.split == "development"),
        key=lambda i: i.item_id,
    )
    missing = set(papers) - {i.paper_id for i in pool}
    if missing or not 0 < pairs <= len(pool):
        raise ReviewError(f"cannot draw {pairs} items; no items for {sorted(missing)}")
    rng = random.Random(seed)
    chosen: list[BenchmarkItem] = []
    covered: set[tuple[str, str]] = set()
    per_paper: Counter[str] = Counter()
    size = Counter((i.answer_class, structure(i)) for i in pool)
    while len(chosen) < pairs:
        left = [i for i in pool if i not in chosen]

        def score(i: BenchmarkItem) -> tuple[int, int, int]:
            stratum = (i.answer_class, structure(i))
            return (-per_paper[i.paper_id], int(stratum not in covered), -size[stratum])

        best = max(score(i) for i in left)
        pick = rng.choice([i for i in left if score(i) == best])
        chosen.append(pick)
        covered.add((pick.answer_class, structure(pick)))
        per_paper[pick.paper_id] += 1
    return sorted(chosen, key=lambda i: i.item_id)


def min_gap(pairs: int, fraction: float = GAP_FRACTION) -> int:
    return min(pairs, math.ceil(fraction * 2 * pairs))


def check_separation(order: list[tuple[str, str]], gap: int) -> None:
    """Every item appears once per representation, at least `gap` positions apart."""
    where: dict[str, dict[str, int]] = {}
    for position, (item_id, representation) in enumerate(order):
        if representation in where.setdefault(item_id, {}):
            raise ReviewError(f"{item_id} repeats {representation}")
        where[item_id][representation] = position
    for item_id, arms in where.items():
        if set(arms) != set(REPRESENTATIONS):
            raise ReviewError(f"{item_id} is not a complete pair")
        if abs(arms["JATS"] - arms["PDF_DOCLING"]) < gap:
            raise ReviewError(f"{item_id} representations are closer than {gap} positions")


def orientations(
    ids: list[str], groups: dict[str, str] | None, rng: random.Random
) -> dict[str, str]:
    """Which representation each item shows first: balanced overall and within each group.

    Inside a group the first arms alternate from a random start; a group with an odd count
    leaves one extra, given to whichever side is behind so the total differs by at most one.
    """
    by_group: dict[str, list[str]] = {}
    for item_id in ids:
        by_group.setdefault((groups or {}).get(item_id, ""), []).append(item_id)
    first: dict[str, str] = {}
    totals = {"JATS": 0, "PDF_DOCLING": 0}
    for group in sorted(by_group):
        members = by_group[group]
        rng.shuffle(members)
        behind = min(totals, key=lambda a: (totals[a], rng.random()))
        other = "PDF_DOCLING" if behind == "JATS" else "JATS"
        for k, item_id in enumerate(members):
            first[item_id] = behind if k % 2 == 0 else other
            totals[first[item_id]] += 1
    return first


def schedule(
    item_ids: list[str],
    seed: int,
    fraction: float = GAP_FRACTION,
    groups: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Seeded order in which the two arms of an item are at least `gap` positions apart.

    Construction, not rejection: item i's first arm is placed at i and its second arm at
    n + (i + shift) % n with shift >= gap, which keeps every distance >= gap. Seeded swaps
    that preserve both the distance and each item's first arm then remove the regular
    layout, so the balanced orientation (see `orientations`) survives.
    """
    if not item_ids or len(set(item_ids)) != len(item_ids):
        raise ReviewError("items must be nonempty and distinct")
    rng = random.Random(seed)
    ids = sorted(item_ids)
    rng.shuffle(ids)
    n = len(ids)
    gap = min_gap(n, fraction)
    shift = rng.randint(gap, n)
    first = orientations(ids, groups, rng)
    order: list[tuple[str, str]] = [("", "")] * (2 * n)
    for i, item_id in enumerate(ids):
        other = "PDF_DOCLING" if first[item_id] == "JATS" else "JATS"
        order[i] = (item_id, first[item_id])
        order[n + (i + shift) % n] = (item_id, other)
    check_separation(order, gap)
    for _ in range(20 * n):
        a, b = rng.sample(range(2 * n), 2)
        order[a], order[b] = order[b], order[a]
        try:
            check_separation(order, gap)
            if first_arms(order) != first:
                raise ReviewError("orientation changed")
        except ReviewError:
            order[a], order[b] = order[b], order[a]
    check_separation(order, gap)
    return order


def first_arms(order: list[tuple[str, str]]) -> dict[str, str]:
    seen: dict[str, str] = {}
    for item_id, representation in order:
        seen.setdefault(item_id, representation)
    return seen


# ---- masked rendering -----------------------------------------------------------------

Part = tuple[str, str]  # ("scaffold" | "source", text)
Line = list[Part]


def S(text: str) -> Part:
    return ("scaffold", text)


def T(text: str) -> Part:
    return ("source", text)


class Analysis:
    """Everything the packet needs about one item x representation unit."""

    def __init__(self, handle: str, seed: int, source: Source) -> None:
        self.handle = handle
        self.source = source
        self.index: Index = _index(source.document, source.canonical)
        self.candidate = locate(source.item, source.document, source.canonical)
        self.nodes = {n.node_id: n for n in source.document.nodes}
        self.handles = self._handles(seed)
        self.container_of = {h: c for c, h in self.handles.items()}

    def _handles(self, seed: int) -> dict[str, str]:
        for width in range(5, 20):
            handles = {
                c: "H" + sha256_bytes(f"{seed}|{self.handle}|{c}".encode())[:width]
                for c in self.index.units
            }
            if len(set(handles.values())) == len(handles):
                return handles
        raise ReviewError("container handles collide")

    def shown(self) -> list[str]:
        """Containers holding candidate text for any component, plus their neighbours."""
        containers = list(self.index.units)
        hit: set[str] = {
            self.index.where[o]
            for s in self.candidate.suggestions
            for o in s.occurrence_ids
            if o in self.index.where
        } | {p.container_node_id for s in self.candidate.sets for p in s.placements}
        for evidence in self.source.item.evidence_sets:
            for component in evidence.components:
                found = [
                    c
                    for c, units in self.index.units.items()
                    if any(_spans(u.text, component.text) for u in units)
                ]
                hit |= set(found[:5])
        wanted = set(hit)
        for c in hit:
            position = containers.index(c)
            wanted |= set(containers[max(0, position - 1) : position + 2])
        return [c for c in containers if c in wanted]

    def render_container(self, container: str) -> list[Line]:
        node = self.nodes[container]
        lines: list[Line] = [[S(f"[{self.handles[container]}] {node.kind}")]]
        scope = _scope_titles_for(self.index, container, self.source.item)
        titles = [t for t in self.index.units if t in scope]
        for title in titles:
            lines.append(
                [S("  section title: "), T(" ".join(u.text for u in self.index.units[title]))]
            )
        units = self.index.units[container]
        if node.table is None:
            return lines + [[S("  "), T(u.text)] for u in units]
        by_cell = {u.cell.node_id: u.text for u in units if u.cell is not None}
        lines += [[S("  "), T(u.text)] for u in units if u.cell is None]
        grid: dict[int, list[Line]] = {}
        for cell in sorted(node.table.cells, key=lambda c: (c.row, c.column)):
            span = (
                f" span {cell.row_span}x{cell.column_span}"
                if (cell.row_span, cell.column_span) != (1, 1)
                else ""
            )
            head = " header" if cell.is_header else ""
            grid.setdefault(cell.row, []).append(
                [
                    S(f"  r{cell.row} c{cell.column}{head}{span}: "),
                    T(by_cell.get(cell.node_id, cell.text)),
                ]
            )
        for row in sorted(grid):
            lines.extend(grid[row])
        return lines

    def render_unit(self) -> list[Line]:
        item = self.source.item
        lines: list[Line] = [[S(f"## {self.handle}")], [S("Question: "), T(item.question)]]
        lines.append([S("Minimum sufficient evidence (any one set suffices):")])
        for number, evidence in enumerate(item.evidence_sets):
            lines.append([S(f"  set {number}")])
            lines += [[S(f"    {c.role}: "), T(c.text)] for c in evidence.components]
        lines.append([S("Locator notes (search aids, not proof of absence):")])
        notes = [
            [
                S(
                    f"  - {s.kind}"
                    + (f" (set {s.set_index})" if s.set_index is not None else "")
                    + ": "
                ),
                T(s.detail),
            ]
            for s in self.candidate.suggestions
        ]
        lines += notes or [[S("  - none")]]
        lines.append([S("Candidate contexts:")])
        shown = self.shown()
        if not shown:
            lines.append([S("  none located; inspect the full section or document")])
        for container in shown:
            lines += self.render_container(container)
        return lines

    def render_section(self, handle: str | None) -> list[Line]:
        """Full canonical content of the section holding a container, or of the document."""
        if handle is None:
            wanted = list(self.index.units)
        else:
            start = self.container_of[handle]
            section = self._section_of(start)
            wanted = [c for c in self.index.units if section is None or self._under(c, section)]
        lines: list[Line] = []
        for container in wanted:
            lines += self.render_container(container)
        return lines

    def _chain(self, node_id: str) -> list[str]:
        chain: list[str] = []
        current: str | None = node_id
        while current is not None:
            chain.append(current)
            current = self.nodes[current].parent_node_id
        return chain

    def _section_of(self, container: str) -> str | None:
        return next((n for n in self._chain(container) if self.nodes[n].kind == "section"), None)

    def _under(self, container: str, section: str) -> bool:
        return section in self._chain(container)

    def identifiers(self) -> set[str]:
        document, source = self.source.document, self.source
        found = {n.node_id for n in document.nodes} | set(self.index.order)
        found |= {
            source.canonical.artifact_id,
            source.canonical.sha256,
            document.source.artifact_id,
        }
        found |= {source.canonical.logical_ref, Path(source.canonical.logical_ref).name}
        found |= {source.item.item_id, source.item.paper_id, source.root.name}
        if document.native_artifact is not None:
            found.add(document.native_artifact.artifact_id)
        return {i for i in found if i}


def to_text(lines: list[Line]) -> str:
    return "\n".join("".join(text for _, text in line) for line in lines)


def scaffold_leaks(lines: list[Line], identifiers: set[str]) -> list[str]:
    """System-generated leaks only: genuine source text may contain any word, even 'PDF'."""
    scaffold = "\n".join(text for line in lines for kind, text in line if kind == "scaffold")
    leaks = [m.group(0) for m in FORBIDDEN_WORDS.finditer(scaffold)]
    # The substring test only skips identifiers that cannot match; the regex decides.
    leaks += [
        i
        for i in identifiers
        if i in scaffold and re.search(rf"(?<!\w){re.escape(i)}(?!\w)", scaffold)
    ]
    return leaks


INSTRUCTIONS = """# Stage A review packet (development rehearsal, masked view)

Judge only the text shown for each unit: it is the stored canonical content, unedited.
Lines after a label are source text; labels such as [H1a2b3] are local handles.
Locator notes are search aids and never proof of absence. Before `absent`,
`scientifically_corrupted` or `unresolved`, inspect the full section or document with
`inspect` (it stays within this unit's own representation).

Labels: preserved | altered_but_sufficient | scientifically_corrupted | absent | unresolved
Order per unit: `start` -> edit review.json -> `finish` -> `guess`.
"""


# ---- packet files ---------------------------------------------------------------------


def dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha256_bytes(data)


def ref_dict(ref: ArtifactRef) -> dict[str, str]:
    return {"artifact_id": ref.artifact_id, "logical_ref": ref.logical_ref, "sha256": ref.sha256}


def template(handle: str) -> dict[str, Any]:
    return {
        "handle": handle,
        "label": None,
        "alteration": None,
        "rationale": None,
        "sets": [],  # [{"set_index": 0, "placements": [{"container": "H..", "components": [
        #   {"role": "answer", "text": "<benchmark text>", "found": null,
        #    "equivalence_note": null, "form": null}]}]}]
        "consulted_original_source": None,
        "started_utc": None,
        "finished_utc": None,
        "representation_guess": None,
        "guess_confidence": None,
        "guessed_utc": None,
    }


def code_version() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_packet(
    sources: dict[tuple[str, str], Source],
    items: list[BenchmarkItem],
    *,
    out: Path,
    seed: int,
    papers: tuple[str, ...],
    pairs: int | None = None,
    fraction: float | None = None,
    role: str = "first_rater",
    created_utc: str,
    code: str | None,
    previous_manifest: Path | None = None,
    include_previously_reviewed: bool = False,
    final_test: bool = False,
    authorization: dict[str, str] | None = None,
) -> None:
    if final_test:
        # Every final-test item x representation unit is reviewed; nothing is sampled.
        if role != "first_rater" or pairs is not None or fraction is not None:
            raise ReviewError("the final-test packet is the complete first-rater queue")
        chosen = sorted(items, key=lambda i: i.item_id)
        if any(i.split != "final_test" for i in chosen):
            raise ReviewError("the final-test packet holds final-test items only")
        if authorization is None:
            raise ReviewError("the final-test packet records its evaluation authorization")
    else:
        if role not in ROLES or (pairs is None) == (fraction is None):
            raise ReviewError("give a known role and exactly one of pairs/fraction")
        eligible = [i for i in items if i.paper_id in papers]
        count = pairs if pairs is not None else math.ceil((fraction or 0) * len(eligible))
        if role == "intra_rater":
            check_washout(previous_manifest, seed, created_utc)
        chosen = select_items(
            items, papers, count, seed, include_previously_reviewed=include_previously_reviewed
        )
    order = schedule(
        [i.item_id for i in chosen], seed, groups={i.item_id: i.paper_id for i in chosen}
    )
    handles = [f"R{position:03d}" for position in range(1, len(order) + 1)]
    rendered = {}
    for handle, key in zip(handles, order, strict=True):
        rendered[handle] = render(Analysis(handle, seed, sources[key]))
    _write_packet(
        out, rendered, order, handles, seed=seed, chosen=len(chosen), created_utc=created_utc,
        code=code, role=role, papers=papers, final_test=final_test,
        include_previously_reviewed=include_previously_reviewed,
        previous_manifest=previous_manifest, authorization=authorization,
    )  # fmt: skip


def render(analysis: "Analysis") -> tuple[list[Line], dict[str, Any]]:
    """The masked lines of one unit and its unmasked provenance record; leaks are refused."""
    lines = analysis.render_unit()
    leaks = scaffold_leaks(lines, analysis.identifiers())
    if leaks:
        raise ReviewError(f"{analysis.handle}: masking leak {sorted(set(leaks))}")
    a = analysis
    record = {
        "item": a.source.item.model_dump(mode="json"),
        "item_sha256": item_digest(a.source.item),
        "representation": a.source.representation,
        "root": str(a.source.root),
        "canonical": ref_dict(a.source.canonical),
        "candidate": ref_dict(stage_a_ref(a.candidate)),
        "containers": {h: c for c, h in a.handles.items()},
        "shown": [a.handles[c] for c in a.shown()],
    }
    return lines, record


def _write_packet(
    out: Path,
    rendered: dict[str, tuple[list[Line], dict[str, Any]]],
    order: list[tuple[str, str]],
    handles: list[str],
    *,
    seed: int,
    chosen: int,
    created_utc: str,
    code: str | None,
    role: str,
    papers: tuple[str, ...],
    final_test: bool,
    include_previously_reviewed: bool,
    previous_manifest: Path | None,
    authorization: dict[str, str] | None,
) -> None:
    view_lines: list[Line] = [[S(INSTRUCTIONS)]]
    for handle in handles:
        view_lines += [*rendered[handle][0], [S("")]]
    reviewer, provenance = out / "reviewer", out / "provenance"
    view_sha = write(reviewer / "masked_view.md", (to_text(view_lines) + "\n").encode())
    review_sha = write(reviewer / "review.json", dumps([template(h) for h in handles]))
    write(reviewer / "inspections.jsonl", b"")
    order_sha = write(
        provenance / "order.json",
        dumps(
            {
                "seed": seed,
                "gap_fraction": GAP_FRACTION,
                "gap": min_gap(chosen),
                "orientation": "balanced overall and within papers",
                "order": [[h, i, r] for h, (i, r) in zip(handles, order, strict=True)],
            }
        ),
    )
    units = {h: rendered[h][1] for h in handles}
    unmasked_sha = write(provenance / "unmasked_evidence.json", dumps(units))
    manifest = {
        "status": (
            FINAL_STATUS
            if final_test
            else "development_pending_human_review"
            if include_previously_reviewed
            else STATUS
        ),
        "role": role,
        "seed": seed,
        "papers": list(papers),
        "excluded_papers": (
            [] if include_previously_reviewed or final_test else sorted(EXCLUDED_PAPERS)
        ),
        "previously_reviewed_papers": sorted(set(papers) & EXCLUDED_PAPERS),
        "publication_requires_human_judgments": True,
        "pairs": chosen,
        "units": len(order),
        "gap": min_gap(chosen),
        "created_utc": created_utc,
        "code_commit": code,
        "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "order_sha256": order_sha,
        "masked_view_sha256": view_sha,
        "review_template_sha256": review_sha,
        "unmasked_evidence_sha256": unmasked_sha,
        "canonical_documents": sorted({u["canonical"]["sha256"] for u in units.values()}),
        "previous_manifest_sha256": (
            sha256_bytes(previous_manifest.read_bytes()) if previous_manifest else None
        ),
        "min_washout_days": 14 if role == "intra_rater" else None,
    }
    if final_test:
        manifest["authorization"] = authorization
    write(provenance / "session_manifest.json", dumps(manifest))


def build_final_packet(
    sources: dict[tuple[str, str], Source],
    items: list[BenchmarkItem],
    *,
    out: Path,
    seed: int,
    created_utc: str,
    code: str | None,
    authorization: dict[str, str],
    stop_after: int | None = None,
) -> int:
    """The complete final-test packet, built one unit at a time in a resumable staging area.

    The seeded plan (order, handles, item and document identities, authorization) is fixed
    first. Each unit is rendered once, grouped by document so its index is built once, and
    staged as a hash-checked record. A restart keeps only records that match the plan; the
    packet is written from the staged records and published by one rename, so `out` either
    does not exist or is complete. Returns the number of units still missing (0 = published).
    """
    if out.exists():
        raise ReviewError("the final-test packet already exists")
    chosen = sorted(items, key=lambda i: i.item_id)
    if any(i.split != "final_test" for i in chosen):
        raise ReviewError("the final-test packet holds final-test items only")
    order = schedule(
        [i.item_id for i in chosen], seed, groups={i.item_id: i.paper_id for i in chosen}
    )
    handles = [f"R{position:03d}" for position in range(1, len(order) + 1)]
    plan = {
        "status": "incomplete_final_test_packet_not_for_review",
        "seed": seed,
        "authorization": authorization,
        "units": {
            h: {
                "item_id": i,
                "representation": r,
                "item_sha256": item_digest(sources[(i, r)].item),
                "canonical_sha256": sources[(i, r)].canonical.sha256,
            }
            for h, (i, r) in zip(handles, order, strict=True)
        },
    }
    staging = out.with_name(out.name + ".incomplete")
    plan_bytes = dumps(plan)
    if (staging / "plan.json").exists():
        if (staging / "plan.json").read_bytes() != plan_bytes:
            raise ReviewError("staged work belongs to another plan; it is not reused")
    else:
        write(staging / "plan.json", plan_bytes)
    plan_sha = sha256_bytes(plan_bytes)

    def staged(handle: str) -> tuple[list[Line], dict[str, Any]] | None:
        path = staging / "units" / f"{handle}.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_bytes())
            payload = value["payload"]
            entry = plan["units"][handle]
            if (
                value["payload_sha256"] != sha256_bytes(dumps(payload))
                or payload["plan_sha256"] != plan_sha
                or payload["handle"] != handle
                or payload["record"]["item_sha256"] != entry["item_sha256"]
                or payload["record"]["canonical"]["sha256"] != entry["canonical_sha256"]
            ):
                raise ValueError("staged unit does not match the plan")
        except (ValueError, KeyError, TypeError):
            path.unlink()  # rebuilt below; an unverified record is never kept
            return None
        lines = [[(kind, text) for kind, text in line] for line in payload["lines"]]
        return lines, payload["record"]

    # Grouping by document keeps one document index in memory and builds it once.
    by_document = sorted(
        zip(handles, order, strict=True), key=lambda x: (sources[x[1]].canonical.sha256, x[0])
    )
    done = 0
    for handle, key in by_document:
        if staged(handle) is not None:
            continue
        if stop_after is not None and done >= stop_after:
            return sum(1 for h in handles if not (staging / "units" / f"{h}.json").exists())
        lines, record = render(Analysis(handle, seed, sources[key]))
        payload = {"plan_sha256": plan_sha, "handle": handle, "lines": lines, "record": record}
        data = dumps({"payload": payload, "payload_sha256": sha256_bytes(dumps(payload))})
        target = staging / "units" / f"{handle}.json"
        write(target.with_suffix(".tmp"), data)
        target.with_suffix(".tmp").replace(target)
        done += 1
    rendered = {}
    for handle in handles:
        unit = staged(handle)
        if unit is None:
            raise ReviewError(f"{handle} failed verification after staging")
        rendered[handle] = unit
    assembled = staging / "packet"
    if assembled.exists():
        shutil.rmtree(assembled)
    _write_packet(
        assembled, rendered, order, handles, seed=seed, chosen=len(chosen),
        created_utc=created_utc, code=code, role="first_rater",
        papers=tuple(sorted({i.paper_id for i in chosen})), final_test=True,
        include_previously_reviewed=False, previous_manifest=None, authorization=authorization,
    )  # fmt: skip
    assembled.rename(out)
    shutil.rmtree(staging)
    return 0


def check_washout(previous: Path | None, seed: int, created_utc: str) -> None:
    if previous is None:
        raise ReviewError("an intra-rater packet needs the first session's manifest")
    first = json.loads(previous.read_bytes())
    gap = parse_utc(created_utc) - parse_utc(first["created_utc"])
    if gap.days < 14 or seed == first["seed"]:
        raise ReviewError("intra-rater re-review needs >= 14 days and a new seed")


def parse_utc(value: str) -> datetime:
    if not value.endswith("Z"):
        raise ReviewError(f"timestamp {value!r} is not UTC with a Z suffix")
    return datetime.fromisoformat(value)


def now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


# ---- loading a packet -----------------------------------------------------------------


def load_provenance(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Unmasked records, verified against the hashes fixed in the session manifest."""
    manifest = json.loads((out / "provenance" / "session_manifest.json").read_bytes())
    data = (out / "provenance" / "unmasked_evidence.json").read_bytes()
    if sha256_bytes(data) != manifest["unmasked_evidence_sha256"]:
        raise ReviewError("unmasked provenance differs from the session manifest")
    return manifest, json.loads(data)


def load_analysis(handle: str, seed: int, record: dict[str, Any]) -> Analysis:
    canonical = ArtifactRef(**record["canonical"])
    root = Path(record["root"])
    document = read_project_json(root / canonical.logical_ref, canonical.sha256, CanonicalDocument)
    item = BenchmarkItem.model_validate(record["item"])
    if item_digest(item) != record["item_sha256"]:
        raise ReviewError(f"{handle}: stored item differs from its recorded hash")
    return Analysis(handle, seed, Source(item, document, root, canonical))


def read_review(out: Path) -> list[dict[str, Any]]:
    data = json.loads((out / "reviewer" / "review.json").read_bytes())
    if not isinstance(data, list):
        raise ReviewError("review.json must be a list")
    return data  # type: ignore[no-any-return]


def save_review(out: Path, records: list[dict[str, Any]]) -> None:
    (out / "reviewer" / "review.json").write_bytes(dumps(records))


def find(records: list[dict[str, Any]], handle: str) -> dict[str, Any]:
    matches = [r for r in records if r.get("handle") == handle]
    if len(matches) != 1:
        raise ReviewError(f"{handle}: expected exactly one record")
    return matches[0]


# ---- timing and inspection commands ---------------------------------------------------


def start(out: Path, handle: str, stamp: str) -> None:
    records = read_review(out)
    record = find(records, handle)
    if record["started_utc"] is not None:
        raise ReviewError(f"{handle} was already started")
    record["started_utc"] = stamp
    save_review(out, records)


def finish(out: Path, handle: str, stamp: str) -> None:
    records = read_review(out)
    record = find(records, handle)
    if record["started_utc"] is None or record["finished_utc"] is not None:
        raise ReviewError(f"{handle} must be started and not yet finished")
    if record["label"] not in LABELS or not (record["rationale"] or "").strip():
        raise ReviewError(f"{handle} needs a label and rationale before finishing")
    record["finished_utc"] = stamp
    save_review(out, records)


def guess(out: Path, handle: str, representation: str, confidence: int, stamp: str) -> None:
    """The representation guess is recorded only after the scientific judgment is finished."""
    records = read_review(out)
    record = find(records, handle)
    if record["finished_utc"] is None or record["guessed_utc"] is not None:
        raise ReviewError(f"{handle} must be finished and not yet guessed")
    if representation not in GUESSES or confidence not in range(1, 6):
        raise ReviewError("guess is JATS, PDF_DOCLING or cannot_tell with confidence 1-5")
    record.update(
        representation_guess=representation, guess_confidence=confidence, guessed_utc=stamp
    )
    save_review(out, records)


def inspect(out: Path, handle: str, container: str | None, stamp: str) -> str:
    manifest, units = load_provenance(out)
    record = units[handle]
    analysis = load_analysis(handle, manifest["seed"], record)
    if container is not None and container not in analysis.container_of:
        raise ReviewError(f"{handle}: unknown container {container}")
    lines = analysis.render_section(container)
    leaks = scaffold_leaks(lines, analysis.identifiers())
    if leaks:
        raise ReviewError(f"{handle}: masking leak {sorted(set(leaks))}")
    event = {
        "handle": handle,
        "scope": "section" if container else "document",
        "container": container,
        "utc": stamp,
    }
    with (out / "reviewer" / "inspections.jsonl").open("a") as log:
        log.write(json.dumps(event, sort_keys=True) + "\n")
    return to_text(lines)


def inspections(out: Path) -> Counter[str]:
    path = out / "reviewer" / "inspections.jsonl"
    return Counter(json.loads(x)["handle"] for x in path.read_text().splitlines() if x)


# ---- validation of completed records --------------------------------------------------


def build_sets(analysis: Analysis, raw: list[dict[str, Any]]) -> list[SetLocus]:
    sets = []
    for entry in raw:
        placements = []
        for placement in entry["placements"]:
            container = analysis.container_of.get(placement["container"])
            if container is None:
                raise ReviewError(f"unknown container {placement['container']}")
            components = []
            for c in placement["components"]:
                found = c.get("found") or c["text"]
                ids: list[str] = []
                for unit in analysis.index.units[container]:
                    for start_, end_ in _spans(unit.text, found):
                        ids += [i for i in unit.ids([(start_, end_)]) if i not in ids]
                if not ids:
                    raise ReviewError(f"{found!r} does not occur in {placement['container']}")
                note, form = c.get("equivalence_note"), c.get("form")
                observed = found if found != c["text"] and not note and not form else None
                components.append(
                    ComponentLocus(
                        role=c["role"],
                        text=c["text"],
                        occurrence_ids=ids,
                        observed_text=observed,
                        equivalence_note=note,
                        form=form,
                    )
                )
            placements.append(Placement(container_node_id=container, components=components))
        sets.append(SetLocus(set_index=entry["set_index"], placements=placements))
    return sets


def complete(record: dict[str, Any], seen_inspections: Counter[str]) -> None:
    handle = record["handle"]
    missing = [
        k
        for k in (
            "label",
            "rationale",
            "consulted_original_source",
            "started_utc",
            "finished_utc",
            "representation_guess",
            "guess_confidence",
            "guessed_utc",
        )
        if record.get(k) in (None, "")
    ]
    if missing:
        raise ReviewError(f"{handle}: incomplete, missing {missing}")
    if record["label"] not in LABELS:
        raise ReviewError(f"{handle}: unknown label {record['label']!r}")
    if record["representation_guess"] not in GUESSES or record["guess_confidence"] not in range(
        1, 6
    ):
        raise ReviewError(f"{handle}: invalid representation guess")
    started, finished, guessed = (
        parse_utc(record[k]) for k in ("started_utc", "finished_utc", "guessed_utc")
    )
    if not started <= finished <= guessed:
        raise ReviewError(f"{handle}: timestamps are out of order")
    if record["label"] == "absent" and not seen_inspections[handle]:
        raise ReviewError(f"{handle}: absent requires an inspected section or document")


def validate_all(out: Path, reviewer: str, role: str = "first_rater") -> dict[str, StageAJudgment]:
    manifest, units = load_provenance(out)
    records = read_review(out)
    handles = [r.get("handle") for r in records]
    if len(set(handles)) != len(handles) or set(handles) != set(units):
        raise ReviewError("review records must cover each packet unit exactly once")
    seen = inspections(out)
    errors, judgments = [], {}
    for record in records:
        handle = record["handle"]
        try:
            complete(record, seen)
            analysis = load_analysis(handle, manifest["seed"], units[handle])
            source = analysis.source
            judgment = StageAJudgment(
                benchmark_version=source.item.benchmark_version,
                corpus_version=source.item.corpus_version,
                item_id=source.item.item_id,
                item_sha256=item_digest(source.item),
                paper_id=source.item.paper_id,
                representation=source.document.representation,
                canonical=source.canonical,
                label=record["label"],
                basis="reviewer",
                candidate=stage_a_ref(analysis.candidate),
                sets=build_sets(analysis, record["sets"]),
                alteration=record["alteration"],
                rationale=record["rationale"],
                reviewer=reviewer,
                at_utc=record["finished_utc"],
            )
            validate_judgment(judgment, source.item, source.document, source.canonical)
            judgments[handle] = judgment
        except (ReviewError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{handle}: {error}")
    if errors:
        raise ReviewError("\n".join(errors))
    body = {
        "status": manifest["status"],
        "role": role,
        "judgments": {h: j.model_dump(mode="json") for h, j in judgments.items()},
    }
    write(out / "reviewer" / f"judgments_{role}.json", dumps(body))
    return judgments


# ---- summary --------------------------------------------------------------------------


def publish_review(
    out: Path, reviewer: str, destination: Path, inventory: Path, role: str = "first_rater"
) -> dict[str, Any]:
    """Freeze completed human records; blank templates and rehearsal queues cannot publish."""
    from biorag_repro.core.storage import publish_bytes
    from biorag_repro.evaluation import stage_a_storage

    manifest, units = load_provenance(out)
    if manifest["status"] not in PUBLISHABLE:
        raise ReviewError("only an explicitly prepared human review can publish")
    judgments = validate_all(out, reviewer, role)
    refs = {}
    for handle, judgment in judgments.items():
        analysis = load_analysis(handle, manifest["seed"], units[handle])
        source = analysis.source
        candidate_ref = stage_a_ref(analysis.candidate)
        if (destination / candidate_ref.logical_ref).exists():
            stage_a_storage.read_candidate(
                candidate_ref, destination, source.item, source.document, source.canonical
            )
        else:
            stage_a_storage.publish_candidate(
                analysis.candidate, source.item, source.document, source.canonical, destination
            )
        ref = stage_a_ref(judgment)
        if (destination / ref.logical_ref).exists():
            stage_a_storage.read_judgment(
                ref, destination, source.item, source.document, source.canonical
            )
        else:
            stage_a_storage.publish_judgment(
                judgment, source.item, source.document, source.canonical, destination
            )
        refs[f"{judgment.item_id}/{judgment.representation}"] = ref.model_dump(mode="json")
    value = {
        "status": "human_judgments_submitted",
        "reviewer": reviewer,
        "role": role,
        "session_manifest_sha256": sha256_bytes(
            (out / "provenance/session_manifest.json").read_bytes()
        ),
        "judgments": refs,
        "stage_b_judgments": {},
    }
    data = dumps(value)
    inventory.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(inventory, data, sha256_bytes(data))
    return value


def quartiles(values: list[float]) -> dict[str, float]:
    q = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else values * 3
    return {
        "min": min(values),
        "q1": q[0],
        "median": q[1],
        "q3": q[2],
        "max": max(values),
        "total": sum(values),
    }


def summarize(out: Path, reviewer: str, role: str = "first_rater") -> dict[str, Any]:
    validate_all(out, reviewer, role)
    manifest, units = load_provenance(out)
    records = {r["handle"]: r for r in read_review(out)}
    seen = inspections(out)
    arm = {h: units[h]["representation"] for h in records}
    seconds = {
        h: (parse_utc(r["finished_utc"]) - parse_utc(r["started_utc"])).total_seconds()
        for h, r in records.items()
    }
    collapse = {
        "preserved": "usable",
        "altered_but_sufficient": "usable",
        "scientifically_corrupted": "failed",
        "absent": "failed",
        "unresolved": "unresolved",
    }
    guesses: dict[str, Counter[str]] = {a: Counter() for a in REPRESENTATIONS}
    confidence: dict[str, list[int]] = {a: [] for a in REPRESENTATIONS}
    for h, r in records.items():
        g = r["representation_guess"]
        guesses[arm[h]][
            "cannot_tell" if g == "cannot_tell" else ("correct" if g == arm[h] else "wrong")
        ] += 1
        confidence[arm[h]].append(r["guess_confidence"])
    median = statistics.median(seconds.values())
    q = quartiles(list(seconds.values()))
    per_arm = {
        a: quartiles([s for h, s in seconds.items() if arm[h] == a])
        for a in REPRESENTATIONS
        if any(arm[h] == a for h in records)
    }
    return {
        "status": STATUS,
        "role": role,
        "units": len(records),
        "labels": dict(sorted(Counter(r["label"] for r in records.values()).items())),
        "labels_by_representation": {
            a: dict(sorted(Counter(r["label"] for h, r in records.items() if arm[h] == a).items()))
            for a in REPRESENTATIONS
        },
        "attribution_collapse": dict(
            sorted(Counter(collapse[r["label"]] for r in records.values()).items())
        ),
        "unresolved": sorted(h for h, r in records.items() if r["label"] == "unresolved"),
        "observed_seconds": q,
        "observed_seconds_by_representation": per_arm,
        "masking": {
            a: {
                **dict(sorted(guesses[a].items())),
                "mean_confidence": statistics.mean(c) if c else None,
            }
            for a, c in confidence.items()
        },
        "consulted_original_source": sum(
            bool(r["consulted_original_source"]) for r in records.values()
        ),
        "units_with_expanded_inspection": sum(1 for h in records if seen[h]),
        "projected_not_observed": {
            "basis": "median and quartiles of observed seconds per judgment, development rehearsal",
            "full_benchmark_units": FULL_BENCHMARK_UNITS,
            "first_rater_hours_median": FULL_BENCHMARK_UNITS * median / 3600,
            "first_rater_hours_q1_q3": [
                FULL_BENCHMARK_UNITS * q["q1"] / 3600,
                FULL_BENCHMARK_UNITS * q["q3"] / 3600,
            ],
            "second_rater_hours_median_20pct": 0.2 * FULL_BENCHMARK_UNITS * median / 3600,
        },
        "packet_seed": manifest["seed"],
    }


# ---- real development inputs ----------------------------------------------------------


def load_pilot(
    data_root: Path, pilot_root: Path, papers: tuple[str, ...], view_sha256: str
) -> tuple[list[BenchmarkItem], dict[tuple[str, str], Source]]:
    _, items = read_development_view(
        data_root / "benchmark/derived/benchmark_v1_development", view_sha256
    )
    items = [i for i in items if i.paper_id in papers]
    sources: dict[tuple[str, str], Source] = {}
    for paper in papers:
        for representation, prefix in (("JATS", "jats"), ("PDF_DOCLING", "pdf")):
            logged = json.loads((pilot_root / "logs" / f"{prefix}_{paper}.out").read_bytes())
            vref = ArtifactRef(**logged[representation])
            validation = read_project_json(
                pilot_root / vref.logical_ref, vref.sha256, Phase4Validation
            )
            canonical = validation.canonical_artifact
            document = read_project_json(
                pilot_root / canonical.logical_ref, canonical.sha256, CanonicalDocument
            )
            for item in items:
                if item.paper_id == paper:
                    sources[(item.item_id, representation)] = Source(
                        item, document, pilot_root, canonical
                    )
    return items, sources


def load_final_test(
    repo: Path, data_root: Path, selection_path: Path, authorization: Path, authorization_sha: str
) -> tuple[list[BenchmarkItem], dict[tuple[str, str], Source]]:
    """All 305 frozen final-test items and both canonical documents of each, after the
    researcher's evaluation authorization is verified. Protected content is read only here."""
    from biorag_repro.benchmark.release import read_final_test_items
    from biorag_repro.evaluation.final_test import check_source, read_authorization

    approved, _ = read_authorization(authorization, authorization_sha, repo)
    check_source(repo, approved.evaluation_source_revision)
    _, items = read_final_test_items(data_root)
    selection = json.loads(selection_path.read_bytes())
    sources: dict[tuple[str, str], Source] = {}
    for item in items:
        for representation in REPRESENTATIONS:
            entry = selection["canonical"][f"{item.paper_id}/{representation}"]
            root = Path(entry["root"])
            validation = read_project_json(
                root / entry["logical_ref"], entry["sha256"], Phase4Validation
            )
            canonical = validation.canonical_artifact
            document = read_project_json(
                root / canonical.logical_ref, canonical.sha256, CanonicalDocument
            )
            sources[(item.item_id, representation)] = Source(item, document, root, canonical)
    return items, sources


def scope_for(out: Path) -> Any:
    """Final-test packets are read and validated only inside `final_test_scope`."""
    from contextlib import nullcontext

    from biorag_repro.evaluation.final_test import EvaluationRefused, read_authorization
    from biorag_repro.evaluation.stage_a import final_test_scope

    path = out / "provenance" / "session_manifest.json"
    manifest = json.loads(path.read_bytes()) if path.is_file() else {}
    if manifest.get("status") != FINAL_STATUS:
        return nullcontext()
    # Every later command on a final-test packet re-verifies the evaluation authorization.
    try:
        recorded = manifest["authorization"]
        read_authorization(
            Path(recorded["path"]), recorded["sha256"], Path(__file__).resolve().parent.parent
        )
    except (EvaluationRefused, KeyError, TypeError) as error:
        raise ReviewError(f"final-test packet authorization does not verify: {error}") from None
    return final_test_scope()


def main(argv: list[str] | None = None) -> int:
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    q = commands.add_parser("queue")
    q.add_argument("--out", type=Path, required=True)
    q.add_argument("--seed", type=int, required=True)
    q.add_argument("--papers", nargs="+", default=list(REHEARSAL_PAPERS))
    q.add_argument("--pairs", type=int)
    q.add_argument("--fraction", type=float)
    q.add_argument("--role", choices=ROLES, default="first_rater")
    q.add_argument("--previous-manifest", type=Path)
    q.add_argument(
        "--data-root", type=Path, default=repo.parent / "bio-rag-evidence-reproducibility-data"
    )
    q.add_argument(
        "--pilot-root", type=Path, default=repo.parent / "bio-rag-evidence-reproducibility-pilot"
    )
    q.add_argument("--view-sha256", default=VIEW_SHA256)
    fq = commands.add_parser("final-queue")
    fq.add_argument("--out", type=Path, required=True)
    fq.add_argument("--seed", type=int, required=True)
    fq.add_argument("--data-root", type=Path, required=True)
    fq.add_argument("--selection", type=Path, required=True)
    fq.add_argument(
        "--authorization", type=Path, default=repo / "approvals/final_test_evaluation.json"
    )
    fq.add_argument("--authorization-sha256", required=True)
    fq.add_argument("--stop-after", type=int, help="build at most this many units, then stop")
    for name in ("start", "finish"):
        c = commands.add_parser(name)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("handle")
    g = commands.add_parser("guess")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("handle")
    g.add_argument("representation", choices=GUESSES)
    g.add_argument("confidence", type=int)
    i = commands.add_parser("inspect")
    i.add_argument("--out", type=Path, required=True)
    i.add_argument("handle")
    i.add_argument("--container", help="section of this container; omit for the whole document")
    for name in ("check", "summary"):
        c = commands.add_parser(name)
        c.add_argument("--out", type=Path, required=True)
        c.add_argument("--reviewer", default="Rama Krishna Pudota")
        c.add_argument("--role", choices=ROLES, default="first_rater")
    pub = commands.add_parser("publish")
    pub.add_argument("--out", type=Path, required=True)
    pub.add_argument("--reviewer", required=True)
    pub.add_argument("--destination", type=Path, required=True)
    pub.add_argument("--inventory", type=Path, required=True)
    pub.add_argument("--role", choices=ROLES, default="first_rater")
    args = parser.parse_args(argv)
    try:
        if args.command == "final-queue":
            from biorag_repro.evaluation.stage_a import final_test_scope

            with final_test_scope():
                items, sources = load_final_test(
                    repo, args.data_root, args.selection, args.authorization,
                    args.authorization_sha256,
                )  # fmt: skip
                missing = build_final_packet(
                    sources,
                    items,
                    out=args.out,
                    seed=args.seed,
                    created_utc=now(),
                    code=code_version(),
                    authorization={
                        "path": str(args.authorization.resolve()),
                        "sha256": args.authorization_sha256,
                    },
                    stop_after=args.stop_after,
                )
            if missing:
                print(f"incomplete: {missing} units still to build; rerun to resume")
                return 2
            print(f"{2 * len(items)} final-test units published")
            return 0
        with scope_for(args.out):
            return _command(args)
    except ReviewError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


def _command(args: argparse.Namespace) -> int:
    try:
        if args.command == "queue":
            papers = tuple(args.papers)
            items, sources = load_pilot(args.data_root, args.pilot_root, papers, args.view_sha256)
            build_packet(
                sources,
                items,
                out=args.out,
                seed=args.seed,
                papers=papers,
                pairs=args.pairs,
                fraction=args.fraction,
                role=args.role,
                created_utc=now(),
                code=code_version(),
                previous_manifest=args.previous_manifest,
            )
        elif args.command == "start":
            start(args.out, args.handle, now())
        elif args.command == "finish":
            finish(args.out, args.handle, now())
        elif args.command == "guess":
            guess(args.out, args.handle, args.representation, args.confidence, now())
        elif args.command == "inspect":
            print(inspect(args.out, args.handle, args.container, now()))
        elif args.command == "publish":
            result = publish_review(
                args.out, args.reviewer, args.destination, args.inventory, args.role
            )
            print(f"{len(result['judgments'])} supplied human judgments published")
        elif args.command == "check":
            print(f"{len(validate_all(args.out, args.reviewer, args.role))} judgments valid")
        else:
            result = summarize(args.out, args.reviewer, args.role)
            data = dumps(result)
            write(args.out / "reviewer" / f"summary_{args.role}.json", data)
            sys.stdout.write(data.decode())
    except ReviewError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
