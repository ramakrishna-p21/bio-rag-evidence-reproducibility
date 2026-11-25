"""Benchmark-wide structural checks and review warnings."""

from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from pydantic import Field

from biorag_repro.benchmark.flags import flag_items
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.review import (
    CandidateState,
    DecisionEvent,
    FlagDisposition,
    validate_history,
)
from biorag_repro.benchmark.split import PaperSplit
from biorag_repro.benchmark.storage import jsonl_sha256
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.corpus.manifest import CorpusManifest, verify_source_bytes


def resolve_jats_path(content: bytes, path: str) -> str:
    """Resolve an indexed original-JATS path without inferring scientific meaning."""
    node = ET.fromstring(content)
    if node.tag.rsplit("}", 1)[-1] != "article" or not path.startswith("/article/"):
        raise ValueError("JATS path must begin at article")
    for part in path.split("/")[2:]:
        tag, _, suffix = part.partition("[")
        if not suffix.endswith("]") or not suffix[:-1].isdigit():
            raise ValueError("JATS path needs indexed element segments")
        index = int(suffix[:-1]) - 1
        children = [child for child in node if child.tag.rsplit("}", 1)[-1] == tag]
        if index < 0 or index >= len(children):
            raise ValueError("JATS path does not resolve")
        node = children[index]
    return " ".join(" ".join(node.itertext()).split())


class Finding(Record):
    severity: Literal["hard_error", "review_warning"]
    rule: str
    item_id: str | None = None
    detail: str


class ValidationReport(Record):
    schema_version: str = "1"
    item_count: int
    items_sha256: str
    split_sha256: str | None = None
    corpus_content_sha256: str | None = None
    findings: list[Finding] = Field(default_factory=list)

    @property
    def hard_errors(self) -> int:
        return sum(f.severity == "hard_error" for f in self.findings)

    @property
    def review_warnings(self) -> int:
        return sum(f.severity == "review_warning" for f in self.findings)


def validate_items(
    items: list[BenchmarkItem],
    *,
    paper_ids: set[str],
    corpus_version: str,
    accepted_only: bool = False,
    manifest: CorpusManifest | None = None,
    data_root: Path | None = None,
    split: PaperSplit | None = None,
    events: list[DecisionEvent] | None = None,
    dispositions: list[FlagDisposition] | None = None,
) -> ValidationReport:
    findings: list[Finding] = []

    def add(
        severity: Literal["hard_error", "review_warning"],
        rule: str,
        item: BenchmarkItem,
        detail: str,
    ) -> None:
        findings.append(Finding(severity=severity, rule=rule, item_id=item.item_id, detail=detail))

    counts = Counter(item.item_id for item in items)
    events_by_item: dict[str, list[DecisionEvent]] = defaultdict(list)
    for event in events or []:
        events_by_item[event.item_id].append(event)
    papers = {paper.paper_id: paper for paper in manifest.papers} if manifest else {}
    for item in items:
        if counts[item.item_id] > 1:
            add("hard_error", "duplicate_item_id", item, "item ID occurs more than once")
        if item.paper_id not in paper_ids or item.corpus_version != corpus_version:
            add("hard_error", "corpus_binding", item, "paper or corpus version differs")
        if accepted_only and item.candidate_status != "accepted":
            add("hard_error", "not_accepted", item, "release contains nonaccepted item")
        if item.candidate_status != "accepted":
            add(
                "review_warning",
                "pending_scientific_review",
                item,
                "candidate is not confirmed gold",
            )
        if item.candidate_status == "accepted" and item.audit_status == "selected_pending":
            add("hard_error", "open_audit", item, "selected audit review remains open")
        if item.jats_status != "present_equivalent" or item.pdf_status != "present_equivalent":
            add(
                "review_warning",
                "source_equivalence",
                item,
                "original-source pair is not confirmed equivalent",
            )
        if split is not None:
            expected_split = (
                "development"
                if item.paper_id in split.development_ids
                else "final_test"
                if item.paper_id in split.final_test_ids
                else None
            )
            if item.split != expected_split:
                add("hard_error", "split_consistency", item, "item split differs from paper split")
        if item.candidate_status == "accepted" and events is not None:
            history = events_by_item.get(item.item_id, [])
            try:
                initial_status: CandidateState = (
                    "proposed" if history and history[0].previous_status == "proposed" else "lead"
                )
                state = validate_history(history, initial_status=initial_status)
            except ValueError as exc:
                add("hard_error", "decision_history", item, str(exc))
            else:
                if state != "accepted" or {event.event_id for event in history} != set(
                    item.decision_ids
                ):
                    add(
                        "hard_error",
                        "decision_history",
                        item,
                        "accepted item lacks matching decision chain",
                    )
        if item.paper_id not in papers:
            continue
        paper = papers[item.paper_id]
        for refs, source, kind in (
            (item.jats_refs, paper.jats, "jats"),
            (item.pdf_refs, paper.pdf, "pdf"),
        ):
            for ref in refs:
                if ref.artifact_id != source.artifact_id or ref.source_sha256 != source.sha256:
                    add(
                        "hard_error",
                        "source_binding",
                        item,
                        f"{kind} reference differs from corpus manifest",
                    )
        if data_root is None:
            continue
        for kind in ("jats", "pdf"):
            source = paper.jats if kind == "jats" else paper.pdf
            source_path = data_root / "corpus/raw" / source.logical_ref
            try:
                verify_source_bytes(paper, kind, source_path)  # type: ignore[arg-type]
                if kind == "jats":
                    content = source_path.read_bytes()
                    for ref in item.jats_refs:
                        if not resolve_jats_path(content, ref.xml_path):
                            raise ValueError("JATS reference resolves to empty text")
                else:
                    import pymupdf  # type: ignore[import-untyped]

                    with pymupdf.open(source_path) as document:
                        for ref in item.pdf_refs:
                            if ref.page > len(document):
                                raise ValueError("PDF page does not exist")
                            if ref.region is not None:
                                page_rect = document[ref.page - 1].rect
                                x0, y0, x1, y1 = ref.region
                                if (
                                    x0 < 0
                                    or y0 < 0
                                    or x0 >= x1
                                    or y0 >= y1
                                    or x1 > page_rect.width
                                    or y1 > page_rect.height
                                ):
                                    raise ValueError("PDF region exceeds page bounds")
            except (OSError, ValueError) as exc:
                add("hard_error", "source_location", item, f"{kind}: {exc}")
    related = (
        [group for group in split.ordered_groups if len(group.paper_ids) > 1] if split else None
    )
    final_pairs = {
        (record.code, tuple(record.item_ids))
        for record in dispositions or []
        if record.disposition == "retain_distinct"
    }
    for flag in flag_items(items, related):
        if (flag.code, tuple(sorted(flag.item_ids))) in final_pairs:
            continue
        for item_id in flag.item_ids:
            resolved = any(
                flag.code in event.resolved_flags for event in events_by_item.get(item_id, [])
            )
            if resolved:
                continue
            severity: Literal["hard_error", "review_warning"] = (
                "hard_error" if events is not None else "review_warning"
            )
            findings.append(
                Finding(severity=severity, rule=flag.code, item_id=item_id, detail=flag.detail)
            )
    return ValidationReport(
        item_count=len(items),
        items_sha256=jsonl_sha256(items),
        split_sha256=sha256_bytes(serialize_project_json(split, PaperSplit)) if split else None,
        corpus_content_sha256=split.corpus_content_sha256 if split else None,
        findings=findings,
    )
