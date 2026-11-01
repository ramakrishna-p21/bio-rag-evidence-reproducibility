"""Explicit scientific review events and paper-grouped worksheet rows."""

from datetime import datetime, timedelta
from typing import Literal

from pydantic import Field, field_validator

from biorag_repro.benchmark.discovery import CandidateLead
from biorag_repro.benchmark.flags import ReviewFlag
from biorag_repro.benchmark.models import Record

CandidateState = Literal["lead", "proposed", "accepted", "excluded"]


class DecisionEvent(Record):
    schema_version: str = "1"
    event_id: str
    item_id: str
    reviewer: str
    at_utc: str
    action: Literal[
        "propose", "accept", "exclude", "correct", "reopen", "audit_pass", "audit_defect"
    ]
    previous_status: CandidateState
    new_status: CandidateState
    reason: str
    evidence: str
    resolved_flags: list[str] = Field(default_factory=list)
    supersedes_event_id: str | None = None

    @field_validator("at_utc")
    @classmethod
    def utc_time(cls, value: str) -> str:
        if not value.endswith("Z") or datetime.fromisoformat(value).utcoffset() != timedelta(0):
            raise ValueError("review timestamp must be UTC with Z suffix")
        return value

    @field_validator("event_id", "item_id", "reviewer", "reason", "evidence")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("review event fields must be nonblank")
        return value


def validate_history(
    events: list[DecisionEvent], *, initial_status: CandidateState = "lead"
) -> CandidateState:
    """Check an append-only per-item decision chain without inferring missing acts."""
    status = initial_status
    seen: set[str] = set()
    item_ids = {event.item_id for event in events}
    if len(item_ids) > 1:
        raise ValueError("review history mixes item IDs")
    for event in events:
        if event.event_id in seen or event.previous_status != status:
            raise ValueError("review event duplicates ID or breaks state chain")
        transition = (event.previous_status, event.new_status)
        allowed = {
            "propose": {("lead", "proposed")},
            "accept": {("proposed", "accepted")},
            "exclude": {("lead", "excluded"), ("proposed", "excluded")},
            "correct": {("proposed", "proposed"), ("accepted", "accepted")},
            "reopen": {("accepted", "proposed"), ("excluded", "proposed")},
            "audit_pass": {("accepted", "accepted")},
            "audit_defect": {("accepted", "accepted")},
        }
        if transition not in allowed[event.action]:
            raise ValueError("review action does not permit this state transition")
        seen.add(event.event_id)
        status = event.new_status
    return status


class WorksheetRow(Record):
    schema_version: str = "1"
    lead_id: str
    paper_id: str
    jats_xml_path: str
    jats_section: str
    jats_text: str
    table_id: str | None
    row_label: str | None
    column_label: str | None
    pdf_search_terms: list[str]
    pdf_hits: list[dict[str, object]] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    proposed_question: str | None = None
    proposed_gold: str | None = None
    minimum_evidence: str | None = None
    source_equivalence: str | None = None
    decision: str | None = None
    reviewer: str | None = None
    decision_reason: str | None = None


def worksheet_rows(
    leads: list[CandidateLead], flags: list[ReviewFlag] | None = None
) -> list[WorksheetRow]:
    codes_by_lead: dict[str, set[str]] = {}
    for flag in flags or []:
        for lead_id in flag.item_ids:
            codes_by_lead.setdefault(lead_id, set()).add(flag.code)
    return [
        WorksheetRow(
            lead_id=lead.lead_id,
            paper_id=lead.paper_id,
            jats_xml_path=lead.xml_path,
            jats_section=lead.section,
            jats_text=lead.text,
            table_id=lead.table_id,
            row_label=lead.row_label,
            column_label=lead.column_label,
            pdf_search_terms=list(dict.fromkeys([*lead.numeric_tokens, *lead.entity_tokens]))[:8],
            flags=sorted(codes_by_lead.get(lead.lead_id, set())),
        )
        for lead in leads
    ]


def review_queue(leads: list[CandidateLead], *, limit: int = 100) -> list[CandidateLead]:
    """Prioritize inspectable leads while retaining the complete discovery output."""
    if limit < 1:
        raise ValueError("review queue limit must be positive")

    def priority(lead: CandidateLead) -> tuple[int, int, str]:
        section = lead.section.casefold()
        location = (
            0
            if "result" in section or section.startswith("3.")
            else 1
            if section == "abstract"
            else 2
        )
        content = (
            0 if lead.numeric_tokens and (lead.relation_tokens or lead.kind == "table_cell") else 1
        )
        return location, content, lead.xml_path

    prose = sorted((lead for lead in leads if lead.kind != "table_cell"), key=priority)
    tables = sorted((lead for lead in leads if lead.kind == "table_cell"), key=priority)
    half = limit // 2
    selected = [*prose[:half], *tables[:half]]
    remaining = sorted([*prose[half:], *tables[half:]], key=priority)
    return [*selected, *remaining[: limit - len(selected)]]
