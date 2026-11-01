"""Review flags that never determine scientific inclusion automatically."""

import re
from collections import defaultdict
from difflib import SequenceMatcher

from pydantic import Field

from biorag_repro.benchmark.discovery import CandidateLead
from biorag_repro.benchmark.models import BenchmarkItem, EntityGold, QuantityGold, Record
from biorag_repro.benchmark.split import PaperGroup


class ReviewFlag(Record):
    code: str
    item_ids: list[str] = Field(min_length=1)
    detail: str


def flag_leads(leads: list[CandidateLead]) -> list[ReviewFlag]:
    by_value: dict[tuple[str, str], list[CandidateLead]] = defaultdict(list)
    for lead in leads:
        for token in set(lead.numeric_tokens):
            by_value[(lead.paper_id, token)].append(lead)
    flags: list[ReviewFlag] = []
    for (paper_id, token), matches in sorted(by_value.items()):
        contexts = {lead.xml_path for lead in matches}
        if len(contexts) > 1:
            flags.append(
                ReviewFlag(
                    code="repeated_value_context",
                    item_ids=sorted(lead.lead_id for lead in matches),
                    detail=f"{paper_id}: {token} occurs at {len(contexts)} locations",
                )
            )
    return flags


def question_key(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


def flag_questions(items: list[tuple[str, str, str]]) -> list[ReviewFlag]:
    """Flag exact normalized wording and answer-bearing fact keys for review."""
    by_question: dict[str, list[str]] = defaultdict(list)
    by_fact: dict[str, list[str]] = defaultdict(list)
    for item_id, question, fact_key in items:
        by_question[question_key(question)].append(item_id)
        by_fact[fact_key].append(item_id)
    flags = []
    for kind, groups in (("duplicate_question", by_question), ("duplicate_fact", by_fact)):
        for key, ids in sorted(groups.items()):
            if key and len(ids) > 1:
                flags.append(ReviewFlag(code=kind, item_ids=sorted(ids), detail=key))
    return flags


def flag_items(
    items: list[BenchmarkItem], related_groups: list[PaperGroup] | None = None
) -> list[ReviewFlag]:
    """Flag possible leakage and duplicate findings for human disposition."""
    flags: list[ReviewFlag] = []
    group_by_paper = {
        paper_id: index
        for index, group in enumerate(related_groups or [])
        for paper_id in group.paper_ids
    }
    for item in items:
        answer = (
            item.gold.value
            if isinstance(item.gold, QuantityGold)
            else item.gold.canonical
            if isinstance(item.gold, EntityGold)
            else None
        )
        if answer and re.search(
            rf"(?<!\w){re.escape(answer.casefold())}(?!\w)", item.question.casefold()
        ):
            flags.append(
                ReviewFlag(
                    code="answer_leakage",
                    item_ids=[item.item_id],
                    detail="question contains canonical answer text",
                )
            )
    ordered = sorted(items, key=lambda item: item.item_id)
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            same_related_group = left.paper_id in group_by_paper and group_by_paper.get(
                left.paper_id
            ) == group_by_paper.get(right.paper_id)
            if left.paper_id != right.paper_id and not same_related_group:
                continue
            ratio = SequenceMatcher(
                None, question_key(left.question), question_key(right.question)
            ).ratio()
            if ratio >= 0.88:
                flags.append(
                    ReviewFlag(
                        code="near_duplicate_question",
                        item_ids=[left.item_id, right.item_id],
                        detail=f"normalized similarity {ratio:.3f}",
                    )
                )
            if left.gold == right.gold and {ref.xml_path for ref in left.jats_refs} & {
                ref.xml_path for ref in right.jats_refs
            }:
                flags.append(
                    ReviewFlag(
                        code="shared_answer_bearing_fact",
                        item_ids=[left.item_id, right.item_id],
                        detail="same gold and overlapping JATS source path",
                    )
                )
    return flags
