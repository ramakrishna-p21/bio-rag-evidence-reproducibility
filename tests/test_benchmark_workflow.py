import pytest
from pydantic import ValidationError
from test_benchmark_models import item

from biorag_repro.benchmark.audit import AuditedItem, AuditReport, sample_audit
from biorag_repro.benchmark.flags import flag_items
from biorag_repro.benchmark.models import BenchmarkItem, EntityGold, QuantityGold, SetGold
from biorag_repro.benchmark.review import DecisionEvent, FlagDisposition, validate_history
from biorag_repro.benchmark.split import PaperGroup
from biorag_repro.benchmark.storage import decode_jsonl, encode_jsonl
from biorag_repro.benchmark.validation import validate_items


def test_audit_replays_and_includes_unflagged() -> None:
    ids = [f"q_{i:03d}" for i in range(120)]
    a = sample_audit(ids, seed=3102027)
    b = sample_audit(list(reversed(ids)), seed=3102027)
    assert a == b
    assert len(a.selected_ids) == 100
    assert len(set(a.selected_ids)) == 100
    assert a.eligible_count == 120
    assert a.eligible_ids_sha256
    assert len(sample_audit(ids[:9], seed=3102027).selected_ids) == 9


def test_audit_report_keeps_pre_repair_population() -> None:
    ids = [f"q_{i:03d}" for i in range(4)]
    draw = sample_audit(ids, seed=3102027)
    report = AuditReport(
        draw=draw,
        proposed_item_ids=ids,
        audited_items=[
            AuditedItem(
                item_id=i,
                decision_event_id=f"audit_{i}",
                defect_found=i == ids[0],
                finding="defect" if i == ids[0] else "no defect",
                repair_event_id="repair_1" if i == ids[0] else None,
            )
            for i in draw.selected_ids
        ],
    )
    assert report.draw.eligible_count == 4
    with pytest.raises(ValidationError):
        AuditReport.model_validate({**report.model_dump(), "proposed_item_ids": ids[1:]})


def test_review_history_rejects_missing_previous_state() -> None:
    event = DecisionEvent(
        event_id="e1",
        item_id="q1",
        reviewer="Rama Krishna Pudota",
        at_utc="2026-11-01T00:00:00Z",
        action="accept",
        previous_status="proposed",
        new_status="accepted",
        reason="Both sources checked",
        evidence="JATS and PDF refs",
    )
    with pytest.raises(ValueError):
        validate_history([event], initial_status="lead")


def test_accepted_persisted_proposal_uses_actual_decision_chain() -> None:
    draft = item(QuantityGold(value="1.2", comparator="eq"))
    now = "2026-11-01T00:00:00Z"
    event = DecisionEvent(
        event_id="accept_1",
        item_id=draft.item_id,
        reviewer="Rama Krishna Pudota",
        at_utc=now,
        action="accept",
        previous_status="proposed",
        new_status="accepted",
        reason="Reviewed both original sources",
        evidence="Persisted proposal SHA-256 and JATS/PDF locations",
    )
    accepted = BenchmarkItem.model_validate(
        {
            **draft.model_dump(),
            "candidate_status": "accepted",
            "review_status": "reviewed",
            "split": "development",
            "jats_status": "present_equivalent",
            "pdf_status": "present_equivalent",
            "decision_ids": [event.event_id],
        }
    )
    report = validate_items(
        [accepted], paper_ids={"P001"}, corpus_version="corpus_v1", events=[event]
    )
    assert report.hard_errors == 0


def test_candidate_is_not_automatically_eligible() -> None:
    draft = item(QuantityGold(value="1.2", comparator="eq"))
    report = validate_items(
        [draft], paper_ids={"P001"}, corpus_version="corpus_v1", accepted_only=False
    )
    assert report.hard_errors == 0
    assert report.review_warnings > 0
    mismatch = validate_items([draft], paper_ids={"P002"}, corpus_version="corpus_v2")
    assert mismatch.hard_errors == 1


def test_accepted_without_paired_source_is_rejected() -> None:
    draft = item(QuantityGold(value="1.2", comparator="eq"))
    with pytest.raises(ValidationError):
        BenchmarkItem.model_validate({**draft.model_dump(), "candidate_status": "accepted"})


def test_leakage_and_near_duplicate_are_flags_only() -> None:
    first = item(QuantityGold(value="1.2", comparator="eq"))
    second = BenchmarkItem.model_validate(
        {
            **first.model_dump(),
            "item_id": "q_002",
            "question": "What was the reported effect for condition A?",
        }
    )
    flagged = flag_items([first, second])
    assert any(flag.code == "near_duplicate_question" for flag in flagged)
    leaked = BenchmarkItem.model_validate(
        {**first.model_dump(), "question": "Was the effect 1.2 under condition A?"}
    )
    assert any(flag.code == "answer_leakage" for flag in flag_items([leaked]))


def test_related_version_duplicate_is_flagged() -> None:
    first = item(QuantityGold(value="1.2", comparator="eq"))
    data = first.model_dump()
    data.update(item_id="q_002", paper_id="P002")
    data["jats_refs"][0]["paper_id"] = "P002"
    data["pdf_refs"][0]["paper_id"] = "P002"
    second = BenchmarkItem.model_validate(data)
    assert flag_items([first, second]) == []
    flags = flag_items(
        [first, second], [PaperGroup(paper_ids=["P001", "P002"], basis="related version")]
    )
    assert any(flag.code == "near_duplicate_question" for flag in flags)


def test_bounded_set_member_in_question_is_flagged_for_leakage() -> None:
    record = item(
        SetGold(
            members=[
                EntityGold(answer_class="biological_entity", canonical="BRCA1"),
                EntityGold(answer_class="biological_entity", canonical="BRCA2"),
            ]
        )
    )
    record.question = "Which genes, including BRCA1, were associated with the condition?"
    assert any(flag.code == "answer_leakage" for flag in flag_items([record]))


def test_scientifically_resolved_duplicate_flag_is_closed() -> None:
    first = item(QuantityGold(value="1.2", comparator="eq"))
    second = BenchmarkItem.model_validate(
        {**first.model_dump(), "item_id": "q_002", "question": first.question}
    )
    first.candidate_status = second.candidate_status = "accepted"
    first.review_status = second.review_status = "reviewed"
    first.jats_status = second.jats_status = "present_equivalent"
    first.pdf_status = second.pdf_status = "present_equivalent"
    first.split = second.split = "development"
    events = []
    for record in (first, second):
        event = DecisionEvent(
            event_id=f"accept_{record.item_id}",
            item_id=record.item_id,
            reviewer="Rama Krishna Pudota",
            at_utc="2026-11-01T00:00:00Z",
            action="accept",
            previous_status="proposed",
            new_status="accepted",
            reason="Distinct scientific findings",
            evidence="Original JATS and PDF reviewed",
            resolved_flags=[
                "duplicate_question",
                "near_duplicate_question",
                "shared_answer_bearing_fact",
            ],
        )
        events.append(event)
        record.decision_ids = [event.event_id]
    report = validate_items(
        [first, second], paper_ids={"P001"}, corpus_version="corpus_v1", events=events
    )
    assert report.hard_errors == 0
    assert report.review_warnings == 0


def test_final_pair_disposition_closes_cross_batch_similarity_without_rewriting_review_events() -> (
    None
):
    first = item(QuantityGold(value="1.2", comparator="eq"))
    second = BenchmarkItem.model_validate(
        {
            **first.model_dump(),
            "item_id": "q_002",
            "gold": QuantityGold(value="2.3", comparator="eq").model_dump(),
        }
    )
    events = []
    for record in (first, second):
        record.candidate_status = "accepted"
        record.review_status = "reviewed"
        record.jats_status = record.pdf_status = "present_equivalent"
        record.split = "development"
        event = DecisionEvent(
            event_id=f"accept_{record.item_id}",
            item_id=record.item_id,
            reviewer="Rama Krishna Pudota",
            at_utc="2026-11-01T00:00:00Z",
            action="accept",
            previous_status="proposed",
            new_status="accepted",
            reason="Both original sources reviewed",
            evidence="Frozen JATS and PDF",
        )
        events.append(event)
        record.decision_ids = [event.event_id]
    pair = sorted([first.item_id, second.item_id])
    disposition = FlagDisposition(
        disposition_id="flag_pair",
        code="near_duplicate_question",
        item_ids=pair,
        reviewer="Rama Krishna Pudota",
        at_utc="2026-11-01T01:00:00Z",
        disposition="retain_distinct",
        reason="Different scientific measurements",
        evidence="Original-source review of both items",
    )
    common = {"paper_ids": {"P001"}, "corpus_version": "corpus_v1", "events": events}
    assert validate_items([first, second], **common).hard_errors == 2
    assert validate_items([first, second], dispositions=[disposition], **common).hard_errors == 0
    conditional = FlagDisposition.model_validate(
        {
            **disposition.model_dump(),
            "disposition": "retain_distinct_conditional",
            "condition": "Await paired PDF review",
        }
    )
    assert validate_items([first, second], dispositions=[conditional], **common).hard_errors == 2


def test_conditional_similarity_disposition_does_not_admit_candidates() -> None:
    record = FlagDisposition(
        disposition_id="flag_001",
        code="near_duplicate_question",
        item_ids=["q_001", "q_002"],
        reviewer="Rama Krishna Pudota",
        at_utc="2026-11-01T04:00:00Z",
        disposition="retain_distinct_conditional",
        reason="The questions target different ecological metrics and onset times.",
        condition="Paired original-PDF evidence must be confirmed for both items.",
        evidence="Frozen batch proposal SHA-256 " + "a" * 64,
    )
    assert record.disposition == "retain_distinct_conditional"
    assert decode_jsonl(encode_jsonl([record]), FlagDisposition) == [record]
    with pytest.raises(ValidationError):
        FlagDisposition.model_validate(
            {**record.model_dump(), "at_utc": "2026-11-01T04:00:00-07:00"}
        )
    with pytest.raises(ValidationError):
        FlagDisposition.model_validate({**record.model_dump(), "item_ids": ["q_001", "q_001"]})
    final = FlagDisposition.model_validate(
        {
            **record.model_dump(),
            "disposition_id": "flag_002",
            "disposition": "retain_distinct",
            "condition": None,
        }
    )
    assert decode_jsonl(encode_jsonl([record, final]), FlagDisposition) == [record, final]
    with pytest.raises(ValidationError):
        FlagDisposition.model_validate({**final.model_dump(), "condition": "still pending"})
