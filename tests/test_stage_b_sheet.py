import json

import pytest

from scripts import stage_b_sheet as sheet

ENTRY = {
    "handle": "B0001",
    "question": "How many were genotyped?",
    "evidence_sets": [[{"role": "subject", "text": "CKCS"}, {"role": "answer", "text": "250"}]],
    "chunks": [
        {"chunk": "C1", "designated_paper": True, "text": "These were genotyped in 250 CKCS; <b>."},
        {"chunk": "C2", "designated_paper": False, "text": "Other paper, 250 dogs."},
    ],
}


def proposal(handle="B0001", **changes):
    value = {
        "handle": handle, "status": "delivered", "reason": None, "flags": [],
        "citations": [
            {"set_index": 0, "component_index": 0, "chunk": "C1", "quote": "CKCS", "occurrence": None},
            {"set_index": 0, "component_index": 1, "chunk": "C1", "quote": "250", "occurrence": None},
        ],
        "conflict_citations": [], "missing_components": [], "unresolved_reason": None,
        "rationale": "stated", "priority": "straightforward",
    }  # fmt: skip
    return value | changes


def files(tmp_path, entries, proposals):
    (tmp_path / "queue.json").write_text(json.dumps(entries))
    (tmp_path / "proposals.json").write_text(json.dumps({"proposals": proposals}))
    review = [
        {"handle": e["handle"], "status": None, "at_utc": None, "rationale": None} for e in entries
    ]
    (tmp_path / "review.json").write_text(json.dumps(review))
    return tmp_path / "queue.json", tmp_path / "proposals.json", tmp_path / "review.json"


def test_machine_check_catches_unmatched_ambiguous_and_incomplete_anchors():
    assert sheet.check(ENTRY, proposal()) == []
    wrong = proposal(citations=[{"set_index": 0, "component_index": 0, "chunk": "C1",
                                 "quote": "missing text", "occurrence": None}])  # fmt: skip
    assert any("not found" in p for p in sheet.check(ENTRY, wrong))
    assert any("no evidence set is fully cited" in p for p in sheet.check(ENTRY, wrong))
    other = proposal(citations=[{"set_index": 0, "component_index": 0, "chunk": "C2",
                                 "quote": "250", "occurrence": None}])  # fmt: skip
    assert any("not a designated-paper chunk" in p for p in sheet.check(ENTRY, other))
    negative = proposal(status="not_delivered", citations=[], missing_components=[])
    assert any("name a missing component" in p for p in sheet.check(ENTRY, negative))


def test_sheet_marks_anchors_inside_the_escaped_full_context(tmp_path):
    queue, props, _ = files(tmp_path, [ENTRY], [proposal()])
    result = sheet.sheet(queue, props, ["B0001"], tmp_path / "s.html")
    page = (tmp_path / "s.html").read_text()
    assert result == {"entries": 1, "machine_check_failures": []}
    assert '<mark class="s">250<sup>s0.1</sup></mark>' in page
    assert "&lt;b&gt;" in page and "Other paper" in page  # whole context shown, text escaped


def test_apply_records_only_verified_entries_and_never_overwrites(tmp_path):
    second = ENTRY | {"handle": "B0002"}
    queue, props, review = files(tmp_path, [ENTRY, second], [proposal(), proposal("B0002")])
    verification = tmp_path / "v.json"
    verification.write_text(json.dumps({"verified": {"B0001": {"action": "accept"}}}))
    log = tmp_path / "log.jsonl"
    assert sheet.apply(queue, props, verification, review, log)["recorded"] == 1
    records = {r["handle"]: r for r in json.loads(review.read_text())}
    assert records["B0001"]["status"] == "delivered" and records["B0001"]["at_utc"]
    assert records["B0002"]["status"] is None  # unverified entries stay open
    with pytest.raises(SystemExit, match="no open review record"):
        sheet.apply(queue, props, verification, review, log)


def test_apply_refuses_an_entry_whose_anchors_no_longer_match(tmp_path):
    bad = proposal(citations=[{"set_index": 0, "component_index": 0, "chunk": "C1",
                               "quote": "absent", "occurrence": None}])  # fmt: skip
    queue, props, review = files(tmp_path, [ENTRY], [bad])
    verification = tmp_path / "v.json"
    verification.write_text(json.dumps({"verified": {"B0001": {"action": "accept"}}}))
    with pytest.raises(SystemExit, match="B0001"):
        sheet.apply(queue, props, verification, review, tmp_path / "log.jsonl")
    assert json.loads(review.read_text())[0]["status"] is None


def test_plan_batches_confirmatory_entries_in_queue_order_and_build_is_resumable(tmp_path):
    entries = [ENTRY | {"handle": f"B{n:04d}"} for n in range(1, 8)]
    arms = ["confirmatory", "dense_robustness", "confirmatory", "confirmatory", "chunk_robustness",
            "confirmatory", "confirmatory"]  # fmt: skip
    mapping = {e["handle"]: [f"q/{a}/JATS"] for e, a in zip(entries, arms, strict=True)}
    queue = tmp_path / "queue.json"
    queue.write_text(json.dumps(entries))
    (tmp_path / "mapping.json").write_text(json.dumps(mapping))
    out = tmp_path / "plan"
    result = sheet.plan(queue, tmp_path / "mapping.json", "confirmatory", out, size=2)
    assert result["batches"] == {"batch_01": 2, "batch_02": 2, "batch_03": 1}
    planned = json.loads((out / "plan.json").read_text())["batches"]
    assert planned["batch_01"] + planned["batch_02"] + planned["batch_03"] == [
        "B0001",
        "B0003",
        "B0004",
        "B0006",
        "B0007",
    ]
    assert "confirmatory" not in (out / "inputs" / "batch_01_entries.json").read_text()
    proposals = tmp_path / "p"
    proposals.mkdir()
    (proposals / "a.json").write_text(
        json.dumps({"proposals": [proposal("B0001"), proposal("B0003")]})
    )
    first = sheet.build(out, queue, proposals)
    assert (
        first["batch_01"]["state"] == "built" and first["batch_02"]["state"] == "awaiting_proposals"
    )
    assert sheet.build(out, queue, proposals)["batch_01"]["state"] == "kept"
