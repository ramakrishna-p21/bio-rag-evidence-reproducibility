"""Evidence from published Stage A/B inventories: strict read-back, no invented judgments."""

import json
from dataclasses import replace

import pytest
from test_stage_a import CANONICAL
from test_stage_b import P_A, P_B, P_C, altered_world, deliver, prose_world

from biorag_repro.evaluation import stage_b, stage_b_storage, storage
from biorag_repro.evaluation.analysis_table import Observation, build_row
from biorag_repro.evaluation.judgment_evidence import EvidenceError, read_evidence


class Published:
    """One item published to a temp root with the inventories that index it."""

    def __init__(self, tmp_path, w, case=None, *, with_b=True):
        self.root, self.w, self.case = tmp_path, w, case
        a, am = storage.publish_judgment(w.stage_a, w.item, w.document, CANONICAL, tmp_path)
        self.a_row = {
            "item_id": w.item.item_id,
            "representation": w.document.representation,
            "judgment": a.model_dump(),
            "manifest": am.model_dump(),
            "label": w.stage_a.label,
            "reviewer": w.stage_a.reviewer,
            "at_utc": w.stage_a.at_utc,
        }
        self.b_row = None
        if case is not None and with_b:
            judged = stage_b.deterministic_judgment(case.evaluate())
            b, bm = stage_b_storage.publish_judgment(judged, case.inputs(), tmp_path)
            self.b_row = {
                "item_id": w.item.item_id,
                "representation": w.document.representation,
                "judgment": b.model_dump(),
                "manifest": bm.model_dump(),
                "status": judged.status,
                "basis": judged.basis,
                "context_provenance": "assembled_only",
                "reviewer": judged.reviewer,
                "at_utc": judged.at_utc,
            }

    def write(self, a_rows=None, b_rows=None):
        a_path, b_path = self.root / "a.json", self.root / "b.json"
        a_path.write_text(json.dumps([self.a_row] if a_rows is None else a_rows))
        rows = [self.b_row] if b_rows is None and self.b_row else (b_rows or [])
        b_path.write_text(json.dumps(rows))
        return a_path, b_path

    def read(self, a_rows=None, b_rows=None, *, items=None, inputs_for=None, b=True):
        a_path, b_path = self.write(a_rows, b_rows)
        return read_evidence(
            stage_a_inventory=a_path,
            stage_b_inventory=b_path if b else None,
            root=self.root,
            items=items if items is not None else [self.w.item],
            document_for=lambda item, rep: (self.w.document, CANONICAL),
            inputs_for=inputs_for or (lambda item, rep, a: self.case.inputs(stage_a=a)),
        )


@pytest.fixture
def published(tmp_path):
    w = prose_world(P_A, P_B, P_C)
    return Published(tmp_path, w, deliver(w, [w.chunk(P_A)]))


def key(p):
    return (p.w.item.item_id, p.w.document.representation)


def test_stage_a_and_b_are_read_back_and_linked(published):
    evidence = published.read()[key(published)]
    assert evidence.stage_a == published.w.stage_a
    assert evidence.stage_b is not None and evidence.stage_b.status == "delivered"
    assert evidence.b_inputs == published.case.inputs()


def test_evidence_is_accepted_by_the_analysis_table(published):
    evidence = published.read()[key(published)]
    observation = Observation(
        item=published.w.item,
        phase="development",
        matrix_arm="confirmatory",
        condition="JATS",
        repeat_index=1,
        resolved_config_sha256="b" * 64,
        retrieval_config_sha256=evidence.stage_b.configuration.resolved_sha256,
        evidence=evidence,
    )
    row, _ = build_row(observation)
    assert row.stage_a_label == "preserved" and row.stage_b_status == "delivered"
    assert row.context_provenance == "assembled_only"


def test_stage_a_without_a_stage_b_judgment_stays_unassessed(tmp_path):
    w = prose_world(P_A, P_B, P_C)
    case = deliver(w, [w.chunk(P_B)])  # evidence not delivered: only a candidate could exist
    p = Published(tmp_path, w, case, with_b=False)
    candidate = case.evaluate()
    assert candidate.determination != "automatic_delivered"
    stage_b_storage.publish_candidate(candidate, case.inputs(), tmp_path)  # exists, not a judgment
    evidence = p.read()[key(p)]
    assert evidence.stage_b is None and evidence.b_inputs is None
    assert evidence.stage_a == w.stage_a


def test_stage_b_inventory_is_optional(published):
    evidence = published.read(b=False)[key(published)]
    assert evidence.stage_b is None


def test_missing_judgment_gives_no_entry_and_nothing_is_invented(published):
    assert published.read(a_rows=[], b_rows=[]) == {}


def test_unresolved_stage_a_is_read_and_has_no_stage_b(tmp_path):
    w = altered_world("unresolved")
    p = Published(tmp_path, w, deliver(w, [w.chunk(P_B)]), with_b=False)
    evidence = p.read()[key(p)]
    assert evidence.stage_a.label == "unresolved" and evidence.stage_b is None


@pytest.mark.parametrize(
    "field,value",
    [("label", "absent"), ("reviewer", "Someone Else"), ("at_utc", "2020-01-01T00:00:00Z")],
)
def test_row_that_disagrees_with_the_stored_stage_a_judgment_is_refused(published, field, value):
    row = {**published.a_row, field: value}
    with pytest.raises(EvidenceError, match="disagrees"):
        published.read(a_rows=[row])


def test_row_that_disagrees_with_the_stored_stage_b_judgment_is_refused(published):
    row = {**published.b_row, "status": "not_delivered"}
    with pytest.raises(EvidenceError, match="disagrees"):
        published.read(b_rows=[row])


def test_item_outside_the_authorized_set_is_refused(published):
    with pytest.raises(EvidenceError, match="not an authorized item"):
        published.read(items=[])


def test_row_for_another_item_or_representation_is_refused(published):
    for change in ({"item_id": "q_other"}, {"representation": "PDF_DOCLING"}):
        with pytest.raises(EvidenceError):
            published.read(a_rows=[{**published.a_row, **change}])


def test_duplicate_rows_are_refused(published):
    with pytest.raises(EvidenceError, match="duplicate Stage A"):
        published.read(a_rows=[published.a_row, published.a_row])
    with pytest.raises(EvidenceError, match="duplicate Stage B"):
        published.read(b_rows=[published.b_row, published.b_row])


def test_stage_b_without_stage_a_is_refused(published):
    with pytest.raises(EvidenceError, match="no Stage A"):
        published.read(a_rows=[])


def test_changed_judgment_bytes_are_refused(published):
    path = published.root / published.a_row["judgment"]["logical_ref"]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="Stage A judgment"):
        published.read()


def test_changed_manifest_bytes_are_refused(published):
    path = published.root / published.b_row["manifest"]["logical_ref"]
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(EvidenceError, match="manifest"):
        published.read()


def test_manifest_of_another_judgment_is_refused(published):
    row = {**published.b_row, "manifest": published.a_row["manifest"]}
    with pytest.raises(EvidenceError, match="manifest"):
        published.read(b_rows=[row])


def test_stage_b_judged_from_other_inputs_is_refused(published):
    other = deliver(published.w, [published.w.chunk(P_B)])
    with pytest.raises(EvidenceError, match="Stage B judgment"):
        published.read(inputs_for=lambda item, rep, a: other.inputs(stage_a=a))


def test_deterministic_stage_a_row_may_omit_reviewer_fields(published):
    row = {k: v for k, v in published.a_row.items() if k not in {"reviewer", "at_utc"}}
    assert published.read(a_rows=[row])[key(published)].stage_a.reviewer is None


def test_unreadable_or_malformed_inventory_is_refused(published, tmp_path):
    a_path, b_path = published.write()
    a_path.write_text("{not json")
    with pytest.raises(EvidenceError, match="not usable"):
        read_evidence(
            stage_a_inventory=a_path,
            stage_b_inventory=b_path,
            root=published.root,
            items=[published.w.item],
            document_for=lambda i, r: (published.w.document, CANONICAL),
            inputs_for=lambda i, r, a: published.case.inputs(stage_a=a),
        )
    with pytest.raises(EvidenceError, match="not usable"):
        published.read(a_rows=[{**published.a_row, "surprise": 1}])


def test_a_claim_of_recorded_generator_input_is_not_an_accepted_provenance(published):
    row = {**published.b_row, "context_provenance": "generator_input_recorded"}
    with pytest.raises(EvidenceError, match="not usable"):
        published.read(b_rows=[row])
    assert replace  # keep import used
