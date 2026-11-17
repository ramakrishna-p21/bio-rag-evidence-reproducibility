"""Synthetic records exercise retrospective provenance, never real reviewer decisions."""

from dataclasses import replace

import pytest
from test_analysis_export import observation, roster_of, world
from test_stage_a import CANONICAL
from test_stage_b import P_A, deliver

from biorag_repro.core.hashing import canonical_json_bytes
from biorag_repro.evaluation import analysis_handoff, stage_b
from biorag_repro.evaluation import stage_a_retrospective as retro
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import Evidence, build_table
from biorag_repro.evaluation.stage_a import stage_a_ref
from biorag_repro.evaluation.stage_a_storage import read_judgment as read_original
from biorag_repro.generation.environment import GitState
from scripts.transcribe_stage_a import source_ref


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    w = world()
    rows = [
        {
            "handle": f"R{i:03d}",
            "researcher_concurred_label": "preserved",
            "researcher_concurred_rationale": "Synthetic fixture only.",
        }
        for i in range(1, 157)
    ]
    raw = {}
    refs = {}

    def put(name, value):
        data = canonical_json_bytes(value)
        raw[name] = data
        ref = refs[name] = source_ref(data)
        target = tmp_path / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return ref

    put(
        "coordinate_map",
        {
            "R001": {
                "item_sha256": w.stage_a.item_sha256,
                "representation": w.document.representation,
                "canonical": CANONICAL.model_dump(),
            }
        },
    )
    put("session_manifest", {"unmasked_evidence_sha256": refs["coordinate_map"].sha256})
    put("approved_decisions", {"synthetic_fixture": True, "records": rows})
    put(
        "authorization",
        {
            "synthetic_fixture": True,
            "protocol": retro.PROTOCOL,
            "approved_decisions_sha256": refs["approved_decisions"].sha256,
        },
    )
    for field, name in (
        ("APPROVED_DECISIONS_SHA256", "approved_decisions"),
        ("SESSION_SHA256", "session_manifest"),
        ("AUTHORIZATION_SHA256", "authorization"),
    ):
        monkeypatch.setattr(retro, field, refs[name].sha256)
    value = retro.RetrospectiveJudgment(
        **{
            k: getattr(w.stage_a, k)
            for k in (
                "benchmark_version",
                "corpus_version",
                "item_id",
                "item_sha256",
                "paper_id",
                "representation",
                "canonical",
            )
        },
        handle="R001",
        label="preserved",
        candidate=w.stage_a.candidate,
        sets=w.stage_a.sets,
        rationale=rows[0]["researcher_concurred_rationale"],
        recorded_at_utc="2026-11-18T00:00:00Z",
        **refs,
    )
    return w, value, tmp_path


def test_retrospective_publication_has_no_fabricated_review_events(synthetic):
    w, value, root = synthetic
    ref, _ = retro.publish_judgment(value, w.item, w.document, CANONICAL, root)
    loaded = retro.read_judgment(ref, root, w.item, w.document, CANONICAL)
    assert loaded == value
    assert loaded.historical_review_at_utc is None and loaded.representation_guess is None
    assert loaded.started_utc is None and loaded.consulted_original_source is None
    assert loaded.eligible_for_review_timing_or_guess_analysis is False
    with pytest.raises(ValueError):
        read_original(ref, root, w.item, w.document, CANONICAL)
    assert stage_a_ref(w.stage_a) != ref


def test_approved_label_and_component_placements_cannot_be_silently_changed(synthetic):
    w, value, root = synthetic
    with pytest.raises(ValueError, match="complete component placements"):
        retro.RetrospectiveJudgment.model_validate(value.model_dump() | {"sets": []})
    changed = value.model_copy(update={"label": "unresolved"})
    with pytest.raises(ValueError, match="changes approved science"):
        retro.validate_retrospective(changed, w.item, w.document, CANONICAL, root)
    with pytest.raises(ValueError):
        retro.RetrospectiveJudgment.model_validate(
            value.model_dump() | {"finished_utc": "2026-11-18T00:00:00Z"}
        )


def test_downstream_revalidates_approval_root_and_preserves_analysis_provenance(synthetic):
    w, value, root = synthetic
    case = deliver(w, [w.chunk(P_A)])
    inputs = replace(case.inputs(), stage_a=value, stage_a_root=root)
    assert stage_b.evaluate(inputs).determination == case.evaluate().determination
    with pytest.raises(ValueError, match="approval artifact root"):
        stage_b.evaluate(replace(inputs, stage_a_root=None))
    obs = observation(
        w, result=None, evidence=Evidence(w.document, CANONICAL, value, stage_a_root=root)
    )
    table = build_table([obs], roster_of([obs]))
    target = root / "analysis"
    _, _, build = publish_table(table, target)
    handoff = analysis_handoff.build_handoff(
        build,
        target,
        code=GitState(status="ok", commit="a" * 40, dirty=False, dirty_paths=0),
        rehearsal=True,
    )
    assert handoff.stage_a_review_protocols == {
        retro.PROTOCOL: [retro.judgment_ref(value).artifact_id]
    }
    assert len(table.rows) == 1 and table.rows[0].stage_a_label == "preserved"
    # Approval bytes are read again rather than trusting a previously loaded value.
    path = root / value.approved_decisions.logical_ref
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        stage_b.evaluate(inputs)


def test_focused_equivalence_needs_registered_approval_and_genuine_component_quotes(
    synthetic, monkeypatch
):
    w, value, root = synthetic
    sets = [s.model_dump(mode="json") for s in value.sets]
    sets[0]["placements"][0]["components"][0]["equivalence_note"] = "Synthetic focused attestation."
    quotes = [
        {
            "set_index": s.set_index,
            "container_node_id": p.container_node_id,
            "role": c.role,
            "component_text": c.text,
            "canonical_quote": c.text,
        }
        for s in value.sets
        for p in s.placements
        for c in p.components
    ]
    body = {
        "synthetic_fixture": True,
        "protocol": retro.PROTOCOL,
        "handle": value.handle,
        "approved_decisions_sha256": value.approved_decisions.sha256,
        "item_sha256": value.item_sha256,
        "canonical_sha256": value.canonical.sha256,
        "sets": sets,
        "quotes": quotes,
    }
    raw = canonical_json_bytes(body)
    ref = source_ref(raw)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(raw)
    changed = retro.RetrospectiveJudgment.model_validate(
        value.model_dump() | {"sets": sets, "component_resolution": ref}
    )
    with pytest.raises(ValueError, match="no registered researcher approval"):
        retro.validate_retrospective(changed, w.item, w.document, CANONICAL, root)
    monkeypatch.setattr(retro, "APPROVED_COMPONENT_RESOLUTIONS", frozenset({ref.sha256}))
    retro.validate_retrospective(changed, w.item, w.document, CANONICAL, root)
    body["quotes"][0]["canonical_quote"] = "This quote does not occur in the synthetic document."
    raw = canonical_json_bytes(body)
    bad_ref = source_ref(raw)
    (root / bad_ref.logical_ref).write_bytes(raw)
    bad = changed.model_copy(update={"component_resolution": bad_ref})
    monkeypatch.setattr(retro, "APPROVED_COMPONENT_RESOLUTIONS", frozenset({bad_ref.sha256}))
    with pytest.raises(ValueError, match="does not carry"):
        retro.validate_retrospective(bad, w.item, w.document, CANONICAL, root)
