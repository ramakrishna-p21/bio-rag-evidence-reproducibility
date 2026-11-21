"""Synthetic decisions for the prospective execution checks."""

import json
from types import SimpleNamespace

import pytest

from biorag_repro.finaltest import authorization as auth


def fixture_report(blocked=None):
    gates = [
        SimpleNamespace(
            gate_id=name,
            status="pending" if name == blocked else "passed",
            verification="unverified" if name == blocked else "verified",
            reason_code="fixture",
        )
        for names in auth.GROUPS.values()
        for name in names
    ]
    gates.append(
        SimpleNamespace(
            gate_id="G07_r_analysis_environment",
            status="passed",
            verification="verified",
            reason_code="fixture",
        )
    )
    return SimpleNamespace(
        gates=gates,
        candidate=SimpleNamespace(head_sha="a" * 40, freeze_record_sha256="b" * 64),
        policy_sha256="c" * 64,
        binding_sha256="d" * 64,
    )


def setup(monkeypatch, blocked=None):
    report = fixture_report(blocked)
    monkeypatch.setattr(auth, "load_policy", lambda _repo: object())
    monkeypatch.setattr(auth, "system_context", lambda _repo, _data: object())
    monkeypatch.setattr(
        auth, "evaluate", lambda _ctx, _policy, **_kwargs: SimpleNamespace(report=report)
    )
    monkeypatch.setattr(
        auth.evidence,
        "_approved_storage",
        lambda _ctx, _policy: (SimpleNamespace(authorized_root="/synthetic/root"), "fixture"),
    )
    monkeypatch.setattr(auth, "verify_history", lambda *_args: [])
    decision = {
        "schema_version": "final_test_execution_v2",
        "researcher": "Rama Krishna Pudota",
        "source_revision": "a" * 40,
        "private_storage_and_transfer_approved": True,
        "frozen_configuration_sha256": "b" * 64,
    }
    monkeypatch.setattr(auth, "read_regular", lambda _repo, _name: json.dumps(decision).encode())
    return decision


def test_four_checks_pass_with_synthetic_evidence_and_verified_analysis_host(monkeypatch, tmp_path):
    setup(monkeypatch)
    result = auth.authorize(tmp_path, tmp_path, "e" * 64)
    assert result.source_revision == "a" * 40
    assert len(result.evidence_sha256) == 64


def test_analysis_host_is_required_by_frozen_start_contract(monkeypatch, tmp_path):
    setup(monkeypatch)
    original = auth.evaluate

    def pending(ctx, policy, **kwargs):
        result = original(ctx, policy, **kwargs)
        for gate in result.report.gates:
            if gate.gate_id == "G07_r_analysis_environment":
                gate.status = "pending"
                gate.verification = "unverified"
        return result

    monkeypatch.setattr(auth, "evaluate", pending)
    with pytest.raises(auth.ExecutionRefused, match="G07_r_analysis_environment"):
        auth.authorize(tmp_path, tmp_path, "e" * 64)


@pytest.mark.parametrize("group", list(auth.GROUPS))
def test_every_execution_group_blocks_when_evidence_is_missing(monkeypatch, tmp_path, group):
    setup(monkeypatch, auth.GROUPS[group][0])
    with pytest.raises(auth.ExecutionRefused, match=group):
        auth.authorize(tmp_path, tmp_path, "e" * 64)


def test_campaign_decision_must_match_live_source(monkeypatch, tmp_path):
    decision = setup(monkeypatch)
    decision["source_revision"] = "f" * 40
    with pytest.raises(auth.ExecutionRefused, match="campaign decision"):
        auth.authorize(tmp_path, tmp_path, "e" * 64)


def test_unanchored_history_blocks_execution(monkeypatch, tmp_path):
    setup(monkeypatch)

    def refused(*_args):
        raise ValueError("latest head differs")

    monkeypatch.setattr(auth, "verify_history", refused)
    with pytest.raises(auth.ExecutionRefused, match="execution_integrity: G13"):
        auth.authorize(tmp_path, tmp_path, "e" * 64)


def test_amended_drive_requires_exact_campaign_selection(monkeypatch, tmp_path):
    decision = setup(monkeypatch)
    with pytest.raises(auth.ExecutionRefused, match="amended campaign custody gate"):
        auth.authorize(tmp_path, tmp_path, "e" * 64, custody_mode="amended_drive")
    decision.update(
        {
            "custody_mode": "amended_drive_v1",
            "campaign_id": "rk_capstone_final_test_2026",
            "drive_root": str(tmp_path / "drive"),
        }
    )
    selected = auth.authorize(tmp_path, tmp_path, "e" * 64, custody_mode="amended_drive")
    assert selected.custody_mode == "amended_drive"
    assert selected.campaign_id == "rk_capstone_final_test_2026"


def test_amended_authorization_recognizes_one_unacknowledged_anchor(monkeypatch, tmp_path):
    from test_finaltest_history_v2 import batch, root_fixture

    from biorag_repro.finaltest import history_v2

    root, storage, anchors, genesis = root_fixture(tmp_path)
    batch_id, _ = batch(root, 1)
    head = history_v2.anchor_batch(root, storage, anchors, batch_id, genesis)
    decision = setup(monkeypatch)
    decision.update(
        {
            "custody_mode": "amended_drive_v1",
            "campaign_id": "rk_capstone_final_test_2026",
            "drive_root": str(tmp_path / "drive"),
        }
    )
    monkeypatch.setattr(auth.evidence, "ROOT_ANCHORS", "independent_anchors")
    monkeypatch.setattr(
        auth.evidence, "_approved_storage", lambda _ctx, _policy: (storage, "fixture")
    )
    monkeypatch.setattr(auth, "verify_history", history_v2.verify_history)
    result = auth.authorize(
        tmp_path,
        tmp_path,
        genesis,
        custody_mode="amended_drive",
        pending_batch_id=batch_id,
    )
    assert result.history_head_sha256 == genesis
    assert result.pending_head_sha256 == head
