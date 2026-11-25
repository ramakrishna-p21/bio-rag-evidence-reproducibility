"""Production genesis is exact, immutable and still requires independent head custody."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from finaltest_support import authorization

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.finaltest import genesis, inventory, layout
from biorag_repro.finaltest.authorization import CampaignDecision, CampaignPreflight


def preflight(tmp_path):
    storage = authorization(tmp_path, "final_test")
    created = "2026-11-21T00:00:00Z"
    head = sha256_bytes(inventory.genesis_records(storage, created).anchor)
    decision = CampaignDecision(
        schema_version="final_test_execution_v2",
        researcher="Rama Krishna Pudota",
        source_revision="a" * 40,
        private_storage_and_transfer_approved=True,
        frozen_configuration_sha256="b" * 64,
        storage_authorization_sha256=layout.authorization_sha256(storage),
        storage_root=storage.authorized_root,
        root_id=layout.root_id_for(storage),
        genesis_created_utc=created,
        genesis_anchor_sha256=head,
        custody_mode="amended_drive_v1",
        campaign_id="rk_capstone_final_test_2026",
        drive_root="/content/drive/MyDrive/RK_Capstone_Final_2026",
    )
    decision_bytes = canonical_json_bytes(decision.model_dump(mode="json", exclude_none=True))
    return CampaignPreflight(SimpleNamespace(), storage, decision, decision_bytes)


def test_authorized_final_test_genesis_is_published_once_and_read_back(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    monkeypatch.setattr(genesis, "campaign_preflight", lambda *_args, **_kwargs: approved)
    decision_sha = sha256_bytes(approved.decision_bytes)
    head = approved.decision.genesis_anchor_sha256

    result = genesis.initialize_final_test_root(
        tmp_path,
        tmp_path / "data",
        decision_sha256=decision_sha,
        expected_genesis_head_sha256=head,
    )
    before = {path: path.read_bytes() for path in result.root.rglob("*") if path.is_file()}
    again = genesis.initialize_final_test_root(
        tmp_path,
        tmp_path / "data",
        decision_sha256=decision_sha,
        expected_genesis_head_sha256=head,
    )
    assert again == result
    assert {path: path.read_bytes() for path in result.root.rglob("*") if path.is_file()} == before
    assert (
        result.anchor.read_bytes()
        == inventory.genesis_records(approved.storage, approved.decision.genesis_created_utc).anchor
    )


@pytest.mark.parametrize("wrong", ["decision", "head"])
def test_genesis_refuses_wrong_explicit_identity_before_writing(monkeypatch, tmp_path, wrong):
    approved = preflight(tmp_path)
    monkeypatch.setattr(genesis, "campaign_preflight", lambda *_args, **_kwargs: approved)
    decision_sha = sha256_bytes(approved.decision_bytes)
    head = approved.decision.genesis_anchor_sha256
    with pytest.raises(genesis.GenesisRefused):
        genesis.initialize_final_test_root(
            tmp_path,
            tmp_path / "data",
            decision_sha256="0" * 64 if wrong == "decision" else decision_sha,
            expected_genesis_head_sha256="0" * 64 if wrong == "head" else head,
        )
    assert not Path(approved.storage.authorized_root).exists()


def test_genesis_preserves_and_refuses_a_partial_root(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    monkeypatch.setattr(genesis, "campaign_preflight", lambda *_args, **_kwargs: approved)
    root = Path(approved.storage.authorized_root)
    root.mkdir()
    (root / "partial").write_text("preserve")
    with pytest.raises(genesis.GenesisRefused, match="partial or inconsistent"):
        genesis.initialize_final_test_root(
            tmp_path,
            tmp_path / "data",
            decision_sha256=sha256_bytes(approved.decision_bytes),
            expected_genesis_head_sha256=approved.decision.genesis_anchor_sha256,
        )
    assert (root / "partial").read_text() == "preserve"


def run(monkeypatch, tmp_path, approved, data=None):
    monkeypatch.setattr(genesis, "campaign_preflight", lambda *_args, **_kwargs: approved)
    return genesis.initialize_final_test_root(
        tmp_path,
        data or tmp_path / "data",
        decision_sha256=sha256_bytes(approved.decision_bytes),
        expected_genesis_head_sha256=approved.decision.genesis_anchor_sha256,
    )


def test_genesis_refuses_a_symlinked_root_and_leaves_the_target_alone(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    target = tmp_path / "elsewhere"
    target.mkdir()
    Path(approved.storage.authorized_root).symlink_to(target)
    with pytest.raises(genesis.GenesisRefused, match="plain directory"):
        run(monkeypatch, tmp_path, approved)
    assert list(target.iterdir()) == []


def test_genesis_refuses_anchors_inside_the_root(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    root = Path(approved.storage.authorized_root)
    with pytest.raises(genesis.GenesisRefused, match="outside the root"):
        run(monkeypatch, tmp_path, approved, data=root)
    assert not root.exists()


def test_genesis_refuses_a_rehearsal_authorization(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    approved = CampaignPreflight(
        approved.report,
        authorization(tmp_path, "rehearsal"),
        approved.decision,
        approved.decision_bytes,
    )
    with pytest.raises(genesis.GenesisRefused, match="final_test"):
        run(monkeypatch, tmp_path, approved)


def test_genesis_refuses_an_interrupted_publication_without_repairing_it(monkeypatch, tmp_path):
    approved = preflight(tmp_path)
    run(monkeypatch, tmp_path, approved)
    root = Path(approved.storage.authorized_root)
    event = root / layout.REGISTRY_DIR / "000001.json"
    event.chmod(0o644)
    event.unlink()
    with pytest.raises(genesis.GenesisRefused):
        run(monkeypatch, tmp_path, approved)
    assert not event.exists()


def test_genesis_refuses_a_root_that_belongs_to_another_authorization(monkeypatch, tmp_path):
    other = authorization(tmp_path, "final_test", storage_label="another storage")
    inventory._initialize(
        Path(other.authorized_root),
        other,
        tmp_path / "data/readiness/final_test_root",
        lambda: "2026-11-21T00:00:00Z",
    )
    approved = preflight(tmp_path)
    with pytest.raises(genesis.GenesisRefused):
        run(monkeypatch, tmp_path, approved)


def test_genesis_does_not_depend_on_a_g13_ready_state():
    """Preflight must not require G13; the root it creates is what makes G13 pass."""
    from biorag_repro.finaltest.authorization import GROUPS

    assert not any("G13" in gate for gates in GROUPS.values() for gate in gates)
