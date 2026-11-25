"""Synthetic externally retained batch history; no protected inputs."""

import json
from pathlib import Path

import pytest
from finaltest_support import authorization

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.finaltest import inventory
from biorag_repro.finaltest.history_v2 import (
    HistoryRefused,
    _verify_history,
    anchor_batch,
    verify_history,
)
from biorag_repro.finaltest.layout import ANCHOR_SUFFIX, root_id_for


def root_fixture(tmp_path):
    auth = authorization(tmp_path, "final_test")
    root = Path(auth.authorized_root)
    anchors = tmp_path / "independent_anchors"
    inventory._initialize(root, auth, anchors, inventory.utc_now)
    head = sha256_bytes((anchors / (root_id_for(auth) + ANCHOR_SUFFIX)).read_bytes())
    return root, auth, anchors, head


def batch(root, number):
    batch_id = f"run_{number:032x}"
    path = root / "batches" / batch_id
    (path / "anchors").mkdir(parents=True)
    (path / "batch.json").write_text(json.dumps({"run_id": batch_id}))
    (path / "anchors" / "000001.json").write_bytes(
        canonical_json_bytes(
            {
                "files": {"batch.json": sha256_file(path / "batch.json")},
                "previous_anchor_sha256": None,
            }
        )
    )
    return batch_id, path


def test_genesis_two_batches_and_independent_latest_head(tmp_path):
    root, auth, anchors, head = root_fixture(tmp_path)
    assert verify_history(root, auth, anchors, head) == []
    first, _ = batch(root, 1)
    head = anchor_batch(root, auth, anchors, first, head)
    second, _ = batch(root, 2)
    head = anchor_batch(root, auth, anchors, second, head)
    assert [r["batch_id"] for r in verify_history(root, auth, anchors, head)] == [first, second]


def test_duplicate_external_anchor_is_refused(tmp_path):
    root, auth, anchors, head = root_fixture(tmp_path)
    batch_id, _ = batch(root, 1)
    head = anchor_batch(root, auth, anchors, batch_id, head)
    with pytest.raises(HistoryRefused, match="duplicate batch anchor"):
        anchor_batch(root, auth, anchors, batch_id, head)


def test_external_anchor_commitment_survives_crash_before_publication(tmp_path, monkeypatch):
    from biorag_repro.finaltest import history_v2
    from biorag_repro.finaltest.custody import SyntheticCustodian

    root, auth, anchors, head = root_fixture(tmp_path)
    batch_id, path = batch(root, 1)
    custodian = SyntheticCustodian(head)
    manifest = "e" * 64
    for name in ("batch.json", "anchors/000001.json"):
        custodian.commit(
            batch_id,
            manifest,
            custodian.authoritative_head(),
            name,
            sha256_file(path / name),
        )
    publish = history_v2.publish_bytes

    def interrupted(*_args, **_kwargs):
        raise KeyboardInterrupt("synthetic termination before external publication")

    monkeypatch.setattr(history_v2, "publish_bytes", interrupted)
    with pytest.raises(KeyboardInterrupt):
        anchor_batch(
            root,
            auth,
            anchors,
            batch_id,
            head,
            custodian=custodian,
            manifest_sha256=manifest,
            custody_genesis_head=head,
        )
    assert len(custodian.entries()) == 3
    assert not list(anchors.glob("*.000001.json"))
    monkeypatch.setattr(history_v2, "publish_bytes", publish)
    recovered = anchor_batch(
        root,
        auth,
        anchors,
        batch_id,
        head,
        custodian=custodian,
        manifest_sha256=manifest,
        custody_genesis_head=head,
    )
    assert len(custodian.entries()) == 3
    assert verify_history(root, auth, anchors, recovered)[0]["batch_id"] == batch_id
    with pytest.raises(HistoryRefused, match="latest head"):
        verify_history(root, auth, anchors, head)


def test_synthetic_4575_observation_publication_and_readback(tmp_path):
    root, auth, anchors, head = root_fixture(tmp_path)
    batch_id, path = batch(root, 1)
    observations = path / "results"
    observations.mkdir()
    for item in range(305):
        for condition in range(15):
            name = f"synthetic_{item:03d}_{condition:02d}.json"
            (observations / name).write_bytes(
                canonical_json_bytes({"item": f"synthetic_{item:03d}", "condition": condition})
            )
    files = {
        p.relative_to(path).as_posix(): sha256_file(p)
        for p in sorted(path.rglob("*"))
        if p.is_file() and p.name != "000001.json"
    }
    (path / "anchors" / "000001.json").write_bytes(
        canonical_json_bytes({"files": files, "previous_anchor_sha256": None})
    )
    head = anchor_batch(root, auth, anchors, batch_id, head)
    assert len(verify_history(root, auth, anchors, head)[0]["files"]) == 4577
    (observations / "synthetic_000_00.json").write_text("changed")
    with pytest.raises(HistoryRefused):
        verify_history(root, auth, anchors, head)


@pytest.mark.parametrize("damage", ["tamper", "rollback", "extra", "anchor_gap"])
def test_history_refuses_damage_and_rollback(tmp_path, damage):
    root, auth, anchors, head = root_fixture(tmp_path)
    first, path = batch(root, 1)
    head = anchor_batch(root, auth, anchors, first, head)
    if damage == "tamper":
        (path / "batch.json").write_text("changed")
    elif damage == "rollback":
        (anchors / f"{root_id_for(auth)}.000001.json").unlink()
    elif damage == "extra":
        (path / "extra.json").write_text("{}")
    else:
        old = anchors / f"{root_id_for(auth)}.000001.json"
        old.rename(anchors / f"{root_id_for(auth)}.000002.json")
    with pytest.raises(HistoryRefused):
        verify_history(root, auth, anchors, head)


def test_unanchored_interrupted_batch_is_not_accepted(tmp_path):
    root, auth, anchors, head = root_fixture(tmp_path)
    batch(root, 1)
    with pytest.raises(HistoryRefused, match="unanchored|missing from its latest internal seal"):
        verify_history(root, auth, anchors, head)


def test_same_batch_continues_only_from_retained_head(tmp_path):
    root, auth, anchors, head = root_fixture(tmp_path)
    batch_id, path = batch(root, 1)
    head = anchor_batch(root, auth, anchors, batch_id, head)
    (path / "attempts").mkdir()
    (path / "attempts" / "000001.json").write_text('{"event":"interrupted"}')
    with pytest.raises(HistoryRefused, match="missing from its latest internal seal"):
        verify_history(root, auth, anchors, head)
    prior = path / "anchors" / "000001.json"
    (path / "anchors" / "000002.json").write_bytes(
        canonical_json_bytes(
            {
                "files": {
                    "batch.json": sha256_file(path / "batch.json"),
                    "attempts/000001.json": sha256_file(path / "attempts" / "000001.json"),
                },
                "previous_anchor_sha256": sha256_file(prior),
            }
        )
    )
    new_head = anchor_batch(root, auth, anchors, batch_id, head)
    assert len(verify_history(root, auth, anchors, new_head)) == 2
    with pytest.raises(HistoryRefused, match="latest head"):
        verify_history(root, auth, anchors, head)


def test_pending_suffix_checks_external_predecessor_before_recovery_write(tmp_path):
    root, auth, anchors, genesis = root_fixture(tmp_path)
    batch_id, path = batch(root, 1)
    current = anchor_batch(root, auth, anchors, batch_id, genesis)
    (path / "attempts").mkdir()
    (path / "attempts/000001.json").write_bytes(b"{}")
    assert len(_verify_history(root, auth, anchors, current, batch_id)) == 1
    with pytest.raises(HistoryRefused, match="latest head"):
        _verify_history(root, auth, anchors, genesis, batch_id)
    assert len(list((path / "anchors").glob("*.json"))) == 1


@pytest.mark.parametrize("damage", ["unsealed_file", "missing_file", "seal_gap", "seal_tamper"])
def test_external_anchor_refuses_incomplete_internal_seal(tmp_path, damage):
    root, auth, anchors, head = root_fixture(tmp_path)
    batch_id, path = batch(root, 1)
    if damage == "unsealed_file":
        (path / "unsealed.json").write_text("{}")
    elif damage == "missing_file":
        (path / "batch.json").unlink()
    elif damage == "seal_gap":
        (path / "anchors" / "000001.json").rename(path / "anchors" / "000002.json")
    else:
        (path / "anchors" / "000001.json").write_text(json.dumps({"files": {}}))
    with pytest.raises(HistoryRefused):
        anchor_batch(root, auth, anchors, batch_id, head)
