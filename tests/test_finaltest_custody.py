"""Synthetic custody checks; no record here authorizes final-test execution."""

from pathlib import Path

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest.custody import CustodyRefused, SyntheticCustodian, verify_custody

GENESIS = "a" * 64
MANIFEST = "b" * 64
BATCH = "run_" + "c" * 32


def commit(custodian, path, data, *, head=None):
    return custodian.commit(
        BATCH, MANIFEST, head or custodian.authoritative_head(), path, sha256_bytes(data)
    )


def test_committed_absent_and_published_committed_are_distinct(tmp_path):
    custodian = SyntheticCustodian(GENESIS)
    commit(custodian, "results/one.json", b"one")
    assert verify_custody(
        custodian, tmp_path, BATCH, MANIFEST, custodian.authoritative_head(), GENESIS
    ) == {"results/one.json": "committed_absent"}
    (tmp_path / "results").mkdir()
    (tmp_path / "results/one.json").write_bytes(b"one")
    assert verify_custody(
        custodian, tmp_path, BATCH, MANIFEST, custodian.authoritative_head(), GENESIS
    ) == {"results/one.json": "published_committed"}


@pytest.mark.parametrize("damage", ["altered", "uncommitted", "missing_proof", "wrong_batch"])
def test_recovery_refuses_untrusted_publication(tmp_path, damage):
    custodian = SyntheticCustodian(GENESIS)
    (tmp_path / "results").mkdir()
    path = tmp_path / "results/one.json"
    path.write_bytes(b"one")
    if damage != "missing_proof":
        commit(custodian, "results/one.json", b"one")
    if damage == "altered":
        path.write_bytes(b"other")
    elif damage == "uncommitted":
        (tmp_path / "results/two.json").write_bytes(b"two")
    with pytest.raises(CustodyRefused):
        verify_custody(
            custodian,
            tmp_path,
            "run_" + "d" * 32 if damage == "wrong_batch" else BATCH,
            MANIFEST,
            custodian.authoritative_head(),
            GENESIS,
        )


def test_authoritative_head_rejects_stale_divergent_and_replayed_history(tmp_path):
    custodian = SyntheticCustodian(GENESIS)
    first = commit(custodian, "results/one.json", b"one")
    with pytest.raises(CustodyRefused, match="stale"):
        commit(custodian, "results/two.json", b"two", head=GENESIS)
    with pytest.raises(CustodyRefused, match="already committed"):
        commit(custodian, "results/one.json", b"one")
    for head in (GENESIS, "f" * 64):
        with pytest.raises(CustodyRefused, match="authoritative head"):
            verify_custody(custodian, tmp_path, BATCH, MANIFEST, head, GENESIS)
    assert first.head == custodian.authoritative_head()


def test_rollback_of_commitment_list_cannot_satisfy_authoritative_head(tmp_path):
    custodian = SyntheticCustodian(GENESIS)
    commit(custodian, "batch.json", b"batch")
    custodian._entries.clear()  # adversarial test of a rolled-back replica
    with pytest.raises(CustodyRefused, match="cannot be reconstructed"):
        verify_custody(
            custodian, Path(tmp_path), BATCH, MANIFEST, custodian.authoritative_head(), GENESIS
        )


def test_crash_after_commit_before_write_remains_committed_absent(tmp_path, monkeypatch):
    from scripts import run_generation as runner

    custodian = SyntheticCustodian(GENESIS)
    batch = tmp_path / BATCH
    token = runner._publication_session.set((batch, custodian, BATCH, MANIFEST))

    def interrupted(*_args):
        raise KeyboardInterrupt("before atomic publication")

    monkeypatch.setattr(runner.os, "link", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            runner.write_once(batch / "anchors/000001.json", b"seal")
    finally:
        runner._publication_session.reset(token)
    assert verify_custody(
        custodian, batch, BATCH, MANIFEST, custodian.authoritative_head(), GENESIS
    ) == {"anchors/000001.json": "committed_absent"}


def test_production_runner_refuses_without_custody_before_protected_read(monkeypatch):
    from types import SimpleNamespace

    from biorag_repro.finaltest import authorization
    from scripts.run_generation import Refused, run

    def forbidden_read(_self):
        raise AssertionError("protected content was read")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    monkeypatch.setattr(authorization, "authorize", lambda *_: object())
    with pytest.raises(Refused, match="independently administered recovery custody"):
        run(
            SimpleNamespace(
                phase="final_test",
                official=True,
                repo=Path("synthetic"),
                data_root=Path("synthetic"),
                expected_head_sha256=GENESIS,
                published_context_root=Path("synthetic"),
            )
        )
