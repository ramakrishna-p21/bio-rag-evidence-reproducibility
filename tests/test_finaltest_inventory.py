"""Storage root, registry chain and inventory verification on synthetic roots."""

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path

import pytest
from finaltest_support import authorization, made, oracle_chain_ok

from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest import inventory, layout
from biorag_repro.finaltest.inventory import (
    AuthorityNotEstablished,
    InitializationRefused,
    RegistryBusy,
    initialize_root,
    registry_lock,
    verify_root,
)

NO_PRIOR = "no_prior_final_test_execution"
INCOMPLETE = "history_inventory_incomplete"
UNAUTH = "unauthorized_final_test_artifact"
CONFLICT = "conflicting_final_test_artifact"
UNVERIFIED = "unverified_final_test_artifact"
NOW = "2026-01-01T00:00:00Z"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def code(root, anchors, auth) -> str:
    return verify_root(root, auth, anchors).code


def authority_cause(root, anchors, auth) -> str:
    with pytest.raises(AuthorityNotEstablished) as raised:
        verify_root(root, auth, anchors)
    return raised.value.cause


def rewrite(path: Path, data: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(data)


def event_file(root: Path, number: int = 1) -> Path:
    return root / "registry" / f"{number:06d}.json"


def append_event(root, auth, number, previous: bytes | None, **changes):
    values = {
        "root_id": layout.root_id_for(auth),
        "sequence": number,
        "event_type": "root_initialized",
        "previous_event_sha256": None if previous is None else sha(previous),
        "evidence": [],
        "created_utc": NOW,
    }
    event = layout.make_event(**(values | changes))
    data = layout.serialize(event)
    event_file(root, number).write_bytes(data)
    return data


# ---- initialization ------------------------------------------------------------------------


def test_fresh_rehearsal_root_is_verified_empty(tmp_path):
    root, anchors, auth = made(tmp_path)
    verdict = verify_root(root, auth, anchors)
    assert verdict.code == NO_PRIOR
    assert verdict.root_kind == "rehearsal"
    assert verdict.root_id == layout.root_id_for(auth)
    assert verdict.registry_head_sequence == 1
    assert oracle_chain_ok(root / "registry")


def test_the_layout_on_disk_is_exactly_the_documented_one(tmp_path):
    root, anchors, auth = made(tmp_path)
    found = sorted(str(p.relative_to(root)) for p in root.rglob("*"))
    assert found == [
        "ROOT.json",
        "locks",
        "locks/registry.lock",
        "registry",
        "registry/000001.json",
    ]
    assert [p.name for p in anchors.iterdir()] == [f"{layout.root_id_for(auth)}.anchor.json"]
    for path in (
        root / "ROOT.json",
        event_file(root),
        anchors / f"{layout.root_id_for(auth)}.anchor.json",
    ):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_genesis_event_cites_the_root_file_and_the_authorization(tmp_path):
    root, _, auth = made(tmp_path)
    event = json.loads(event_file(root).read_bytes())
    cited = {c["role"]: c["sha256"] for c in event["evidence"]}
    assert cited == {
        "root_file": sha((root / "ROOT.json").read_bytes()),
        "storage_authorization": layout.authorization_sha256(auth),
    }
    assert event["previous_event_sha256"] is None and event["sequence"] == 1


def test_reinitializing_a_verified_root_is_a_read_back(tmp_path):
    root, anchors, auth = made(tmp_path)
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    again = initialize_root(root, auth, anchors)
    assert again.code == NO_PRIOR
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == before


def test_a_final_test_root_cannot_be_initialized_here(tmp_path):
    auth = authorization(tmp_path, "final_test")
    with pytest.raises(InitializationRefused, match="rehearsal"):
        initialize_root(Path(auth.authorized_root), auth, tmp_path / "anchors")
    assert not Path(auth.authorized_root).exists() and not (tmp_path / "anchors").exists()


def test_initialization_needs_the_authorized_location(tmp_path):
    auth = authorization(tmp_path)
    other = tmp_path / "somewhere_else"
    with pytest.raises(InitializationRefused, match="location"):
        initialize_root(other, auth, tmp_path / "anchors")
    assert not other.exists()


def test_the_anchor_must_lie_outside_the_root(tmp_path):
    auth = authorization(tmp_path)
    root = Path(auth.authorized_root)
    with pytest.raises(InitializationRefused, match="outside"):
        initialize_root(root, auth, root / "anchors")
    assert not root.exists()


def test_a_non_empty_unrelated_directory_is_not_initialized_over(tmp_path):
    auth = authorization(tmp_path)
    root = Path(auth.authorized_root)
    root.mkdir()
    (root / "stray.txt").write_text("x")
    with pytest.raises(InitializationRefused):
        initialize_root(root, auth, tmp_path / "anchors")
    assert sorted(p.name for p in root.iterdir()) == ["stray.txt"]
    assert not (tmp_path / "anchors").exists()


def test_an_existing_empty_directory_may_be_initialized(tmp_path):
    auth = authorization(tmp_path)
    root = Path(auth.authorized_root)
    root.mkdir()
    assert initialize_root(root, auth, tmp_path / "anchors").code == NO_PRIOR


# ---- authority: the approved expectation, not the files, decides ---------------------------


def test_missing_and_empty_roots_prove_nothing(tmp_path):
    auth = authorization(tmp_path)
    root, anchors = Path(auth.authorized_root), tmp_path / "anchors"
    assert authority_cause(root, anchors, auth) == "not_initialized"
    root.mkdir()
    assert authority_cause(root, anchors, auth) == "not_initialized"


def test_a_caller_cannot_choose_another_root(tmp_path):
    root, anchors, auth = made(tmp_path)
    copy = tmp_path / "copy" / "final_test_v1"
    copy.parent.mkdir()
    os.system(f"cp -a '{root}' '{copy}'")
    assert authority_cause(copy, anchors, auth) == "root_location"


def test_expected_identity_from_another_approval_is_refused(tmp_path):
    root, anchors, _auth = made(tmp_path)
    other = authorization(tmp_path, corpus_content_sha256="a" * 64)
    assert authority_cause(root, anchors, other) == "root_identity"


def test_an_authorization_for_another_kind_is_refused(tmp_path):
    root, anchors, _ = made(tmp_path, "rehearsal")
    final = authorization(tmp_path, "final_test")
    assert authority_cause(root, anchors, final) == "root_identity"


def test_root_kind_recorded_in_the_root_must_match_the_approval(tmp_path):
    root, anchors, auth = made(tmp_path)
    record = json.loads((root / "ROOT.json").read_bytes())
    record["root_kind"] = "final_test"
    rewrite(root / "ROOT.json", layout.serialize(layout.RootRecord.model_validate(record)))
    assert authority_cause(root, anchors, auth) == "root_kind"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_id", "c" * 64),
        ("corpus_content_sha256", "c" * 64),
        ("benchmark_release_sha256", "c" * 64),
        ("authorization_sha256", "c" * 64),
    ],
)
def test_root_record_must_agree_with_the_approval(tmp_path, field, value):
    root, anchors, auth = made(tmp_path)
    record = json.loads((root / "ROOT.json").read_bytes()) | {field: value}
    rewrite(root / "ROOT.json", layout.serialize(layout.RootRecord.model_validate(record)))
    assert authority_cause(root, anchors, auth) in {"root_identity", "scope"}


def test_unsupported_root_version_is_unverified_not_authority(tmp_path):
    root, anchors, auth = made(tmp_path)
    record = json.loads((root / "ROOT.json").read_bytes()) | {"layout_version": "final_test_v2"}
    rewrite(root / "ROOT.json", (json.dumps(record, sort_keys=True, indent=2) + "\n").encode())
    assert code(root, anchors, auth) == UNVERIFIED


def test_symlinked_root_is_refused(tmp_path):
    root, anchors, _auth = made(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(root)
    linked = authorization(tmp_path, authorized_root=str(link))
    assert authority_cause(link, anchors, linked) == "root_location"


def test_symlinked_parent_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "parent").symlink_to(real)
    auth = authorization(tmp_path, authorized_root=str(tmp_path / "parent" / "final_test_v1"))
    (real / "final_test_v1").mkdir()
    (real / "final_test_v1" / "x").write_text("x")
    assert (
        authority_cause(Path(auth.authorized_root), tmp_path / "anchors", auth) == "root_location"
    )


def test_relative_or_traversing_root_paths_are_refused(tmp_path):
    root, anchors, auth = made(tmp_path)
    assert authority_cause(Path("final_test_v1"), anchors, auth) == "root_location"
    traversing = root / ".." / "final_test_v1"
    assert authority_cause(traversing, anchors, auth) == "root_location"


# ---- partial initialization ----------------------------------------------------------------


class Stop(Exception):
    pass


def failing_after(count, monkeypatch):
    calls = {"n": 0}

    def wrapper(*args, **kwargs):
        if calls["n"] == count:
            raise Stop
        calls["n"] += 1
        return publish_bytes(*args, **kwargs)

    monkeypatch.setattr(inventory, "publish_bytes", wrapper)


@pytest.mark.parametrize("written", [0, 1, 2])
def test_interruption_after_each_write_is_never_authoritative(tmp_path, monkeypatch, written):
    auth = authorization(tmp_path)
    root, anchors = Path(auth.authorized_root), tmp_path / "anchors"
    failing_after(written, monkeypatch)
    with pytest.raises(Stop):
        initialize_root(root, auth, anchors)
    monkeypatch.undo()
    snapshot = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert code(root, anchors, auth) == INCOMPLETE
    with pytest.raises(InitializationRefused, match="partial"):
        initialize_root(root, auth, anchors)  # no silent adoption or cleanup
    assert {p: p.read_bytes() for p in root.rglob("*") if p.is_file()} == snapshot


def test_missing_genesis_record_is_incomplete(tmp_path):
    root, anchors, auth = made(tmp_path)
    (root / "ROOT.json").unlink()
    assert code(root, anchors, auth) == INCOMPLETE


def test_missing_anchor_is_incomplete(tmp_path):
    root, anchors, auth = made(tmp_path)
    next(anchors.iterdir()).unlink()
    assert code(root, anchors, auth) == INCOMPLETE


def test_missing_genesis_event_is_incomplete(tmp_path):
    root, anchors, auth = made(tmp_path)
    event_file(root).unlink()
    assert code(root, anchors, auth) == INCOMPLETE


def test_orphan_temporary_publication_is_incomplete(tmp_path):
    root, anchors, auth = made(tmp_path)
    (root / "registry" / ".000002.json.abcd1234.tmp").write_bytes(b"{")
    assert code(root, anchors, auth) == INCOMPLETE
    (root / "registry" / ".000002.json.abcd1234.tmp").unlink()
    (root / ".ROOT.json.zz9.tmp").write_bytes(b"{")
    assert code(root, anchors, auth) == INCOMPLETE


def test_incomplete_outranks_other_findings(tmp_path):
    root, anchors, auth = made(tmp_path)
    (root / "surplus").mkdir()
    next(anchors.iterdir()).unlink()
    assert code(root, anchors, auth) == INCOMPLETE


# ---- registry chain ------------------------------------------------------------------------


def valid_second_event(root, auth):
    first = event_file(root).read_bytes()
    return first, append_event(root, auth, 2, first)


def test_a_second_genesis_event_is_a_conflict(tmp_path):
    root, anchors, auth = made(tmp_path)
    valid_second_event(root, auth)
    assert oracle_chain_ok(root / "registry")  # chain is sound; the type sequence is not
    assert code(root, anchors, auth) == CONFLICT


def test_sequence_gap_is_incomplete(tmp_path):
    root, anchors, auth = made(tmp_path)
    first = event_file(root).read_bytes()
    append_event(root, auth, 3, first)
    assert code(root, anchors, auth) == INCOMPLETE


def test_broken_previous_hash_is_unverified(tmp_path):
    root, _, auth = made(tmp_path)
    append_event(root, auth, 2, b"not the first event")
    assert not oracle_chain_ok(root / "registry")
    files = {n: event_file(root, n).read_bytes() for n in (1, 2)}
    problems = inventory.check_registry(layout.root_id_for(auth), files).problems
    assert ("unverified", "registry event 2 breaks the hash chain") in problems


def test_duplicate_sequence_number_is_a_conflict(tmp_path):
    root, anchors, auth = made(tmp_path)
    event_file(root, 2).write_bytes(event_file(root).read_bytes())
    assert code(root, anchors, auth) == CONFLICT


def test_foreign_event_is_a_conflict(tmp_path):
    root, anchors, auth = made(tmp_path)
    first = event_file(root).read_bytes()
    append_event(root, auth, 2, first, root_id="d" * 64)
    assert code(root, anchors, auth) == CONFLICT


def test_edited_event_bytes_are_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    data = event_file(root).read_bytes()
    rewrite(event_file(root), data.replace(b"\n", b" \n", 1))
    assert code(root, anchors, auth) == UNVERIFIED


def test_event_edited_and_reserialized_disagrees_with_the_anchor(tmp_path):
    root, anchors, auth = made(tmp_path)
    event = json.loads(event_file(root).read_bytes())
    again = layout.make_event(
        root_id=event["root_id"],
        sequence=1,
        event_type="root_initialized",
        previous_event_sha256=None,
        evidence=[layout.EvidenceCitation.model_validate(c) for c in event["evidence"]],
        created_utc="2027-02-02T00:00:00Z",
    )
    rewrite(event_file(root), layout.serialize(again))
    assert code(root, anchors, auth) == CONFLICT


def test_truncated_event_is_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    rewrite(event_file(root), event_file(root).read_bytes()[:40])
    assert code(root, anchors, auth) == UNVERIFIED


def test_malformed_json_and_duplicate_keys_are_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    rewrite(event_file(root), b'{"sequence": 1, "sequence": 1}')
    assert code(root, anchors, auth) == UNVERIFIED
    rewrite(event_file(root), b"not json")
    assert code(root, anchors, auth) == UNVERIFIED


def test_unexpected_event_type_is_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    first = event_file(root).read_bytes()
    event = json.loads(first) | {"sequence": 2, "event_type": "batch_authorized"}
    event_file(root, 2).write_text(json.dumps(event, sort_keys=True, indent=2) + "\n")
    assert code(root, anchors, auth) == UNVERIFIED


def test_filename_and_sequence_must_agree(tmp_path):
    root, anchors, auth = made(tmp_path)
    first = event_file(root).read_bytes()
    data = append_event(root, auth, 2, first)
    event_file(root, 2).rename(event_file(root, 5))
    assert data and code(root, anchors, auth) in {CONFLICT, INCOMPLETE}


def test_anchor_must_agree_with_the_root_file(tmp_path):
    root, anchors, auth = made(tmp_path)
    anchor_path = next(anchors.iterdir())
    anchor = json.loads(anchor_path.read_bytes()) | {"root_file_sha256": "e" * 64}
    rewrite(anchor_path, layout.serialize(layout.RootAnchor.model_validate(anchor)))
    assert code(root, anchors, auth) == CONFLICT


def test_anchor_for_another_identity_is_refused(tmp_path):
    root, anchors, auth = made(tmp_path)
    anchor_path = next(anchors.iterdir())
    anchor = json.loads(anchor_path.read_bytes()) | {"root_id": "e" * 64}
    rewrite(anchor_path, layout.serialize(layout.RootAnchor.model_validate(anchor)))
    assert authority_cause(root, anchors, auth) == "anchor_identity"


def test_event_citing_a_different_root_file_is_a_conflict(tmp_path):
    root, anchors, auth = made(tmp_path)
    record = json.loads((root / "ROOT.json").read_bytes()) | {"created_utc": "2031-01-01T00:00:00Z"}
    data = layout.serialize(layout.RootRecord.model_validate(record))
    rewrite(root / "ROOT.json", data)
    assert code(root, anchors, auth) == CONFLICT


# ---- inventory completeness ---------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        lambda root: (root / "batches").mkdir(),
        lambda root: (root / "batches").mkdir() or (root / "batches" / "b1").mkdir(),
        lambda root: (root / "extra.txt").write_text("x"),
        lambda root: (root / "registry" / "notes.txt").write_text("x"),
        lambda root: (root / "registry" / "0001.json").write_text("{}"),
        lambda root: (root / "registry" / "sub").mkdir(),
        lambda root: (root / "locks" / "other.lock").write_text(""),
        lambda root: (root / "locks" / "registry.lock").write_text("data"),
    ],
    ids=[
        "reserved-dir",
        "reserved-dir-with-batch",
        "root-file",
        "registry-file",
        "malformed-event-name",
        "registry-subdir",
        "other-lock",
        "lock-with-content",
    ],
)
def test_unexpected_entries_are_unauthorized(tmp_path, make):
    root, anchors, auth = made(tmp_path)
    make(root)
    assert code(root, anchors, auth) == UNAUTH


def test_symlink_in_the_authoritative_scope_is_unauthorized(tmp_path):
    root, anchors, auth = made(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"{}")
    (root / "registry" / "000002.json").symlink_to(outside)
    assert code(root, anchors, auth) == UNAUTH


def test_symlink_replacing_a_required_file_is_never_followed(tmp_path):
    root, anchors, auth = made(tmp_path)
    data = event_file(root).read_bytes()
    event_file(root).unlink()
    target = tmp_path / "elsewhere.json"
    target.write_bytes(data)
    event_file(root).symlink_to(target)
    verdict = verify_root(root, auth, anchors)
    assert verdict.code == INCOMPLETE  # the genesis event is not verified, so it is missing
    assert "unexpected: unexpected link registry/000001.json" in verdict.problems


def test_directory_replacing_a_required_file_is_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    (root / "ROOT.json").unlink()
    (root / "ROOT.json").mkdir()
    assert code(root, anchors, auth) == UNVERIFIED


@pytest.mark.skipif(os.geteuid() == 0, reason="permissions are not enforced for root")
def test_unreadable_required_artifact_is_unverified(tmp_path):
    root, anchors, auth = made(tmp_path)
    event_file(root).chmod(0)
    try:
        assert code(root, anchors, auth) == UNVERIFIED
    finally:
        event_file(root).chmod(0o444)


def test_authority_failures_outrank_artifact_findings(tmp_path):
    root, anchors, _auth = made(tmp_path)
    (root / "extra.txt").write_text("x")
    other = authorization(tmp_path, corpus_content_sha256="a" * 64)
    assert authority_cause(root, anchors, other) == "root_identity"


def test_inventory_snapshot_lists_every_entry_with_its_hash(tmp_path):
    root, anchors, auth = made(tmp_path)
    verdict = verify_root(root, auth, anchors)
    paths = [e["path"] for e in verdict.inventory["entries"]]
    assert paths == sorted(paths)
    assert set(paths) == {
        "ROOT.json",
        "locks",
        "locks/registry.lock",
        "registry",
        "registry/000001.json",
    }
    by_path = {e["path"]: e for e in verdict.inventory["entries"]}
    assert by_path["registry/000001.json"]["sha256"] == sha(event_file(root).read_bytes())
    assert verdict.inventory["registry_head"] == {
        "sequence": 1,
        "event_file_sha256": sha(event_file(root).read_bytes()),
    }
    assert verdict.inventory["classification"] == NO_PRIOR
    assert str(tmp_path) not in json.dumps(verdict.inventory)


# ---- locking -------------------------------------------------------------------------------


def test_registry_lock_excludes_a_second_holder_on_this_host(tmp_path):
    root, _, _ = made(tmp_path)
    with registry_lock(root), pytest.raises(RegistryBusy), registry_lock(root):
        pass
    with registry_lock(root):  # released afterwards
        pass


def test_registry_lock_sees_a_foreign_flock(tmp_path):
    root, _, _ = made(tmp_path)
    descriptor = os.open(root / "locks" / "registry.lock", os.O_RDWR)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        with pytest.raises(RegistryBusy), registry_lock(root):
            pass
    finally:
        os.close(descriptor)


def test_lock_file_is_a_known_operational_file_and_stays_empty(tmp_path):
    root, anchors, auth = made(tmp_path)
    with registry_lock(root):
        pass
    assert (root / "locks" / "registry.lock").stat().st_size == 0
    assert code(root, anchors, auth) == NO_PRIOR


def test_event_citing_other_records_is_a_conflict_even_when_the_anchor_follows(tmp_path):
    root, anchors, auth = made(tmp_path)
    event = json.loads(event_file(root).read_bytes())
    forged = layout.make_event(
        root_id=event["root_id"],
        sequence=1,
        event_type="root_initialized",
        previous_event_sha256=None,
        evidence=[
            layout.EvidenceCitation(role="root_file", sha256="9" * 64),
            layout.EvidenceCitation(role="storage_authorization", sha256="8" * 64),
        ],
        created_utc=NOW,
    )
    data = layout.serialize(forged)
    rewrite(event_file(root), data)
    anchor_path = next(anchors.iterdir())
    anchor = json.loads(anchor_path.read_bytes()) | {"genesis_event_sha256": sha(data)}
    rewrite(anchor_path, layout.serialize(layout.RootAnchor.model_validate(anchor)))
    verdict = verify_root(root, auth, anchors)
    assert verdict.code == CONFLICT
    assert verdict.problems == ("conflicting: the genesis event cites other root records",)


def test_registry_findings_name_foreign_and_misnumbered_events(tmp_path):
    root, _, auth = made(tmp_path)
    root_id = layout.root_id_for(auth)
    first = event_file(root).read_bytes()
    foreign = layout.serialize(
        layout.make_event(
            root_id="d" * 64,
            sequence=1,
            event_type="root_initialized",
            previous_event_sha256=None,
            evidence=[],
            created_utc=NOW,
        )
    )
    assert inventory.check_registry(root_id, {1: foreign}).problems == [
        ("conflicting", "registry event 1 belongs to another root")
    ]
    second = layout.serialize(
        layout.make_event(
            root_id=root_id,
            sequence=2,
            event_type="root_initialized",
            previous_event_sha256=sha(first),
            evidence=[],
            created_utc=NOW,
        )
    )
    assert inventory.check_registry(root_id, {1: second}).problems == [
        ("conflicting", "registry event 1 claims 2")
    ]


def test_initialization_reads_back_and_refuses_an_unverified_result(tmp_path, monkeypatch):
    auth = authorization(tmp_path)
    real = inventory.verify_root

    def damaged(root, authorization, anchors):
        verdict = real(root, authorization, anchors)
        return type(verdict)(**{**verdict.__dict__, "code": UNVERIFIED})

    monkeypatch.setattr(inventory, "verify_root", damaged)
    with pytest.raises(InitializationRefused, match="does not verify"):
        initialize_root(Path(auth.authorized_root), auth, tmp_path / "anchors")
