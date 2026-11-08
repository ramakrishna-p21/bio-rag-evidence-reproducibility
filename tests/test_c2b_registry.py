"""Rehearsal-only registry events and the verifier's view of them (P10-C2B).

A final-test root must be judged exactly as before: the two new event types and the new files
stay unauthorized artifacts there. A rehearsal root accepts them only when every cited file
exists, hashes as cited and refers to the right batch.
"""

import json
from pathlib import Path

import pytest
from finaltest_campaign_support import sha
from finaltest_start_support import ARM, make_world
from finaltest_support import authorization

from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.inventory import verify_root
from biorag_repro.finaltest.layout import EvidenceCitation
from biorag_repro.finaltest.registry import ProtocolRefused
from biorag_repro.finaltest.session import open_session
from biorag_repro.finaltest.start import _start_with

NOTE = inventory.HEAD_NOT_ESTABLISHED_TEXT
UNAUTH = "unauthorized_final_test_artifact"
CONFLICT = "conflicting_final_test_artifact"
INCOMPLETE = "history_inventory_incomplete"


def started_world(tmp_path):
    w = make_world(tmp_path)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
    return w, started


def verdict_of(w):
    return verify_root(w.root, w.auth, w.anchors, w.witnesses)


def edit(path: Path, data: bytes) -> None:
    path.chmod(0o644)
    path.write_bytes(data)


def raw_event(w, event_type, evidence):
    root_id = layout.root_id_for(w.auth)
    return registry._append(w.root, w.witnesses, root_id, event_type, evidence, w.clock)


def cited(event_type, **values):
    return [
        EvidenceCitation(role=r, sha256=values.get(r, sha(r)))
        for r in layout.REHEARSAL_ROLES[event_type]
    ]


# ---- final-test roots are judged as before --------------------------------------------------


def test_a_final_test_root_still_rejects_the_new_events_files_and_directories(tmp_path):
    base = tmp_path / "final"
    base.mkdir()
    auth = authorization(base, "final_test")
    anchors, witnesses = base / "anchors", base / "witnesses"
    inventory._initialize(Path(auth.authorized_root), auth, anchors, inventory.utc_now)
    root = Path(auth.authorized_root)
    root_id = layout.root_id_for(auth)
    registry._ensure_genesis_witness(root, witnesses, root_id, inventory.utc_now)
    clean = verify_root(root, auth, anchors, witnesses)
    assert clean.code == "no_prior_final_test_execution"

    registry._append(root, witnesses, root_id, "batch_authorized",
                     cited("batch_authorized"), inventory.utc_now)  # fmt: skip
    (root / "runs" / ("a" * 32)).mkdir(parents=True)
    (root / "runs" / ("a" * 32) / "run.json").write_bytes(b"{}")
    (root / layout.SESSION_LOCK_FILE).write_bytes(b"")
    verdict = verify_root(root, auth, anchors, witnesses)
    assert verdict.code == UNAUTH
    assert any("unauthorized type batch_authorized" in p for p in verdict.problems)
    assert any("unexpected" in p and "runs" in p for p in verdict.problems)
    assert any("locks/session.lock" in p for p in verdict.problems)
    paths = {e["path"] for e in verdict.inventory["entries"]}
    assert "runs" in paths and not any(p.startswith("runs/") for p in paths)  # never entered
    assert verdict.registry_head_sequence == 2


# ---- the rehearsal verifier -----------------------------------------------------------------


def test_a_complete_start_verifies_with_only_the_head_note(tmp_path):
    w, started = started_world(tmp_path)
    verdict = verdict_of(w)
    assert verdict.problems == (NOTE,) and verdict.code == INCOMPLETE
    assert verdict.batches[started.batch_id].state == "authorized"
    entries = {e["path"] for e in verdict.inventory["entries"]}
    assert f"runs/{started.run_id}" in entries
    assert f"batches/{started.batch_id}/inputs/binding.json" in entries
    assert f"batches/{started.batch_id}/citations/000001.json" in entries
    assert layout.SESSION_LOCK_FILE in entries


def paths(w, started):
    base = w.root / "batches" / started.batch_id
    return {
        "binding": base / "inputs/binding.json",
        "rosters": sorted((base / "inputs/rosters").glob("*.json")),
        "citation": base / "citations/000001.json",
        "run": w.root / "runs" / started.run_id / "run.json",
        "inputs": base / "inputs",
        "citations": base / "citations",
    }


def test_a_changed_binding_citation_or_run_record_is_a_conflict(tmp_path):
    for name in ("binding", "citation"):
        w, started = started_world(tmp_path / name)
        path = paths(w, started)[name]
        edit(path, path.read_bytes() + b" ")
        verdict = verdict_of(w)
        assert verdict.code in (CONFLICT, INCOMPLETE) and verdict.problems != (NOTE,)
        assert verdict.batches[started.batch_id].state != "authorized"
        assert any("differs from its event" in p for p in verdict.problems)


def test_a_changed_run_record_no_longer_matches_the_authorization(tmp_path):
    w, started = started_world(tmp_path)
    path = paths(w, started)["run"]
    edit(path, path.read_bytes() + b" ")
    verdict = verdict_of(w)
    assert verdict.batches[started.batch_id].state == "inputs_bound"
    assert any("has no run record" in p for p in verdict.problems)
    assert any("is not an authorized run" in p for p in verdict.problems)


def test_a_missing_roster_or_citation_is_incomplete_not_repaired(tmp_path):
    w, started = started_world(tmp_path / "a")
    paths(w, started)["rosters"][0].unlink()
    verdict = verdict_of(w)
    assert any("is missing" in p and "roster" in p for p in verdict.problems)
    w, started = started_world(tmp_path / "b")
    paths(w, started)["citation"].unlink()
    verdict = verdict_of(w)
    assert any("no citation file" in p for p in verdict.problems)
    assert verdict.batches[started.batch_id].state == "inputs_bound"


def test_extra_files_and_directories_are_unauthorized(tmp_path):
    w, started = started_world(tmp_path / "roster")
    p = paths(w, started)
    (p["rosters"][0].parent / ("c" * 64 + ".json")).write_bytes(b"{}")
    assert any("not in the binding" in x for x in verdict_of(w).problems)
    w, started = started_world(tmp_path / "citation")
    p = paths(w, started)
    (p["citations"] / "000002.json").write_bytes(b"{}")
    assert any("000002.json" in x and "not authorized" in x for x in verdict_of(w).problems)
    w, started = started_world(tmp_path / "run")
    (w.root / "runs" / ("b" * 32)).mkdir()
    (w.root / "runs" / ("b" * 32) / "run.json").write_bytes(b"{}")
    assert any("is not an authorized run" in x for x in verdict_of(w).problems)
    w, started = started_world(tmp_path / "stray")
    (w.root / "runs" / started.run_id / "x").write_bytes(
        b""
    )  # inside a run: not this verifier's job
    assert verdict_of(w).problems == (NOTE,)
    w, started = started_world(tmp_path / "lock")
    (w.root / layout.SESSION_LOCK_FILE).write_bytes(b"content")
    assert any("lock file has content" in x for x in verdict_of(w).problems)


def test_a_temporary_file_in_the_new_directories_is_an_unfinished_publication(tmp_path):
    for where in ("inputs", "citations"):
        w, started = started_world(tmp_path / where)
        (paths(w, started)[where] / ".x.json.tmp").write_bytes(b"x")
        assert any("unfinished publication" in p for p in verdict_of(w).problems)


def events_of(w):
    return [json.loads(p.read_bytes()) for p in sorted((w.root / "registry").glob("0*"))]


def test_a_repeated_authorization_or_inputs_event_is_a_conflict(tmp_path):
    for index, event_type in ((3, "batch_inputs_bound"), (4, "batch_authorized")):
        w, _ = started_world(tmp_path / event_type)
        again = [EvidenceCitation(**c) for c in events_of(w)[index]["evidence"]]
        raw_event(w, event_type, again)
        verdict = verdict_of(w)
        assert verdict.code == CONFLICT and any("follows batch" in p for p in verdict.problems)


def test_an_authorization_must_cite_the_inputs_event_that_precedes_it(tmp_path):
    _, started = started_world(tmp_path)
    batch = started.batch_id
    w2 = make_world(tmp_path / "second")
    b2 = registry.reserve_batch(w2.root, w2.auth, w2.anchors, w2.witnesses, w2.request.plan, ARM)
    registry.materialize_batch(w2.root, w2.auth, w2.anchors, w2.witnesses, b2)
    materialized = registry.registry_head(w2.root)[1]
    raw_event(w2, "batch_inputs_bound",
              cited("batch_inputs_bound", batch_id=b2, materialization_event=sha("other")))  # fmt: skip
    assert any("another materialization" in p for p in verdict_of(w2).problems)
    assert batch and materialized


def test_a_rehearsal_event_with_the_wrong_roles_is_a_conflict(tmp_path):
    w = make_world(tmp_path)
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    registry.materialize_batch(w.root, w.auth, w.anchors, w.witnesses, batch)
    raw_event(w, "batch_inputs_bound", [EvidenceCitation(role="batch_id", sha256=batch)])
    assert any("does not cite" in p for p in verdict_of(w).problems)


def test_inputs_bound_for_a_batch_that_was_never_materialized_is_a_conflict(tmp_path):
    w = make_world(tmp_path)
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    values = {"batch_id": batch}
    raw_event(w, "batch_inputs_bound", cited("batch_inputs_bound", **values))
    verdict = verdict_of(w)
    assert any("follows batch" in p for p in verdict.problems)  # the unfinished batch ranks first


def test_an_inputs_event_for_an_unreserved_batch_or_without_a_binding(tmp_path):
    w = make_world(tmp_path / "a")
    registry._ensure_genesis_witness(w.root, w.witnesses, layout.root_id_for(w.auth), w.clock)
    raw_event(w, "batch_inputs_bound", cited("batch_inputs_bound"))
    assert any("unreserved batch" in p for p in verdict_of(w).problems)

    w = make_world(tmp_path / "b")
    batch = registry.reserve_batch(w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    registry.materialize_batch(w.root, w.auth, w.anchors, w.witnesses, batch)
    materialized = registry.registry_head(w.root)[1]
    raw_event(
        w,
        "batch_inputs_bound",
        cited("batch_inputs_bound", batch_id=batch, materialization_event=materialized),
    )
    verdict = verdict_of(w)
    assert verdict.code == INCOMPLETE and any("no binding file" in p for p in verdict.problems)


def test_the_publishers_enforce_event_types_roles_and_file_names(tmp_path):
    w = make_world(tmp_path)
    with inventory.registry_lock(w.root) as held:
        with pytest.raises(ProtocolRefused, match="not a rehearsal event"):
            registry.append_event_locked(held, w.root, w.auth, w.witnesses, "batch_resumed", [])
        with pytest.raises(ProtocolRefused, match="must cite exactly"):
            registry.append_event_locked(
                held, w.root, w.auth, w.witnesses, "batch_authorized",
                [EvidenceCitation(role="batch_id", sha256=sha("b"))],
            )  # fmt: skip
        for bad in ("batches/x/inputs/binding.json", f"batches/{'a' * 64}/manifest.json", "runs/x"):
            with pytest.raises(ProtocolRefused, match="not a batch file"):
                registry.publish_batch_file(held, w.root, w.auth, bad, b"x")
    assert not (w.root / "batches").exists()


# ---- the lock split keeps the old interfaces ------------------------------------------------


def test_the_locked_variants_need_the_lock_of_their_own_root(tmp_path):
    w = make_world(tmp_path)
    other = inventory.HeldLock(tmp_path / "elsewhere")
    with pytest.raises(ProtocolRefused, match="lock held is not the lock"):
        registry.reserve_locked(other, w.root, w.auth, w.anchors, w.witnesses, w.request.plan, ARM)
    with pytest.raises(ProtocolRefused, match="lock held is not the lock"):
        registry.materialize_locked(other, w.root, w.auth, w.anchors, w.witnesses, "a" * 64)
    assert not (w.root / "batches").exists()


def test_a_prepared_manifest_must_be_the_manifest_of_the_plan_and_arm(tmp_path):
    from biorag_repro.finaltest.batches import make_manifest

    w = make_world(tmp_path)
    root_id = layout.root_id_for(w.auth)
    good = make_manifest(root_id, w.request.plan, ARM, "2026-11-08T00:00:00Z")
    forged = good.model_copy(update={"expected_count": good.expected_count + 1})
    with inventory.registry_lock(w.root) as held:
        with pytest.raises(ProtocolRefused, match="prepared manifest"):
            registry.reserve_locked(held, w.root, w.auth, w.anchors, w.witnesses,
                                    w.request.plan, ARM, manifest=forged)  # fmt: skip
        assert not (w.root / "batches").exists()
        batch = registry.reserve_locked(held, w.root, w.auth, w.anchors, w.witnesses,
                                        w.request.plan, ARM, manifest=good)  # fmt: skip
    assert (w.root / "batches" / batch / "manifest.json").read_bytes() == layout.serialize(good)


def test_the_wrapper_and_the_locked_function_publish_the_same_bytes(tmp_path):
    fixed = lambda: "2026-11-08T00:00:00Z"
    a, b = make_world(tmp_path / "a"), make_world(tmp_path / "b")
    batch_a = registry.reserve_batch(
        a.root, a.auth, a.anchors, a.witnesses, a.request.plan, ARM, now=fixed
    )
    with inventory.registry_lock(b.root) as held:
        batch_b = registry.reserve_locked(held, b.root, b.auth, b.anchors, b.witnesses,
                                          b.request.plan, ARM, now=fixed)  # fmt: skip

    def event(w):
        data = json.loads((w.root / "registry/000002.json").read_bytes())
        data.pop("root_id"), data.pop("identity_sha256"), data.pop("previous_event_sha256")
        data["evidence"] = [c["role"] for c in data["evidence"]]
        return data

    assert event(a) == event(b)
    manifest_a = json.loads((a.root / "batches" / batch_a / "manifest.json").read_bytes())
    manifest_b = json.loads((b.root / "batches" / batch_b / "manifest.json").read_bytes())
    for key in ("root_id", "batch_id"):
        manifest_a.pop(key), manifest_b.pop(key)
    assert manifest_a == manifest_b
