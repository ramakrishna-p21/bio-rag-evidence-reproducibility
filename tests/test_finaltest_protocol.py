"""Batch reservation, witnesses, recovery and rollback on synthetic rehearsal roots."""

import hashlib
import multiprocessing
import os
import shutil
import stat
from pathlib import Path

import pytest
from finaltest_campaign_support import CONFIGS, SEEDS, population, sha
from finaltest_support import authorization, made, oracle_chain_ok

from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest import inventory, layout, registry
from biorag_repro.finaltest.batches import (
    batch_id_for,
    observed_state_sha256,
    ownership_sha256,
)
from biorag_repro.finaltest.campaign import ARMS, build_plan, plan_sha256
from biorag_repro.finaltest.inventory import RegistryBusy, verify_root
from biorag_repro.finaltest.layout import EvidenceCitation, HeadCheckpoint
from biorag_repro.finaltest.registry import (
    ProtocolRefused,
    close_incomplete,
    materialize_batch,
    reserve_batch,
    witness_head,
)

NO_PRIOR = "no_prior_final_test_execution"
INCOMPLETE = "history_inventory_incomplete"
UNAUTH = "unauthorized_final_test_artifact"
CONFLICT = "conflicting_final_test_artifact"
UNVERIFIED = "unverified_final_test_artifact"
NOTE = inventory.HEAD_NOT_ESTABLISHED_TEXT


class Env:
    def __init__(self, base: Path):
        self.root, self.anchors, self.auth = made(base)
        self.witnesses = base / "witnesses"
        self.plan = build_plan(population(), CONFIGS, SEEDS)
        self.root_id = layout.root_id_for(self.auth)

    def verify(self, checkpoint=None):
        return verify_root(self.root, self.auth, self.anchors, self.witnesses, checkpoint)

    def reserve(self, arm="closed_book", plan=None):
        return reserve_batch(
            self.root, self.auth, self.anchors, self.witnesses, plan or self.plan, arm
        )

    def reserve_full(self, arm):
        batch = self.reserve(arm)
        self.materialize(batch)
        return batch

    def materialize(self, batch):
        materialize_batch(self.root, self.auth, self.anchors, self.witnesses, batch)

    def close(self, batch):
        close_incomplete(self.root, self.auth, self.anchors, self.witnesses, batch)

    def batch_id(self, arm="closed_book", plan=None):
        return batch_id_for(self.root_id, plan_sha256(plan or self.plan), arm)

    def event_files(self):
        return sorted((self.root / "registry").glob("0*.json"))

    def raw_event(self, event_type, evidence):
        registry._ensure_genesis_witness(
            self.root, self.witnesses, self.root_id, lambda: "2026-01-02T00:00:00Z"
        )
        return registry._append(
            self.root,
            self.witnesses,
            self.root_id,
            event_type,
            evidence,
            lambda: "2026-01-02T00:00:00Z",
        )


@pytest.fixture
def env(tmp_path):
    return Env(tmp_path)


class Crash(Exception):
    pass


def boom(*args, **kwargs):
    raise Crash


# ---- the happy path and what it does not establish -------------------------------------


def test_reserve_and_materialize_publish_the_expected_records(env):
    batch = env.reserve()
    manifest = env.root / "batches" / batch / "manifest.json"
    env.materialize(batch)
    files = env.event_files()
    assert [f.name for f in files] == ["000001.json", "000002.json", "000003.json"]
    assert oracle_chain_ok(env.root / "registry")
    assert stat.S_IMODE(manifest.stat().st_mode) == 0o444
    reserved = layout.RegistryEvent.model_validate_json(files[1].read_bytes())
    cited = {c.role: c.sha256 for c in reserved.evidence}
    assert reserved.event_type == "batch_reserved"
    assert cited["batch_manifest"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert cited["batch_id"] == batch == env.batch_id()
    assert cited["campaign_plan"] == plan_sha256(env.plan)
    assert cited["ownership"] == ownership_sha256(
        env.auth.benchmark_release_sha256, "closed_book", CONFIGS["closed_book"]
    )
    verdict = env.verify()
    assert verdict.batches[batch].state == "materialized"
    assert sorted(p.name for p in env.witnesses.iterdir()) == [
        f"{env.root_id}.00000{n}.witness.json" for n in (1, 2, 3)
    ]


def test_a_structural_history_is_never_a_positive_class_or_an_authorization(env):
    batch = env.reserve()
    env.materialize(batch)
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert verdict.head_authority == "not_established"
    assert verdict.problems == (NOTE,)
    assert not any(e.event_type.startswith("batch_auth") for e in _events(env))


def _events(env):
    return [layout.RegistryEvent.model_validate_json(f.read_bytes()) for f in env.event_files()]


def test_a_genesis_only_root_stays_positive_with_or_without_a_witness_store(env):
    assert env.verify().code == NO_PRIOR
    assert verify_root(env.root, env.auth, env.anchors).code == NO_PRIOR


def test_all_five_arms_can_be_reserved_with_distinct_ids_and_claims(env):
    ids = {env.reserve_full(arm) for arm in ARMS}
    assert len(ids) == 5
    verdict = env.verify()
    assert len({b.ownership_sha256 for b in verdict.batches.values()}) == 5
    assert {b.state for b in verdict.batches.values()} == {"materialized"}


def test_a_second_reservation_waits_for_the_unfinished_one(env):
    env.reserve("closed_book")
    with pytest.raises(ProtocolRefused, match="does not verify cleanly"):
        env.reserve("confirmatory")


# ---- refusals ----------------------------------------------------------------------------


def test_a_second_reservation_of_an_arm_is_refused(env):
    env.reserve_full("closed_book")
    with pytest.raises(ProtocolRefused, match="one reservation per arm"):
        env.reserve()


def test_a_closed_arm_cannot_be_reserved_again(env):
    batch = env.reserve()
    env.close(batch)
    with pytest.raises(ProtocolRefused, match="one reservation per arm"):
        env.reserve()


def test_partial_completion_followed_by_replacement_is_blocked_by_the_key_claim(env):
    """The old batch may hold observations; closure must not release its keys."""
    old = env.reserve()
    env.close(old)
    other_plan = build_plan(population(), CONFIGS, sha("another seed registry"))
    assert plan_sha256(other_plan) != plan_sha256(env.plan)
    with pytest.raises(ProtocolRefused, match="claimed"):
        env.reserve("closed_book", other_plan)
    assert env.verify().batches[old].state == "closed_incomplete"


def test_a_different_campaign_plan_cannot_be_added_for_another_arm(env):
    env.reserve_full("closed_book")
    other_plan = build_plan(population(), CONFIGS, sha("another seed registry"))
    with pytest.raises(ProtocolRefused, match="another campaign plan"):
        env.reserve("confirmatory", other_plan)


def test_a_plan_for_another_benchmark_is_refused(env):
    other = build_plan(population(benchmark_release_sha256=sha("x")), CONFIGS, SEEDS)
    with pytest.raises(ProtocolRefused, match="another benchmark"):
        env.reserve("closed_book", other)
    with pytest.raises(ProtocolRefused, match="no arm"):
        env.reserve("nonexistent")


def test_nothing_is_published_into_a_final_test_root(tmp_path):
    auth = authorization(tmp_path, "final_test")
    root, anchors = Path(auth.authorized_root), tmp_path / "anchors"
    inventory._initialize(root, auth, anchors, lambda: "2026-01-01T00:00:00Z")
    witnesses = tmp_path / "w"
    plan = build_plan(population(), CONFIGS, SEEDS)
    before = sorted(p.name for p in root.rglob("*"))
    for call in (
        lambda: reserve_batch(root, auth, anchors, witnesses, plan, "closed_book"),
        lambda: materialize_batch(root, auth, anchors, witnesses, "a" * 64),
        lambda: close_incomplete(root, auth, anchors, witnesses, "a" * 64),
        lambda: witness_head(root, auth, anchors, witnesses),
    ):
        with pytest.raises(ProtocolRefused, match="rehearsal"):
            call()
    assert sorted(p.name for p in root.rglob("*")) == before
    assert not witnesses.exists()


# ---- crashes at every publication boundary ------------------------------------------------


def test_crash_before_reservation_leaves_nothing(env, monkeypatch):
    monkeypatch.setattr(registry, "_append", boom)
    with pytest.raises(Crash):
        env.reserve()
    assert env.verify().code == NO_PRIOR
    assert [f.name for f in env.event_files()] == ["000001.json"]


def test_crash_after_the_event_before_its_witness_is_a_recoverable_state(env, monkeypatch):
    real = registry._publish_witness
    calls = []

    def crash_on_second(*args):
        calls.append(args)
        if len(calls) == 2:  # the genesis witness is first; the event's witness is second
            raise Crash
        real(*args)

    monkeypatch.setattr(registry, "_publish_witness", crash_on_second)
    with pytest.raises(Crash):
        env.reserve()
    monkeypatch.undo()
    verdict = env.verify()
    assert verdict.unwitnessed_head
    assert verdict.code == INCOMPLETE
    # nothing may be appended to a head whose witness is missing
    batch = env.batch_id()
    with pytest.raises(ProtocolRefused):
        env.close(batch)
    with pytest.raises(ProtocolRefused):
        env.reserve("confirmatory")
    assert len(env.event_files()) == 2
    assert witness_head(env.root, env.auth, env.anchors, env.witnesses) == 2
    assert not env.verify().unwitnessed_head
    env.close(batch)  # reservation without manifest, now reconcilable
    assert env.verify().batches[batch].state == "closed_incomplete"


def test_crash_after_reservation_before_the_manifest_then_explicit_closure(env, monkeypatch):
    monkeypatch.setattr(registry, "_publish_manifest", boom)
    with pytest.raises(Crash):
        env.reserve()
    monkeypatch.undo()
    batch = env.batch_id()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert verdict.reconcile[batch][0] == "reserved_unpublished"
    with pytest.raises(ProtocolRefused):
        env.materialize(batch)
    env.close(batch)
    closed = env.verify()
    assert closed.batches[batch].state == "closed_incomplete"
    assert closed.problems == (NOTE,)
    assert not (env.root / "batches" / batch).exists()  # nothing was invented to cover it
    with pytest.raises(ProtocolRefused, match="one reservation per arm"):
        env.reserve()


def test_crash_while_writing_the_manifest_blocks_and_preserves_the_evidence(env, monkeypatch):
    def partial(target, data, digest, **kwargs):
        (target.parent / f".{target.name}.partial.tmp").write_bytes(data[:10])
        raise Crash

    monkeypatch.setattr(registry, "publish_bytes", partial)
    # the manifest is the first publish_bytes call after the event; let the event through
    real = publish_bytes
    seen = []

    def selective(target, data, digest, **kwargs):
        seen.append(target.name)
        if target.name == "manifest.json":
            return partial(target, data, digest)
        return real(target, data, digest, **kwargs)

    monkeypatch.setattr(registry, "publish_bytes", selective)
    with pytest.raises(Crash):
        env.reserve()
    monkeypatch.undo()
    batch = env.batch_id()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert batch not in verdict.reconcile or verdict.reconcile[batch][0] != "reserved_published"
    assert any("unfinished publication" in p for p in verdict.problems)
    with pytest.raises(ProtocolRefused):
        env.close(batch)
    leftovers = list((env.root / "batches" / batch).iterdir())
    assert [p.name for p in leftovers] == [".manifest.json.partial.tmp"]


def test_crash_after_the_manifest_before_materialization(env, monkeypatch):
    batch = env.reserve()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert verdict.reconcile[batch][0] == "reserved_published"
    assert verdict.batches[batch].state == "reserved"  # never authorized or materialized
    env.materialize(batch)
    assert env.verify().batches[batch].state == "materialized"


def test_an_unfinished_reservation_with_a_manifest_can_be_closed_keeping_the_file(env):
    batch = env.reserve()
    manifest = env.root / "batches" / batch / "manifest.json"
    data = manifest.read_bytes()
    env.close(batch)
    assert manifest.read_bytes() == data
    closure = _events(env)[-1]
    cited = {c.role: c.sha256 for c in closure.evidence}
    assert closure.event_type == "batch_closed_incomplete"
    assert cited["observed_state"] == observed_state_sha256(hashlib.sha256(data).hexdigest(), False)
    assert env.verify().problems == (NOTE,)


def test_a_materialized_batch_is_not_closed_by_this_protocol(env):
    batch = env.reserve()
    env.materialize(batch)
    with pytest.raises(ProtocolRefused, match="not an unfinished"):
        env.close(batch)


def test_interrupted_registry_append_blocks_further_events(env):
    env.reserve()
    (env.root / "registry" / ".000003.json.abc.tmp").write_bytes(b"{")
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    with pytest.raises(ProtocolRefused):
        env.reserve("confirmatory")
    with pytest.raises(ProtocolRefused):
        env.close(env.batch_id())


def test_a_competing_writer_cannot_reuse_a_sequence_number(env, monkeypatch):
    env.reserve_full("closed_book")
    stale = registry._registry_files(env.root)
    monkeypatch.setattr(registry, "_registry_files", lambda root: {1: stale[1]})
    with pytest.raises(ProtocolRefused, match="competing writer"):
        registry._append(
            env.root,
            env.witnesses,
            env.root_id,
            "batch_reserved",
            _reserve_cited(env, sha("b")),
            lambda: "2026-01-02T00:00:00Z",
        )
    assert len(env.event_files()) == 3


def test_a_held_lock_refuses_a_second_writer(env):
    with inventory.registry_lock(env.root), pytest.raises(RegistryBusy):
        env.reserve()
    assert len(env.event_files()) == 1


def _reserve_in_process(args):
    base, arm = args
    env = Env.__new__(Env)
    env.root, env.auth = Path(base) / "final_test_v1", authorization(Path(base))
    env.anchors, env.witnesses = Path(base) / "anchors", Path(base) / "witnesses"
    env.plan = build_plan(population(), CONFIGS, SEEDS)
    batch = None
    for _ in range(2000):
        try:
            batch = batch or env.reserve(arm)
            env.materialize(batch)
            os._exit(0)
        except (RegistryBusy, ProtocolRefused):
            continue  # another writer holds the lock or has an unfinished reservation
    os._exit(3)


def test_competing_processes_end_with_one_consistent_registry(tmp_path):
    env = Env(tmp_path)
    ctx = multiprocessing.get_context("fork")
    workers = [
        ctx.Process(target=_reserve_in_process, args=((str(tmp_path), arm),)) for arm in ARMS
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(60)
    assert [w.exitcode for w in workers] == [0] * 5
    verdict = env.verify()
    assert sorted(verdict.batches) == sorted(env.batch_id(a) for a in ARMS)
    assert verdict.problems == (NOTE,)
    assert [f.name for f in env.event_files()] == [f"{n:06d}.json" for n in range(1, 12)]
    assert {b.state for b in verdict.batches.values()} == {"materialized"}
    assert oracle_chain_ok(env.root / "registry")


# ---- orphans, tampering, reserved and malformed events -----------------------------------


def test_a_manifest_without_a_reservation_is_an_unauthorized_artifact(env):
    batch = env.reserve_full("closed_book")
    other = env.batch_id("confirmatory")
    (env.root / "batches" / other).mkdir()
    shutil.copy(env.root / "batches" / batch / "manifest.json", env.root / "batches" / other)
    verdict = env.verify()
    assert verdict.code == UNAUTH
    assert any("unreserved batch" in p for p in verdict.problems)


def test_complete_looking_files_do_not_make_an_incomplete_batch_authorized(env):
    """All files exist, but the registry never materialized the batch."""
    batch = env.reserve()
    assert (env.root / "batches" / batch / "manifest.json").exists()
    verdict = env.verify()
    assert verdict.batches[batch].state == "reserved"
    assert verdict.code == INCOMPLETE


def test_a_modified_manifest_is_a_conflict(env):
    batch = env.reserve()
    env.materialize(batch)
    path = env.root / "batches" / batch / "manifest.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b'"expected_count": 15', b'"expected_count": 16'))
    verdict = env.verify()
    assert verdict.code == CONFLICT
    assert any("differs from its reservation" in p for p in verdict.problems)


def test_a_published_manifest_cannot_be_republished_over(env):
    batch = env.reserve()
    with pytest.raises(FileExistsError):
        registry._publish_manifest(env.root, batch, b"{}")


def test_a_missing_manifest_of_a_materialized_batch_is_incomplete(env):
    batch = env.reserve()
    env.materialize(batch)
    (env.root / "batches" / batch / "manifest.json").unlink()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert not verdict.reconcile
    with pytest.raises(ProtocolRefused):
        env.close(batch)


def test_future_event_types_are_unauthorized_artifacts(env):
    batch = env.reserve()
    env.materialize(batch)
    env.raw_event("batch_resumed", [EvidenceCitation(role="readiness_report", sha256=sha("r"))])
    verdict = env.verify()
    assert verdict.code == UNAUTH
    assert any("unauthorized type batch_resumed" in p for p in verdict.problems)


def _reserve_cited(env, batch, plan=None, claim=None, manifest=None):
    plan = plan or env.plan
    return [
        EvidenceCitation(role="campaign_plan", sha256=plan_sha256(plan)),
        EvidenceCitation(role="batch_id", sha256=batch),
        EvidenceCitation(role="ownership", sha256=claim or sha("claim")),
        EvidenceCitation(role="batch_manifest", sha256=manifest or sha("m")),
    ]


def test_a_batch_event_with_the_wrong_evidence_roles_is_invalid(env):
    with pytest.raises(ValueError, match="must cite exactly"):
        layout.make_event(
            root_id=env.root_id,
            sequence=2,
            event_type="batch_reserved",
            previous_event_sha256=sha("p"),
            evidence=[EvidenceCitation(role="batch_id", sha256=sha("b"))],
            created_utc="2026-01-02T00:00:00Z",
        )


def test_the_same_batch_reserved_twice_is_a_conflict(env):
    batch = env.reserve_full("closed_book")
    env.raw_event("batch_reserved", _reserve_cited(env, batch, claim=sha("other claim")))
    verdict = env.verify()
    assert verdict.code == CONFLICT
    assert any("is reserved twice" in p for p in verdict.problems)


def test_two_batches_claiming_the_same_keys_are_a_conflict(env):
    batch = env.reserve_full("closed_book")
    claim = env.verify().batches[batch].ownership_sha256
    env.raw_event("batch_reserved", _reserve_cited(env, sha("b2"), claim=claim))
    verdict = env.verify()
    assert any("claim the same keys" in p for p in verdict.problems)


def test_a_second_campaign_plan_in_the_registry_is_a_conflict(env):
    env.reserve_full("closed_book")
    other = build_plan(population(), CONFIGS, sha("other"))
    env.raw_event("batch_reserved", _reserve_cited(env, sha("b2"), plan=other))
    assert any("more than one campaign plan" in p for p in env.verify().problems)


def test_materialization_without_a_reservation_is_a_conflict(env):
    env.reserve_full("closed_book")
    env.raw_event(
        "batch_materialized",
        [
            EvidenceCitation(role="batch_id", sha256=sha("b")),
            EvidenceCitation(role="batch_manifest", sha256=sha("m")),
            EvidenceCitation(role="reservation_event", sha256=sha("r")),
        ],
    )
    assert env.verify().code == CONFLICT


def test_closing_with_an_impossible_observed_state_is_a_conflict(env):
    batch = env.reserve("closed_book")
    reservation = hashlib.sha256(env.event_files()[1].read_bytes()).hexdigest()
    env.raw_event(
        "batch_closed_incomplete",
        [
            EvidenceCitation(role="batch_id", sha256=batch),
            EvidenceCitation(role="reservation_event", sha256=reservation),
            EvidenceCitation(role="observed_state", sha256=observed_state_sha256(sha("zz"), True)),
        ],
    )
    assert any("impossible state" in p for p in env.verify().problems)


def test_events_after_closure_are_a_conflict(env):
    batch = env.reserve()
    env.close(batch)
    reservation = hashlib.sha256(env.event_files()[1].read_bytes()).hexdigest()
    manifest = env.verify().batches[batch].manifest_sha256
    env.raw_event(
        "batch_materialized",
        [
            EvidenceCitation(role="batch_id", sha256=batch),
            EvidenceCitation(role="batch_manifest", sha256=manifest),
            EvidenceCitation(role="reservation_event", sha256=reservation),
        ],
    )
    assert env.verify().code == CONFLICT


def test_the_genesis_event_must_be_first_and_unique(env):
    env.reserve_full("closed_book")
    files = env.event_files()
    assert layout.RegistryEvent.model_validate_json(files[0].read_bytes()).event_type == (
        "root_initialized"
    )
    env.raw_event("root_initialized", [EvidenceCitation(role="root_file", sha256=sha("r"))])
    assert env.verify().code == CONFLICT


def test_a_runs_directory_is_an_unauthorized_artifact(env):
    batch = env.reserve()
    env.materialize(batch)
    (env.root / "runs").mkdir()
    assert env.verify().code == UNAUTH


# ---- witnesses, rollback and checkpoints --------------------------------------------------


def _snapshot(env, where):
    shutil.copytree(env.root, where / "root")
    shutil.copytree(env.witnesses, where / "witnesses")


def _restore(env, where):
    shutil.rmtree(env.root)
    shutil.rmtree(env.witnesses)
    shutil.copytree(where / "root", env.root)
    shutil.copytree(where / "witnesses", env.witnesses)


def test_truncating_only_the_registry_is_detected_against_the_witnesses(env):
    env.reserve_full("closed_book")
    env.event_files()[-1].unlink()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE  # the shortened registry also leaves a batch unfinished
    assert any("beyond the registry head" in p for p in verdict.problems)


def test_a_diverged_registry_event_is_detected_against_its_witness(env):
    env.reserve_full("closed_book")
    files = env.event_files()
    other = layout.make_event(
        root_id=env.root_id,
        sequence=3,
        event_type="batch_reserved",
        previous_event_sha256=hashlib.sha256(files[1].read_bytes()).hexdigest(),
        evidence=_reserve_cited(env, sha("zz")),
        created_utc="2026-01-03T00:00:00Z",
    )
    files[2].chmod(0o644)
    files[2].write_bytes(layout.serialize(other))
    verdict = env.verify()
    assert any("witness 3 records a different event" in p for p in verdict.problems)


def test_a_missing_middle_witness_is_incomplete_not_recoverable_by_witness_head(env):
    env.reserve_full("closed_book")
    env.reserve_full("confirmatory")
    (env.witnesses / f"{env.root_id}.000002.witness.json").unlink()
    verdict = env.verify()
    assert verdict.code == INCOMPLETE
    assert not verdict.unwitnessed_head
    with pytest.raises(ProtocolRefused):
        witness_head(env.root, env.auth, env.anchors, env.witnesses)


def test_a_witness_that_does_not_chain_is_a_conflict(env):
    env.reserve_full("closed_book")
    env.reserve_full("confirmatory")
    path = env.witnesses / f"{env.root_id}.000003.witness.json"
    witness = layout.HeadWitness.model_validate_json(path.read_bytes())
    bad = witness.model_copy(update={"previous_witness_sha256": sha("nope")})
    path.chmod(0o644)
    path.write_bytes(layout.serialize(bad))
    assert any("breaks the witness chain" in p for p in env.verify().problems)


def test_a_witness_for_another_root_or_unparsable_is_not_believed(env):
    env.reserve_full("closed_book")
    path = env.witnesses / f"{env.root_id}.000003.witness.json"
    path.chmod(0o644)
    path.write_bytes(b"{}")
    assert env.verify().code == UNVERIFIED


def test_a_coordinated_rollback_of_registry_and_witnesses_is_not_detected_without_a_checkpoint(
    env, tmp_path
):
    """The documented limit: both stores sit under the same data root."""
    env.reserve_full("closed_book")
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    _snapshot(env, earlier)
    second = env.reserve_full("confirmatory")
    later_head = HeadCheckpoint(
        root_id=env.root_id,
        sequence=5,
        event_file_sha256=hashlib.sha256(env.event_files()[4].read_bytes()).hexdigest(),
    )
    _restore(env, earlier)
    # the chain is internally valid and consistent with its witnesses, so only the head
    # authority note remains and the history is still not a positive class
    verdict = env.verify()
    assert second not in verdict.batches
    assert verdict.checkpoint_continuity is None
    assert verdict.head_authority == "not_established"
    assert verdict.code != NO_PRIOR
    # a checkpoint retained elsewhere is what exposes the rollback
    detected = env.verify(later_head)
    assert detected.checkpoint_continuity is False
    assert detected.code == CONFLICT


def test_rolling_back_to_genesis_looks_like_a_fresh_root_without_a_checkpoint(env):
    batch = env.reserve()
    head = HeadCheckpoint(
        root_id=env.root_id,
        sequence=2,
        event_file_sha256=hashlib.sha256(env.event_files()[1].read_bytes()).hexdigest(),
    )
    env.event_files()[1].unlink()
    shutil.rmtree(env.root / "batches")
    shutil.rmtree(env.witnesses)
    assert env.verify().code == NO_PRIOR  # the limit: genesis is anchored, later heads are not
    assert env.verify(head).code == CONFLICT
    assert batch not in env.verify().batches


def test_a_matching_checkpoint_is_continuity_not_authority(env):
    batch = env.reserve()
    env.materialize(batch)
    head = HeadCheckpoint(
        root_id=env.root_id,
        sequence=3,
        event_file_sha256=hashlib.sha256(env.event_files()[2].read_bytes()).hexdigest(),
    )
    verdict = env.verify(head)
    assert verdict.checkpoint_continuity is True
    assert verdict.head_authority == "not_established"
    assert verdict.code == INCOMPLETE


def test_an_older_checkpoint_is_continuous_with_a_longer_registry(env):
    env.reserve_full("closed_book")
    old = HeadCheckpoint(
        root_id=env.root_id,
        sequence=2,
        event_file_sha256=hashlib.sha256(env.event_files()[1].read_bytes()).hexdigest(),
    )
    env.reserve_full("confirmatory")
    assert env.verify(old).checkpoint_continuity is True


def test_a_checkpoint_for_another_root_or_event_is_a_conflict(env):
    env.reserve_full("closed_book")
    foreign = HeadCheckpoint(root_id=sha("other"), sequence=2, event_file_sha256=sha("e"))
    assert env.verify(foreign).code == CONFLICT
    wrong = HeadCheckpoint(root_id=env.root_id, sequence=2, event_file_sha256=sha("e"))
    assert env.verify(wrong).code == CONFLICT


def test_an_empty_directory_of_an_unreserved_batch_is_an_unauthorized_artifact(env):
    env.reserve_full("closed_book")
    (env.root / "batches" / env.batch_id("confirmatory")).mkdir()
    verdict = env.verify()
    assert verdict.code == UNAUTH
    assert any("directory of the unreserved batch" in p for p in verdict.problems)


def test_the_first_registry_event_must_be_the_genesis_event(env):
    first = env.root / "registry" / "000001.json"
    wrong = layout.make_event(
        root_id=env.root_id,
        sequence=1,
        event_type="batch_reserved",
        previous_event_sha256=None,
        evidence=_reserve_cited(env, sha("b")),
        created_utc="2026-01-01T00:00:00Z",
    )
    first.chmod(0o644)
    first.write_bytes(layout.serialize(wrong))
    assert "conflicting: the first event is not the genesis event" in env.verify().problems


def test_witness_head_refuses_when_no_head_witness_is_missing(env):
    env.reserve_full("closed_book")
    with pytest.raises(ProtocolRefused, match="single unwitnessed"):
        witness_head(env.root, env.auth, env.anchors, env.witnesses)
    assert len(list(env.witnesses.iterdir())) == 3
