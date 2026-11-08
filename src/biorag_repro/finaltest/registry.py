"""Structural batch protocol for rehearsal roots: reserve, materialize, close, witness.

Nothing here authorizes or runs anything. A reservation records that a matrix arm's observation
keys are claimed; the manifest records what the batch expects; neither is permission. Every
operation refuses a `final_test` root: publishing into a real root is a separately reviewed act.

Order of one reservation, all under the registry lock, after the whole root has been verified:

    1. append `batch_reserved`, citing the digest of the manifest bytes decided in advance
    2. publish `batches/<id>/manifest.json`
    3. append `batch_materialized`

The registry is written first so that every interruption leaves something the verifier
recognizes: a reservation without a manifest, or a manifest the reservation already names.
Files that exist without a reservation are unauthorized artifacts, never adopted. Each event
is followed by a witness; a missing head witness is its own recoverable state.

Recovery is explicit and never repairs: `close_incomplete` appends an immutable closure for a
reservation whose only problem is that it was not finished, and keeps the evidence and the key
claim. Closure is not permission to reuse the arm: one reservation per arm per campaign is the
rule here, and replacing a batch that may already hold observations needs a supersession
protocol that does not exist. Anything else that is incomplete stays blocked.
"""

import os
import re
from collections.abc import Callable
from pathlib import Path

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_bytes
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.batches import (
    BatchManifest,
    batch_id_for,
    make_manifest,
    observed_state_sha256,
    ownership_sha256,
)
from biorag_repro.finaltest.campaign import CampaignPlan, plan_sha256
from biorag_repro.finaltest.inventory import (
    HEAD_NOT_ESTABLISHED_TEXT,
    HeldLock,
    RootVerdict,
    registry_lock,
    verify_root,
)
from biorag_repro.finaltest.layout import (
    EvidenceCitation,
    HeadWitness,
    StorageAuthorization,
)
from biorag_repro.generation.environment import utc_now
from biorag_repro.readiness.files import read_regular


class ProtocolRefused(Exception):
    """The operation was not performed; the root is unchanged by it."""


def _publish(target: Path, data: bytes) -> None:
    publish_bytes(target, data, sha256_bytes(data))
    target.chmod(0o444)


def _require_rehearsal(authorization: StorageAuthorization) -> None:
    if authorization.root_kind != "rehearsal":
        raise ProtocolRefused("batch events are published only into rehearsal roots in this pass")


def _clean(verdict: RootVerdict, allowed: set[str] | None = None) -> None:
    """Every problem must be accounted for; the unestablished head is the known limit."""
    left = [p for p in verdict.problems if p != HEAD_NOT_ESTABLISHED_TEXT]
    if allowed is not None:
        left = [p for p in left if p not in allowed]
    if left:
        raise ProtocolRefused(f"the root does not verify cleanly ({verdict.code}): {left[:3]}")


def _registry_files(root: Path) -> dict[int, bytes]:
    directory = root / layout.REGISTRY_DIR
    return {
        int(name[:6]): read_regular(root, f"{layout.REGISTRY_DIR}/{name}")
        for name in sorted(os.listdir(directory))
        if layout.EVENT_NAME.match(name)
    }


def registry_head(root: Path) -> tuple[int, str]:
    """Sequence and file hash of the newest registry event, read without following links."""
    files = _registry_files(root)
    number = max(files)
    return number, sha256_bytes(files[number])


def _witness_path(witness_dir: Path, root_id: str, number: int) -> Path:
    return witness_dir / f"{root_id}.{number:06d}{layout.WITNESS_SUFFIX}"


def _publish_witness(
    root: Path, witness_dir: Path, root_id: str, number: int, now: Callable[[], str]
) -> None:
    event_bytes = _registry_files(root)[number]
    event = layout.RegistryEvent.model_validate_json(event_bytes)
    previous = None
    if number > 1:
        previous = sha256_bytes(
            read_regular(witness_dir, _witness_path(witness_dir, root_id, number - 1).name)
        )
    witness = HeadWitness(
        layout_version=layout.LAYOUT_VERSION,
        root_id=root_id,
        sequence=number,
        event_file_sha256=sha256_bytes(event_bytes),
        event_identity_sha256=event.identity_sha256,
        previous_witness_sha256=previous,
        created_utc=now(),
    )
    witness_dir.mkdir(parents=True, exist_ok=True)
    _publish(_witness_path(witness_dir, root_id, number), layout.serialize(witness))


def _append(
    root: Path,
    witness_dir: Path,
    root_id: str,
    event_type: str,
    evidence: list[EvidenceCitation],
    now: Callable[[], str],
) -> int:
    files = _registry_files(root)
    number = max(files) + 1
    event = layout.make_event(
        root_id=root_id,
        sequence=number,
        event_type=event_type,
        previous_event_sha256=sha256_bytes(files[number - 1]),
        evidence=evidence,
        created_utc=now(),
    )
    target = root / layout.REGISTRY_DIR / f"{number:06d}.json"
    try:
        _publish(target, layout.serialize(event))
    except FileExistsError:
        raise ProtocolRefused(
            f"registry event {number} already exists (competing writer)"
        ) from None
    _publish_witness(root, witness_dir, root_id, number, now)
    return number


def _publish_manifest(root: Path, batch_id: str, data: bytes) -> None:
    batches = root / layout.BATCHES_DIR
    batches.mkdir(exist_ok=True)
    (batches / batch_id).mkdir()
    _publish(batches / batch_id / layout.MANIFEST_NAME, data)


def _verified(
    root: Path, authorization: StorageAuthorization, anchors: Path, witnesses: Path
) -> RootVerdict:
    return verify_root(root, authorization, anchors, witnesses)


def _ensure_genesis_witness(
    root: Path, witnesses: Path, root_id: str, now: Callable[[], str]
) -> None:
    if not _witness_path(witnesses, root_id, 1).exists():
        _publish_witness(root, witnesses, root_id, 1, now)


def reserve_batch(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    plan: CampaignPlan,
    matrix_arm: str,
    *,
    now: Callable[[], str] = utc_now,
) -> str:
    """Reserve one matrix arm and publish its manifest. Returns the batch id."""
    _require_rehearsal(authorization)
    identity = _reservation_identity(authorization, plan, matrix_arm)  # before taking the lock
    with registry_lock(root) as held:
        return reserve_locked(
            held, root, authorization, anchors, witnesses, plan, matrix_arm,
            now=now, identity=identity,
        )  # fmt: skip


def _reservation_identity(
    authorization: StorageAuthorization, plan: CampaignPlan, matrix_arm: str
) -> tuple[str, str, str]:
    """Root id, plan hash and batch id; refuses a plan that is not for this root and arm."""
    if (plan.benchmark_release_sha256, plan.corpus_content_sha256) != (
        authorization.benchmark_release_sha256,
        authorization.corpus_content_sha256,
    ):
        raise ProtocolRefused("the plan is for another benchmark or corpus than the root")
    if matrix_arm not in {b.matrix_arm for b in plan.batches}:
        raise ProtocolRefused(f"the plan has no arm {matrix_arm}")
    root_id = layout.root_id_for(authorization)
    plan_sha = plan_sha256(plan)
    return root_id, plan_sha, batch_id_for(root_id, plan_sha, matrix_arm)


def reserve_locked(
    held: HeldLock,
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    plan: CampaignPlan,
    matrix_arm: str,
    *,
    now: Callable[[], str] = utc_now,
    manifest: BatchManifest | None = None,
    identity: tuple[str, str, str] | None = None,
) -> str:
    """`reserve_batch` for a caller that already holds the registry lock.

    A `manifest` decided in advance (so that inputs can be bound to its digest before anything
    is written) must be exactly the manifest this plan and arm produce.
    """
    _require_rehearsal(authorization)
    _require_held(held, root)
    root_id, plan_sha, batch_id = identity or _reservation_identity(authorization, plan, matrix_arm)
    verdict = _verified(root, authorization, anchors, witnesses)
    _clean(verdict)
    if verdict.registry_head_sequence == 1:
        _ensure_genesis_witness(root, witnesses, root_id, now)
    if batch_id in verdict.batches:
        raise ProtocolRefused("this arm is already reserved; one reservation per arm")
    if manifest is None:
        manifest = make_manifest(root_id, plan, matrix_arm, now())
    elif manifest != make_manifest(root_id, plan, matrix_arm, manifest.created_utc):
        raise ProtocolRefused("the prepared manifest is not the manifest of this plan and arm")
    claim = ownership_sha256(
        authorization.benchmark_release_sha256, matrix_arm, manifest.resolved_config_sha256
    )
    for batch in verdict.batches.values():
        if batch.ownership_sha256 == claim:
            raise ProtocolRefused("the observation keys of this arm are already claimed")
        if batch.campaign_plan_sha256 != plan_sha:
            raise ProtocolRefused("the root already holds batches of another campaign plan")
    data = layout.serialize(manifest)
    _append(
        root,
        witnesses,
        root_id,
        "batch_reserved",
        [
            EvidenceCitation(role="campaign_plan", sha256=plan_sha),
            EvidenceCitation(role="batch_id", sha256=batch_id),
            EvidenceCitation(role="ownership", sha256=claim),
            EvidenceCitation(role="batch_manifest", sha256=sha256_bytes(data)),
        ],
        now,
    )
    _publish_manifest(root, batch_id, data)
    return batch_id


def _require_held(held: HeldLock, root: Path) -> None:
    if held.root != root:
        raise ProtocolRefused("the registry lock held is not the lock of this root")


_BATCH_FILE = re.compile(
    r"^batches/[0-9a-f]{64}/(inputs/binding\.json|inputs/rosters/[0-9a-f]{64}\.json|citations/000001\.json)$"
)


def append_event_locked(
    held: HeldLock,
    root: Path,
    authorization: StorageAuthorization,
    witnesses: Path,
    event_type: str,
    evidence: list[EvidenceCitation],
    *,
    now: Callable[[], str] = utc_now,
) -> int:
    """Append `batch_inputs_bound` or `batch_authorized` to a rehearsal root, then witness it.

    Only the two rehearsal event types, only with their exact evidence roles, only into a
    rehearsal root and only under the registry lock. The caller has verified the root.
    """
    _require_rehearsal(authorization)
    _require_held(held, root)
    if event_type not in layout.REHEARSAL_ROLES:
        raise ProtocolRefused(f"{event_type} is not a rehearsal event")
    if [c.role for c in evidence] != layout.REHEARSAL_ROLES[event_type]:
        raise ProtocolRefused(
            f"{event_type} must cite exactly {layout.REHEARSAL_ROLES[event_type]}"
        )
    return _append(root, witnesses, layout.root_id_for(authorization), event_type, evidence, now)


def publish_batch_file(
    held: HeldLock, root: Path, authorization: StorageAuthorization, relative: str, data: bytes
) -> None:
    """Publish one input, roster or citation file of a rehearsal batch, read-only, no clobber."""
    _require_rehearsal(authorization)
    _require_held(held, root)
    if _BATCH_FILE.fullmatch(relative) is None:
        raise ProtocolRefused(f"{relative} is not a batch file this protocol publishes")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    _publish(target, data)


def _reservation_sha(root: Path, batch_id: str, verdict: RootVerdict) -> str:
    for data in _registry_files(root).values():
        event = layout.RegistryEvent.model_validate_json(data)
        cited = {c.role: c.sha256 for c in event.evidence}
        if event.event_type == "batch_reserved" and cited["batch_id"] == batch_id:
            return sha256_bytes(data)
    raise ProtocolRefused("no reservation found")  # pragma: no cover - verified beforehand


def materialize_batch(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    batch_id: str,
    *,
    now: Callable[[], str] = utc_now,
) -> None:
    """Record that a reserved batch's manifest is published. This is not authorization."""
    _require_rehearsal(authorization)
    with registry_lock(root) as held:
        materialize_locked(held, root, authorization, anchors, witnesses, batch_id, now=now)


def materialize_locked(
    held: HeldLock,
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    batch_id: str,
    *,
    now: Callable[[], str] = utc_now,
) -> None:
    _require_rehearsal(authorization)
    _require_held(held, root)
    root_id = layout.root_id_for(authorization)
    verdict = _verified(root, authorization, anchors, witnesses)
    found = verdict.reconcile.get(batch_id)
    if found is None or found[0] != "reserved_published":
        raise ProtocolRefused("the batch is not reserved with a published manifest")
    _clean(verdict, {found[1]})
    _append(
        root,
        witnesses,
        root_id,
        "batch_materialized",
        [
            EvidenceCitation(role="batch_id", sha256=batch_id),
            EvidenceCitation(
                role="batch_manifest", sha256=verdict.batches[batch_id].manifest_sha256
            ),
            EvidenceCitation(
                role="reservation_event", sha256=_reservation_sha(root, batch_id, verdict)
            ),
        ],
        now,
    )


def close_incomplete(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    batch_id: str,
    *,
    now: Callable[[], str] = utc_now,
) -> None:
    """Close a reservation that was never materialized, preserving all of its evidence.

    Refused unless the whole root verifies and the only open problem is that this one
    reservation is unfinished (manifest absent, or present and matching its reservation).
    The head must already be witnessed. The batch's key claim stays in force.
    """
    _require_rehearsal(authorization)
    root_id = layout.root_id_for(authorization)
    with registry_lock(root):
        verdict = _verified(root, authorization, anchors, witnesses)
        found = verdict.reconcile.get(batch_id)
        if found is None:
            raise ProtocolRefused("the batch is not an unfinished reservation")
        _clean(verdict, {found[1]})
        manifest_sha = None
        if found[0] == "reserved_published":
            manifest_sha = verdict.batches[batch_id].manifest_sha256
        _append(
            root,
            witnesses,
            root_id,
            "batch_closed_incomplete",
            [
                EvidenceCitation(role="batch_id", sha256=batch_id),
                EvidenceCitation(
                    role="reservation_event", sha256=_reservation_sha(root, batch_id, verdict)
                ),
                EvidenceCitation(
                    role="observed_state", sha256=observed_state_sha256(manifest_sha, False)
                ),
            ],
            now,
        )


def witness_head(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    *,
    now: Callable[[], str] = utc_now,
) -> int:
    """Publish the missing witness of the registry head, and only that.

    Allowed only when the root otherwise verifies and the head is the single unwitnessed
    event. It appends nothing to the registry.
    """
    _require_rehearsal(authorization)
    root_id = layout.root_id_for(authorization)
    with registry_lock(root):
        verdict = _verified(root, authorization, anchors, witnesses)
        head = verdict.registry_head_sequence
        if not verdict.unwitnessed_head or head is None:
            raise ProtocolRefused("the registry head is not the single unwitnessed event")
        # unfinished reservations stay unfinished: this only adds the missing witness
        accounted = {f"incomplete: registry event {head} has no witness"}
        accounted |= {text for _, text in verdict.reconcile.values()}
        _clean(verdict, accounted)
        _publish_witness(root, witnesses, root_id, head, now)
    return head
