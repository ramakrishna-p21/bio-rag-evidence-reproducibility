"""Classification contract for final-test history (gate G13).

The classes are the ones fixed in `models.HISTORY_PASS_CODES` and `HISTORY_FAIL_CODES`. This
function only says which class an inventory falls into. It cannot make an inventory
authoritative: a `HistoryInventory` built by a caller, or a digest handed in beside it, is
not a trust root. The only production caller is `finaltest.inventory.verify_root`, which
builds the inventory from files it read itself, after checking the root against the approved
storage authorization (G11). The readiness adapters never call this module directly.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HistoryEntry:
    batch_id: str
    kind: Literal["observation", "attempt", "registry", "unexpected"]
    verified: bool


@dataclass(frozen=True)
class HistoryInventory:
    complete: bool  # coverage of every required location, as established by the verifier
    entries: tuple[HistoryEntry, ...]
    authorized_batch: str | None
    approved_batches: frozenset[str]  # other batches whose history is accepted
    contradictions: bool  # entries contradict each other or the authorized batch


INCOMPLETE = "history_inventory_incomplete"


def classify_history(inventory: HistoryInventory) -> str:
    """A history reason code, or `history_inventory_incomplete` when nothing can be said."""
    if not inventory.complete:
        return INCOMPLETE
    if inventory.contradictions:
        return "conflicting_final_test_artifact"
    allowed = set(inventory.approved_batches)
    if inventory.authorized_batch is not None:
        allowed.add(inventory.authorized_batch)
    if any(entry.batch_id not in allowed for entry in inventory.entries):
        return "unauthorized_final_test_artifact"
    if any(not entry.verified for entry in inventory.entries):
        return "unverified_final_test_artifact"
    if not inventory.entries:
        return "no_prior_final_test_execution"
    if all(entry.batch_id == inventory.authorized_batch for entry in inventory.entries):
        return "same_batch_valid_history"
    return "other_approved_batch_valid_history"
