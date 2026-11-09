"""What a claimed latest registry head does and does not establish.

Five things are kept apart, because each answers a different question:

* a *claim* about the latest head (`HeadAttestation`): a record of a sequence number and the
  SHA-256 of one event file. Its JSON can be valid and its content identity exact, and it still
  says nothing about who wrote it or whether it is current;
* internal chain validity: every event parses, the hashes link, nothing is out of place;
* consistency of the registry with its witnesses: catches a rewrite of one store;
* continuity with the claim: the registry still holds the claimed event;
* independent authorization of the claim: this module never establishes it.

The registry, the witnesses and the batch manifests all live under one data root. Anyone who
can rewrite that root can rewind all three together, and the result is indistinguishable from
the honest registry as it was. Only a head retained where that person cannot reach it can
expose such a rollback, and nothing in this repository is such a place. So the public path
returns `authority = "not_established"` for every head past genesis, whatever the claim says.
Genesis alone is anchored, by the out-of-tree anchor checked in `inventory`.

`HeadAssessment.grants_authorization` is always False. Nothing here is consumed by execution
code; it exists so that a later pass has an exactly specified claim to compare against.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self

from pydantic import model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.inventory import RootVerdict, verify_root
from biorag_repro.finaltest.layout import (
    LAYOUT_VERSION,
    FrozenRecord,
    Sha256,
    StorageAuthorization,
)
from biorag_repro.readiness.models import HISTORY_FAIL_CODES

# Where a claimed head came from. No value names an authority: that would be a statement the
# record cannot back. `test_fixture` marks a claim built inside a test.
Provenance = Literal["unverified_claim", "test_fixture"]
Authority = Literal["genesis_anchored", "not_established", "fixture_only"]

_EVENT_PATH: Final = re.compile(r"^registry/(\d{6})\.json$")


class HeadAttestation(FrozenRecord):
    """A claim about one registry event. It is evidence of a claim and nothing more."""

    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"] = LAYOUT_VERSION
    root_id: Sha256
    sequence: int
    event_file_sha256: Sha256
    previous_sequence: int | None = None
    previous_event_file_sha256: Sha256 | None = None
    campaign_plan_sha256: Sha256 | None = None
    batch_id: Sha256 | None = None
    provenance: Provenance

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if self.sequence < 1:
            raise ValueError("registry sequence starts at 1")
        if (self.previous_sequence is None) != (self.previous_event_file_sha256 is None):
            raise ValueError("a predecessor needs both its sequence and its event hash")
        if self.previous_sequence is not None and not 1 <= self.previous_sequence < self.sequence:
            raise ValueError("the predecessor must be an earlier event")
        if self.batch_id is not None and self.campaign_plan_sha256 is None:
            raise ValueError("a batch scope names the campaign plan it belongs to")
        return self


def attestation_sha256(attestation: HeadAttestation) -> str:
    """Identity of the claim: SHA-256 of its stored canonical bytes."""
    return sha256_bytes(layout.serialize(attestation))


def parse_attestation(data: bytes) -> HeadAttestation:
    """Parse stored bytes. Anything that is not exactly the canonical form is refused."""
    attestation = HeadAttestation.model_validate_json(data, strict=True)
    if layout.serialize(attestation) != data:
        raise ValueError("head attestation is not in canonical project form")
    return attestation


@dataclass(frozen=True)
class HeadAssessment:
    classification: str
    authority: Authority
    root_id: str
    registry_head_sequence: int | None
    chain_valid: bool
    witnesses_consistent: bool
    unwitnessed_head: bool
    claim_in_registry: bool | None  # None: no claim was supplied
    problems: tuple[str, ...]

    @property
    def grants_authorization(self) -> Literal[False]:
        return False


def _event_hashes(verdict: RootVerdict) -> dict[int, str]:
    found: dict[int, str] = {}
    for entry in verdict.inventory["entries"]:
        match = _EVENT_PATH.match(entry["path"])
        if match and entry["sha256"] is not None:
            found[int(match.group(1))] = entry["sha256"]
    return found


def _contains(events: dict[int, str], sequence: int | None, digest: str | None) -> bool:
    return sequence is not None and events.get(sequence) == digest


def _claim_classification(
    claim: HeadAttestation, verdict: RootVerdict, events: dict[int, str]
) -> tuple[str, bool]:
    """(classification, claim_in_registry) for a claim against a sound, witnessed registry."""
    head = verdict.registry_head_sequence
    if claim.root_id != verdict.root_id:
        return "claim_for_another_root", False
    held = _contains(events, claim.sequence, claim.event_file_sha256)
    if head is not None and claim.sequence > head:
        return "registry_behind_claim", False
    if not held:
        return "claim_not_in_registry", False
    if claim.previous_sequence is not None and not _contains(
        events, claim.previous_sequence, claim.previous_event_file_sha256
    ):
        return "claim_lineage_conflict", True
    if claim.batch_id is not None:
        batch = verdict.batches.get(claim.batch_id)
        if batch is None or batch.campaign_plan_sha256 != claim.campaign_plan_sha256:
            return "claim_scope_conflict", True
    if claim.sequence < (head or 0):
        return "registry_ahead_of_claim", True
    return "claim_matches_head", True


def _assess(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    claim: HeadAttestation | None,
    *,
    fixture_trust: bool,
) -> HeadAssessment:
    structure = verify_root(root, authorization, anchors)
    witnessed = verify_root(root, authorization, anchors, witnesses)
    extra = tuple(p for p in witnessed.problems if p not in structure.problems)
    chain_valid = structure.code not in HISTORY_FAIL_CODES
    head = witnessed.registry_head_sequence
    events = _event_hashes(witnessed)
    held = None if claim is None else _contains(events, claim.sequence, claim.event_file_sha256)
    anchored = witnessed.head_authority == "genesis_anchored"

    if not chain_valid:
        classification = "chain_invalid"
    elif extra:  # an unwitnessed head is one of the extra problems
        classification = "witness_inconsistent"
    elif claim is None:
        classification = "genesis_anchored" if anchored else "head_unestablished"
    else:
        classification, held = _claim_classification(claim, witnessed, events)
        if classification == "claim_matches_head" and anchored:
            classification = "genesis_anchored"

    authority: Authority = (
        "genesis_anchored" if classification == "genesis_anchored" else ("not_established")
    )
    if (
        fixture_trust
        and claim is not None
        and claim.provenance == "test_fixture"
        and classification == "claim_matches_head"
    ):
        authority = "fixture_only"
    return HeadAssessment(
        classification=classification,
        authority=authority,
        root_id=witnessed.root_id,
        registry_head_sequence=head,
        chain_valid=chain_valid,
        witnesses_consistent=not extra,
        unwitnessed_head=witnessed.unwitnessed_head,
        claim_in_registry=held,
        problems=witnessed.problems,
    )


def assess_head(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    claim: HeadAttestation | None = None,
) -> HeadAssessment:
    """Read-only. Say what the registry, its witnesses and a claim establish, and no more.

    It takes no lock, so a concurrent writer can make the answer stale at once; a caller that
    needs a stable view holds the registry lock itself. The claim is never an authority root.
    """
    return _assess(root, authorization, anchors, witnesses, claim, fixture_trust=False)


def _assess_with_fixture_trust(
    root: Path,
    authorization: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    claim: HeadAttestation | None = None,
) -> HeadAssessment:
    """Test seam: believe a claim whose provenance is `test_fixture`. No production caller."""
    return _assess(root, authorization, anchors, witnesses, claim, fixture_trust=True)
