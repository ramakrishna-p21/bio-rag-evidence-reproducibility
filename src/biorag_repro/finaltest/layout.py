"""Versioned layout and typed records of the final-test storage root.

A root holds `ROOT.json`, an append-only registry of hash-chained events and a lock
directory. No execution path exists in this version, so a valid root holds the genesis event
only; the event types after genesis are kept so that stored events still parse, and the
verifier reports any of them as unauthorized.

    final_test_v1/
        ROOT.json            genesis record of the root
        registry/NNNNNN.json registry events, one file each
        locks/registry.lock  empty file used only for flock

What is trusted. `StorageAuthorization` is the evidence artifact that policy slot G11
(`storage_authorization`) is meant to approve: its canonical-JSON SHA-256 is the slot value.
It names the exact location, root kind, corpus and benchmark identity and scientific matrix
the root is for, and the root identity is derived from those fields. `ROOT.json` and the
anchor are written by the same software that writes the registry, so they are consistency
and tamper evidence relative to that approved expectation. They do not authorize anything,
and a SHA-256 chain shows tampering relative to a trusted digest, not who wrote the bytes.
"""

import os
import re
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, StringConstraints, field_validator, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.generation.environment import check_utc

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
RootKind = Literal["rehearsal", "final_test"]

LAYOUT_VERSION: Final = "final_test_v1"
MATRIX_ID: Final = "experiment_plan_v1"
ROOT_FILE: Final = "ROOT.json"
REGISTRY_DIR: Final = "registry"
LOCKS_DIR: Final = "locks"
LOCK_FILE: Final = "locks/registry.lock"
ANCHOR_SUFFIX: Final = ".anchor.json"
# Event types the structural protocol can publish. Reservation is not authorization.
OPERATIONAL_EVENTS: Final = (
    "root_initialized",
    "batch_reserved",
    "batch_materialized",
    "batch_closed_incomplete",
)
# Published only into rehearsal roots. In a final-test root the verifier treats them
# like any other event that nothing is authorized to publish.
REHEARSAL_EVENTS: Final = (
    "batch_inputs_bound",
    "batch_authorized",
    "batch_resumed",
    "batch_completed",
)
EVENT_TYPES: Final = OPERATIONAL_EVENTS + REHEARSAL_EVENTS
# Exact evidence roles, in order, of the batch events, kept so that stored events parse as
# before. The genesis event's evidence is checked against the approved authorization by the
# inventory verifier.
REQUIRED_ROLES: Final = {
    "batch_reserved": ["campaign_plan", "batch_id", "ownership", "batch_manifest"],
    "batch_materialized": ["batch_id", "batch_manifest", "reservation_event"],
    "batch_closed_incomplete": ["batch_id", "reservation_event", "observed_state"],
}
EVENT_NAME: Final = re.compile(r"^(\d{6})\.json$")


class StorageAuthorization(Record):
    """Evidence for policy slot G11: the storage location and scope approved for a root."""

    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"]
    root_kind: RootKind
    authorized_root: Nonblank  # absolute path of the `final_test_v1` directory
    storage_label: Nonblank  # human description of the storage; approved but not an identity
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256  # SHA-256 of the benchmark release record bytes
    matrix_id: Literal["experiment_plan_v1"]

    @field_validator("authorized_root")
    @classmethod
    def absolute_and_normal(cls, value: str) -> str:
        if not os.path.isabs(value) or os.path.normpath(value) != value:
            raise ValueError("authorized_root must be an absolute, normalized path")
        return value


def authorization_sha256(authorization: StorageAuthorization) -> str:
    """The policy-slot value: SHA-256 of the canonical JSON of the authorization."""
    return sha256_bytes(canonical_json_bytes(authorization.model_dump(mode="json")))


def root_id_for(authorization: StorageAuthorization) -> str:
    """Root identity derived from the approved fields; it is never a free value."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "layout_version": authorization.layout_version,
                "root_kind": authorization.root_kind,
                "authorized_root": authorization.authorized_root,
                "corpus_content_sha256": authorization.corpus_content_sha256,
                "benchmark_release_sha256": authorization.benchmark_release_sha256,
                "matrix_id": authorization.matrix_id,
            }
        )
    )


class RootRecord(Record):
    """`ROOT.json`: the genesis record. Written once; the timestamp is informational."""

    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"]
    root_kind: RootKind
    root_id: Sha256
    authorization_sha256: Sha256
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256
    matrix_id: Literal["experiment_plan_v1"]
    created_utc: str

    _utc = field_validator("created_utc")(check_utc)


class RootAnchor(Record):
    """Kept outside the root. It pins the exact genesis bytes; it is not an approval."""

    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"]
    root_kind: RootKind
    root_id: Sha256
    authorization_sha256: Sha256
    root_file_sha256: Sha256
    genesis_event_sha256: Sha256
    created_utc: str

    _utc = field_validator("created_utc")(check_utc)


class EvidenceCitation(Record):
    role: Nonblank
    sha256: Sha256


def event_identity(
    *,
    root_id: str,
    sequence: int,
    event_type: str,
    previous_event_sha256: str | None,
    evidence: list[EvidenceCitation],
    schema_version: str = "1",
    layout_version: str = LAYOUT_VERSION,
) -> str:
    """Identity of an event. The informational timestamp is not part of it."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": schema_version,
                "layout_version": layout_version,
                "root_id": root_id,
                "sequence": sequence,
                "event_type": event_type,
                "previous_event_sha256": previous_event_sha256,
                "evidence": [c.model_dump(mode="json") for c in evidence],
            }
        )
    )


class RegistryEvent(Record):
    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"]
    root_id: Sha256
    sequence: int
    event_type: Literal[
        "root_initialized",
        "batch_reserved",
        "batch_materialized",
        "batch_closed_incomplete",
        "batch_inputs_bound",
        "batch_authorized",
        "batch_resumed",
        "batch_completed",
    ]
    previous_event_sha256: Sha256 | None
    identity_sha256: Sha256
    evidence: list[EvidenceCitation]
    created_utc: str  # informational; excluded from the identity

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.sequence < 1:
            raise ValueError("sequence starts at 1")
        if (self.sequence == 1) != (self.previous_event_sha256 is None):
            raise ValueError("only the genesis event has no previous event")
        expected = event_identity(
            root_id=self.root_id,
            sequence=self.sequence,
            event_type=self.event_type,
            previous_event_sha256=self.previous_event_sha256,
            evidence=self.evidence,
            schema_version=self.schema_version,
            layout_version=self.layout_version,
        )
        if self.identity_sha256 != expected:
            raise ValueError("event identity does not match its content")
        roles = REQUIRED_ROLES.get(self.event_type)
        if roles is not None and [c.role for c in self.evidence] != roles:
            raise ValueError(f"{self.event_type} must cite exactly {roles}")
        return self


def make_event(
    *,
    root_id: str,
    sequence: int,
    event_type: str,
    previous_event_sha256: str | None,
    evidence: list[EvidenceCitation],
    created_utc: str,
) -> RegistryEvent:
    return RegistryEvent.model_validate(
        {
            "schema_version": "1",
            "layout_version": LAYOUT_VERSION,
            "root_id": root_id,
            "sequence": sequence,
            "event_type": event_type,
            "previous_event_sha256": previous_event_sha256,
            "identity_sha256": event_identity(
                root_id=root_id,
                sequence=sequence,
                event_type=event_type,
                previous_event_sha256=previous_event_sha256,
                evidence=evidence,
            ),
            "evidence": [c.model_dump(mode="json") for c in evidence],
            "created_utc": created_utc,
        }
    )


def serialize(record: BaseModel) -> bytes:
    """Project JSON bytes. A stored record must equal this serialization exactly."""
    return serialize_project_json(record, type(record))
