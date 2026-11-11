"""Versioned layout and typed records of the final-test storage root.

A root holds `ROOT.json`, an append-only registry of hash-chained events, a lock directory
and the batch records the registry cites.

    final_test_v1/
        ROOT.json            genesis record of the root
        registry/NNNNNN.json registry events, one file each
        locks/registry.lock  empty file used only for flock

    batches/<id>/manifest.json   structural batch record (one per matrix arm)
    runs/<run_id>/run.json       run record; the run's observation store lives beside it

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

from pydantic import BaseModel, ConfigDict, StringConstraints, field_validator, model_validator

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
BATCHES_DIR: Final = "batches"
MANIFEST_NAME: Final = "manifest.json"
WITNESS_SUFFIX: Final = ".witness.json"
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
# Not part of `REQUIRED_ROLES`: the event model must parse these events exactly as before so
# that a final-test root is judged by the same rules. The rehearsal verifier and publisher
# enforce the roles. `inputs_binding` is the hash of the binding file; `run_record` is the hash
# of the run record bytes decided before the event is written (they contain the run id).
REHEARSAL_ROLES: Final = {
    "batch_inputs_bound": ["batch_id", "materialization_event", "inputs_binding"],
    "batch_authorized": [
        "batch_id",
        "inputs_bound_event",
        "authorization_citation",
        "run_record",
        "readiness_report",
    ],
    # The event that authorizes a resume. `previous_authorization_event` is the file hash of the
    # latest earlier batch_authorized or batch_resumed event of the same batch.
    "batch_resumed": [
        "batch_id",
        "previous_authorization_event",
        "continuity_assessment",
        "authorization_citation",
        "readiness_report",
        "run_record",
    ],
    # The event that ends a batch. `authorization_basis` is the digest of the record that shows
    # who may publish the completion: the batch's latest citation when the session that ran the
    # batch publishes it, the continuity assessment of a fresh report when a new session does.
    "batch_completed": [
        "batch_id",
        "previous_authorization_event",
        "authorization_basis",
        "completion_record",
        "readiness_report",
        "run_record",
    ],
}
# Rehearsal-only files, relative to the root.
SESSION_LOCK_FILE: Final = "locks/session.lock"
# Who held the session last (pid, process start, host, open or closed). Written when the first
# attempt of a session begins; read by the next session to decide whether recovery is safe.
HOLDER_FILE: Final = "locks/session.holder"
INPUTS_DIR: Final = "inputs"
ROSTERS_DIR: Final = "rosters"
BINDING_NAME: Final = "binding.json"
CITATIONS_DIR: Final = "citations"
CITATION_NAME: Final = "000001.json"
# Batch-level completion evidence (rehearsal roots only). The continuity file exists only when
# the completion was authorized by a fresh report.
COMPLETION_NAME: Final = "completion.json"
COMPLETION_CONTINUITY_NAME: Final = "completion.continuity.json"


def citation_name(ordinal: int) -> str:
    """The citation of the initial authorization is 1; the n-th resume is n + 1."""
    return f"{ordinal:06d}.json"


def continuity_name(ordinal: int) -> str:
    """The continuity assessment cited by the citation of the same ordinal."""
    return f"{ordinal:06d}.continuity.json"


RUNS_DIR: Final = "runs"
# Exact evidence roles, in order, of the batch events. The values are SHA-256 digests; batch_id
# and ownership are derived identities (see batches.py), not file hashes. The genesis event's
# evidence is checked against the approved authorization by the inventory verifier.
REQUIRED_ROLES: Final = {
    "batch_reserved": ["campaign_plan", "batch_id", "ownership", "batch_manifest"],
    "batch_materialized": ["batch_id", "batch_manifest", "reservation_event"],
    "batch_closed_incomplete": ["batch_id", "reservation_event", "observed_state"],
}
EVENT_NAME: Final = re.compile(r"^(\d{6})\.json$")


class FrozenRecord(Record):
    """A record whose published value cannot be changed in memory either."""

    model_config = ConfigDict(frozen=True)


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


class HeadWitness(FrozenRecord):
    """Consistency record for one registry event, kept outside the root.

    A witness shows that the registry once held this event. It is written by the same software
    that writes the registry and lives under the same data root, so it detects inconsistency
    between two storage structures, not a coordinated rollback of both, and it is not an
    authority for the latest head.
    """

    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"]
    root_id: Sha256
    sequence: int
    event_file_sha256: Sha256
    event_identity_sha256: Sha256
    previous_witness_sha256: Sha256 | None
    created_utc: str

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def chain_start(self) -> Self:
        if self.sequence < 1 or (self.sequence == 1) != (self.previous_witness_sha256 is None):
            raise ValueError("only the first witness has no previous witness")
        return self


class HeadCheckpoint(FrozenRecord):
    """A latest-head claim from a source the caller trusts independently of the root.

    The batch layout defines the shape and checks continuity against it, but no source in this repository
    can establish that a checkpoint is the latest authorized head, so a checkpoint never makes
    a history authoritative.
    """

    root_id: Sha256
    sequence: int
    event_file_sha256: Sha256


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
