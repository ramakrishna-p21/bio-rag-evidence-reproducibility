"""What an adapter found, and how that becomes a gate result.

Three questions are kept apart for every gate:

* integrity: were the referenced bytes read and hashed?
* contract validity: do they satisfy the frozen structural and scientific rules?
* policy approval: does the verified identity subject equal the digest the policy approves?

`to_result` is the only place these meet. A subject that is technically valid but has no
approval is `pending`; one that differs from a populated approval is `failed`; a violated
contract is `failed` whether or not an approval exists; something that could not be
inspected is `unavailable`; a prerequisite that does not exist yet is `pending`. The observed
subject is stored as evidence and is never copied into the policy.
"""

from dataclasses import dataclass, field
from typing import Any, Literal

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness.models import (
    PASS_CODE,
    EvidenceRef,
    GateId,
    GateResult,
    evidence_digest,
)
from biorag_repro.readiness.policy import GATE_SLOT, ApprovalPolicy

State = Literal["ok", "failed", "unavailable", "absent"]
PENDING_APPROVAL = "verified_pending_policy_approval"
POLICY_MISMATCH = "policy_mismatch"


@dataclass
class Gathered:
    """`ok`: checks passed and `subject` is the identity to compare with the policy.
    `failed`: a contract is violated. `unavailable`: the evidence could not be inspected.
    `absent`: the prerequisite does not exist yet; any `refs` gathered are kept."""

    state: State
    reason: str
    scope: str
    detail: str | None = None
    subject: dict[str, Any] | None = None
    refs: list[EvidenceRef] = field(default_factory=list)
    snapshots: dict[str, bytes] = field(default_factory=dict)
    facts: dict[str, Any] = field(default_factory=dict)  # candidate identities that were read

    def add_snapshot(self, kind: str, value: Any) -> str:
        """Keep an observation as canonical JSON bytes and cite it; returns its SHA-256."""
        data = canonical_json_bytes(value)
        digest = sha256_bytes(data)
        self.snapshots[digest] = data
        self.refs.append(
            EvidenceRef(
                kind=kind,
                locator=f"snapshot/{digest}.json",
                sha256=digest,
                subject_kind="stored_file_bytes",
            )
        )
        return digest


def file_ref(kind: str, locator: str, data: bytes) -> EvidenceRef:
    return EvidenceRef(
        kind=kind, locator=locator, sha256=sha256_bytes(data), subject_kind="stored_file_bytes"
    )


def to_result(gate: GateId, gathered: Gathered, policy: ApprovalPolicy) -> GateResult:
    slot = GATE_SLOT[gate]
    approved = None if slot is None else getattr(policy.slots, slot)
    refs = list(gathered.refs)
    status: Literal["passed", "failed", "pending", "unavailable"]
    reason = gathered.reason
    expected = None
    if gathered.state == "ok":
        if gathered.subject is None:
            raise ValueError(f"{gate}: an ok result needs its identity subject")
        subject = Gathered("ok", reason, gathered.scope)
        digest = subject.add_snapshot("observed_subject", gathered.subject)
        gathered.snapshots.update(subject.snapshots)
        refs += subject.refs
        if approved is None:
            status, reason = "pending", PENDING_APPROVAL
        elif approved == digest:
            status, reason, expected = "passed", PASS_CODE, approved
        else:
            status, reason, expected = "failed", POLICY_MISMATCH, approved
    elif gathered.state == "failed":
        status = "failed"
    elif gathered.state == "unavailable":
        status = "unavailable"
    else:
        status = "pending"
    return GateResult(
        gate_id=gate,
        status=status,
        verification="verified" if status in ("passed", "failed") else "unverified",
        scope=gathered.scope,
        policy_expectation=slot,
        expected_sha256=expected,
        reason_code=reason,
        detail=gathered.detail,
        evidence=refs,
        evidence_digest=evidence_digest(refs) if refs else None,
    )
