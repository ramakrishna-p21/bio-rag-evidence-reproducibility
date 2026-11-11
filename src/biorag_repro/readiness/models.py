"""Readiness-report contract: gate results, evidence references and the deterministic binding.

A gate result is a claim by an evaluator. It becomes a *passed* gate only if it carries
structurally valid positive evidence; whether that evidence matches an approval in the
tracked policy is decided in `decide`. Nothing in this module reads a file, Git, an
environment or a device.

The recorded `decision` and `blockers` of a report are results, not inputs. The report model
recomputes both from the gate results and refuses a report that disagrees, so a forged READY
fails validation.
"""

import re
from collections.abc import Sequence
from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record
from biorag_repro.generation.environment import check_utc

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Code = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]

GateId = Literal[
    "G01_corpus_sources",
    "G02_benchmark_freeze",
    "G03_final_test_configuration_freeze",
    "G04_generator_qualification",
    "G05_model_parser_identities",
    "G06_python_environment",
    "G07_r_analysis_environment",
    "G08_seed_registry",
    "G09_scientific_source_revision",
    "G10_development_restart_recovery",
    "G11_storage_authorization",
    "G12_shared_gpu_authorization",
    "G13_final_test_history_integrity",
]
MANDATORY_GATES: Final[tuple[GateId, ...]] = (
    "G01_corpus_sources",
    "G02_benchmark_freeze",
    "G03_final_test_configuration_freeze",
    "G04_generator_qualification",
    "G05_model_parser_identities",
    "G06_python_environment",
    "G07_r_analysis_environment",
    "G08_seed_registry",
    "G09_scientific_source_revision",
    "G10_development_restart_recovery",
    "G11_storage_authorization",
    "G12_shared_gpu_authorization",
    "G13_final_test_history_integrity",
)
HISTORY_GATE: Final = "G13_final_test_history_integrity"

# There is no "skipped" or "not applicable" status: a check that did not run is `unavailable`.
GateStatus = Literal["passed", "failed", "pending", "unavailable"]
Verification = Literal["verified", "unverified"]
Decision = Literal["READY", "NOT_READY"]

PASS_CODE: Final = "verified_against_policy"
# G13 is about the validity of final-test history, not about the absence of artifacts. Each
# passing code names a history class a verifier may establish; a bare empty directory or a
# report saying "clean" establishes none of them. Meanings are the contract for the future
# verifier (P9-B/P10); none of them is evaluated here.
HISTORY_PASS_CODES: Final[dict[str, str]] = {
    "no_prior_final_test_execution": (
        "a verified inventory of the final-test locations shows no observation or attempt"
    ),
    "same_batch_valid_history": (
        "every observation and attempt belongs to the authorized batch and verifies; completed "
        "observations are preserved and never regenerated, interrupted attempts are eligible "
        "for authorized resume"
    ),
    "other_approved_batch_valid_history": (
        "prior artifacts verify and belong to another batch listed as approved"
    ),
}
HISTORY_FAIL_CODES: Final[dict[str, str]] = {
    "unauthorized_final_test_artifact": "an artifact belongs to no authorized batch",
    "conflicting_final_test_artifact": "artifacts contradict each other or the authorized batch",
    "unverified_final_test_artifact": "an artifact exists that could not be verified",
}
PASS_CODES: Final[frozenset[str]] = frozenset({PASS_CODE, *HISTORY_PASS_CODES})
# A passing reason is whitelisted, so words such as "skipped" or "waived" can never pass.

EVALUATOR_LIMITATIONS: Final[tuple[str, ...]] = (
    (
        "A matching SHA-256 shows byte integrity relative to a trusted expected digest; it does "
        "not show scientific approval, who produced the bytes, or when."
    ),
    (
        "The approval policy is a tracked Python module. Its Git history gives versioning and "
        "auditability, not cryptographic proof of who authorized a change."
    ),
    (
        "Code that can be modified by someone with repository write access can change this "
        "evaluator and its policy; detection relies on recorded code and policy identities."
    ),
    (
        "No trusted clock or signature service exists. Timestamps and nonces are informational "
        "and are not part of the readiness binding."
    ),
    "Evidence produced by the executing process describes that process's own account.",
    (
        "A decision under a synthetic fixture policy establishes decision logic only and has no "
        "authority over real execution."
    ),
)


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


class EvidenceRef(Record):
    """One piece of evidence, with the exact kind of bytes its digest covers."""

    kind: Code
    locator: Nonblank  # logical reference, never a machine-specific absolute path
    sha256: Sha256
    subject_kind: Literal["stored_file_bytes", "canonical_json"]

    @field_validator("locator")
    @classmethod
    def logical(cls, value: str) -> str:
        if value.startswith(("/", "~")) or re.match(r"^[A-Za-z]:[\\/]", value) or "\\" in value:
            raise ValueError("an evidence locator is a logical reference, not an absolute path")
        return value


def evidence_digest(refs: Sequence[EvidenceRef]) -> str:
    """Identity of the evidence set: the canonical JSON of its references, in the given order."""
    return _digest([r.model_dump(mode="json") for r in refs])


class GateResult(Record):
    gate_id: GateId
    status: GateStatus
    # `verified`: a verifier established this result. `unverified`: it is a claim or an absence.
    verification: Verification
    scope: Nonblank  # what was verified and where, e.g. the data root or the execution host
    policy_expectation: Code | None  # the policy slot compared against; None only for G13
    expected_sha256: Sha256 | None  # that slot's value at evaluation time
    reason_code: Code
    detail: str | None
    evidence: list[EvidenceRef]
    evidence_digest: Sha256 | None  # derived from `evidence`; never a free value

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.evidence_digest != (evidence_digest(self.evidence) if self.evidence else None):
            raise ValueError("evidence digest is not the digest of the evidence references")
        history = self.gate_id == HISTORY_GATE
        if self.status == "passed":
            if self.verification != "verified":
                raise ValueError("a passed gate must be verified")
            if not self.evidence:
                raise ValueError("a passed gate needs positive evidence")
            allowed = set(HISTORY_PASS_CODES) if history else {PASS_CODE}
            if self.reason_code not in allowed:
                raise ValueError(f"{self.gate_id} cannot pass with reason {self.reason_code}")
            if history != (self.policy_expectation is None and self.expected_sha256 is None):
                raise ValueError("only the history gate passes without a policy expectation")
            if not history and (self.policy_expectation is None or self.expected_sha256 is None):
                raise ValueError("a passed gate names the policy expectation it matched")
            return self
        if self.reason_code in PASS_CODES:
            raise ValueError("a gate that did not pass cannot carry a passing reason")
        if self.status in ("pending", "unavailable") and self.verification != "unverified":
            raise ValueError(f"a {self.status} gate has not been verified")
        if self.reason_code in HISTORY_FAIL_CODES and not history:
            raise ValueError("history failure codes belong to the history gate")
        return self


class Candidate(Record):
    """Scientific identities the readiness decision is about. `None` means not yet established."""

    head_sha: GitRevision | None
    scientific_source_identity_sha256: Sha256 | None
    freeze_record_sha256: Sha256 | None
    configuration_sha256_by_arm: dict[Nonblank, Sha256]
    benchmark_release_sha256: Sha256 | None
    corpus_content_sha256: Sha256 | None
    python_lock_sha256: Sha256 | None
    renv_lock_sha256: Sha256 | None
    generation_seed_registry_sha256: Sha256 | None
    benchmark_seed_registry_sha256: Sha256 | None


class HostBinding(Record):
    hostname: Nonblank
    operating_system: Nonblank
    machine: Nonblank
    gpu_names: list[Nonblank]


class EvaluatorIdentity(Record):
    version: Nonblank
    code_sha256: Sha256


class Blocker(Record):
    gate_id: GateId
    status: Literal["failed", "pending", "unavailable"]
    reason_code: Code


class GateSetError(ValueError):
    """The gate results are not exactly the mandatory set."""


def check_gate_set(gate_ids: Sequence[str]) -> None:
    seen: set[str] = set()
    for gate in gate_ids:
        if gate not in MANDATORY_GATES:
            raise GateSetError(f"unexpected gate {gate!r}")
        if gate in seen:
            raise GateSetError(f"duplicate gate {gate}")
        seen.add(gate)
    if missing := [g for g in MANDATORY_GATES if g not in seen]:
        raise GateSetError("missing mandatory gates: " + ", ".join(missing))


def derive_blockers(results: Sequence[GateResult]) -> list[Blocker]:
    order = {gate: i for i, gate in enumerate(MANDATORY_GATES)}
    return [
        Blocker(gate_id=r.gate_id, status=r.status, reason_code=r.reason_code)
        for r in sorted(results, key=lambda r: order[r.gate_id])
        if r.status != "passed"
    ]


def derive_decision(blockers: Sequence[Blocker]) -> Decision:
    return "NOT_READY" if blockers else "READY"


def compute_binding(
    *,
    requested_run_mode: str,
    policy_sha256: str,
    evaluator: EvaluatorIdentity,
    candidate: Candidate,
    host: HostBinding,
    results: Sequence[GateResult],
) -> str:
    """Deterministic identity of what was evaluated and found.

    Covers gate membership, every gate's status, verification, scope, expectation, reason and
    evidence, the candidate, the host, the policy and the evaluator. It excludes the decision
    and blockers (derived), warnings, timestamps, the nonce and the report's own hash.
    """
    order = {gate: i for i, gate in enumerate(MANDATORY_GATES)}
    return _digest(
        {
            "kind": "readiness_binding",
            "schema_version": "1",
            "requested_run_mode": requested_run_mode,
            "mandatory_gates": list(MANDATORY_GATES),
            "policy_sha256": policy_sha256,
            "evaluator": evaluator.model_dump(mode="json"),
            "candidate": candidate.model_dump(mode="json"),
            "host": host.model_dump(mode="json"),
            "gates": [
                r.model_dump(mode="json", exclude={"detail"})
                for r in sorted(results, key=lambda r: order[r.gate_id])
            ],
        }
    )


class ReadinessReport(Record):
    schema_version: Literal["1"] = "1"
    kind: Literal["readiness_report"] = "readiness_report"
    requested_run_mode: Literal["final_test"]
    policy_sha256: Sha256
    evaluator: EvaluatorIdentity
    candidate: Candidate
    host: HostBinding
    gates: list[GateResult]
    blockers: list[Blocker]  # recorded result; recomputed on validation
    decision: Decision  # recorded result, never an authorization input
    warnings: list[Nonblank]  # informational; cannot affect the decision
    binding_sha256: Sha256
    limitations: list[Nonblank]
    created_at_utc: str  # informational
    nonce: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]  # informational

    @field_validator("created_at_utc")
    @classmethod
    def utc(cls, value: str) -> str:
        return check_utc(value)

    @model_validator(mode="after")
    def recomputed(self) -> Self:
        check_gate_set([g.gate_id for g in self.gates])
        blockers = derive_blockers(self.gates)
        if self.blockers != blockers:
            raise ValueError("recorded blockers differ from the gate results")
        if self.decision != derive_decision(blockers):
            raise ValueError("recorded decision differs from the gate results")
        if self.limitations != list(EVALUATOR_LIMITATIONS):
            raise ValueError("limitations differ from the fixed statement")
        expected = compute_binding(
            requested_run_mode=self.requested_run_mode,
            policy_sha256=self.policy_sha256,
            evaluator=self.evaluator,
            candidate=self.candidate,
            host=self.host,
            results=self.gates,
        )
        if self.binding_sha256 != expected:
            raise ValueError("binding differs from the evaluated content")
        return self
