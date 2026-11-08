"""Synthetic readiness fixtures. A fixture policy exists only in tests; the package has none."""

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.readiness.decide import build_report
from biorag_repro.readiness.models import (
    MANDATORY_GATES,
    Candidate,
    EvaluatorIdentity,
    EvidenceRef,
    GateResult,
    HostBinding,
    ReadinessReport,
    evidence_digest,
)
from biorag_repro.readiness.policy import GATE_SLOT, SLOT_NAMES, ApprovalPolicy, ApprovalSlots


def digest(label: str) -> str:
    return sha256_bytes(label.encode())


def fixture_policy(version: str = "fixture-policy-0") -> ApprovalPolicy:
    slots = ApprovalSlots(**{name: digest(f"slot:{name}") for name in SLOT_NAMES})
    return ApprovalPolicy(policy_version=version, slots=slots)


def evidence(gate: str) -> list[EvidenceRef]:
    return [
        EvidenceRef(
            kind="fixture_artifact",
            locator=f"fixture/{gate}.json",
            sha256=digest(f"evidence:{gate}"),
            subject_kind="stored_file_bytes",
        )
    ]


def passed(gate: str, policy: ApprovalPolicy) -> GateResult:
    slot = GATE_SLOT[gate]
    refs = evidence(gate)
    reason = "no_prior_final_test_execution" if slot is None else "verified_against_policy"
    return GateResult(
        gate_id=gate,
        status="passed",
        verification="verified",
        scope="fixture",
        policy_expectation=slot,
        expected_sha256=None if slot is None else getattr(policy.slots, slot),
        reason_code=reason,
        detail=None,
        evidence=refs,
        evidence_digest=evidence_digest(refs),
    )


def blocked(gate: str, status: str, reason: str = "fixture_blocker") -> GateResult:
    return GateResult(
        gate_id=gate,
        status=status,
        verification="verified" if status == "failed" else "unverified",
        scope="fixture",
        policy_expectation=GATE_SLOT[gate],
        expected_sha256=None,
        reason_code=reason,
        detail=None,
        evidence=[],
        evidence_digest=None,
    )


def all_passed(policy: ApprovalPolicy) -> list[GateResult]:
    return [passed(gate, policy) for gate in MANDATORY_GATES]


def with_status(policy: ApprovalPolicy, gate: str, status: str) -> list[GateResult]:
    return [blocked(g, status) if g == gate else passed(g, policy) for g in MANDATORY_GATES]


def candidate(**overrides: object) -> Candidate:
    values: dict[str, object] = {
        "head_sha": "a" * 40,
        "scientific_source_identity_sha256": digest("source"),
        "freeze_record_sha256": digest("freeze"),
        "configuration_sha256_by_arm": {"confirmatory": digest("config:confirmatory")},
        "benchmark_release_sha256": digest("benchmark"),
        "corpus_content_sha256": digest("corpus"),
        "python_lock_sha256": digest("python-lock"),
        "renv_lock_sha256": digest("renv-lock"),
        "generation_seed_registry_sha256": digest("generation-seeds"),
        "benchmark_seed_registry_sha256": digest("benchmark-seeds"),
    }
    values.update(overrides)
    return Candidate(**values)


def host(**overrides: object) -> HostBinding:
    values: dict[str, object] = {
        "hostname": "fixture-host",
        "operating_system": "Linux",
        "machine": "x86_64",
        "gpu_names": ["Fixture GPU"],
    }
    values.update(overrides)
    return HostBinding(**values)


def evaluator(**overrides: object) -> EvaluatorIdentity:
    values: dict[str, object] = {"version": "fixture-evaluator-0", "code_sha256": digest("code")}
    values.update(overrides)
    return EvaluatorIdentity(**values)


def report(
    policy: ApprovalPolicy, results: list[GateResult] | None = None, **parts: object
) -> ReadinessReport:
    arguments: dict[str, object] = {
        "policy": policy,
        "results": all_passed(policy) if results is None else results,
        "candidate": candidate(),
        "host": host(),
        "evaluator": evaluator(),
        "warnings": [],
        "created_at_utc": "2026-11-07T00:00:00.000000Z",
        "nonce": "0" * 32,
    }
    arguments.update(parts)
    return build_report(**arguments)  # type: ignore[arg-type]
