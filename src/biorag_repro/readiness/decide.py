"""The readiness decision: a pure function of validated gate results and the policy.

Outcomes are kept apart:

* `decide` returns READY or NOT_READY for a structurally valid set of results. NOT_READY is a
  correct answer, including when every gate is pending.
* `ReportInvalid` means the input cannot be a readiness result at all: a missing, duplicate
  or unexpected gate, or a pass that the policy does not back. It is never turned into a
  decision, and it is not the same thing as NOT_READY.
* An evaluator that cannot run is a different failure again (P9-B) and produces no report.

READY requires every mandatory gate to have passed with positive evidence that matches a
policy slot. Warnings, decision fields supplied by a caller, and missing input cannot
change that.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import ValidationError

from biorag_repro.readiness.models import (
    EVALUATOR_LIMITATIONS,
    Blocker,
    Candidate,
    Decision,
    EvaluatorIdentity,
    GateResult,
    GateSetError,
    HostBinding,
    ReadinessReport,
    check_gate_set,
    compute_binding,
    derive_blockers,
    derive_decision,
)
from biorag_repro.readiness.policy import GATE_SLOT, ApprovalPolicy, policy_sha256


class ReportInvalid(ValueError):
    """The results or report are not a valid readiness input; no decision exists."""


@dataclass(frozen=True)
class Outcome:
    decision: Decision
    blockers: tuple[Blocker, ...]


def _check_against_policy(results: Sequence[GateResult], policy: ApprovalPolicy) -> None:
    for result in results:
        if result.status != "passed":
            continue
        slot = GATE_SLOT[result.gate_id]
        approved = None if slot is None else getattr(policy.slots, slot)
        if result.policy_expectation != slot:
            raise ReportInvalid(f"{result.gate_id} passed against the wrong policy slot")
        if slot is not None and (approved is None or result.expected_sha256 != approved):
            raise ReportInvalid(f"{result.gate_id} passed without a matching policy approval")


def decide(results: Sequence[GateResult], policy: ApprovalPolicy) -> Outcome:
    for result in results:
        if not isinstance(result, GateResult):
            raise ReportInvalid("a gate result is not a GateResult")
    try:
        check_gate_set([r.gate_id for r in results])
    except GateSetError as error:
        raise ReportInvalid(str(error)) from error
    # A result built without validation (model_construct, model_copy) is checked again.
    for result in results:
        try:
            GateResult.model_validate(result.model_dump(mode="python"))
        except ValidationError as error:
            raise ReportInvalid(f"{result.gate_id} is not a valid gate result: {error}") from error
    _check_against_policy(results, policy)
    blockers = derive_blockers(results)
    return Outcome(derive_decision(blockers), tuple(blockers))


def build_report(
    *,
    policy: ApprovalPolicy,
    results: Sequence[GateResult],
    candidate: Candidate,
    host: HostBinding,
    evaluator: EvaluatorIdentity,
    warnings: Sequence[str],
    created_at_utc: str,
    nonce: str,
) -> ReadinessReport:
    """Assemble a report whose decision, blockers and binding come from the results alone."""
    outcome = decide(results, policy)
    digest = policy_sha256(policy)
    return ReadinessReport(
        requested_run_mode="final_test",
        policy_sha256=digest,
        evaluator=evaluator,
        candidate=candidate,
        host=host,
        gates=list(results),
        blockers=list(outcome.blockers),
        decision=outcome.decision,
        warnings=list(warnings),
        binding_sha256=compute_binding(
            requested_run_mode="final_test",
            policy_sha256=digest,
            evaluator=evaluator,
            candidate=candidate,
            host=host,
            results=results,
        ),
        limitations=list(EVALUATOR_LIMITATIONS),
        created_at_utc=created_at_utc,
        nonce=nonce,
    )


def verify_report(report: ReadinessReport, policy: ApprovalPolicy) -> Outcome:
    """Re-derive the outcome of a report against a policy. The recorded decision is not used.

    A report is valid for this policy only if it was produced under exactly this policy and
    every passed gate still matches the policy's approval. A model that skipped validation
    (for example `model_construct`) is re-checked here.
    """
    if not isinstance(report, ReadinessReport):
        raise ReportInvalid("not a readiness report")
    if report.policy_sha256 != policy_sha256(policy):
        raise ReportInvalid("the report was produced under a different policy")
    outcome = decide(report.gates, policy)
    binding = compute_binding(
        requested_run_mode=report.requested_run_mode,
        policy_sha256=report.policy_sha256,
        evaluator=report.evaluator,
        candidate=report.candidate,
        host=report.host,
        results=report.gates,
    )
    if (
        report.binding_sha256 != binding
        or report.decision != outcome.decision
        or tuple(report.blockers) != outcome.blockers
        or report.limitations != list(EVALUATOR_LIMITATIONS)
    ):
        raise ReportInvalid("the recorded result differs from what the gate results imply")
    return outcome
