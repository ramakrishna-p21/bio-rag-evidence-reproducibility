"""Authorization boundary for final-test execution. Phase 10 calls it; nothing here runs anything.

`authorize_final_test` joins two independent checks and refuses unless both hold:

* the stored report is verified as a historical assessment (`publish.read_report`: bytes,
  manifest, policy, recomputed decision and retained evidence) and its recomputed decision is
  READY;
* a fresh evaluation of the present checkout and data root under the same policy is READY, and
  its binding equals the stored one. The binding covers the policy, evaluator, candidate,
  host and every gate's status, expectation, reason and evidence, so a changed source,
  specification, configuration, seed, lock, host or approval cannot be matched by an older
  report. Only the timestamp, nonce, warnings and gate prose differ without effect.

The live evaluation alone decides what is true now: nothing from the stored report's checkout
evidence is trusted. `compare_current` is informational and is not used.

The result is an auditable handoff, not a credential. It is a plain Python object that any
code in the same process can construct, so it proves nothing to a caller that receives it
from elsewhere. Phase 10 must call this function itself, before creating a final-test batch
and before every resume, and must record `report_sha256` in the batch manifest.

Limits: there is no lock between the checks and the later use, so evidence can change after
they return (the stored report is read again after the live evaluation to catch a change
during the check, nothing more). The real project cannot be authorized: the real policy has
no approval and gate G13 is unavailable until Phase 10 defines its inventory authority.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from biorag_repro.generation.environment import utc_now
from biorag_repro.readiness import publish
from biorag_repro.readiness.decide import ReportInvalid, verify_report
from biorag_repro.readiness.evaluate import Evaluation, evaluate, system_context
from biorag_repro.readiness.models import Candidate, HostBinding, ReadinessReport
from biorag_repro.readiness.policy import REAL_POLICY, ApprovalPolicy

Reason = Literal[
    "report_incomplete",
    "report_unverifiable",
    "stored_not_ready",
    "live_evaluation_failed",
    "live_invalid",
    "live_not_ready",
    "identity_mismatch",
    "report_changed_during_check",
]

SCOPE: Final = "final_test_batch_start"


class AuthorizationRefused(ValueError):
    """Authorization was refused. `reason` says which requirement failed."""

    def __init__(self, reason: Reason, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason: Reason = reason


@dataclass(frozen=True)
class FinalTestAuthorization:
    report_sha256: str
    manifest_sha256: str
    policy_sha256: str
    evaluator_code_sha256: str
    report_binding_sha256: str
    live_binding_sha256: str
    candidate: Candidate
    host: HostBinding
    scope: str
    verified_at_utc: str  # informational


def _difference(stored: ReadinessReport, live: ReadinessReport) -> str | None:
    """Name the first authorization-relevant part that differs, or None if the bindings agree.

    Policy identity and run mode are not compared here: `verify_report` already requires both
    reports to carry the verifying policy, and the run mode has a single allowed value.
    """
    if stored.evaluator != live.evaluator:
        return "evaluator"
    for name in Candidate.model_fields:
        if getattr(stored.candidate, name) != getattr(live.candidate, name):
            return f"candidate.{name}"
    if stored.host != live.host:
        return "host"
    if stored.binding_sha256 != live.binding_sha256:
        return "gates"
    return None


def _authorize(
    data_root: Path,
    report_sha256: str,
    *,
    repo: Path,
    policy: ApprovalPolicy,
    evaluate_fn: Callable[[ApprovalPolicy], Evaluation],
) -> FinalTestAuthorization:
    try:
        stored = publish.read_report(data_root, report_sha256, policy)
    except publish.PartialPublication as error:
        raise AuthorizationRefused("report_incomplete", str(error)) from error
    except publish.ReadbackError as error:
        raise AuthorizationRefused("report_unverifiable", str(error)) from error
    if stored.outcome.decision != "READY":
        gates = ", ".join(b.gate_id for b in stored.outcome.blockers)
        raise AuthorizationRefused("stored_not_ready", gates)

    try:
        live = evaluate_fn(policy)
    except Exception as error:  # any failure to evaluate now means no authority
        raise AuthorizationRefused("live_evaluation_failed", str(error)[:300]) from error
    try:
        outcome = verify_report(live.report, policy)
    except ReportInvalid as error:
        raise AuthorizationRefused("live_invalid", str(error)[:300]) from error
    if outcome.decision != "READY":
        gates = ", ".join(b.gate_id for b in outcome.blockers)
        raise AuthorizationRefused("live_not_ready", gates)
    if (part := _difference(stored.report, live.report)) is not None:
        raise AuthorizationRefused("identity_mismatch", part)

    # read_report ties the report and manifest to the hash, so a clean second read is the same one
    try:
        publish.read_report(data_root, report_sha256, policy)
    except (publish.PublicationError, publish.ReadbackError) as error:
        raise AuthorizationRefused("report_changed_during_check", str(error)) from error

    report = stored.report
    return FinalTestAuthorization(
        report_sha256=report_sha256,
        manifest_sha256=stored.manifest_sha256,
        policy_sha256=report.policy_sha256,
        evaluator_code_sha256=report.evaluator.code_sha256,
        report_binding_sha256=report.binding_sha256,
        live_binding_sha256=live.report.binding_sha256,
        candidate=report.candidate,
        host=report.host,
        scope=SCOPE,
        verified_at_utc=utc_now(),
    )


def authorize_final_test(
    data_root: Path, report_sha256: str, *, repo: Path
) -> FinalTestAuthorization:
    """Authorize starting or resuming final-test work, or raise `AuthorizationRefused`.

    Always uses the tracked approval policy and the real evaluator.
    """
    return _authorize(
        data_root,
        report_sha256,
        repo=repo,
        policy=REAL_POLICY,
        evaluate_fn=lambda policy: evaluate(system_context(repo, data_root), policy),
    )
