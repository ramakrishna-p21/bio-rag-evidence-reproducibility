"""Scientific continuity between an initial and a later readiness assessment of one batch.

The Phase 9 guard compares one stored report with one live evaluation. It does not compare two
authorizations with each other, so a seed approval, configuration or benchmark that changed
between a batch's first authorization and a resume would pass if the later report were READY.
This module is that comparison. It is a pure function of two reports and two batch scopes; it
authorizes nothing and is not called by any execution code.

Every difference is classified, never skipped:

* `identical`;
* `compatible_recorded`: differs, the difference is expected, and both values are kept in the
  finding. Only two cases exist: the commit id when the content identity of the scientific
  source (the Git blob list of the G09 scope plus the frozen specification bytes) is the same
  and was established on both sides, and the G13 history evidence, which must evolve with the
  registry but must independently hold a passing history class in both assessments;
* `blocking`: anything else, including every value that is not established.

The commit id is not ignored. The G09 identity follows content, not commits, so a descendant
commit with the same scope content is recorded and allowed, while a descendant with other
content blocks. The evaluator, the policy hash and the host have no reviewed rule that allows
a change, so a change blocks. The evaluator and `readiness/policy.py` sit inside the G09 scope,
so changing either also changes the source identity; they are still compared on their own.

G09's evidence digest is the one approved-gate digest that is not compared: its evidence
contains the commit id (handled above) while the gate's approved subject, compared through
`expected_sha256`, contains the content.
"""

from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.layout import FrozenRecord, Sha256
from biorag_repro.readiness.models import (
    HISTORY_GATE,
    MANDATORY_GATES,
    ReadinessReport,
)

Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Verdict = Literal["identical", "compatible_recorded", "blocking"]
SOURCE_GATE: Final = "G09_scientific_source_revision"

# candidate field -> component name
_SCIENTIFIC: Final = {
    "corpus_content_sha256": "corpus",
    "benchmark_release_sha256": "benchmark",
    "freeze_record_sha256": "freeze_record",
    "python_lock_sha256": "python_lock",
    "renv_lock_sha256": "renv_lock",
    "generation_seed_registry_sha256": "generation_seed_registry",
    "benchmark_seed_registry_sha256": "benchmark_seed_registry",
    "scientific_source_identity_sha256": "scientific_source_identity",
}
_SCOPE_FIELDS: Final = (
    "campaign_plan_sha256",
    "batch_id",
    "matrix_arm",
    "resolved_config_sha256",
    "ownership_sha256",
    "key_set_sha256",
    "inputs_set_sha256",
)


class BatchScope(FrozenRecord):
    """The batch identities an assessment is about. `inputs_set_sha256` is None until bound."""

    campaign_plan_sha256: Sha256
    batch_id: Sha256
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256
    ownership_sha256: Sha256
    key_set_sha256: Sha256
    inputs_set_sha256: Sha256 | None


class ComponentFinding(FrozenRecord):
    component: Nonblank
    verdict: Verdict
    initial: str | None
    later: str | None
    note: str


class ContinuityAssessment(FrozenRecord):
    schema_version: Literal["1"] = "1"
    findings: list[ComponentFinding]
    verdict: Literal["continuous", "blocked"]
    initial_binding_sha256: Sha256
    later_binding_sha256: Sha256
    initial_scientific_identity_sha256: Sha256
    later_scientific_identity_sha256: Sha256

    @model_validator(mode="after")
    def consistent(self) -> Self:
        blocking = any(f.verdict == "blocking" for f in self.findings)
        same = self.initial_scientific_identity_sha256 == self.later_scientific_identity_sha256
        if (self.verdict == "continuous") != (not blocking and same):
            raise ValueError("verdict does not follow from the findings and identity digests")
        return self


def _digest(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _approval_gates() -> list[str]:
    return [g for g in MANDATORY_GATES if g != HISTORY_GATE]


def _gate_parts(report: ReadinessReport, gate: str) -> tuple[str, str, str | None, str | None]:
    result = next(g for g in report.gates if g.gate_id == gate)
    evidence = None if gate == SOURCE_GATE else result.evidence_digest
    return result.status, result.reason_code, result.expected_sha256, evidence


def scientific_identity_sha256(report: ReadinessReport, scope: BatchScope) -> str:
    """Digest of everything that must not change between authorizations of one batch.

    Excludes the commit id, the evaluator, the policy hash, the host, G13 and volatile report
    fields; each of those is judged separately by `compare_assessments`.
    """
    candidate = report.candidate.model_dump(mode="json")
    candidate.pop("head_sha")
    gates = {}
    for gate in _approval_gates():
        status, reason, expected, evidence = _gate_parts(report, gate)
        gates[gate] = {
            "status": status,
            "reason_code": reason,
            "expected_sha256": expected,
            "evidence_digest": evidence,
        }
    return _digest(
        {
            "kind": "final_test_scientific_identity",
            "schema_version": "1",
            "candidate": candidate,
            "gates": gates,
            "batch": scope.model_dump(mode="json"),
        }
    )


def _same(
    component: str, initial: str | None, later: str | None, note: str = ""
) -> ComponentFinding:
    if initial is None or later is None:
        return ComponentFinding(
            component=component,
            verdict="blocking",
            initial=initial,
            later=later,
            note="not established" + (f": {note}" if note else ""),
        )
    verdict: Verdict = "identical" if initial == later else "blocking"
    return ComponentFinding(
        component=component, verdict=verdict, initial=initial, later=later, note=note
    )


def _configuration_finding(initial: ReadinessReport, later: ReadinessReport) -> ComponentFinding:
    a, b = (
        initial.candidate.configuration_sha256_by_arm,
        later.candidate.configuration_sha256_by_arm,
    )
    differing = sorted(arm for arm in a.keys() | b.keys() if a.get(arm) != b.get(arm))
    return ComponentFinding(
        component="configuration_by_arm",
        verdict="blocking" if differing or not a else "identical",
        initial=_digest(a),
        later=_digest(b),
        note=", ".join(differing) if differing else ("no configuration" if not a else ""),
    )


def _history_finding(initial: ReadinessReport, later: ReadinessReport) -> ComponentFinding:
    gates = [next(g for g in r.gates if g.gate_id == HISTORY_GATE) for r in (initial, later)]
    # a passed G13 is verified and carries a history class (GateResult enforces both)
    holds = all(g.status == "passed" for g in gates)
    if not holds:
        verdict: Verdict = "blocking"
    elif (gates[0].reason_code, gates[0].evidence_digest) == (
        gates[1].reason_code,
        gates[1].evidence_digest,
    ):
        verdict = "identical"
    else:
        verdict = "compatible_recorded"
    return ComponentFinding(
        component="g13_history",
        verdict=verdict,
        initial=f"{gates[0].status}/{gates[0].reason_code}/{gates[0].evidence_digest}",
        later=f"{gates[1].status}/{gates[1].reason_code}/{gates[1].evidence_digest}",
        note="history evidence evolves with the registry and must hold independently each time",
    )


def _commit_finding(initial: ReadinessReport, later: ReadinessReport) -> ComponentFinding:
    a, b = initial.candidate, later.candidate
    finding = _same("head_sha", a.head_sha, b.head_sha)
    if finding.verdict != "blocking" or a.head_sha is None or b.head_sha is None:
        return finding
    same_content = (
        a.scientific_source_identity_sha256 is not None
        and a.scientific_source_identity_sha256 == b.scientific_source_identity_sha256
    )
    if not same_content:
        return finding.model_copy(update={"note": "the scientific source content also differs"})
    return finding.model_copy(
        update={
            "verdict": "compatible_recorded",
            "note": "new commit, identical scientific source content",
        }
    )


def compare_assessments(
    initial: ReadinessReport,
    later: ReadinessReport,
    initial_scope: BatchScope,
    later_scope: BatchScope,
) -> ContinuityAssessment:
    """Classify every authorization-relevant difference between two verified reports.

    The caller has already verified both reports against the policy each was produced under.
    """
    findings: list[ComponentFinding] = []
    ready = (initial.decision, later.decision)
    findings.append(
        ComponentFinding(
            component="decision",
            verdict="identical" if ready == ("READY", "READY") else "blocking",
            initial=initial.decision,
            later=later.decision,
            note="both assessments must be READY",
        )
    )
    for field, component in _SCIENTIFIC.items():
        findings.append(
            _same(component, getattr(initial.candidate, field), getattr(later.candidate, field))
        )
    findings.append(_configuration_finding(initial, later))
    for gate in _approval_gates():
        a, b = _gate_parts(initial, gate), _gate_parts(later, gate)
        verdict: Verdict = "identical" if a == b else "blocking"
        findings.append(
            ComponentFinding(
                component=f"approval:{gate}",
                verdict=verdict,
                initial=_digest(list(a)),
                later=_digest(list(b)),
                note="status, reason, approved digest and evidence",
            )
        )
    findings.append(_history_finding(initial, later))
    findings.append(_commit_finding(initial, later))
    findings.append(
        _same(
            "evaluator",
            _digest(initial.evaluator.model_dump(mode="json")),
            _digest(later.evaluator.model_dump(mode="json")),
        )
    )
    findings.append(_same("policy", initial.policy_sha256, later.policy_sha256))
    findings.append(
        _same(
            "host",
            _digest(initial.host.model_dump(mode="json")),
            _digest(later.host.model_dump(mode="json")),
        )
    )
    for field in _SCOPE_FIELDS:
        a, b = getattr(initial_scope, field), getattr(later_scope, field)
        if field == "inputs_set_sha256":
            verdict = "identical" if a == b else "blocking"  # None on both sides is "not yet bound"
            findings.append(
                ComponentFinding(
                    component=f"batch:{field}", verdict=verdict, initial=a, later=b, note=""
                )
            )
        else:
            findings.append(_same(f"batch:{field}", a, b))
    approved = all(
        r.candidate.configuration_sha256_by_arm.get(s.matrix_arm) == s.resolved_config_sha256
        for r, s in ((initial, initial_scope), (later, later_scope))
    )
    findings.append(
        ComponentFinding(
            component="batch:config_is_approved_for_arm",
            verdict="identical" if approved else "blocking",
            initial=initial_scope.resolved_config_sha256,
            later=later_scope.resolved_config_sha256,
            note="the batch configuration must be the approved one for its arm",
        )
    )
    blocked = any(f.verdict == "blocking" for f in findings)
    before = scientific_identity_sha256(initial, initial_scope)
    after = scientific_identity_sha256(later, later_scope)
    return ContinuityAssessment(
        findings=findings,
        verdict="blocked" if blocked else "continuous",
        initial_binding_sha256=initial.binding_sha256,
        later_binding_sha256=later.binding_sha256,
        initial_scientific_identity_sha256=before,
        later_scientific_identity_sha256=after,
    )


def continuity_assessment_sha256(assessment: ContinuityAssessment) -> str:
    """Identity of the stored assessment: SHA-256 of its canonical project bytes."""
    return sha256_bytes(layout.serialize(assessment))
