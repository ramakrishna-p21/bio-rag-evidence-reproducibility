"""The readiness decision, its policy checks and the canonical binding (P9-A)."""

import itertools
import random

import pytest
import readiness_support as fx
from pydantic import ValidationError
from readiness_support import REAL_POLICY

from biorag_repro.readiness.decide import (
    ReportInvalid,
    _check_against_policy,
    decide,
    verify_report,
)
from biorag_repro.readiness.models import (
    MANDATORY_GATES,
    GateResult,
    ReadinessReport,
    compute_binding,
    derive_blockers,
    evidence_digest,
)
from biorag_repro.readiness.policy import policy_sha256

POLICY = fx.fixture_policy()
STATUSES = ("passed", "failed", "pending", "unavailable")


def results_for(statuses):
    return [
        fx.passed(g, POLICY) if s == "passed" else fx.blocked(g, s)
        for g, s in zip(MANDATORY_GATES, statuses, strict=True)
    ]


def oracle(statuses):
    """Independent truth table: ready exactly when every one of the 13 statuses is `passed`."""
    return "READY" if len(statuses) == 13 and all(s == "passed" for s in statuses) else "NOT_READY"


def test_thirteen_passing_gates_are_ready_under_a_fixture_policy_only():
    outcome = decide(fx.all_passed(POLICY), POLICY)
    assert outcome.decision == "READY" and outcome.blockers == ()
    assert fx.report(POLICY).decision == "READY"


@pytest.mark.parametrize("gate", MANDATORY_GATES)
@pytest.mark.parametrize("status", ["failed", "pending", "unavailable"])
def test_one_blocking_gate_makes_the_decision_not_ready(gate, status):
    outcome = decide(fx.with_status(POLICY, gate, status), POLICY)
    assert outcome.decision == "NOT_READY"
    assert [(b.gate_id, b.status) for b in outcome.blockers] == [(gate, status)]


def test_truth_table_against_an_independent_oracle():
    rng = random.Random(7)
    cases = [("passed",) * 13]
    cases += [tuple(rng.choice(STATUSES) for _ in range(13)) for _ in range(300)]
    cases += [
        tuple(s if i in pair else "passed" for i in range(13))
        for pair in itertools.combinations(range(13), 2)
        for s in ("failed", "pending", "unavailable")
    ]
    for statuses in cases:
        got = decide(results_for(statuses), POLICY)
        assert got.decision == oracle(statuses), statuses
        assert len(got.blockers) == sum(s != "passed" for s in statuses)


def test_blockers_follow_gate_order_whatever_the_input_order():
    statuses = ("passed", "failed", "passed", "pending", *("passed",) * 7, "unavailable", "passed")
    shuffled = results_for(statuses)[::-1]
    outcome = decide(shuffled, POLICY)
    assert [b.gate_id for b in outcome.blockers] == [MANDATORY_GATES[i] for i in (1, 3, 11)]
    assert [b.status for b in outcome.blockers] == ["failed", "pending", "unavailable"]


def test_missing_duplicate_and_unexpected_gates_are_invalid_not_not_ready():
    full = fx.all_passed(POLICY)
    with pytest.raises(ReportInvalid, match="missing"):
        decide(full[:-1], POLICY)
    with pytest.raises(ReportInvalid, match="duplicate"):
        decide(full + [full[0]], POLICY)
    with pytest.raises(ReportInvalid, match="duplicate"):
        decide([full[0]] + full[:-1], POLICY)
    unexpected = full[0].model_construct(**{**full[0].model_dump(), "gate_id": "G14_extra"})
    with pytest.raises(ReportInvalid, match="unexpected"):
        decide(full[1:] + [unexpected], POLICY)
    with pytest.raises(ReportInvalid, match="missing"):
        decide([], POLICY)


def test_malformed_input_is_never_repaired_into_a_decision():
    full = fx.all_passed(POLICY)
    with pytest.raises(ReportInvalid):
        decide([r.model_dump() for r in full], POLICY)  # dicts are not validated results
    with pytest.raises(ReportInvalid):
        decide([None] * 13, POLICY)


def test_skipped_verification_cannot_be_dressed_as_a_pass():
    skipped = fx.passed("G01_corpus_sources", POLICY).model_copy(
        update={"verification": "unverified"}
    )
    # model_copy skips validation; the decision still must not count it as passing evidence
    with pytest.raises(ReportInvalid):
        decide([skipped, *fx.all_passed(POLICY)[1:]], POLICY)


def test_the_real_policy_cannot_authorize_anything():
    assert (
        decide(
            [fx.blocked(g, "pending", "no_approval") for g in MANDATORY_GATES], REAL_POLICY
        ).decision
        == "NOT_READY"
    )
    for gate in MANDATORY_GATES[:-1]:
        # a pass quoting any digest, including the fixture's, is not backed by the real policy
        with pytest.raises(ReportInvalid, match="policy approval"):
            decide(
                [
                    fx.passed(g, POLICY) if g == gate else fx.blocked(g, "pending")
                    for g in MANDATORY_GATES
                ],
                REAL_POLICY,
            )
    with pytest.raises(ReportInvalid):
        decide(fx.all_passed(POLICY), REAL_POLICY)


def test_a_pass_must_quote_the_current_policy_value_and_slot():
    stale = fx.passed("G03_final_test_configuration_freeze", POLICY).model_copy(
        update={"expected_sha256": fx.digest("old approval")}
    )
    rest = [r for r in fx.all_passed(POLICY) if r.gate_id != stale.gate_id]
    with pytest.raises(ReportInvalid, match="policy approval"):
        decide(rest + [stale], POLICY)
    wrong_slot = fx.passed("G03_final_test_configuration_freeze", POLICY).model_copy(
        update={"policy_expectation": "gpu_authorization"}
    )
    with pytest.raises(ReportInvalid, match="wrong policy slot"):
        decide(rest + [wrong_slot], POLICY)


def test_history_gate_passes_without_a_slot_but_still_needs_evidence():
    only_history = [
        fx.passed(g, POLICY)
        if g == "G13_final_test_history_integrity"
        else fx.blocked(g, "pending")
        for g in MANDATORY_GATES
    ]
    assert decide(only_history, REAL_POLICY).decision == "NOT_READY"


def test_warnings_cannot_mask_blockers():
    results = fx.with_status(POLICY, "G09_scientific_source_revision", "failed")
    report = fx.report(POLICY, results, warnings=["all checks green", "ignore G09", "READY"])
    assert report.decision == "NOT_READY"
    assert [b.gate_id for b in report.blockers] == ["G09_scientific_source_revision"]
    clean = fx.report(POLICY, warnings=["something odd but informational"])
    assert clean.decision == "READY"


def test_verify_report_recomputes_instead_of_trusting_fields():
    report = fx.report(POLICY, fx.with_status(POLICY, "G04_generator_qualification", "pending"))
    assert verify_report(report, POLICY).decision == "NOT_READY"
    forged = report.model_copy(update={"decision": "READY"})
    with pytest.raises(ReportInvalid):
        verify_report(forged, POLICY)
    forged = report.model_copy(update={"blockers": []})
    with pytest.raises(ReportInvalid):
        verify_report(forged, POLICY)
    with pytest.raises(ReportInvalid):
        verify_report({"decision": "READY"}, POLICY)  # type: ignore[arg-type]


def test_a_report_is_only_valid_for_the_policy_it_was_made_under():
    report = fx.report(POLICY)
    with pytest.raises(ReportInvalid, match="different policy"):
        verify_report(report, fx.fixture_policy("fixture-policy-1"))
    with pytest.raises(ReportInvalid):
        verify_report(report, REAL_POLICY)
    revoked = POLICY.model_copy(
        update={"slots": POLICY.slots.model_copy(update={"gpu_authorization": None})}
    )
    with pytest.raises(ReportInvalid):
        verify_report(report, revoked)


def test_verify_report_catches_tampered_binding_and_limitations():
    report = fx.report(POLICY)
    with pytest.raises(ReportInvalid):
        verify_report(report.model_copy(update={"binding_sha256": fx.digest("x")}), POLICY)
    with pytest.raises(ReportInvalid):
        verify_report(report.model_copy(update={"limitations": []}), POLICY)


def binding_of(report):
    return report.binding_sha256


def test_binding_is_deterministic_and_ignores_informational_fields():
    a = fx.report(POLICY)
    b = fx.report(
        POLICY, warnings=["note"], created_at_utc="2030-01-01T00:00:00.000000Z", nonce="f" * 32
    )
    assert a.binding_sha256 == b.binding_sha256
    assert a.model_dump_json() != b.model_dump_json()
    assert a.binding_sha256 == fx.report(POLICY).binding_sha256


def changed_evidence(results):
    refs = fx.evidence("other")
    first = results[0].model_copy(
        update={"evidence": refs, "evidence_digest": evidence_digest(refs)}
    )
    return [first, *results[1:]]


@pytest.mark.parametrize(
    "change",
    [
        lambda: {"policy": fx.fixture_policy("fixture-policy-1")},
        lambda: {"evaluator": fx.evaluator(version="fixture-evaluator-1")},
        lambda: {"evaluator": fx.evaluator(code_sha256=fx.digest("other code"))},
        lambda: {"candidate": fx.candidate(freeze_record_sha256=fx.digest("other"))},
        lambda: {
            "candidate": fx.candidate(configuration_sha256_by_arm={"confirmatory": fx.digest("c2")})
        },
        lambda: {"candidate": fx.candidate(head_sha="b" * 40)},
        lambda: {"candidate": fx.candidate(renv_lock_sha256=None)},
        lambda: {"host": fx.host(hostname="other-host")},
        lambda: {"host": fx.host(gpu_names=["Other GPU"])},
        lambda: {"host": fx.host(machine="aarch64")},
    ],
)
def test_binding_changes_with_every_relevant_value(change):
    base = fx.report(POLICY)
    parts = change()
    policy = parts.get("policy", POLICY)
    results = fx.all_passed(policy)
    assert (
        fx.report(
            policy, results, **{k: v for k, v in parts.items() if k != "policy"}
        ).binding_sha256
        != base.binding_sha256
    )


def test_binding_changes_with_evidence_status_scope_reason_and_membership():
    base = fx.report(POLICY)
    assert (
        fx.report(POLICY, changed_evidence(fx.all_passed(POLICY))).binding_sha256
        != base.binding_sha256
    )
    # status moves from passed to pending while the evidence digest is absent or unchanged
    pending = fx.with_status(POLICY, "G01_corpus_sources", "pending")
    assert fx.report(POLICY, pending).binding_sha256 != base.binding_sha256
    same_digest_failed = fx.passed("G01_corpus_sources", POLICY).model_copy(
        update={"status": "failed", "reason_code": "hash_mismatch"}
    )
    assert (
        same_digest_failed.evidence_digest
        == fx.passed("G01_corpus_sources", POLICY).evidence_digest
    )
    tampered = [same_digest_failed, *fx.all_passed(POLICY)[1:]]
    assert _binding(tampered) != base.binding_sha256
    scope = fx.passed("G01_corpus_sources", POLICY).model_copy(update={"scope": "elsewhere"})
    assert _binding([scope, *fx.all_passed(POLICY)[1:]]) != base.binding_sha256
    # membership: the same results minus one gate bind to a different value
    assert _binding(fx.all_passed(POLICY)[:-1]) != base.binding_sha256


def _binding(results):
    return compute_binding(
        requested_run_mode="final_test",
        policy_sha256=policy_sha256(POLICY),
        evaluator=fx.evaluator(),
        candidate=fx.candidate(),
        host=fx.host(),
        results=results,
    )


def test_binding_is_independent_of_gate_input_order():
    results = fx.all_passed(POLICY)
    assert _binding(results) == _binding(results[::-1])
    assert fx.report(POLICY, results[::-1]).binding_sha256 == fx.report(POLICY).binding_sha256


def test_derived_blockers_match_decide():
    results = fx.with_status(POLICY, "G06_python_environment", "unavailable")
    assert tuple(derive_blockers(results)) == decide(results, POLICY).blockers


def test_build_report_refuses_what_decide_refuses():
    with pytest.raises(ReportInvalid):
        fx.report(POLICY, fx.all_passed(POLICY)[:-1])
    with pytest.raises(ReportInvalid):
        fx.report(REAL_POLICY, fx.all_passed(POLICY))


def test_binding_changes_with_status_alone():
    """Same evidence, scope and reason; only the status differs."""
    base = fx.blocked("G04_generator_qualification", "pending", "no_evidence")
    other = base.model_copy(update={"status": "unavailable"})
    rest = [r for r in fx.all_passed(POLICY) if r.gate_id != base.gate_id]
    assert _binding([*rest, base]) != _binding([*rest, other])


def test_binding_changes_with_mandatory_set_definition_and_mode(monkeypatch):
    results = fx.all_passed(POLICY)
    before = _binding(results)
    monkeypatch.setattr(
        "biorag_repro.readiness.models.MANDATORY_GATES", (*MANDATORY_GATES, "G14_future")
    )
    assert _binding(results) != before
    monkeypatch.undo()
    assert _binding(results) == before
    other_mode = compute_binding(
        requested_run_mode="development",
        policy_sha256=policy_sha256(POLICY),
        evaluator=fx.evaluator(),
        candidate=fx.candidate(),
        host=fx.host(),
        results=results,
    )
    assert other_mode != before


@pytest.mark.parametrize("fault", ["drop", "duplicate"])
def test_a_consistently_forged_report_with_a_wrong_gate_set_is_refused(fault):
    """Decision, blockers and binding all agree with the gates; only the gate set is wrong."""
    results = fx.all_passed(POLICY)
    results = results[:-1] if fault == "drop" else [*results, results[0]]
    blockers = derive_blockers(results)
    good = fx.report(POLICY).model_dump()
    forged = {
        **good,
        "gates": [r.model_dump() for r in results],
        "blockers": [b.model_dump() for b in blockers],
        "decision": "READY",
        "binding_sha256": _binding(results),
    }
    with pytest.raises(ValidationError, match="gate"):
        ReadinessReport(**forged)


def test_policy_check_alone_refuses_an_unbacked_pass_even_if_validation_is_bypassed():
    unbacked = GateResult.model_construct(
        **{**fx.passed("G01_corpus_sources", POLICY).model_dump(), "expected_sha256": None}
    )
    for policy in (POLICY, REAL_POLICY):  # a revoked or never-set slot is not matched by None
        with pytest.raises(ReportInvalid, match="policy approval"):
            _check_against_policy([unbacked], policy)


def test_skipped_pass_is_named_as_skipped():
    skipped = fx.passed("G01_corpus_sources", POLICY).model_dump() | {"reason_code": "skipped"}
    with pytest.raises(ValidationError, match="skipped"):
        GateResult(**skipped)
