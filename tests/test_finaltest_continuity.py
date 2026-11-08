"""Scientific continuity between an initial and a later readiness assessment of one batch."""

import hashlib
import json

import pytest
from readiness_support import (
    all_passed,
    candidate,
    digest,
    evaluator,
    evidence,
    fixture_policy,
    host,
    passed,
    report,
)

from biorag_repro.finaltest import layout
from biorag_repro.finaltest.continuity import (
    BatchScope,
    ContinuityAssessment,
    compare_assessments,
    continuity_assessment_sha256,
    scientific_identity_sha256,
)
from biorag_repro.readiness.models import MANDATORY_GATES, EvidenceRef, evidence_digest
from biorag_repro.readiness.policy import GATE_SLOT, SLOT_NAMES, ApprovalPolicy, ApprovalSlots

POLICY = fixture_policy()
G09 = "G09_scientific_source_revision"
G13 = "G13_final_test_history_integrity"
APPROVAL_GATES = [g for g in MANDATORY_GATES if g != G13]


def scope(**changes):
    values = {
        "campaign_plan_sha256": digest("plan"),
        "batch_id": digest("batch"),
        "matrix_arm": "confirmatory",
        "resolved_config_sha256": digest("config:confirmatory"),
        "ownership_sha256": digest("ownership"),
        "key_set_sha256": digest("keys"),
        "inputs_set_sha256": None,
    }
    return BatchScope(**(values | changes))


def results(policy=POLICY, **replaced):
    return [replaced.get(r.gate_id, r) for r in all_passed(policy)]


def make(policy=POLICY, nonce="0" * 32, replaced=None, **parts):
    return report(policy, results(policy, **(replaced or {})), nonce=nonce, **parts)


def g13(label, reason="same_batch_valid_history"):
    refs = [
        EvidenceRef(
            kind="fixture_artifact",
            locator=f"fixture/{label}.json",
            sha256=digest(label),
            subject_kind="stored_file_bytes",
        )
    ]
    base = passed(G13, POLICY)
    return base.model_copy(
        update={"reason_code": reason, "evidence": refs, "evidence_digest": evidence_digest(refs)}
    )


def assess(initial, later, a=None, b=None):
    return compare_assessments(initial, later, a or scope(), b or scope())


def blocked_components(result):
    return {f.component for f in result.findings if f.verdict == "blocking"}


def policy_with(**slots):
    values = {name: digest(f"slot:{name}") for name in SLOT_NAMES} | slots
    return ApprovalPolicy(policy_version="fixture-policy-0", slots=ApprovalSlots(**values))


# ---- what stays the same ---------------------------------------------------------------------


def test_the_same_assessment_repeated_is_continuous():
    result = assess(make(), make(nonce="1" * 32))
    assert result.verdict == "continuous"
    assert blocked_components(result) == set()


def test_a_legitimate_registry_append_changes_g13_and_remains_continuous():
    initial = make(replaced={G13: g13("genesis", "no_prior_final_test_execution")})
    later = make(replaced={G13: g13("after-first-append")})
    result = assess(initial, later)
    assert initial.binding_sha256 != later.binding_sha256  # the binding is not loosened
    finding = next(f for f in result.findings if f.component == "g13_history")
    assert finding.verdict == "compatible_recorded"
    assert result.verdict == "continuous"


@pytest.mark.parametrize(
    ("status", "verification", "reason"),
    [
        ("failed", "verified", "unauthorized_final_test_artifact"),
        ("failed", "verified", "conflicting_final_test_artifact"),
        ("unavailable", "unverified", "history_inventory_incomplete"),
        ("unavailable", "unverified", "final_test_inventory_authority_not_established"),
    ],
)
def test_g13_is_not_dropped_it_must_hold_in_the_later_assessment(status, verification, reason):
    failing = g13("x", reason).model_copy(
        update={
            "status": status,
            "verification": verification,
            "evidence": [],
            "evidence_digest": None,
        }
    )
    later = report(POLICY, results(**{G13: failing}))
    result = assess(make(), later)
    assert result.verdict == "blocked" and "g13_history" in blocked_components(result)


def test_a_not_ready_later_assessment_is_blocked():
    from readiness_support import with_status

    later = report(POLICY, with_status(POLICY, "G02_benchmark_freeze", "failed"))
    result = assess(make(), later)
    assert result.verdict == "blocked" and "decision" in blocked_components(result)


# ---- scientific identity: any change blocks --------------------------------------------------

SCIENTIFIC_FIELDS = {
    "corpus_content_sha256": "corpus",
    "benchmark_release_sha256": "benchmark",
    "freeze_record_sha256": "freeze_record",
    "python_lock_sha256": "python_lock",
    "renv_lock_sha256": "renv_lock",
    "generation_seed_registry_sha256": "generation_seed_registry",
    "benchmark_seed_registry_sha256": "benchmark_seed_registry",
    "scientific_source_identity_sha256": "scientific_source_identity",
}


@pytest.mark.parametrize(("field", "component"), sorted(SCIENTIFIC_FIELDS.items()))
def test_a_changed_scientific_identity_blocks(field, component):
    later = make(candidate=candidate(**{field: digest("changed")}))
    result = assess(make(), later)
    assert result.verdict == "blocked"
    assert component in blocked_components(result)


@pytest.mark.parametrize("field", sorted(SCIENTIFIC_FIELDS))
def test_an_unestablished_identity_blocks_even_when_both_sides_agree(field):
    side = make(candidate=candidate(**{field: None}))
    result = assess(side, make(candidate=candidate(**{field: None})))
    assert result.verdict == "blocked"
    assert SCIENTIFIC_FIELDS[field] in blocked_components(result)


def test_a_changed_configuration_for_any_arm_blocks():
    two = {"confirmatory": digest("config:confirmatory"), "closed_book": digest("config:cb")}
    initial = make(candidate=candidate(configuration_sha256_by_arm=two))
    for later_map in (
        two | {"closed_book": digest("config:other")},
        {"confirmatory": two["confirmatory"]},  # an arm removed
        two | {"dense_robustness": digest("config:dense")},  # an arm added
    ):
        later = make(candidate=candidate(configuration_sha256_by_arm=later_map))
        result = assess(initial, later)
        assert result.verdict == "blocked"
        assert "configuration_by_arm" in blocked_components(result)


@pytest.mark.parametrize("gate", APPROVAL_GATES)
def test_a_changed_approval_of_any_gate_blocks(gate):
    later_policy = policy_with(**{GATE_SLOT[gate]: digest("a different approval")})
    result = assess(make(), make(later_policy))
    assert f"approval:{gate}" in blocked_components(result)
    assert result.verdict == "blocked"


@pytest.mark.parametrize("gate", [g for g in APPROVAL_GATES if g != G09])
def test_changed_evidence_of_an_approved_gate_blocks_except_for_the_commit_bound_g09(gate):
    refs = evidence(gate) + [
        EvidenceRef(
            kind="fixture_artifact",
            locator=f"fixture/{gate}-more.json",
            sha256=digest("more"),
            subject_kind="stored_file_bytes",
        )
    ]
    changed = passed(gate, POLICY).model_copy(
        update={"evidence": refs, "evidence_digest": evidence_digest(refs)}
    )
    result = assess(make(), make(replaced={gate: changed}))
    assert f"approval:{gate}" in blocked_components(result)


# ---- the operational identities are not silently excluded --------------------------------------


def test_a_new_commit_with_identical_scientific_source_is_recorded_not_ignored():
    initial = make(candidate=candidate(head_sha="a" * 40))
    later = make(candidate=candidate(head_sha="b" * 40))
    result = assess(initial, later)
    finding = next(f for f in result.findings if f.component == "head_sha")
    assert (finding.initial, finding.later) == ("a" * 40, "b" * 40)
    assert finding.verdict == "compatible_recorded"
    assert result.verdict == "continuous"


def test_a_new_commit_with_changed_scientific_source_is_blocked():
    initial = make(candidate=candidate(head_sha="a" * 40))
    later = make(
        candidate=candidate(head_sha="b" * 40, scientific_source_identity_sha256=digest("new"))
    )
    result = assess(initial, later)
    assert {"head_sha", "scientific_source_identity"} <= blocked_components(result)


def test_a_new_commit_is_blocked_when_the_source_identity_is_not_established():
    kwargs = {"scientific_source_identity_sha256": None}
    initial = make(candidate=candidate(head_sha="a" * 40, **kwargs))
    later = make(candidate=candidate(head_sha="b" * 40, **kwargs))
    assert "head_sha" in blocked_components(assess(initial, later))


def test_a_missing_commit_id_is_blocked():
    result = assess(
        make(candidate=candidate(head_sha=None)), make(candidate=candidate(head_sha=None))
    )
    assert "head_sha" in blocked_components(result)


def test_a_changed_evaluator_is_blocked():
    later = make(evaluator=evaluator(code_sha256=digest("other evaluator")))
    assert "evaluator" in blocked_components(assess(make(), later))
    later = make(evaluator=evaluator(version="fixture-evaluator-1"))
    assert "evaluator" in blocked_components(assess(make(), later))


def test_a_changed_policy_identity_is_blocked_even_when_every_approval_is_the_same():
    later = make(ApprovalPolicy(policy_version="fixture-policy-1", slots=POLICY.slots))
    result = assess(make(), later)
    assert blocked_components(result) == {"policy"}


def test_a_changed_host_is_blocked_until_a_rule_for_it_exists():
    for later_host in (
        host(hostname="other-host"),
        host(gpu_names=["Fixture GPU", "Second GPU"]),
        host(operating_system="Darwin"),
        host(machine="aarch64"),
    ):
        assert "host" in blocked_components(assess(make(), make(host=later_host)))


# ---- the batch -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["campaign_plan_sha256", "batch_id", "ownership_sha256", "key_set_sha256", "matrix_arm"],
)
def test_a_different_batch_identity_blocks(field):
    value = "closed_book" if field == "matrix_arm" else digest("another")
    result = assess(make(), make(), scope(), scope(**{field: value}))
    assert result.verdict == "blocked"
    assert f"batch:{field}" in blocked_components(result)


def test_the_batch_configuration_must_be_the_approved_one_for_its_arm():
    result = assess(make(), make(), scope(), scope(resolved_config_sha256=digest("other")))
    assert result.verdict == "blocked"
    assert "batch:resolved_config_sha256" in blocked_components(result)
    wrong = scope(resolved_config_sha256=digest("not approved"))
    assert "batch:config_is_approved_for_arm" in blocked_components(
        assess(make(), make(), wrong, wrong)
    )


def test_unbound_inputs_are_allowed_only_when_both_sides_are_unbound_or_equal():
    bound = scope(inputs_set_sha256=digest("inputs"))
    assert assess(make(), make(), scope(), scope()).verdict == "continuous"
    assert assess(make(), make(), bound, bound).verdict == "continuous"
    assert "batch:inputs_set_sha256" in blocked_components(assess(make(), make(), scope(), bound))
    assert "batch:inputs_set_sha256" in blocked_components(assess(make(), make(), bound, scope()))
    other = scope(inputs_set_sha256=digest("other inputs"))
    assert "batch:inputs_set_sha256" in blocked_components(assess(make(), make(), bound, other))


# ---- the digest -------------------------------------------------------------------------------


def oracle_identity(rep, batch):
    """The definition of the scientific identity, restated with plain json and hashlib."""
    cand = json.loads(rep.candidate.model_dump_json())
    cand.pop("head_sha")
    gates = {}
    for g in rep.gates:
        if g.gate_id == G13:
            continue
        gates[g.gate_id] = {
            "status": g.status,
            "reason_code": g.reason_code,
            "expected_sha256": g.expected_sha256,
            "evidence_digest": None if g.gate_id == G09 else g.evidence_digest,
        }
    payload = {
        "kind": "final_test_scientific_identity",
        "schema_version": "1",
        "candidate": cand,
        "gates": gates,
        "batch": json.loads(batch.model_dump_json()),
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def test_the_identity_digest_matches_an_independent_restatement():
    rep = make()
    assert scientific_identity_sha256(rep, scope()) == oracle_identity(rep, scope())


def test_the_identity_digest_ignores_only_volatile_and_separately_judged_parts():
    base = scientific_identity_sha256(make(), scope())
    same = [
        make(nonce="2" * 32),
        make(candidate=candidate(head_sha="c" * 40)),
        make(replaced={G13: g13("another history")}),
        make(evaluator=evaluator(code_sha256=digest("x"))),
        make(host=host(hostname="elsewhere")),
    ]
    assert {scientific_identity_sha256(r, scope()) for r in same} == {base}


@pytest.mark.parametrize("field", sorted(SCIENTIFIC_FIELDS))
def test_every_scientific_field_moves_the_digest(field):
    other = make(candidate=candidate(**{field: digest("different")}))
    assert scientific_identity_sha256(other, scope()) != scientific_identity_sha256(make(), scope())


def test_a_batch_change_moves_the_digest():
    assert scientific_identity_sha256(make(), scope(batch_id=digest("x"))) != (
        scientific_identity_sha256(make(), scope())
    )


def test_a_continuous_assessment_pairs_equal_digests_and_a_blocked_one_does_not_pretend():
    ok = assess(make(), make(nonce="3" * 32))
    assert ok.initial_scientific_identity_sha256 == ok.later_scientific_identity_sha256
    bad = assess(make(), make(candidate=candidate(corpus_content_sha256=digest("c2"))))
    assert bad.initial_scientific_identity_sha256 != bad.later_scientific_identity_sha256


def test_the_assessment_is_a_stable_canonical_record():
    result = assess(make(), make(nonce="4" * 32))
    assert (
        continuity_assessment_sha256(result) == hashlib.sha256(layout.serialize(result)).hexdigest()
    )
    again = assess(make(), make(nonce="5" * 32))  # the nonce is not part of any finding
    assert again == result
    assert ContinuityAssessment.model_validate_json(layout.serialize(result)) == result


def test_every_blocked_assessment_names_what_blocked_it_and_a_continuous_one_names_nothing():
    ok = assess(make(), make())
    assert ok.verdict == "continuous" and not blocked_components(ok)
    bad = assess(make(), make(host=host(hostname="x")))
    assert bad.verdict == "blocked" and blocked_components(bad) == {"host"}


# ---- details of single findings -----------------------------------------------------------------


def finding(result, component):
    return next(f for f in result.findings if f.component == component)


def test_a_value_established_on_one_side_only_is_named_as_not_established():
    later = make(candidate=candidate(corpus_content_sha256=None))
    got = finding(assess(make(), later), "corpus")
    assert got.verdict == "blocking" and got.note.startswith("not established")
    got = finding(assess(later, make()), "corpus")
    assert got.verdict == "blocking" and got.note.startswith("not established")


def test_an_empty_configuration_is_blocked_even_when_both_sides_agree():
    empty = make(candidate=candidate(configuration_sha256_by_arm={}))
    got = finding(assess(empty, empty), "configuration_by_arm")
    assert got.verdict == "blocking" and got.note == "no configuration"


def test_g13_with_the_same_class_and_evidence_is_identical_and_other_evidence_is_recorded():
    same = make(replaced={G13: g13("one")})
    assert finding(assess(same, make(replaced={G13: g13("one")})), "g13_history").verdict == (
        "identical"
    )
    other = make(replaced={G13: g13("two")})
    assert finding(assess(same, other), "g13_history").verdict == "compatible_recorded"


def test_the_batch_configuration_is_checked_against_the_approval_on_each_side_separately():
    good, wrong = scope(), scope(resolved_config_sha256=digest("not approved"))
    for a, b in ((wrong, good), (good, wrong)):
        got = finding(assess(make(), make(), a, b), "batch:config_is_approved_for_arm")
        assert got.verdict == "blocking"


def test_a_stored_assessment_cannot_contradict_itself():
    ok = assess(make(), make())
    blocked = assess(make(), make(host=host(hostname="x")))
    base = ok.model_dump(mode="json")
    with pytest.raises(ValueError):  # continuous, yet a finding blocks
        ContinuityAssessment.model_validate(
            base | {"findings": blocked.model_dump(mode="json")["findings"]}
        )
    with pytest.raises(ValueError):  # continuous, yet the identities differ
        ContinuityAssessment.model_validate(
            base | {"later_scientific_identity_sha256": digest("x")}
        )
    with pytest.raises(ValueError):  # blocked, though nothing blocks and the identities agree
        ContinuityAssessment.model_validate(base | {"verdict": "blocked"})
