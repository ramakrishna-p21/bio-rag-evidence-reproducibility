"""Evidence integrity, contract validity and policy approval are three different questions."""

import pytest
from readiness_support import digest, fixture_policy

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness.decide import decide
from biorag_repro.readiness.gather import Gathered, to_result
from biorag_repro.readiness.policy import REAL_POLICY, ApprovalPolicy, ApprovalSlots

GATE = "G01_corpus_sources"
SUBJECT = {"kind": "x", "n": 1}
SUBJECT_SHA = sha256_bytes(canonical_json_bytes(SUBJECT))


def with_slot(value):
    return ApprovalPolicy(policy_version="t", slots=ApprovalSlots(corpus_identity=value))


def ok():
    return Gathered("ok", "checks_passed", "scope", subject=dict(SUBJECT))


def test_valid_evidence_with_unset_slot_is_pending_and_keeps_the_observed_subject():
    result = to_result(GATE, ok(), REAL_POLICY)
    assert (result.status, result.verification) == ("pending", "unverified")
    assert result.reason_code == "verified_pending_policy_approval"
    assert result.expected_sha256 is None
    assert SUBJECT_SHA in {r.sha256 for r in result.evidence}


def test_observed_subject_is_not_written_into_the_policy():
    to_result(GATE, ok(), REAL_POLICY)
    assert REAL_POLICY.slots.corpus_identity is None


def test_populated_slot_equal_to_subject_passes():
    policy = with_slot(SUBJECT_SHA)
    result = to_result(GATE, ok(), policy)
    assert (result.status, result.verification) == ("passed", "verified")
    assert result.expected_sha256 == SUBJECT_SHA


def test_populated_slot_that_differs_fails_with_a_policy_mismatch():
    result = to_result(GATE, ok(), with_slot(digest("other")))
    assert result.status == "failed" and result.reason_code == "policy_mismatch"
    assert result.expected_sha256 == digest("other")


@pytest.mark.parametrize("slot", [None, SUBJECT_SHA, digest("other")])
def test_contract_violation_fails_whatever_the_slot_says(slot):
    gathered = Gathered("failed", "contract_violated", "scope")
    result = to_result(GATE, gathered, with_slot(slot))
    assert result.status == "failed" and result.reason_code == "contract_violated"


@pytest.mark.parametrize("slot", [None, SUBJECT_SHA])
def test_uninspectable_evidence_is_unavailable_never_pending_or_passed(slot):
    result = to_result(GATE, Gathered("unavailable", "tool_missing", "s"), with_slot(slot))
    assert (result.status, result.verification) == ("unavailable", "unverified")


@pytest.mark.parametrize("slot", [None, SUBJECT_SHA])
def test_missing_prerequisite_is_pending_even_if_a_slot_is_populated(slot):
    result = to_result(GATE, Gathered("absent", "freeze_absent", "s"), with_slot(slot))
    assert result.status == "pending" and result.evidence == []


def test_ok_without_a_subject_is_an_error():
    with pytest.raises(ValueError):
        to_result(GATE, Gathered("ok", "x", "s"), REAL_POLICY)


def test_results_feed_the_existing_decision_without_a_ready_outcome():
    from biorag_repro.readiness.models import MANDATORY_GATES

    results = [to_result(g, ok(), REAL_POLICY) for g in MANDATORY_GATES]
    assert decide(results, REAL_POLICY).decision == "NOT_READY"
    assert fixture_policy()
