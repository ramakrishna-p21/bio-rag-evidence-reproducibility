"""Historical authorization citations: identity, chain and the continuity they must cite."""

import ast
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError
from readiness_support import digest, fixture_policy, host, report
from test_finaltest_continuity import make, scope

import biorag_repro
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    CitationChainError,
    citation_sha256,
    verify_citation_chain,
)
from biorag_repro.finaltest.continuity import (
    compare_assessments,
    continuity_assessment_sha256,
    scientific_identity_sha256,
)

SRC = Path(biorag_repro.__file__).parent
NOW = "2026-11-08T00:00:00Z"


def initial_citation(rep=None, batch=None, **changes):
    rep = rep or make()
    batch = batch or scope()
    values = {
        "kind": "initial",
        "batch_id": batch.batch_id,
        "initial_report_sha256": digest("report-1"),
        "readiness_report_sha256": digest("report-1"),
        "policy_sha256": rep.policy_sha256,
        "evaluator_code_sha256": rep.evaluator.code_sha256,
        "report_binding_sha256": rep.binding_sha256,
        "head_sha": rep.candidate.head_sha,
        "scientific_source_identity_sha256": rep.candidate.scientific_source_identity_sha256,
        "scientific_identity_sha256": scientific_identity_sha256(rep, batch),
        "registry_head_sequence": 1,
        "registry_head_event_sha256": digest("event-1"),
        "head_authority": "genesis_anchored",
        "inputs_set_sha256": batch.inputs_set_sha256,
        "previous_citation_sha256": None,
        "continuity_assessment_sha256": None,
        "created_utc": NOW,
    }
    return AuthorizationCitation.model_validate(values | changes)


def resume_pair(previous, label="2", rep=None, batch=None, head=5, **changes):
    """(citation, assessment) for a resume that follows `previous`."""
    first_rep, rep, batch = make(), rep or make(nonce="9" * 32), batch or scope()
    assessment = compare_assessments(first_rep, rep, batch, batch)
    values = {
        "kind": "resume",
        "readiness_report_sha256": digest(f"report-{label}"),
        "report_binding_sha256": rep.binding_sha256,
        "registry_head_sequence": head,
        "registry_head_event_sha256": digest(f"event-{head}"),
        "head_authority": "not_established",
        "previous_citation_sha256": citation_sha256(previous),
        "continuity_assessment_sha256": continuity_assessment_sha256(assessment),
        "scientific_identity_sha256": assessment.later_scientific_identity_sha256,
    }
    base = previous.model_dump(mode="json") | values | changes
    return AuthorizationCitation.model_validate(base), assessment


def index(*assessments):
    return {continuity_assessment_sha256(a): a for a in assessments}


# ---- the record --------------------------------------------------------------------------------


def test_identity_is_the_sha256_of_the_stored_bytes():
    first = initial_citation()
    assert citation_sha256(first) == hashlib.sha256(layout.serialize(first)).hexdigest()
    assert citation_sha256(first) != citation_sha256(
        initial_citation(created_utc="2026-11-09T00:00:00Z")
    )


def test_the_record_carries_what_a_later_audit_needs():
    names = set(AuthorizationCitation.model_fields)
    assert {
        "initial_report_sha256",
        "readiness_report_sha256",
        "policy_sha256",
        "evaluator_code_sha256",
        "report_binding_sha256",
        "head_sha",
        "scientific_source_identity_sha256",
        "scientific_identity_sha256",
        "registry_head_sequence",
        "registry_head_event_sha256",
        "head_authority",
        "inputs_set_sha256",
        "previous_citation_sha256",
        "continuity_assessment_sha256",
    } <= names
    assert not names & {"approved", "authorized", "credential", "token", "valid_until"}


def test_an_initial_citation_has_no_predecessor_and_no_continuity_assessment():
    first = initial_citation()
    assert first.guard_scope == "final_test_batch_start"
    with pytest.raises(ValidationError):
        initial_citation(previous_citation_sha256=digest("x"))
    with pytest.raises(ValidationError):
        initial_citation(continuity_assessment_sha256=digest("x"))
    with pytest.raises(ValidationError):
        initial_citation(initial_report_sha256=digest("another report"))
    with pytest.raises(ValidationError):
        initial_citation(guard_scope="anything")


def test_a_resume_needs_a_predecessor_an_assessment_and_its_own_report():
    first = initial_citation()
    citation, _ = resume_pair(first)
    assert citation.kind == "resume"
    base = citation.model_dump(mode="json")
    for broken in (
        {"previous_citation_sha256": None},
        {"continuity_assessment_sha256": None},
        {"readiness_report_sha256": first.readiness_report_sha256},  # re-uses the old report
    ):
        with pytest.raises(ValidationError):
            AuthorizationCitation.model_validate(base | broken)


def test_genesis_authority_belongs_only_to_the_genesis_head():
    with pytest.raises(ValidationError):
        initial_citation(registry_head_sequence=3)  # claims genesis authority beyond genesis
    assert initial_citation(registry_head_sequence=3, head_authority="not_established")


def test_no_authority_label_names_independent_authority():
    with pytest.raises(ValidationError):
        initial_citation(head_authority="independent_authority")
    with pytest.raises(ValidationError):
        initial_citation(head_authority="fixture_only")


def test_the_head_must_be_a_full_commit_id():
    with pytest.raises(ValidationError):
        initial_citation(head_sha="abc123")


# ---- the chain ---------------------------------------------------------------------------------


def test_a_valid_chain_verifies():
    first = initial_citation()
    second, a2 = resume_pair(first, "2", head=5)
    third, a3 = resume_pair(second, "3", head=8)
    verify_citation_chain([first, second, third], index(a2, a3))
    verify_citation_chain([first], {})


def test_the_chain_may_not_be_empty_or_start_with_a_resume():
    first = initial_citation()
    second, a2 = resume_pair(first)
    with pytest.raises(CitationChainError):
        verify_citation_chain([], {})
    with pytest.raises(CitationChainError):
        verify_citation_chain([second], index(a2))


def test_a_broken_predecessor_link_is_refused():
    first = initial_citation()
    second, a2 = resume_pair(first, previous_citation_sha256=digest("someone else"))
    with pytest.raises(CitationChainError, match="predecessor"):
        verify_citation_chain([first, second], index(a2))


def test_rewriting_an_earlier_citation_breaks_every_later_link():
    first = initial_citation()
    second, a2 = resume_pair(first)
    rewritten = initial_citation(created_utc="2027-01-01T00:00:00Z")
    with pytest.raises(CitationChainError, match="predecessor"):
        verify_citation_chain([rewritten, second], index(a2))


def test_a_citation_of_another_batch_is_refused():
    first = initial_citation()
    second, a2 = resume_pair(first, batch_id=digest("another batch"))
    with pytest.raises(CitationChainError, match="batch"):
        verify_citation_chain([first, second], index(a2))


def test_the_original_report_identity_must_not_move():
    first = initial_citation()
    second, a2 = resume_pair(first, initial_report_sha256=digest("another original"))
    with pytest.raises(CitationChainError, match="original report"):
        verify_citation_chain([first, second], index(a2))


def test_a_report_can_be_cited_only_once():
    first = initial_citation()
    second, a2 = resume_pair(first, "2", head=5)
    third, a3 = resume_pair(second, "2", head=8)  # the same report again
    with pytest.raises(CitationChainError, match="report"):
        verify_citation_chain([first, second, third], index(a2, a3))


def test_the_registry_head_must_move_forward_at_each_resume():
    first = initial_citation(registry_head_sequence=3, head_authority="not_established")
    same, a2 = resume_pair(first, "2", head=3)
    with pytest.raises(CitationChainError, match="head"):
        verify_citation_chain([first, same], index(a2))
    earlier, a3 = resume_pair(first, "3", head=2)
    with pytest.raises(CitationChainError, match="head"):
        verify_citation_chain([first, earlier], index(a3))


def test_a_changed_scientific_identity_is_refused_at_a_resume():
    first = initial_citation()
    second, a2 = resume_pair(first, scientific_identity_sha256=digest("drifted"))
    with pytest.raises(CitationChainError, match="scientific identity"):
        verify_citation_chain([first, second], index(a2))


def test_the_cited_assessment_must_exist_match_and_be_continuous():
    first = initial_citation()
    second, a2 = resume_pair(first)
    with pytest.raises(CitationChainError, match="assessment"):
        verify_citation_chain([first, second], {})
    other = compare_assessments(make(), make(host=host(hostname="x")), scope(), scope())
    with pytest.raises(CitationChainError, match="assessment"):
        verify_citation_chain([first, second], {continuity_assessment_sha256(a2): other})


def test_a_blocked_assessment_cannot_back_a_resume():
    # a host change blocks without moving the scientific-identity digest, so only the verdict can refuse it
    blocked = compare_assessments(make(), make(host=host(hostname="elsewhere")), scope(), scope())
    assert blocked.verdict == "blocked"
    first = initial_citation()
    forged = first.model_dump(mode="json") | {
        "kind": "resume",
        "readiness_report_sha256": digest("report-2"),
        "registry_head_sequence": 5,
        "registry_head_event_sha256": digest("event-5"),
        "head_authority": "not_established",
        "previous_citation_sha256": citation_sha256(first),
        "continuity_assessment_sha256": continuity_assessment_sha256(blocked),
        "scientific_identity_sha256": blocked.later_scientific_identity_sha256,
    }
    with pytest.raises(CitationChainError, match="continuous"):
        verify_citation_chain([first, AuthorizationCitation.model_validate(forged)], index(blocked))


def test_a_changed_policy_or_evaluator_in_the_later_report_is_visible_in_its_citation():
    first = initial_citation()
    later_policy = fixture_policy("fixture-policy-1")
    later = report(later_policy)
    assert later.policy_sha256 != first.policy_sha256
    assessment = compare_assessments(make(), later, scope(), scope())
    assert assessment.verdict == "blocked"  # the citation could not be backed by it


# ---- a citation is a record, not a credential -------------------------------------------------


def test_the_citation_type_is_used_only_by_the_contract_modules_and_nothing_executes_it():
    # audit.py reads citations to compare a stored run with them; it verifies and grants nothing
    allowed = {"batches.py", "audit.py", "inventory.py", "start.py"}
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        used = any(
            (isinstance(n, ast.Name) and n.id in {"AuthorizationCitation", "verify_citation_chain"})
            or (
                isinstance(n, ast.Attribute)
                and n.attr in {"AuthorizationCitation", "verify_citation_chain"}
            )
            or (
                isinstance(n, ast.alias)
                and n.name in {"AuthorizationCitation", "verify_citation_chain"}
            )
            for n in ast.walk(tree)
        )
        assert not used or path.name in allowed, path


def test_only_the_first_citation_of_a_history_is_initial():
    first = initial_citation()
    another = initial_citation(
        initial_report_sha256=digest("report-9"), readiness_report_sha256=digest("report-9")
    )
    with pytest.raises(CitationChainError, match="only the first"):
        verify_citation_chain([first, another], {})


def test_the_cited_assessment_must_be_about_this_citations_identity_and_report():
    first = initial_citation()
    other_batch = scope(batch_id=digest("another batch"))
    elsewhere = compare_assessments(make(), make(), other_batch, other_batch)
    assert elsewhere.verdict == "continuous"
    about_other_batch, _ = resume_pair(
        first, continuity_assessment_sha256=continuity_assessment_sha256(elsewhere)
    )
    with pytest.raises(CitationChainError, match="other reports"):
        verify_citation_chain([first, about_other_batch], index(elsewhere))
    second, a2 = resume_pair(first, report_binding_sha256=digest("some other report"))
    with pytest.raises(CitationChainError, match="other reports"):
        verify_citation_chain([first, second], index(a2))
