"""Existing behavior: how stored reports, policy versions and source revisions interact.

These tests describe the Phase 9 contract as it is. They do not change it. They exist so that a
later design that adds an approval, or commits between batches, starts from facts.
"""

import subprocess
from pathlib import Path

import pytest
import readiness_support as support
from test_readiness_guard import Fixture, build, refused

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import git_state, publish
from biorag_repro.readiness.files import loads_strict
from biorag_repro.readiness.policy import REAL_POLICY, ApprovalPolicy, policy_sha256
from biorag_repro.readiness.publish import ReadbackError


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def other_version(policy: ApprovalPolicy) -> ApprovalPolicy:
    """The same approvals under another policy version: no scientific value moved."""
    return ApprovalPolicy(policy_version="fixture-policy-1", slots=policy.slots)


# ---- reports are verified against the policy supplied, which is the installed one -----------


def test_a_report_verifies_only_against_the_policy_it_was_produced_under(fx):
    assert publish.read_report(fx.data, fx.sha, fx.policy).outcome.decision == "READY"
    newer = other_version(fx.policy)
    assert policy_sha256(newer) != policy_sha256(fx.policy)
    with pytest.raises(ReadbackError, match="different policy"):
        publish.read_report(fx.data, fx.sha, newer)


def test_a_policy_change_with_no_scientific_change_still_makes_the_old_report_unverifiable(fx):
    # the installed policy is the only one the production reader is given
    with pytest.raises(ReadbackError, match="different policy"):
        publish.read_report(fx.data, fx.sha, other_version(fx.policy))


def test_a_changed_approval_value_is_refused_the_same_way(fx):
    values = fx.policy.slots.model_dump()
    name = next(iter(values))
    values[name] = support.digest("a newer approval")
    changed = ApprovalPolicy(
        policy_version=fx.policy.policy_version, slots=type(fx.policy.slots)(**values)
    )
    with pytest.raises(ReadbackError, match="different policy"):
        publish.read_report(fx.data, fx.sha, changed)


# ---- the policy used is retained, so history stays auditable -----------------------------------


def retained_policy(fx) -> ApprovalPolicy:
    path = fx.data / "readiness/evidence" / (policy_sha256(fx.policy) + ".json")
    return ApprovalPolicy.model_validate(loads_strict(path.read_bytes()))


def test_the_policy_used_is_stored_with_the_report_and_audits_it_after_a_policy_change(fx):
    retained = retained_policy(fx)
    assert policy_sha256(retained) == policy_sha256(fx.policy)
    audited = publish.read_report(fx.data, fx.sha, retained)
    assert audited.outcome.decision == "READY" and audited.verified_evidence > 0


def test_the_retained_copy_is_checked_against_the_policy_used_for_verification(fx):
    path = fx.data / "readiness/evidence" / (policy_sha256(fx.policy) + ".json")
    other = other_version(fx.policy)
    path.chmod(0o644)
    path.write_bytes(canonical_json_bytes(other.model_dump(mode="json")))
    with pytest.raises(ReadbackError, match="stored policy copy differs"):
        publish.read_report(fx.data, fx.sha, fx.policy)


def test_a_retained_policy_that_does_not_hash_to_the_report_cannot_verify_it(fx):
    # the stored copy is only trusted through the hash the report names
    with pytest.raises(ReadbackError, match="different policy"):
        publish.read_report(fx.data, fx.sha, other_version(retained_policy(fx)))


# ---- the guard refuses an outdated policy without destroying the record ------------------------


def test_the_guard_refuses_a_report_from_an_earlier_policy_and_leaves_it_intact(fx):
    before = (fx.data / "readiness/reports" / (fx.sha + ".json")).read_bytes()
    newer = other_version(fx.policy)
    refused(fx, "report_unverifiable", "different policy", policy=newer)
    assert (fx.data / "readiness/reports" / (fx.sha + ".json")).read_bytes() == before
    assert publish.read_report(fx.data, fx.sha, retained_policy(fx)).outcome.decision == "READY"


def test_a_fresh_report_under_the_new_policy_authorizes_where_the_old_one_does_not(fx):
    newer = other_version(fx.policy)
    fresh = build(newer, tag="new")
    sha = publish.publish_report(fresh, fx.data).report_sha256
    from biorag_repro.readiness import guard

    result = guard._authorize(
        fx.data, sha, repo=fx.data, policy=newer, evaluate_fn=lambda policy: build(newer, tag="new")
    )
    assert result.report_sha256 == sha and result.policy_sha256 == policy_sha256(newer)
    assert result.report_sha256 != fx.sha


def test_the_guard_does_not_compare_policy_identity_between_two_authorizations(fx):
    # one stored report against one live evaluation: nothing relates two authorizations
    # to each other, which is why continuity is a separate contract
    newer = other_version(fx.policy)
    sha = publish.publish_report(build(newer, tag="new"), fx.data).report_sha256
    from biorag_repro.readiness import guard

    first = fx.authorize()
    second = guard._authorize(
        fx.data, sha, repo=fx.data, policy=newer, evaluate_fn=lambda p: build(newer, tag="new")
    )
    assert first.policy_sha256 != second.policy_sha256


def test_the_evaluator_identity_is_not_part_of_historical_verification(fx, tmp_path):
    evaluation = build(
        fx.policy, tag="x", evaluator=support.evaluator(code_sha256=support.digest("old"))
    )
    other_root = tmp_path / "other_root"
    other_root.mkdir()
    sha = publish.publish_report(evaluation, other_root).report_sha256
    readback = publish.read_report(other_root, sha, fx.policy)
    assert readback.report.evaluator.code_sha256 == support.digest("old")


# ---- source revisions: the scientific identity follows content, not the commit ------------------


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    for name, text in {
        "src/a.py": "a = 1\n",
        "pyproject.toml": "[project]\n",
        "docs/n.md": "n\n",
    }.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "one")
    return tmp_path


def test_a_commit_outside_the_scope_changes_the_commit_but_not_the_scope_identity(repo):
    before = git_state.probe(repo)
    (repo / "docs/n.md").write_text("changed\n")
    git(repo, "commit", "-qam", "docs only")
    after = git_state.probe(repo)
    assert after.head != before.head
    assert after.entries == before.entries


def test_a_commit_inside_the_scope_changes_the_scope_identity(repo):
    before = git_state.probe(repo)
    (repo / "src/a.py").write_text("a = 2\n")
    git(repo, "commit", "-qam", "code")
    after = git_state.probe(repo)
    assert after.head != before.head and after.entries != before.entries


def test_a_commit_that_restores_the_same_content_restores_the_same_scope_identity(repo):
    before = git_state.probe(repo)
    (repo / "src/a.py").write_text("a = 2\n")
    git(repo, "commit", "-qam", "change")
    (repo / "src/a.py").write_text("a = 1\n")
    git(repo, "commit", "-qam", "restore")
    assert git_state.probe(repo).entries == before.entries


def test_the_scientific_scope_contains_the_policy_and_the_evaluator():
    """Recorded because it constrains any design that commits an approval between batches.

    The G09 slot approves a digest over the blob list of this scope. The policy module holds
    that slot, so a policy change is a scope change, and the slot value would have to be a
    fixed point of its own file's hash.
    """
    for path in (
        "src/biorag_repro/readiness/policy.py",
        "src/biorag_repro/readiness/evaluate.py",
        "src/biorag_repro/finaltest/inventory.py",
        "src/biorag_repro/finaltest/head.py",
    ):
        assert git_state._in_scope(path)
    assert not git_state._in_scope("tests/test_finaltest_head.py")
    assert not git_state._in_scope("docs/notes/changes.md")


# ---- the real project is untouched ---------------------------------------------------------------


def test_every_real_approval_slot_is_still_unset():
    assert all(value is None for value in REAL_POLICY.slots.model_dump().values())
    assert len(REAL_POLICY.slots.model_dump()) == 12


def test_the_policy_hash_is_a_function_of_the_whole_policy():
    base = support.fixture_policy()
    assert sha256_bytes(canonical_json_bytes(base.model_dump(mode="json"))) == policy_sha256(base)
