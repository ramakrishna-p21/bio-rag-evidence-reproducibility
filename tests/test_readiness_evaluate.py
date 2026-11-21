import pytest
from readiness_support import REAL_POLICY, digest
from readiness_world import build_world, write

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import evidence
from biorag_repro.readiness.evaluate import EvaluationError, evaluate, evaluator_identity
from biorag_repro.readiness.models import MANDATORY_GATES
from biorag_repro.readiness.policy import ApprovalPolicy, ApprovalSlots, policy_sha256


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def statuses(report):
    return {g.gate_id: (g.status, g.reason_code) for g in report.gates}


def test_real_policy_gives_a_complete_not_ready_report(world):
    result = evaluate(world.ctx, REAL_POLICY)
    report = result.report
    assert report.decision == "NOT_READY"
    assert [g.gate_id for g in report.gates] == list(MANDATORY_GATES)
    assert len(report.blockers) == 13  # fixture identities differ from the real approvals
    got = statuses(report)
    assert got["G01_corpus_sources"] == ("failed", "policy_mismatch")
    assert got["G02_benchmark_freeze"] == ("failed", "policy_mismatch")
    assert got["G09_scientific_source_revision"][0] == "failed"
    assert got["G13_final_test_history_integrity"] == (
        "unavailable",
        "final_test_inventory_authority_not_established",
    )
    assert report.policy_sha256 == policy_sha256(REAL_POLICY)
    assert REAL_POLICY.slots.final_test_freeze is None


def test_candidate_records_what_was_read(world):
    candidate = evaluate(world.ctx, REAL_POLICY).report.candidate
    assert candidate.head_sha == world.commit
    assert candidate.scientific_source_identity_sha256 is not None
    assert candidate.renv_lock_sha256 == sha256_bytes(
        (world.repo / "analysis/renv.lock").read_bytes()
    )
    assert candidate.benchmark_release_sha256 == sha256_bytes(
        (world.data / evidence.BENCHMARK_RELEASE).read_bytes()
    )
    assert candidate.freeze_record_sha256 is None and candidate.python_lock_sha256 is None


def test_policy_copy_and_observations_are_captured(world):
    result = evaluate(world.ctx, REAL_POLICY)
    assert result.snapshots[policy_sha256(REAL_POLICY)] == canonical_json_bytes(
        REAL_POLICY.model_dump(mode="json")
    )
    cited = {
        r.sha256
        for g in result.report.gates
        for r in g.evidence
        if r.locator.startswith("snapshot/")
    }
    assert cited <= set(result.snapshots)
    assert all(sha256_bytes(v) == k for k, v in result.snapshots.items())


def test_binding_ignores_time_and_nonce_but_not_content(world):
    first = evaluate(world.ctx, REAL_POLICY).report
    second = evaluate(world.ctx, REAL_POLICY).report
    assert first.nonce != second.nonce and first.binding_sha256 == second.binding_sha256
    write(world.repo / "src/extra.py", "q = 1\n")
    third = evaluate(world.ctx, REAL_POLICY).report
    assert third.binding_sha256 != first.binding_sha256
    assert statuses(third)["G09_scientific_source_revision"] == (
        "failed",
        "scientific_source_not_clean",
    )


def test_a_policy_that_approves_everything_still_cannot_make_ready(world):
    """Eleven gates have no verifiable subject yet and G13 has no authority."""
    first = evaluate(world.ctx, REAL_POLICY).report
    subjects = {}
    for gate in first.gates:
        for ref in gate.evidence:
            if ref.kind == "observed_subject":
                subjects[gate.gate_id] = ref.sha256
    slots = {
        "corpus_identity": subjects["G01_corpus_sources"],
        "benchmark_identity": subjects["G02_benchmark_freeze"],
        "scientific_source_identity": subjects["G09_scientific_source_revision"],
    }
    for name in ApprovalSlots.model_fields:
        slots.setdefault(name, digest(name))
    policy = ApprovalPolicy(policy_version="t", slots=ApprovalSlots(**slots))
    report = evaluate(world.ctx, policy).report
    got = statuses(report)
    assert got["G01_corpus_sources"][0] == "passed"
    assert got["G02_benchmark_freeze"][0] == "passed"
    assert got["G09_scientific_source_revision"][0] == "passed"
    assert report.decision == "NOT_READY"
    assert got["G13_final_test_history_integrity"][0] == "unavailable"


def test_populated_slot_with_a_different_subject_fails_not_pending(world):
    policy = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(corpus_identity=digest("something else"))
    )
    got = statuses(evaluate(world.ctx, policy).report)
    assert got["G01_corpus_sources"] == ("failed", "policy_mismatch")


def test_changed_source_bytes_make_the_gate_fail_even_with_unset_slot(world):
    next((world.data / "corpus/raw").rglob("pdf_5.bin")).write_bytes(b"%PDF-changed")
    got = statuses(evaluate(world.ctx, REAL_POLICY).report)
    assert got["G01_corpus_sources"][0] == "failed"


def test_an_adapter_that_crashes_is_an_evaluation_error_not_not_ready(world, monkeypatch):
    def crash():
        raise TypeError("bug")

    monkeypatch.setattr("biorag_repro.readiness.production.qualification", lambda ctx: crash())
    with pytest.raises(EvaluationError):
        evaluate(world.ctx, REAL_POLICY)


def test_a_gate_missing_from_the_gathered_set_is_an_evaluation_error(world, monkeypatch):
    import biorag_repro.readiness.evaluate as ev

    real = ev.gather_all

    def without_one(ctx):
        found = real(ctx)
        del found["G05_model_parser_identities"]
        return found

    monkeypatch.setattr(ev, "gather_all", without_one)
    with pytest.raises(EvaluationError):
        evaluate(world.ctx, REAL_POLICY)


def test_evaluator_identity_follows_the_package_source():
    assert evaluator_identity() == evaluator_identity()
    assert len(evaluator_identity().code_sha256) == 64
