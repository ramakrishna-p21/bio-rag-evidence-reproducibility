import ast
import inspect
import json
import shutil
from pathlib import Path

import pytest
import readiness_support as support
from readiness_world import build_world

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import guard, publish
from biorag_repro.readiness.evaluate import Evaluation, EvaluationError
from biorag_repro.readiness.guard import (
    AuthorizationRefused,
    FinalTestAuthorization,
    authorize_final_test,
)
from biorag_repro.readiness.models import (
    MANDATORY_GATES,
    Candidate,
    EvidenceRef,
    GateResult,
    evidence_digest,
)
from biorag_repro.readiness.policy import GATE_SLOT, REAL_POLICY, ApprovalPolicy, policy_sha256


def build(policy: ApprovalPolicy, *, tag: str = "", blocked: dict[str, str] | None = None, **parts):
    """A fixture evaluation whose evidence is real, retained snapshot bytes."""
    blocked = blocked or {}
    snapshots: dict[str, bytes] = {}
    results: list[GateResult] = []
    for gate in MANDATORY_GATES:
        if gate in blocked:
            reason = (
                "final_test_inventory_authority_not_established"
                if gate == "G13_final_test_history_integrity"
                else "fixture_blocker"
            )
            results.append(support.blocked(gate, blocked[gate], reason))
            continue
        body = canonical_json_bytes({"gate": gate, "tag": tag})
        sha = sha256_bytes(body)
        snapshots[sha] = body
        refs = [
            EvidenceRef(
                kind="fixture_snapshot",
                locator=f"snapshot/{sha}.json",
                sha256=sha,
                subject_kind="canonical_json",
            )
        ]
        slot = GATE_SLOT[gate]
        results.append(
            GateResult(
                gate_id=gate,
                status="passed",
                verification="verified",
                scope="fixture",
                policy_expectation=slot,
                expected_sha256=None if slot is None else getattr(policy.slots, slot),
                reason_code="no_prior_final_test_execution"
                if slot is None
                else "verified_against_policy",
                detail=None,
                evidence=refs,
                evidence_digest=evidence_digest(refs),
            )
        )
    report = support.report(policy, results, **parts)
    snapshots[policy_sha256(policy)] = canonical_json_bytes(policy.model_dump(mode="json"))
    return Evaluation(report, snapshots)


class Fixture:
    def __init__(self, root: Path):
        self.data = root
        self.policy = support.fixture_policy()
        stored = build(self.policy)
        self.sha = publish.publish_report(stored, root).report_sha256
        self.live = build(self.policy)
        self.calls = 0

    def evaluate(self, policy):
        self.calls += 1
        if isinstance(self.live, Exception):
            raise self.live
        return self.live

    def authorize(self, **overrides):
        arguments = {
            "repo": self.data,
            "policy": self.policy,
            "evaluate_fn": self.evaluate,
        }
        arguments.update(overrides)
        return guard._authorize(self.data, self.sha, **arguments)

    def report_path(self, suffix=".json", sha=None):
        return self.data / "readiness/reports" / ((sha or self.sha) + suffix)


@pytest.fixture
def fx(tmp_path):
    return Fixture(tmp_path)


def refused(fx, reason, contains="", **overrides):
    with pytest.raises(AuthorizationRefused) as caught:
        fx.authorize(**overrides)
    assert caught.value.reason == reason, str(caught.value)
    assert contains in str(caught.value)


# ---- the positive branch (synthetic policy, synthetic live evaluation) ---------------------


def test_matching_ready_report_and_live_evaluation_authorize(fx):
    result = fx.authorize()
    assert isinstance(result, FinalTestAuthorization)
    assert result.report_sha256 == fx.sha
    assert result.policy_sha256 == policy_sha256(fx.policy)
    assert result.report_binding_sha256 == result.live_binding_sha256
    assert result.candidate == support.candidate()
    assert result.host == support.host()
    assert result.scope == "final_test_batch_start"
    assert result.evaluator_code_sha256 == support.evaluator().code_sha256
    assert fx.calls == 1  # a live evaluation happened for this request


def test_every_request_evaluates_again(fx):
    fx.authorize()
    fx.authorize()
    assert fx.calls == 2


def test_result_is_immutable(fx):
    result = fx.authorize()
    with pytest.raises(AttributeError):
        result.scope = "other"  # type: ignore[misc]


# ---- stored report: historical verification -------------------------------------------------


def test_missing_report(fx):
    fx.report_path().unlink()
    refused(fx, "report_unverifiable", "no report")


def test_corrupted_report_bytes(fx):
    fx.report_path().write_bytes(fx.report_path().read_bytes() + b" ")
    refused(fx, "report_unverifiable", "SHA-256")


def test_requested_hash_that_names_another_report(fx):
    other = publish.publish_report(build(fx.policy, tag="other"), fx.data).report_sha256
    fx.sha = other
    fx.report_path().write_bytes(fx.report_path(sha=fx.sha).read_bytes() + b"\n")
    refused(fx, "report_unverifiable", "SHA-256")


def test_missing_manifest_is_incomplete(fx):
    fx.report_path(".build.json").unlink()
    refused(fx, "report_incomplete")


def test_corrupted_manifest(fx):
    fx.report_path(".build.json").write_bytes(b"{}")
    refused(fx, "report_unverifiable", "manifest")


def test_manifest_of_another_report(fx):
    other = publish.publish_report(build(fx.policy, tag="other"), fx.data).report_sha256
    shutil.copy(fx.report_path(".build.json", sha=other), fx.report_path(".build.json"))
    refused(fx, "report_unverifiable", "manifest")


def test_altered_retained_evidence(fx):
    snapshot = next((fx.data / "readiness/evidence").iterdir())
    snapshot.write_bytes(snapshot.read_bytes() + b" ")
    refused(fx, "report_unverifiable", "differs")


def test_stored_not_ready(tmp_path):
    fx = Fixture(tmp_path)
    stored = build(fx.policy, blocked={"G03_final_test_configuration_freeze": "pending"})
    fx.sha = publish.publish_report(stored, tmp_path).report_sha256
    refused(fx, "stored_not_ready")
    assert fx.calls == 0  # a stored NOT_READY never reaches the live evaluation


def _tampered(fx, change):
    data = json.loads(fx.report_path().read_bytes())
    change(data)
    content = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    sha = sha256_bytes(content)
    fx.report_path(sha=sha).write_bytes(content)
    fx.sha = sha


def test_forged_ready_field_on_a_not_ready_report(tmp_path):
    fx = Fixture(tmp_path)
    stored = build(fx.policy, blocked={"G12_shared_gpu_authorization": "pending"})
    fx.sha = publish.publish_report(stored, tmp_path).report_sha256
    _tampered(fx, lambda d: d.update(decision="READY", blockers=[]))
    refused(fx, "report_unverifiable")


def test_report_missing_a_gate(fx):
    _tampered(fx, lambda d: d["gates"].pop())
    refused(fx, "report_unverifiable", "missing mandatory gates")


def test_report_with_a_duplicate_gate(fx):
    _tampered(fx, lambda d: d["gates"].append(d["gates"][0]))
    refused(fx, "report_unverifiable", "duplicate gate")


def test_report_with_an_unexpected_gate(fx):
    _tampered(fx, lambda d: d["gates"][0].update(gate_id="G99_extra"))
    refused(fx, "report_unverifiable")


def test_report_produced_under_another_policy(fx):
    other = fx.policy.model_copy(update={"policy_version": "fixture-policy-1"})
    refused(fx, "report_unverifiable", "different policy", policy=other)


def test_unset_approval_slot_in_the_verifying_policy(fx):
    slots = fx.policy.slots.model_copy(update={"corpus_identity": None})
    refused(fx, "report_unverifiable", policy=fx.policy.model_copy(update={"slots": slots}))


def test_changed_approval_digest_in_the_verifying_policy(fx):
    slots = fx.policy.slots.model_copy(update={"benchmark_identity": support.digest("new")})
    refused(fx, "report_unverifiable", policy=fx.policy.model_copy(update={"slots": slots}))


def test_report_made_without_a_final_test_run_mode_is_not_a_report(fx):
    _tampered(fx, lambda d: d.update(requested_run_mode="development"))
    refused(fx, "report_unverifiable")


# ---- live evaluation ------------------------------------------------------------------------


@pytest.mark.parametrize("gate", MANDATORY_GATES)
@pytest.mark.parametrize("status", ["failed", "pending", "unavailable"])
def test_live_not_ready_for_any_gate_refuses(fx, gate, status):
    fx.live = build(fx.policy, blocked={gate: status})
    refused(fx, "live_not_ready", gate)


def test_g13_unavailable_live_refuses_with_its_reason(fx):
    fx.live = build(fx.policy, blocked={"G13_final_test_history_integrity": "unavailable"})
    refused(fx, "live_not_ready", "G13_final_test_history_integrity")


def test_stored_ready_then_live_not_ready(fx):
    fx.live = build(fx.policy, blocked={"G09_scientific_source_revision": "failed"})
    refused(fx, "live_not_ready")
    assert fx.calls == 1


def test_live_evaluation_failure(fx):
    fx.live = EvaluationError("probe crashed")
    refused(fx, "live_evaluation_failed", "probe crashed")


def test_unexpected_live_exception_still_refuses(fx):
    fx.live = RuntimeError("boom")
    refused(fx, "live_evaluation_failed")


def test_live_report_with_a_forged_decision(fx):
    honest = build(fx.policy, blocked={"G10_development_restart_recovery": "pending"})
    forged = honest.report.model_copy(update={"decision": "READY", "blockers": []})
    fx.live = Evaluation(forged, honest.snapshots)
    refused(fx, "live_invalid")


@pytest.mark.parametrize("field", list(Candidate.model_fields))
def test_every_candidate_identity_must_match(fx, field):
    value = (
        {"confirmatory": support.digest("other")}
        if field == "configuration_sha256_by_arm"
        else ("b" * 40 if field == "head_sha" else support.digest(f"changed {field}"))
    )
    fx.live = build(fx.policy, candidate=support.candidate(**{field: value}))
    refused(fx, "identity_mismatch", f"candidate.{field}")


def test_candidate_identity_that_was_absent_before_must_match_too(fx):
    fx.live = build(fx.policy, candidate=support.candidate(freeze_record_sha256=None))
    refused(fx, "identity_mismatch", "candidate.freeze_record_sha256")


@pytest.mark.parametrize(
    "host",
    [
        {"hostname": "another-host"},
        {"operating_system": "Darwin"},
        {"machine": "arm64"},
        {"gpu_names": ["Other GPU"]},
    ],
)
def test_execution_host_must_match(fx, host):
    fx.live = build(fx.policy, host=support.host(**host))
    refused(fx, "identity_mismatch", "host")


def test_evaluator_code_must_match(fx):
    fx.live = build(fx.policy, evaluator=support.evaluator(code_sha256=support.digest("new")))
    refused(fx, "identity_mismatch", "evaluator")


def test_evaluator_version_must_match(fx):
    fx.live = build(fx.policy, evaluator=support.evaluator(version="fixture-evaluator-1"))
    refused(fx, "identity_mismatch", "evaluator")


def test_live_policy_must_match(fx):
    other = fx.policy.model_copy(update={"policy_version": "fixture-policy-1"})
    fx.live = build(other)
    refused(fx, "live_invalid", "different policy")


def test_changed_evidence_in_any_gate_is_a_mismatch_not_an_ignored_difference(fx):
    fx.live = build(fx.policy, tag="source drifted")
    refused(fx, "identity_mismatch", "gates")


def test_informational_fields_may_differ(fx):
    fx.live = build(
        fx.policy,
        warnings=["a fresh note"],
        created_at_utc="2027-01-01T00:00:00.000000Z",
        nonce="1" * 32,
    )
    assert fx.authorize().report_binding_sha256 == fx.authorize().live_binding_sha256


def test_stale_binding_alone_is_refused(fx):
    live = build(fx.policy)
    stale = live.report.model_copy(update={"binding_sha256": support.digest("old")})
    fx.live = Evaluation(stale, live.snapshots)
    refused(fx, "live_invalid")


def test_report_replaced_while_the_live_check_runs(fx):
    evidence_file = next((fx.data / "readiness/evidence").iterdir())

    def tamper(policy):
        evidence_file.write_bytes(b"changed during the check")
        return fx.live

    refused(fx, "report_changed_during_check", evaluate_fn=tamper)


def test_a_comparison_with_the_present_checkout_is_not_a_substitute(fx):
    # compare_current can say "same" for head, lock and release while the live gates differ
    fx.live = build(fx.policy, tag="drift")
    refused(fx, "identity_mismatch", "gates")


# ---- production binding ---------------------------------------------------------------------


def test_public_interface_takes_no_policy_or_evaluator():
    names = set(inspect.signature(authorize_final_test).parameters)
    assert names == {"data_root", "report_sha256", "repo"}


def test_private_seams_are_keyword_only():
    parameters = inspect.signature(guard._authorize).parameters
    assert parameters["policy"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["evaluate_fn"].kind is inspect.Parameter.KEYWORD_ONLY


def test_public_function_uses_the_real_policy_and_reads_no_environment():
    source = inspect.getsource(guard)
    tree = ast.parse(source)
    public = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "authorize_final_test"
    )
    names = {n.id for n in ast.walk(public) if isinstance(n, ast.Name)}
    assert "REAL_POLICY" in names
    attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert not attributes & {"environ", "getenv", "argv"} and "os" not in imported


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def test_fixture_policy_report_offered_to_the_production_function(world):
    fixture = support.fixture_policy()
    sha = publish.publish_report(build(fixture), world.data).report_sha256
    with pytest.raises(AuthorizationRefused) as caught:
        authorize_final_test(world.data, sha, repo=world.repo)
    assert caught.value.reason == "report_unverifiable"
    assert "different policy" in str(caught.value)


def test_real_evaluation_of_the_synthetic_world_is_refused_at_the_stored_report(world):
    from biorag_repro.readiness.evaluate import evaluate

    published = publish.publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    with pytest.raises(AuthorizationRefused) as caught:
        authorize_final_test(world.data, published.report_sha256, repo=world.repo)
    assert caught.value.reason == "stored_not_ready"


def test_real_evaluator_cannot_become_ready_even_under_a_fixture_policy(world):
    # G13 is wired to "unavailable"; no fixture policy can change that
    from biorag_repro.readiness.evaluate import evaluate

    evaluation = evaluate(world.ctx, support.fixture_policy())
    assert evaluation.report.decision == "NOT_READY"
    g13 = next(g for g in evaluation.report.gates if g.gate_id.startswith("G13"))
    assert (g13.status, g13.reason_code) == (
        "unavailable",
        "final_test_inventory_authority_not_established",
    )


def test_guard_does_not_use_the_history_classifier():
    assert "readiness.history" not in inspect.getsource(guard)
