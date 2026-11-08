"""Readiness gate-result, policy and report contracts (P9-A). Pure: no files, Git or devices."""

import ast
import json
import re
from pathlib import Path

import pytest
import readiness_support as fx
from pydantic import ValidationError
from readiness_support import REAL_POLICY

from biorag_repro.readiness import decide as decide_module
from biorag_repro.readiness import models, policy
from biorag_repro.readiness.models import (
    HISTORY_FAIL_CODES,
    HISTORY_PASS_CODES,
    MANDATORY_GATES,
    GateResult,
    ReadinessReport,
    evidence_digest,
)

GATE13 = "G13_final_test_history_integrity"
STATUSES = ("passed", "failed", "pending", "unavailable")
POLICY = fx.fixture_policy()


def result(gate="G01_corpus_sources", **changes):
    base = fx.passed(gate, POLICY).model_dump()
    base.update(changes)
    return GateResult(**base)


def test_the_thirteen_mandatory_gates_are_fixed():
    assert len(MANDATORY_GATES) == 13 == len(set(MANDATORY_GATES))
    assert [g[:3] for g in MANDATORY_GATES] == [f"G{i:02d}" for i in range(1, 14)]
    names = [g[4:] for g in MANDATORY_GATES]
    assert names == [
        "corpus_sources",
        "benchmark_freeze",
        "final_test_configuration_freeze",
        "generator_qualification",
        "model_parser_identities",
        "python_environment",
        "r_analysis_environment",
        "seed_registry",
        "scientific_source_revision",
        "development_restart_recovery",
        "storage_authorization",
        "shared_gpu_authorization",
        "final_test_history_integrity",
    ]


def test_status_vocabulary_has_no_skip_state():
    assert set(models.GateStatus.__args__) == set(STATUSES)
    with pytest.raises(ValidationError):
        result(status="skipped")
    with pytest.raises(ValidationError):
        result(status="not_applicable")


@pytest.mark.parametrize("status", STATUSES)
def test_every_status_is_representable(status):
    gate = "G01_corpus_sources"
    made = fx.passed(gate, POLICY) if status == "passed" else fx.blocked(gate, status)
    assert made.status == status


def test_blocked_result_may_carry_evidence_of_what_was_found():
    refs = fx.evidence("G01_corpus_sources")
    kept = result(
        status="failed",
        reason_code="hash_mismatch",
        evidence=refs,
        evidence_digest=evidence_digest(refs),
    )
    assert kept.evidence_digest is not None and kept.status == "failed"


def test_slots_are_mapped_one_to_one_and_all_unset_in_the_real_policy():
    assert len(policy.SLOT_NAMES) == 12
    assert policy.GATE_SLOT[GATE13] is None
    slots = [s for s in policy.GATE_SLOT.values() if s]
    assert sorted(slots) == sorted(policy.SLOT_NAMES)
    assert all(getattr(REAL_POLICY.slots, name) is None for name in policy.SLOT_NAMES)


@pytest.mark.parametrize(
    "name",
    [
        "primary_bootstrap",
        "qwen_checkpoint",
        "environment_lock",
        "gpu",
        "storage",
        "qualification",
        "freeze",
    ],
)
def test_specific_production_approvals_stay_unset(name):
    matching = [n for n in policy.SLOT_NAMES if name in n or n in name]
    for slot in matching:
        assert getattr(REAL_POLICY.slots, slot) is None
    assert REAL_POLICY.slots == policy.ApprovalSlots()


def test_policy_is_frozen_and_strict():
    with pytest.raises(ValidationError):
        REAL_POLICY.slots.final_test_freeze = fx.digest("x")
    with pytest.raises(ValidationError):
        policy.ApprovalSlots(approved=True)
    with pytest.raises(ValidationError):
        policy.ApprovalSlots(final_test_freeze="ABC")


def test_policy_identity_changes_with_any_slot_or_version():
    base = policy.policy_sha256(POLICY)
    assert base == policy.policy_sha256(fx.fixture_policy())
    assert base != policy.policy_sha256(fx.fixture_policy("fixture-policy-1"))
    slots = POLICY.slots.model_copy(update={"gpu_authorization": fx.digest("other")})
    assert base != policy.policy_sha256(
        policy.ApprovalPolicy(policy_version="fixture-policy-0", slots=slots)
    )
    assert policy.policy_sha256(REAL_POLICY) != base


def test_package_offers_no_way_to_select_a_policy():
    """The contract modules read nothing; no module defines a fixture policy.

    P9-B added modules that do read files and run Git; they are checked by the readiness CLI
    tests (no environment access, no policy parameter on `main`).
    """
    root = Path(models.__file__).parent
    imported, names = set(), set()
    for path in (root / n for n in ("models.py", "policy.py", "decide.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Name):
                names.add(node.id)
    assert not imported & {"os", "sys", "argparse", "subprocess", "pathlib", "tomllib", "json"}
    assert not names & {"environ", "getenv", "argv", "open", "read_text", "read_bytes"}
    assert [n for n in names if "fixture" in n.lower() or "synthetic" in n.lower()] == []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not [
            n.name
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef | ast.ClassDef | ast.Assign)
            and "fixture" in getattr(n, "name", "").lower()
        ], path.name


@pytest.mark.parametrize(
    "changes",
    [
        {"verification": "unverified"},
        {"evidence": [], "evidence_digest": None},
        {"reason_code": "skipped"},
        {"reason_code": "waived"},
        {"reason_code": "something_else"},
        {"expected_sha256": None},
        {"policy_expectation": None},
        {"evidence_digest": fx.digest("not the evidence")},
        {"evidence_digest": None},
    ],
)
def test_a_pass_needs_structurally_valid_positive_evidence(changes):
    with pytest.raises(ValidationError):
        result(**changes)


@pytest.mark.parametrize("status", ["failed", "pending", "unavailable"])
def test_a_non_pass_cannot_use_a_passing_reason_or_stray_digest(status):
    with pytest.raises(ValidationError):
        fx.blocked("G01_corpus_sources", status, reason="verified_against_policy")
    with pytest.raises(ValidationError):
        fx.blocked(
            "G13_final_test_history_integrity", status, reason="no_prior_final_test_execution"
        )
    with pytest.raises(ValidationError):
        GateResult(
            **{
                **fx.blocked("G01_corpus_sources", status).model_dump(),
                "evidence_digest": fx.digest("a"),
            }
        )


@pytest.mark.parametrize("status", ["pending", "unavailable"])
def test_pending_and_unavailable_cannot_claim_verification(status):
    with pytest.raises(ValidationError):
        GateResult(
            **{**fx.blocked("G01_corpus_sources", status).model_dump(), "verification": "verified"}
        )


def test_history_gate_semantics():
    for code in HISTORY_PASS_CODES:
        made = result(GATE13, reason_code=code, policy_expectation=None, expected_sha256=None)
        assert made.status == "passed"
        # an empty inventory is still evidence: a pass without any is refused
        with pytest.raises(ValidationError):
            result(
                GATE13,
                reason_code=code,
                policy_expectation=None,
                expected_sha256=None,
                evidence=[],
                evidence_digest=None,
            )
    with pytest.raises(ValidationError):  # the generic code does not describe a valid history
        result(
            GATE13,
            reason_code="verified_against_policy",
            policy_expectation=None,
            expected_sha256=None,
        )
    with pytest.raises(ValidationError):  # history codes are not usable elsewhere
        result("G01_corpus_sources", reason_code="no_prior_final_test_execution")
    with pytest.raises(ValidationError):  # G13 does not pass against a slot
        result(
            GATE13,
            reason_code="no_prior_final_test_execution",
            policy_expectation="gpu_authorization",
            expected_sha256=fx.digest("slot:gpu_authorization"),
        )
    for code in HISTORY_FAIL_CODES:
        failed = fx.blocked(GATE13, "failed", code)
        assert failed.status == "failed"
        with pytest.raises(ValidationError):
            fx.blocked("G01_corpus_sources", "failed", code)
    assert set(HISTORY_PASS_CODES) == {
        "no_prior_final_test_execution",
        "same_batch_valid_history",
        "other_approved_batch_valid_history",
    }


@pytest.mark.parametrize(
    "bad",
    [
        {"sha256": "A" * 64},
        {"sha256": "a" * 63},
        {"locator": "/home/user/file.json"},
        {"locator": "C:\\data\\file.json"},
        {"locator": "~/x"},
        {"locator": " "},
        {"subject_kind": "mixture"},
        {"kind": "Not Snake"},
    ],
)
def test_evidence_reference_is_strict(bad):
    base = fx.evidence("G01_corpus_sources")[0].model_dump()
    with pytest.raises(ValidationError):
        models.EvidenceRef(**{**base, **bad})


def test_unknown_fields_and_ids_are_refused():
    with pytest.raises(ValidationError):
        result(approved=True)
    with pytest.raises(ValidationError):
        result(gate_id="G14_extra")
    with pytest.raises(ValidationError):
        result(sha256="x")


def test_candidate_host_and_evaluator_are_strict():
    with pytest.raises(ValidationError):
        fx.candidate(head_sha="abc")
    with pytest.raises(ValidationError):
        fx.candidate(corpus_content_sha256="G" * 64)
    with pytest.raises(ValidationError):
        fx.host(gpu_names=[""])
    with pytest.raises(ValidationError):
        fx.evaluator(code_sha256="1")
    with pytest.raises(ValidationError):
        fx.candidate(extra=1)
    assert fx.candidate(head_sha=None, corpus_content_sha256=None).head_sha is None


def test_report_round_trips_through_canonical_json():
    report = fx.report(POLICY)
    text = report.model_dump_json()
    assert ReadinessReport.model_validate_json(text) == report
    assert json.loads(text)["decision"] == "READY"
    assert (
        ReadinessReport.model_validate_json(json.dumps(json.loads(text), sort_keys=True)) == report
    )


def test_report_requires_final_test_mode_and_fixed_limitations():
    data = fx.report(POLICY).model_dump()
    for change in (
        {"requested_run_mode": "development"},
        {"limitations": data["limitations"][:-1]},
        {"limitations": []},
        {"created_at_utc": "2026-11-07T00:00:00+01:00"},
        {"nonce": "xyz"},
        {"unexpected": 1},
    ):
        with pytest.raises(ValidationError):
            ReadinessReport(**{**data, **change})


def test_forged_ready_report_fails_validation():
    pending = fx.report(POLICY, fx.with_status(POLICY, "G04_generator_qualification", "pending"))
    forged = pending.model_dump()
    forged["decision"] = "READY"
    with pytest.raises(ValidationError, match="decision"):
        ReadinessReport(**forged)
    forged = pending.model_dump()
    forged["blockers"] = []
    with pytest.raises(ValidationError, match="blockers"):
        ReadinessReport(**forged)


def test_report_with_edited_gate_or_binding_fails_validation():
    data = fx.report(POLICY).model_dump()
    edited = json.loads(json.dumps(data))
    edited["gates"][0]["scope"] = "somewhere else"
    with pytest.raises(ValidationError, match="binding"):
        ReadinessReport(**edited)
    edited = json.loads(json.dumps(data))
    edited["binding_sha256"] = fx.digest("forged")
    with pytest.raises(ValidationError, match="binding"):
        ReadinessReport(**edited)


@pytest.mark.parametrize("mutation", ["drop", "duplicate", "extra"])
def test_report_gate_set_must_be_exact(mutation):
    data = fx.report(POLICY).model_dump()
    if mutation == "drop":
        data["gates"] = data["gates"][:-1]
    elif mutation == "duplicate":
        data["gates"] = data["gates"] + [data["gates"][0]]
    else:
        data["gates"] = data["gates"] + [{**data["gates"][0], "gate_id": "G14_extra"}]
    with pytest.raises(ValidationError):
        ReadinessReport(**data)


def test_exported_schema_matches_the_model():
    from scripts.export_schemas import SCHEMA_DIR, SCHEMAS, schema_bytes

    assert SCHEMAS["readiness_report.schema.json"] is ReadinessReport
    path = SCHEMA_DIR / "readiness_report.schema.json"
    assert path.read_bytes() == schema_bytes(ReadinessReport)
    text = path.read_text()
    assert re.search(r'"G13_final_test_history_integrity"', text)
    assert decide_module  # the schema describes the report; the decision rule lives in decide
