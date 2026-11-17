"""Scientific choices do not supply missing technical approvals."""

import json
import shutil
from pathlib import Path

import pytest

from biorag_repro.evaluation.seeds import APPROVED_REGISTRY_SHA256, read_analysis_seed
from biorag_repro.generation import config
from biorag_repro.readiness import evidence
from biorag_repro.readiness.approvals import PolicyError, load_policy
from biorag_repro.readiness.gather import to_result
from biorag_repro.readiness.policy import ApprovalPolicy, ApprovalSlots

ROOT = Path(__file__).resolve().parents[1]


def copy_registration(tmp_path):
    for name in (
        "configs/analysis_seed.json",
        "approvals/researcher_decisions.json",
        "approvals/policy.json",
    ):
        target = tmp_path / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    return tmp_path


def test_exact_checkpoint_seed_and_unset_technical_approvals():
    assert config.APPROVED_CHECKPOINTS == {
        (
            "Qwen/Qwen3-8B",
            "b968826d9c46dd6066d109eabc6255188de91218",
        ): "4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad"
    }
    seed = read_analysis_seed(ROOT)
    assert (seed.name, seed.value, seed.resamples) == ("primary_bootstrap", 1572911995, 10000)
    assert config.APPROVED_ENVIRONMENT_LOCKS == frozenset()
    assert all(v is None for v in load_policy(ROOT).slots.model_dump().values())
    assert len(APPROVED_REGISTRY_SHA256) == 64


def test_seed_or_authorization_drift_fails(tmp_path):
    root = copy_registration(tmp_path)
    path = root / "configs/analysis_seed.json"
    original = path.read_bytes()
    value = json.loads(original)
    value["value"] += 1
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="approved registry"):
        read_analysis_seed(root)
    path.write_bytes(original)
    (root / "approvals/researcher_decisions.json").write_text("{}")
    with pytest.raises(ValueError, match="decision record"):
        read_analysis_seed(root)
    with pytest.raises(PolicyError, match="authority"):
        load_policy(root)


def test_checkpoint_registration_does_not_pass_technical_gates():
    policy = ApprovalPolicy(policy_version="2", slots=ApprovalSlots())
    for gate, g in (
        ("G04_generator_qualification", evidence.generator_qualification()),
        ("G05_model_parser_identities", evidence.model_parser_identities()),
    ):
        assert to_result(gate, g, policy).status == "pending"


def test_g08_registers_subject_but_does_not_approve_it(tmp_path, monkeypatch):
    from readiness_world import build_world

    world = build_world(tmp_path, monkeypatch)
    for name in ("configs/analysis_seed.json", "approvals/researcher_decisions.json"):
        target = world.repo / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    release, _ = evidence.read_benchmark_record(world.ctx)
    result = evidence.seed_registry(world.ctx, release)
    assert result.state == "ok"
    assert result.subject["analysis_seed_registry_sha256"] == APPROVED_REGISTRY_SHA256
    policy = ApprovalPolicy(policy_version="2", slots=ApprovalSlots())
    assert to_result("G08_seed_registry", result, policy).status == "pending"
