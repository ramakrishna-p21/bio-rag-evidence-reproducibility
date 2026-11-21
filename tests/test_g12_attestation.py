"""G12 keeps reported advisor guidance and researcher permission distinct."""

import json
from types import SimpleNamespace

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import evidence


def put(root, name, data):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def records(tmp_path, monkeypatch, restriction="confirmed_compliant"):
    repo, data = tmp_path / "repo", tmp_path / "data"
    amendment = b"synthetic amendment"
    monkeypatch.setattr(evidence, "GPU_AMENDMENT_SHA256", sha256_bytes(amendment))
    put(repo, evidence.GPU_AMENDMENT, amendment)
    advisor = {
        "schema_version": "g12_researcher_guidance_v1",
        "researcher": "Rama Krishna Pudota",
        "attested_on": "2026-11-20",
        "advisor": "Professor William Andreopoulos",
        "source": "researcher_reported_verbal_guidance",
        "guidance": "conditional_support_for_colab",
        "condition": "scientifically_valid_and_defensible",
        "colab_gpu_use": True,
        "restartable_batches": None,
        "study_material_placement": None,
    }
    put(data, evidence.GPU_ATTESTATION, canonical_json_bytes(advisor))
    storage = b"synthetic storage"
    monkeypatch.setattr(evidence, "GPU_G11_SHA256", sha256_bytes(storage))
    put(data, f"{evidence.STORAGE_APPROVALS}/{evidence.GPU_G11_SHA256}.json", storage)
    campaign = {
        "campaign_id": "rk_capstone_final_test_2026",
        "execution_root": "/content/rk_capstone_final_2026",
        "drive_root": "/content/drive/MyDrive/RK_Capstone_Final_2026",
        "custody_root": "/home/ramakrishnap/colab_transfer/official_custody_heads",
        "g11_authorization_sha256": evidence.GPU_G11_SHA256,
    }
    campaign_bytes = canonical_json_bytes(campaign)
    monkeypatch.setattr(evidence, "GPU_CAMPAIGN_DECISION_SHA256", sha256_bytes(campaign_bytes))
    put(data, evidence.GPU_CAMPAIGN_DECISION, campaign_bytes)
    operational = {
        "schema_version": "g12_operational_v1",
        "researcher": "Rama Krishna Pudota",
        "attested_on": "2026-11-20",
        "campaign_id": campaign["campaign_id"],
        "platform": "researcher_controlled_colab_pro",
        "gpu_model": "NVIDIA L4",
        "execution_root": campaign["execution_root"],
        "drive_root": campaign["drive_root"],
        "custody_root": campaign["custody_root"],
        "g11_authorization_sha256": evidence.GPU_G11_SHA256,
        "campaign_storage_decision_sha256": evidence.GPU_CAMPAIGN_DECISION_SHA256,
        "restartable_batches_authorized": True,
        "restricted_material_authorized": True,
        "external_restrictions_status": restriction,
    }
    put(data, evidence.GPU_OPERATIONAL, canonical_json_bytes(operational))
    return SimpleNamespace(repo=repo, data_root=data), advisor


def test_distinct_authorities_can_produce_technical_subject(tmp_path, monkeypatch):
    ctx, _ = records(tmp_path, monkeypatch)
    found = evidence.shared_gpu_authorization(ctx)
    assert (found.state, found.reason) == ("ok", "checks_passed")
    assert found.subject["campaign_id"] == "rk_capstone_final_test_2026"
    assert len(found.refs) == 5


def test_external_restrictions_pending_or_conflicting(tmp_path, monkeypatch):
    ctx, _ = records(tmp_path, monkeypatch, "pending_account_specific_review")
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_external_restrictions_unconfirmed"
    path = ctx.data_root / evidence.GPU_OPERATIONAL
    value = json.loads(path.read_bytes())
    path.write_bytes(canonical_json_bytes(value | {"external_restrictions_status": "conflict"}))
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_external_restriction_conflict"


def test_advisor_scope_fabrication_and_changed_authority_fail(tmp_path, monkeypatch):
    ctx, advisor = records(tmp_path, monkeypatch)
    put(
        ctx.data_root,
        evidence.GPU_ATTESTATION,
        canonical_json_bytes(advisor | {"restartable_batches": True}),
    )
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_attestation_invalid"
    put(ctx.data_root, evidence.GPU_ATTESTATION, canonical_json_bytes(advisor))
    put(ctx.repo, evidence.GPU_AMENDMENT, b"changed")
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_amendment_changed"


def test_missing_or_changed_operational_evidence_fails_closed(tmp_path, monkeypatch):
    ctx, _ = records(tmp_path, monkeypatch)
    path = ctx.data_root / evidence.GPU_OPERATIONAL
    path.unlink()
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_operational_authorization_absent"
    put(ctx.data_root, evidence.GPU_OPERATIONAL, b"{}")
    assert evidence.shared_gpu_authorization(ctx).reason == "gpu_operational_authorization_invalid"
