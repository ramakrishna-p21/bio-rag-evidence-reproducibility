import json
import os

import pytest
from readiness_support import REAL_POLICY, fixture_policy
from readiness_world import build_world, git, write

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.readiness import publish
from biorag_repro.readiness.evaluate import evaluate
from biorag_repro.readiness.policy import policy_sha256
from biorag_repro.readiness.publish import (
    PartialPublication,
    PublicationError,
    ReadbackError,
    compare_current,
    publish_report,
    read_report,
)


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


@pytest.fixture
def published(world):
    evaluation = evaluate(world.ctx, REAL_POLICY)
    return world, evaluation, publish_report(evaluation, world.data)


def report_file(world, published, suffix=".json"):
    return world.data / "readiness/reports" / (published.report_sha256 + suffix)


def test_round_trip_verifies_report_manifest_and_retained_evidence(published):
    world, evaluation, result = published
    back = read_report(world.data, result.report_sha256, REAL_POLICY)
    assert back.outcome.decision == "NOT_READY"
    assert back.report == evaluation.report
    assert back.manifest_sha256 == result.manifest_sha256
    assert back.not_retained == []
    assert back.verified_evidence > 100  # corpus sources, benchmark artifacts and snapshots
    assert back.not_historical > 0  # repo files are never called retained
    assert result.report_sha256 == sha256_bytes(report_file(world, result).read_bytes())


def test_published_report_has_no_self_hash_or_absolute_path(published):
    world, _, result = published
    text = report_file(world, result).read_text()
    assert result.report_sha256 not in text
    assert str(world.data) not in text and str(world.repo) not in text


def test_publishing_the_same_evaluation_again_is_identical_and_idempotent(published):
    world, evaluation, first = published
    second = publish_report(evaluation, world.data)
    assert second.report_sha256 == first.report_sha256 and second.created == []
    assert len(second.reused) == len(first.created)


def test_existing_conflicting_evidence_is_never_replaced(world):
    evaluation = evaluate(world.ctx, REAL_POLICY)
    publish_report(evaluation, world.data)
    victim = world.data / f"readiness/evidence/{policy_sha256(REAL_POLICY)}.json"
    victim.write_bytes(b"{}")
    with pytest.raises(PublicationError, match="different bytes"):
        publish_report(evaluation, world.data)
    assert victim.read_bytes() == b"{}"


def test_snapshot_not_matching_its_hash_is_refused(world):
    evaluation = evaluate(world.ctx, REAL_POLICY)
    key = next(iter(evaluation.snapshots))
    evaluation.snapshots[key] = b"tampered"
    with pytest.raises(PublicationError):
        publish_report(evaluation, world.data)
    assert not (world.data / "readiness").exists()


def test_report_without_manifest_is_partial_and_publishing_again_completes_it(published):
    world, evaluation, result = published
    report_file(world, result, ".build.json").unlink()
    with pytest.raises(PartialPublication):
        read_report(world.data, result.report_sha256, REAL_POLICY)
    again = publish_report(evaluation, world.data)
    assert again.created == [f"readiness/reports/{result.report_sha256}.build.json"]
    read_report(world.data, result.report_sha256, REAL_POLICY)


def test_manifest_without_report_is_refused_and_completed_by_republishing(published):
    world, evaluation, result = published
    report_file(world, result).unlink()
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)
    publish_report(evaluation, world.data)
    read_report(world.data, result.report_sha256, REAL_POLICY)


def test_corrupted_manifest_is_detected(published):
    world, _, result = published
    path = report_file(world, result, ".build.json")
    path.write_bytes(path.read_bytes().replace(b"readiness_report_v1", b"other_transform", 1))
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)
    path.write_bytes(b"not json")
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_manifest_of_another_report_is_detected(published, world):
    _, _, result = published
    other = evaluate(world.ctx, REAL_POLICY)
    other_result = publish_report(other, world.data)
    victim = report_file(world, result, ".build.json")
    victim.write_bytes(report_file(world, other_result, ".build.json").read_bytes())
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_edited_report_bytes_are_detected(published):
    world, _, result = published
    path = report_file(world, result)
    path.write_bytes(path.read_bytes().replace(b"NOT_READY", b"READY\t", 1))
    with pytest.raises(ReadbackError, match="SHA-256"):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_forged_ready_report_stored_under_its_own_hash_is_rejected(published):
    world, _, result = published
    forged = json.loads(report_file(world, result).read_text())
    forged["decision"] = "READY"
    forged["blockers"] = []
    data = (json.dumps(forged, sort_keys=True, indent=2) + "\n").encode()
    digest = sha256_bytes(data)
    write(world.data / f"readiness/reports/{digest}.json", data)
    with pytest.raises(ReadbackError, match="invalid report"):
        read_report(world.data, digest, REAL_POLICY)


def test_forged_gate_status_with_recomputed_report_still_fails_the_binding(published):
    world, _, result = published
    forged = json.loads(report_file(world, result).read_text())
    forged["gates"][0]["status"] = "passed"
    data = (json.dumps(forged, sort_keys=True, indent=2) + "\n").encode()
    digest = sha256_bytes(data)
    write(world.data / f"readiness/reports/{digest}.json", data)
    with pytest.raises(ReadbackError):
        read_report(world.data, digest, REAL_POLICY)


def test_duplicate_keys_in_a_stored_report_are_rejected(published):
    world, _, result = published
    text = report_file(world, result).read_text().replace('"kind"', '"kind": "x", "kind"', 1)
    data = text.encode()
    digest = sha256_bytes(data)
    write(world.data / f"readiness/reports/{digest}.json", data)
    with pytest.raises(ReadbackError):
        read_report(world.data, digest, REAL_POLICY)


def test_altered_snapshot_is_an_error_but_a_missing_one_is_only_not_retained(published):
    world, evaluation, result = published
    cited = next(
        r for g in evaluation.report.gates for r in g.evidence if r.kind == "git_scope_state"
    )
    path = world.data / "readiness/evidence" / cited.locator.removeprefix("snapshot/")
    original = path.read_bytes()
    path.unlink()
    back = read_report(world.data, result.report_sha256, REAL_POLICY)
    assert cited.locator in back.not_retained
    path.write_bytes(original + b" ")
    with pytest.raises(ReadbackError, match="differs"):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_altered_data_root_evidence_is_an_error(published):
    world, _, result = published
    next((world.data / "corpus/raw").rglob("jats_2.bin")).write_bytes(b"<article>x</article>")
    with pytest.raises(ReadbackError, match="differs"):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_missing_policy_copy_fails_readback(published):
    world, _, result = published
    (world.data / f"readiness/evidence/{policy_sha256(REAL_POLICY)}.json").unlink()
    with pytest.raises(ReadbackError, match="policy"):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_symlinked_report_and_symlinked_directory_are_refused(published, tmp_path):
    world, evaluation, result = published
    path = report_file(world, result)
    moved = tmp_path / "moved.json"
    moved.write_bytes(path.read_bytes())
    path.unlink()
    os.symlink(moved, path)
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)
    other = tmp_path / "other"
    other.mkdir()
    (other / "x").mkdir()
    victim = other / "data"
    victim.mkdir()
    os.symlink(tmp_path / "elsewhere", victim / "readiness")
    (tmp_path / "elsewhere").mkdir()
    with pytest.raises(PublicationError, match="plain directory"):
        publish_report(evaluation, victim)
    assert list((tmp_path / "elsewhere").iterdir()) == []


def test_data_root_must_exist(world, tmp_path):
    with pytest.raises(PublicationError):
        publish_report(evaluate(world.ctx, REAL_POLICY), tmp_path / "absent")


def test_report_stays_valid_when_the_checkout_changes_and_compare_says_so(published):
    world, _, result = published
    write(world.repo / "src/code.py", "x = 99\n")
    git(world.repo, "commit", "-qam", "later")
    back = read_report(world.data, result.report_sha256, REAL_POLICY)
    assert back.outcome.decision == "NOT_READY"
    now = compare_current(back.report, world.repo, world.data)
    assert now["head"] == "different" and now["renv_lock"] == "same"
    write(world.repo / "analysis/renv.lock", '{"R":{"Version":"1"},"Packages":{}}')
    assert compare_current(back.report, world.repo, world.data)["renv_lock"] == "different"


def test_a_not_ready_report_is_not_a_credential_for_a_fixture_policy(world):
    evaluation = evaluate(world.ctx, fixture_policy())
    result = publish_report(evaluation, world.data)
    with pytest.raises(ReadbackError):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_policy_copy_with_other_content_fails_readback(published):
    world, _, result = published
    path = world.data / f"readiness/evidence/{policy_sha256(REAL_POLICY)}.json"
    path.write_bytes(b'{"policy_version":"1"}')
    with pytest.raises(ReadbackError, match="policy"):
        read_report(world.data, result.report_sha256, REAL_POLICY)


def test_valid_but_non_canonical_report_bytes_are_refused(published):
    world, _, result = published
    pretty = json.dumps(json.loads(report_file(world, result).read_text()), indent=4).encode()
    digest = sha256_bytes(pretty)
    write(world.data / f"readiness/reports/{digest}.json", pretty)
    with pytest.raises(ReadbackError, match="canonical"):
        read_report(world.data, digest, REAL_POLICY)


def test_publication_reads_the_report_back_before_returning(world, monkeypatch):
    calls = []
    real = publish.read_report
    monkeypatch.setattr(publish, "read_report", lambda *a: calls.append(a) or real(*a))
    publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
    assert len(calls) == 1


def test_publication_fails_if_the_stored_report_does_not_verify(world, monkeypatch):
    def refuse(*args):
        raise ReadbackError("simulated")

    monkeypatch.setattr(publish, "read_report", refuse)
    with pytest.raises(ReadbackError):
        publish_report(evaluate(world.ctx, REAL_POLICY), world.data)
