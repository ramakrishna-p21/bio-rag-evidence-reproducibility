"""Prospective production identities and fail-closed retained evidence readers."""

from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import evidence, production
from biorag_repro.readiness.evaluate import system_context
from biorag_repro.readiness.files import FileMissing

REPO = Path(__file__).resolve().parent.parent


def context():
    return system_context(REPO, REPO / "artifacts")


def test_g03_stays_pending_while_the_reviewed_source_identity_is_stale():
    ctx = context()
    source = evidence.scientific_source(ctx)
    assert (source.state, source.reason) == ("failed", "scientific_source_not_clean")
    gathered = production.production_freeze(ctx, source)
    assert (gathered.state, gathered.reason, gathered.subject) == (
        "failed",
        "production_freeze_invalid",
        None,
    )
    assert gathered.detail == "execution source has no verified G09 subject"


def test_g03_refuses_altered_production_identity(monkeypatch):
    ctx = context()
    original = production.read_regular

    def changed(root, relative):
        data = original(root, relative)
        if relative.endswith("g03_production_freeze_2026_11_20.json"):
            import json

            value = json.loads(data)
            value["arm_sha256"]["confirmatory"] = "0" * 64
            return canonical_json_bytes(value)
        return data

    monkeypatch.setattr(production, "read_regular", changed)
    assert production.production_freeze(ctx, evidence.scientific_source(ctx)).state == "failed"


def test_g04_original_archive_change_fails(monkeypatch):
    ctx = context()
    original = production.read_regular

    def changed(root, relative):
        data = original(root, relative)
        if relative.endswith("S1-20261115T205406Z-1-001.zip"):
            return data + b"changed"
        return data

    monkeypatch.setattr(production, "read_regular", changed)
    assert production.qualification(ctx).state == "failed"


def test_g04_missing_archive_has_no_subject(monkeypatch):
    ctx = context()
    original = production.read_regular

    def missing(root, relative):
        if relative.endswith("S1-20261115T205406Z-1-001.zip"):
            raise FileMissing(relative)
        return original(root, relative)

    monkeypatch.setattr(production, "read_regular", missing)
    out = production.qualification(ctx)
    assert out.state == "absent" and out.subject is None


def test_g05_missing_asset_fails(monkeypatch):
    ctx = context()
    real_hash = production.sha256_file

    def changed(path):
        if path.name == "tokenizer.json":
            return "0" * 64
        return real_hash(path)

    monkeypatch.setattr(production, "sha256_file", changed)
    assert production.model_parser(ctx).state == "failed"


def test_g05_missing_asset_has_no_subject(monkeypatch):
    ctx = context()
    real_is_file = Path.is_file

    def missing(path):
        if path.name == "tokenizer.json":
            return False
        return real_is_file(path)

    monkeypatch.setattr(Path, "is_file", missing)
    out = production.model_parser(ctx)
    assert out.state == "failed" and out.subject is None


def test_g07_wrong_host_fails_after_designation(monkeypatch):
    ctx = context()
    sources = {
        path: sha256_bytes((REPO / path).read_bytes())
        for path in [*evidence._r_files(REPO), "analysis/run_analysis.R"]
    }
    monkeypatch.setattr(production.socket, "gethostname", lambda: "wrong-host")
    try:
        production.host_designation(
            ctx, sources, sha256_bytes((REPO / "analysis/renv.lock").read_bytes())
        )
    except evidence.FileProblem as error:
        assert "host" in str(error)
    else:
        raise AssertionError("wrong analysis host was accepted")


def test_g07_altered_designation_is_rejected(monkeypatch):
    ctx = context()
    original = production.read_regular

    def changed(root, relative):
        data = original(root, relative)
        if relative.endswith("g07_researcher_designation_evidence_2026_11_20.json"):
            return data + b" "
        return data

    monkeypatch.setattr(production, "read_regular", changed)
    try:
        production.host_designation(ctx, {}, "0" * 64)
    except evidence.FileProblem as error:
        assert "designation" in str(error)
    else:
        raise AssertionError("altered analysis designation was accepted")
