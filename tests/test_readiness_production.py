"""Prospective production identities and fail-closed retained evidence readers."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import evidence, production
from biorag_repro.readiness.evaluate import system_context
from biorag_repro.readiness.evidence import Probes
from biorag_repro.readiness.files import FileMissing
from biorag_repro.readiness.gather import Gathered

REPO = Path(__file__).resolve().parent.parent


def context():
    return system_context(REPO, REPO / "artifacts")


def test_g03_stays_pending_while_the_reviewed_source_identity_is_stale():
    ctx = context()
    source = Gathered("failed", "scientific_source_not_clean", "source")
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


IDENTITY = "a" * 64


def _lock():
    data = (REPO / "analysis/renv.lock").read_bytes()
    lock = json.loads(data)
    sources = {
        path: sha256_bytes((REPO / path).read_bytes())
        for path in [*evidence._r_files(REPO), *evidence.R_SOURCES]
    }
    return data, lock, sources


def _subject():
    data, lock, sources = _lock()
    designation = (
        REPO / "docs/reports/g07_researcher_designation_evidence_2026_11_20.json"
    ).read_bytes()
    return {
        "kind": "designated_analysis_environment_v1",
        "designation_sha256": sha256_bytes(designation),
        "host": "YOGA-C940",
        "renv_lock_sha256": sha256_bytes(data),
        "analysis_sources_sha256": sources,
        "r_version": lock["R"]["Version"],
        "packages": {
            name: {"version": pkg["Version"], "library_index": "1"}
            for name, pkg in lock["Packages"].items()
        },
        "blas_execution_pin": {"OPENBLAS_CORETYPE": "HASWELL", "OPENBLAS_NUM_THREADS": "1"},
    }


def _record(**changes):
    subject = changes.pop("subject", None) or _subject()
    value = {
        "schema_version": "g07_analysis_host_evidence_v1",
        "producing_host": "YOGA-C940",
        "platform": "WSL Ubuntu 22.04.5",
        "execution_source_identity_sha256": IDENTITY,
        "subject": subject,
        "subject_sha256": sha256_bytes(canonical_json_bytes(subject)),
    }
    return {**value, **changes}


def _colab(tmp_path, monkeypatch, record=None, *, raw=None):
    """This checkout evaluated on the generation host, with the record in its data root."""
    monkeypatch.setattr(production.socket, "gethostname", lambda: "colab-l4")
    if record is not None or raw is not None:
        target = tmp_path / evidence.G07_HOST_RECORD
        target.parent.mkdir(parents=True)
        target.write_bytes(raw if raw is not None else canonical_json_bytes(record))

    def no_r(directory, packages):
        raise OSError("Rscript is not installed on this host")

    ctx = system_context(REPO, tmp_path)
    return replace(ctx, probes=Probes(inventory=ctx.probes.inventory, r_runtime=no_r))


def _source(identity=IDENTITY):
    return Gathered(
        "ok", "checks_passed", "source", facts={"execution_source_identity_sha256": identity}
    )


def test_g07_accepts_retained_evidence_from_the_designated_host(tmp_path, monkeypatch):
    ctx = _colab(tmp_path, monkeypatch, _record())
    g = evidence.r_environment(ctx, _source())
    assert (g.state, g.reason) == ("ok", "checks_passed")
    assert g.subject == _subject()


def test_g07_without_retained_evidence_is_still_not_established_off_host(tmp_path, monkeypatch):
    ctx = _colab(tmp_path, monkeypatch)
    g = evidence.r_environment(ctx, _source())
    assert (g.state, g.reason) == ("unavailable", "r_runtime_probe_unavailable")


def test_g07_on_the_designated_host_ignores_any_retained_record(tmp_path, monkeypatch):
    ctx = _colab(tmp_path, monkeypatch, raw=b"not even json")
    monkeypatch.setattr(production.socket, "gethostname", lambda: "YOGA-C940")
    monkeypatch.setattr(production.platform, "release", lambda: "6.0-microsoft-standard-WSL2")
    g = evidence.r_environment(ctx, _source())
    assert (g.state, g.reason) == ("unavailable", "r_runtime_probe_unavailable")


def _altered_subject(**changes):
    return {**_subject(), **changes}


def _altered_package():
    subject = _subject()
    name = next(iter(subject["packages"]))
    subject["packages"][name] = {"version": "0.0.0", "library_index": "1"}
    return subject


@pytest.mark.parametrize(
    ("label", "record", "reason"),
    [
        ("altered environment identity", lambda: _record(subject=_altered_package()), "invalid"),
        (
            "subject altered after hashing",
            lambda: _record(subject_sha256="0" * 64),
            "invalid",
        ),
        (
            "other R version",
            lambda: _record(subject=_altered_subject(r_version="3.0.0")),
            "invalid",
        ),
        (
            "other lock",
            lambda: _record(subject=_altered_subject(renv_lock_sha256="0" * 64)),
            "invalid",
        ),
        (
            "other analysis sources",
            lambda: _record(subject=_altered_subject(analysis_sources_sha256={"x": "0" * 64})),
            "invalid",
        ),
        ("generation host as source", lambda: _record(producing_host="colab-l4"), "invalid"),
        (
            "subject names another host",
            lambda: _record(subject=_altered_subject(host="colab-l4")),
            "invalid",
        ),
        (
            "unsupported schema",
            lambda: _record(schema_version="g07_analysis_host_evidence_v2"),
            "invalid",
        ),
        ("unknown field", lambda: {**_record(), "extra": 1}, "invalid"),
        (
            "stale source identity",
            lambda: _record(execution_source_identity_sha256="b" * 64),
            "stale",
        ),
    ],
)
def test_g07_rejects_altered_substituted_or_stale_evidence(
    tmp_path, monkeypatch, label, record, reason
):
    ctx = _colab(tmp_path, monkeypatch, record())
    g = evidence.r_environment(ctx, _source())
    assert g.state == "failed" and g.subject is None, label
    assert g.reason == f"analysis_host_evidence_{reason}", label


def test_g07_evidence_needs_a_verified_source_identity(tmp_path, monkeypatch):
    ctx = _colab(tmp_path, monkeypatch, _record())
    assert evidence.r_environment(ctx).reason == "analysis_host_evidence_invalid"
    unverified = Gathered("failed", "scientific_source_not_clean", "source")
    assert evidence.r_environment(ctx, unverified).reason == "analysis_host_evidence_invalid"


def test_g07_evidence_cannot_be_exported_from_another_host(tmp_path, monkeypatch):
    ctx = _colab(tmp_path, monkeypatch, _record())
    with pytest.raises(ValueError):
        evidence.export_analysis_host_evidence(ctx, _source())


def test_retained_g07_evidence_does_not_stand_in_for_the_live_g06_check(tmp_path, monkeypatch):
    def no_inventory():
        raise RuntimeError("no live inventory")

    ctx = _colab(tmp_path, monkeypatch, _record())
    ctx = replace(ctx, probes=replace(ctx.probes, inventory=no_inventory))
    g = evidence.python_environment(ctx)
    assert (g.state, g.reason) == ("unavailable", "environment_inventory_unavailable")


def test_the_designated_host_exports_evidence_that_the_generation_host_accepts(
    tmp_path, monkeypatch
):
    _, lock, _ = _lock()
    library = str((REPO / "analysis/renv/library").resolve())
    lines = [f"BIORAG\tlibpaths\t{library}", f"BIORAG\trversion\t{lock['R']['Version']}"]
    lines += [f"BIORAG\tpkg\t{n}\t{p['Version']}\t1" for n, p in lock["Packages"].items()]
    live = _colab(tmp_path, monkeypatch)
    live = replace(live, probes=replace(live.probes, r_runtime=lambda d, p: "\n".join(lines)))
    monkeypatch.setattr(production.socket, "gethostname", lambda: "YOGA-C940")
    monkeypatch.setattr(production.platform, "release", lambda: "6.0-microsoft-standard-WSL2")
    monkeypatch.setattr(
        production.Path,
        "read_text",
        lambda self, *a, **k: 'VERSION_ID="22.04"' if str(self) == "/etc/os-release" else "",
    )
    exported = evidence.export_analysis_host_evidence(live, _source())
    monkeypatch.undo()
    target = tmp_path / evidence.G07_HOST_RECORD
    target.parent.mkdir(parents=True)
    target.write_bytes(exported)
    monkeypatch.setattr(production.socket, "gethostname", lambda: "colab-l4")
    g = evidence.r_environment(live, _source())
    assert (g.state, g.reason) == ("ok", "checks_passed")
