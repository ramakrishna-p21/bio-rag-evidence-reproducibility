"""Gate adapters on a tiny synthetic checkout and data root."""

import hashlib
import json
import pathlib
import re
import shutil
from dataclasses import replace

import pytest
from readiness_support import REAL_POLICY, digest
from readiness_world import R_OUTPUT, build_world, git, write

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.readiness import evidence
from biorag_repro.readiness.evaluate import gather_all
from biorag_repro.readiness.gather import to_result
from biorag_repro.readiness.policy import ApprovalPolicy, ApprovalSlots


@pytest.fixture
def world(tmp_path, monkeypatch):
    return build_world(tmp_path, monkeypatch)


def gathered(world, gate):
    return gather_all(world.ctx)[gate]


def subject_digest(g):
    return sha256_bytes(canonical_json_bytes(g.subject))


# ---- G01 ----------------------------------------------------------------------------------


def test_g01_checks_every_source_and_names_the_exact_release(world):
    g = gathered(world, "G01_corpus_sources")
    assert g.state == "ok" and g.detail == "50 papers, 100 source files"
    kinds = [r.kind for r in g.refs]
    assert kinds.count("corpus_source_jats") == 50 and kinds.count("corpus_source_pdf") == 50
    # independent oracle: the hashes of the files on disk
    on_disk = sorted(
        hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (world.data / "corpus/raw").rglob("*.bin")
    )
    assert on_disk == sorted(r.sha256 for r in g.refs if r.kind.startswith("corpus_source"))
    assert len(g.subject["sources"]) == 100


def test_g01_source_bytes_changed_fails(world):
    victim = next((world.data / "corpus/raw").rglob("jats_7.bin"))
    victim.write_bytes(b"<article>changed</article>")
    g = gathered(world, "G01_corpus_sources")
    assert (g.state, g.reason) == ("failed", "source_bytes_differ")


def test_g01_wrong_magic_with_matching_hash_still_fails(world, monkeypatch):
    victim = next((world.data / "corpus/raw").rglob("pdf_3.bin"))
    victim.write_bytes(b"%PDF-1.4\n%%EOF\n")  # unchanged: sanity
    assert gathered(world, "G01_corpus_sources").state == "ok"
    victim.write_bytes(b"not a pdf at all")
    assert gathered(world, "G01_corpus_sources").state == "failed"


def test_g01_missing_source_and_symlinked_source_fail(world):
    victim = next((world.data / "corpus/raw").rglob("jats_9.bin"))
    original = victim.read_bytes()
    victim.unlink()
    assert gathered(world, "G01_corpus_sources").reason == "corpus_artifact_missing"
    elsewhere = world.data.parent / "elsewhere.bin"
    elsewhere.write_bytes(original)
    victim.symlink_to(elsewhere)
    assert gathered(world, "G01_corpus_sources").reason == "corpus_artifact_invalid"


def test_g01_without_the_benchmark_record_is_unavailable(world):
    (world.data / evidence.BENCHMARK_RELEASE).unlink()
    assert gathered(world, "G01_corpus_sources").state == "unavailable"


def test_g01_altered_release_record_fails(world):
    path = next((world.data / "corpus/releases").rglob("release.json"))
    path.write_bytes(path.read_bytes() + b" ")
    assert gathered(world, "G01_corpus_sources").state == "failed"


def test_g01_self_consistent_but_unapproved_stays_pending(world):
    result = to_result("G01_corpus_sources", gathered(world, "G01_corpus_sources"), REAL_POLICY)
    assert result.status == "pending" and result.reason_code == "verified_pending_policy_approval"


# ---- G02 ----------------------------------------------------------------------------------


def test_g02_rederives_counts_from_the_stored_items(world):
    g = gathered(world, "G02_benchmark_freeze")
    assert g.state == "ok"
    assert (g.subject["item_count"], g.subject["development_items"]) == (5, 2)
    assert g.subject["final_test_items"] == 3


def test_g02_runs_the_existing_release_verifier(world, monkeypatch):
    calls = []
    monkeypatch.setattr(evidence, "verify_release_inputs", lambda r, d: calls.append(d))
    assert gathered(world, "G02_benchmark_freeze").state == "ok" and calls == [world.data]


def test_g02_verifier_rejection_is_a_failure_not_pending(world, monkeypatch):
    def refuse(release, root):
        raise ValueError("final benchmark validation has open findings")

    monkeypatch.setattr(evidence, "verify_release_inputs", refuse)
    g = gathered(world, "G02_benchmark_freeze")
    assert (g.state, g.reason) == ("failed", "benchmark_release_verification_failed")


def test_g02_missing_dependency_is_unavailable_not_skipped(world, monkeypatch):
    def missing(release, root):
        raise ModuleNotFoundError("No module named 'fitz'")

    monkeypatch.setattr(evidence, "verify_release_inputs", missing)
    g = gathered(world, "G02_benchmark_freeze")
    assert (g.state, g.reason) == ("unavailable", "benchmark_validation_dependency_missing")


def test_g02_changed_items_fail_before_the_verifier_is_trusted(world):
    path = world.data / "benchmark/releases/benchmark_v1/items.jsonl"
    path.write_bytes(path.read_bytes().replace(b"TP53", b"TP54", 1))
    assert gathered(world, "G02_benchmark_freeze").state == "failed"


def test_g02_item_count_disagreement_fails(world):
    from biorag_repro.benchmark.release import BenchmarkRelease
    from biorag_repro.core.storage import serialize_project_json

    path = world.data / evidence.BENCHMARK_RELEASE
    release = BenchmarkRelease.model_validate_json(path.read_bytes())
    path.write_bytes(
        serialize_project_json(release.model_copy(update={"item_count": 6}), BenchmarkRelease)
    )
    assert gathered(world, "G02_benchmark_freeze").state == "failed"


def test_g02_implementation_commit_must_be_in_history(world):
    from biorag_repro.benchmark.release import BenchmarkRelease
    from biorag_repro.core.storage import serialize_project_json

    path = world.data / evidence.BENCHMARK_RELEASE
    release = BenchmarkRelease.model_validate_json(path.read_bytes())
    path.write_bytes(
        serialize_project_json(
            release.model_copy(update={"implementation_git_sha": "1" * 40}), BenchmarkRelease
        )
    )
    assert gathered(world, "G02_benchmark_freeze").reason == "implementation_commit_not_in_history"


def test_g02_does_not_leak_item_text_into_evidence_or_detail(world):
    g = gathered(world, "G02_benchmark_freeze")
    blob = json.dumps([r.model_dump() for r in g.refs]) + (g.detail or "")
    assert "TP53" not in blob and "Question about" not in blob


def test_g02_hash_only_requires_exact_approved_witness_without_decoding(world, monkeypatch):
    from biorag_repro.benchmark.release import BenchmarkRelease

    raw = (world.data / evidence.BENCHMARK_RELEASE).read_bytes()
    release = BenchmarkRelease.model_validate_json(raw)

    def forbidden(*args):
        raise AssertionError("protected items were decoded")

    monkeypatch.setattr(evidence, "decode_jsonl", forbidden)
    monkeypatch.setattr(evidence, "verify_release_inputs", forbidden)
    gate = gather_all(world.ctx, g02_hash_only=True)["G02_benchmark_freeze"]
    assert gate.reason == "benchmark_count_witness_missing"
    witness = {
        "schema_version": "benchmark_count_witness_v1",
        "researcher": "Rama Krishna Pudota",
        "approved": True,
        "benchmark_version": release.benchmark_version,
        "release_record_sha256": sha256_bytes(raw),
        "items_sha256": release.items.sha256,
        "split_sha256": release.split.sha256,
        "validation_report_sha256": release.validation_report.sha256,
        "item_count": release.item_count,
        "development_items": 2,
        "final_test_items": 3,
    }
    approval = world.repo / "approvals/benchmark_count_witness_v1.json"
    approval.parent.mkdir(exist_ok=True)
    approval.write_text(json.dumps(witness))
    assert gather_all(world.ctx, g02_hash_only=True)["G02_benchmark_freeze"].state == "ok"
    witness["approved"] = False
    approval.write_text(json.dumps(witness))
    assert gather_all(world.ctx, g02_hash_only=True)["G02_benchmark_freeze"].state == "failed"


# ---- G06 ----------------------------------------------------------------------------------


def test_g06_provisional_lock_is_pending_with_inventory_snapshot(world):
    g = gathered(world, "G06_python_environment")
    assert (g.state, g.reason) == ("absent", "environment_lock_not_approved")
    assert {"python_inventory", "environment_verification", "environment_spec"} <= {
        r.kind for r in g.refs
    }


def test_g06_inventory_probe_failure_is_unavailable(world):
    def boom():
        raise OSError("no")

    world.ctx = world.probes(inventory=boom)
    assert gathered(world, "G06_python_environment").state == "unavailable"


def test_g06_unusable_spec_fails(world):
    (world.repo / "configs/environment.toml").write_text("status = ")
    assert gathered(world, "G06_python_environment").reason == "environment_spec_unusable"


def test_g06_approved_lock_that_matches_passes_only_against_a_matching_slot(world, monkeypatch):
    from synthetic_environment import LOCK

    (world.repo / "configs/environment.toml").write_bytes(LOCK)
    lock_sha = sha256_bytes(LOCK)
    assert gathered(world, "G06_python_environment").reason == "environment_lock_not_approved"
    monkeypatch.setattr(
        "biorag_repro.generation.config.APPROVED_ENVIRONMENT_LOCKS", frozenset({lock_sha})
    )
    g = gathered(world, "G06_python_environment")
    assert g.state == "ok" and g.subject == {
        "kind": "python_environment_lock_v1",
        "lock_sha256": lock_sha,
    }
    slot = sha256_bytes(canonical_json_bytes(g.subject))
    policy = ApprovalPolicy(policy_version="t", slots=ApprovalSlots(python_environment_lock=slot))
    assert to_result("G06_python_environment", g, policy).status == "passed"
    wrong = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(python_environment_lock=digest("x"))
    )
    assert to_result("G06_python_environment", g, wrong).reason_code == "policy_mismatch"


def test_g06_approved_lock_that_differs_from_the_machine_fails(world, monkeypatch):
    from synthetic_environment import LOCK, inventory

    (world.repo / "configs/environment.toml").write_bytes(LOCK)
    monkeypatch.setattr(
        "biorag_repro.generation.config.APPROVED_ENVIRONMENT_LOCKS",
        frozenset({sha256_bytes(LOCK)}),
    )
    world.ctx = world.probes(inventory=lambda: inventory(packages={"torch": "0.0.1"}))
    g = gathered(world, "G06_python_environment")
    assert (g.state, g.reason) == ("failed", "installed_environment_differs_from_approved_lock")


# ---- G07 ----------------------------------------------------------------------------------


def test_g07_matching_runtime_stays_pending_because_no_host_is_designated(world):
    g = gathered(world, "G07_r_analysis_environment")
    assert (g.state, g.reason) == ("absent", "analysis_host_not_designated")
    kinds = {r.kind for r in g.refs}
    assert {"renv_lock", "analysis_sources", "r_inspection"} <= kinds


def test_g07_package_version_difference_fails(world):
    out = R_OUTPUT.replace("2.6.1", "2.6.0")
    world.ctx = world.probes(r_runtime=lambda d, p: out.format(lib=d / "renv/library/x"))
    assert (
        gathered(world, "G07_r_analysis_environment").reason
        == "installed_r_packages_differ_from_lock"
    )


def test_g07_missing_package_fails(world):
    out = "\n".join(l for l in R_OUTPUT.splitlines() if "pkg" not in l)
    world.ctx = world.probes(r_runtime=lambda d, p: out.format(lib=d / "renv/library/x"))
    assert gathered(world, "G07_r_analysis_environment").state == "failed"


def test_g07_r_version_difference_fails(world):
    out = R_OUTPUT.replace("4.6.1", "4.5.0")
    world.ctx = world.probes(r_runtime=lambda d, p: out.format(lib=d / "renv/library/x"))
    assert gathered(world, "G07_r_analysis_environment").reason == "r_version_differs_from_lock"


def test_g07_unrun_r_is_unavailable_and_claims_no_runtime_conformity(world):
    def boom(d, p):
        raise FileNotFoundError("Rscript")

    world.ctx = world.probes(r_runtime=boom)
    g = gathered(world, "G07_r_analysis_environment")
    assert (g.state, g.reason) == ("unavailable", "r_runtime_probe_unavailable")
    assert "not verified" in g.detail


def test_g07_inactive_project_library_is_unavailable(world):
    world.ctx = world.probes(r_runtime=lambda d, p: R_OUTPUT.format(lib="/usr/lib/R"))
    assert gathered(world, "G07_r_analysis_environment").reason == "r_project_library_not_active"


def test_g07_incomplete_probe_output_is_unavailable(world):
    world.ctx = world.probes(r_runtime=lambda d, p: "garbage")
    assert gathered(world, "G07_r_analysis_environment").state == "unavailable"


def test_g07_lock_with_duplicate_keys_is_unusable(world):
    (world.repo / "analysis/renv.lock").write_text('{"R":{"Version":"1"},"R":{"Version":"2"}}')
    assert gathered(world, "G07_r_analysis_environment").reason == "analysis_lock_unusable"


def test_g07_with_a_populated_slot_there_is_still_nothing_to_approve(world):
    g = gathered(world, "G07_r_analysis_environment")
    policy = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(analysis_environment=digest("x"))
    )
    assert to_result("G07_r_analysis_environment", g, policy).status == "pending"


# ---- G08 ----------------------------------------------------------------------------------


def test_g08_verified_construction_seeds_do_not_approve_a_production_seed(world):
    g = gathered(world, "G08_seed_registry")
    assert (g.state, g.reason) == ("absent", "production_analysis_seed_unapproved")
    assert g.refs and g.refs[0].kind == "benchmark_seed_registry"


def test_g08_corrupt_registry_fails(world):
    path = world.data / "benchmark/working/seeds"
    path.write_bytes(path.read_bytes() + b" ")
    assert gathered(world, "G08_seed_registry").state == "failed"


def test_no_integer_seed_value_is_special_cased():
    import pathlib

    text = "".join(p.read_text() for p in pathlib.Path(evidence.__file__).parent.glob("*.py"))
    assert "20260601" not in text


# ---- G09 ----------------------------------------------------------------------------------


def test_g09_clean_tree_records_tracked_and_frozen_identities(world):
    g = gathered(world, "G09_scientific_source_revision")
    assert g.state == "ok" and g.facts["head_sha"] == world.commit
    names = {r.locator for r in g.refs if r.kind == "frozen_specification"}
    assert len(names) == 12 and all(n.startswith("repo/docs/frozen/") for n in names)
    assert g.subject["frozen_specifications"]["ARCHITECTURE.md"] == sha256_bytes(
        b"frozen ARCHITECTURE\n"
    )


def test_g09_frozen_specification_change_changes_the_subject_without_git_noticing(world):
    before = subject_digest(gathered(world, "G09_scientific_source_revision"))
    write(world.repo / "docs/frozen/BENCHMARK_SPEC.md", "edited\n")
    assert git(world.repo, "status", "--porcelain") == ""
    after = subject_digest(gathered(world, "G09_scientific_source_revision"))
    assert before != after


def test_g09_drive_amendment_change_changes_the_subject(world):
    before = subject_digest(gathered(world, "G09_scientific_source_revision"))
    write(world.repo / "docs/frozen/DRIVE_CAMPAIGN_CUSTODY_AMENDMENT_V1.md", "edited\n")
    assert subject_digest(gathered(world, "G09_scientific_source_revision")) != before


def test_g09_changed_frozen_specification_fails_a_populated_slot(world):
    g = gathered(world, "G09_scientific_source_revision")
    policy = ApprovalPolicy(
        policy_version="t",
        slots=ApprovalSlots(scientific_source_identity=subject_digest(g)),
    )
    assert to_result("G09_scientific_source_revision", g, policy).status == "passed"
    write(world.repo / "docs/frozen/BENCHMARK_SPEC.md", "edited\n")
    g2 = gathered(world, "G09_scientific_source_revision")
    assert to_result("G09_scientific_source_revision", g2, policy).reason_code == "policy_mismatch"


def test_g09_missing_frozen_specification_is_unavailable(world):
    (world.repo / "docs/frozen/ARCHITECTURE.md").unlink()
    g = gathered(world, "G09_scientific_source_revision")
    assert (g.state, g.reason) == ("unavailable", "frozen_specification_unavailable")


def test_g09_descendant_commit_with_other_content_is_a_different_identity(world):
    first = subject_digest(gathered(world, "G09_scientific_source_revision"))
    write(world.repo / "src/code.py", "x = 2\n")
    git(world.repo, "commit", "-qam", "change")
    g = gathered(world, "G09_scientific_source_revision")
    assert g.state == "ok" and subject_digest(g) != first and g.facts["head_sha"] != world.commit


def test_g09_untracked_and_modified_scientific_source_fail(world):
    write(world.repo / "src/odd nameé.py", "z = 1\n")
    g = gathered(world, "G09_scientific_source_revision")
    assert (g.state, g.reason) == ("failed", "scientific_source_not_clean")
    (world.repo / "src/odd nameé.py").unlink()
    write(world.repo / "configs/environment.toml", "# edited\n")
    assert gathered(world, "G09_scientific_source_revision").state == "failed"


def test_g09_ignored_runtime_output_is_not_a_failure(world):
    write(world.repo / "out/run.txt", "x")
    assert gathered(world, "G09_scientific_source_revision").state == "ok"


def test_g09_untracked_outside_scope_only_warns(world):
    write(world.repo / "notes.txt", "x")
    world.ctx = world.probes()
    assert gathered(world, "G09_scientific_source_revision").state == "ok"
    assert any("outside the scientific scope" in w for w in world.ctx.warnings)


def test_g09_not_a_repository_is_unavailable(world, tmp_path):
    world.ctx = replace(world.ctx, repo=tmp_path / "nowhere")
    assert gathered(world, "G09_scientific_source_revision").state == "unavailable"


# ---- the gates with no evidence yet -------------------------------------------------------

NO_EVIDENCE = {
    "G03_final_test_configuration_freeze": "final_test_freeze_absent",
    "G04_generator_qualification": "independent_phase7_evidence_absent",
    "G05_model_parser_identities": "checkpoint_and_parser_identities_not_approved",
    "G10_development_restart_recovery": "official_restart_evidence_absent",
    "G12_shared_gpu_authorization": "gpu_authorization_absent",
}


@pytest.mark.parametrize(("gate", "reason"), NO_EVIDENCE.items())
def test_unevidenced_gates_stay_pending_even_when_their_slot_is_populated(world, gate, reason):
    from biorag_repro.readiness.policy import GATE_SLOT

    g = gathered(world, gate)
    assert (g.state, g.reason) == ("absent", reason)
    populated = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(**{GATE_SLOT[gate]: digest("approved")})
    )
    assert to_result(gate, g, populated).status == "pending"


def test_local_gpu_hardware_never_approves_the_shared_gpu_gate(world):
    from synthetic_environment import inventory

    world.ctx = world.probes(inventory=lambda: inventory())  # reports a GPU
    g = gathered(world, "G12_shared_gpu_authorization")
    assert to_result("G12_shared_gpu_authorization", g, REAL_POLICY).status == "pending"


def test_g11_without_approved_record_fails_closed(world):
    policy = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(storage_authorization=digest("missing"))
    )
    gathered = evidence.storage_authorization(world.ctx, policy)
    assert gathered.state == "failed"
    assert to_result("G11_storage_authorization", gathered, policy).status == "failed"


def test_g13_is_unavailable_whatever_the_data_root_contains(world, tmp_path):
    assert gathered(world, "G13_final_test_history_integrity").reason == (
        "final_test_inventory_authority_not_established"
    )
    (world.data / "evaluation").mkdir()  # an empty or populated location changes nothing
    assert gathered(world, "G13_final_test_history_integrity").state == "unavailable"


# ---- G03 reader -----------------------------------------------------------------------------


def test_g03_reads_a_future_freeze_record_without_creating_one():
    record = json.dumps({"schema_version": "1", "settings_sha256": {"retrieval": digest("r")}})
    g = evidence.final_test_freeze(record.encode())
    assert g.state == "ok" and g.subject["settings_sha256"] == {"retrieval": digest("r")}
    policy = ApprovalPolicy(
        policy_version="t", slots=ApprovalSlots(final_test_freeze=subject_digest(g))
    )
    assert to_result("G03_final_test_configuration_freeze", g, policy).status == "passed"
    assert to_result("G03_final_test_configuration_freeze", g, REAL_POLICY).status == "pending"


@pytest.mark.parametrize(
    "bad", [b"{}", b'{"schema_version":"1","settings_sha256":{}}', b"nope", b'{"a":1,"a":2}']
)
def test_g03_malformed_record_fails(bad):
    assert evidence.final_test_freeze(bad).state == "failed"


# ---- cases the first mutation run showed were untested ---------------------------------------


def _rewrite_release_ref(world, field, new_content, path):
    """Replace an artifact's bytes and make the release record cite the new bytes."""
    from biorag_repro.benchmark.release import BenchmarkRelease
    from biorag_repro.core.storage import serialize_project_json

    record_path = world.data / evidence.BENCHMARK_RELEASE
    release = BenchmarkRelease.model_validate_json(record_path.read_bytes())
    (world.data / path).write_bytes(new_content)
    ref = getattr(release, field).model_copy(update={"sha256": sha256_bytes(new_content)})
    record_path.write_bytes(
        serialize_project_json(release.model_copy(update={field: ref}), BenchmarkRelease)
    )


def test_g08_registry_with_matching_hash_but_invalid_content_fails(world):
    _rewrite_release_ref(world, "seed_registry", b'{"entries": []}\n', "benchmark/working/seeds")
    assert gathered(world, "G08_seed_registry").reason == "seed_registry_invalid"


def test_g01_source_with_matching_hash_but_wrong_media_type_fails(world, monkeypatch):
    """The recorded hash is right; the bytes are not a PDF. The format check must still run."""
    from biorag_repro.corpus import manifest as corpus

    original = corpus.verify_source_bytes
    seen = []

    def spy(paper, kind, path):
        seen.append((paper.paper_id, kind))
        if paper.paper_id == "P004" and kind == "pdf":
            raise ValueError("PDF media type does not match bytes")
        original(paper, kind, path)

    monkeypatch.setattr(evidence, "verify_source_bytes", spy)
    g = gathered(world, "G01_corpus_sources")
    assert (g.state, g.reason) == ("failed", "corpus_artifact_invalid")
    assert ("P001", "jats") in seen


def test_g01_corpus_validation_errors_fail_the_gate(world, monkeypatch):
    from biorag_repro.corpus.manifest import Finding

    real = evidence.validate_corpus

    def with_error(manifest):
        report = real(manifest)
        finding = report.findings[0] if report.findings else None
        assert finding is None or isinstance(finding, Finding)
        return report.model_copy(
            update={
                "findings": [
                    Finding(
                        rule="x",
                        severity="hard_error",
                        paper_id="P001",
                        representation=None,
                        artifact_id=None,
                    ),
                    *report.findings,
                ]
            }
        )

    monkeypatch.setattr(evidence, "validate_corpus", with_error)
    g = gathered(world, "G01_corpus_sources")
    assert (g.state, g.reason) == ("failed", "corpus_validation_failed")


def test_g01_recomputed_content_hash_must_equal_the_recorded_one(world, monkeypatch):
    monkeypatch.setattr(evidence, "corpus_content_hash", lambda manifest: "0" * 64)
    assert gathered(world, "G01_corpus_sources").reason == "corpus_identity_differs"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript is not installed")
def test_the_r_probe_reports_description_versions_as_written(tmp_path):
    """DESCRIPTION text such as 0.22-5 must not be rewritten to 0.22.5 (as packageVersion does)."""
    observed = evidence.parse_r_runtime(evidence.run_r_runtime(tmp_path, ["base", "lattice"]))
    assert observed["libpaths"] and observed["packages"]["base"]["version"] == observed["rversion"]
    assert "-" not in observed["packages"]["base"]["version"]
    description = next(
        (pathlib.Path(p) / "lattice" / "DESCRIPTION")
        for p in observed["libpaths"]
        if (pathlib.Path(p) / "lattice" / "DESCRIPTION").is_file()
    )
    written = re.search(r"^Version: (.+)$", description.read_text(), re.MULTILINE).group(1)
    assert observed["packages"]["lattice"]["version"] == written


def test_parse_r_runtime_ignores_startup_noise_and_rejects_incomplete_output():
    out = evidence.parse_r_runtime(
        "# Bootstrapping renv\nBIORAG\tlibpaths\t/a\t/b\nBIORAG\trversion\t4.6.1\n"
        "BIORAG\tpkg\tzoo\t1.9-1\t1\nBIORAG\tpkg\tbroken\n"
    )
    assert out == {
        "libpaths": ["/a", "/b"],
        "rversion": "4.6.1",
        "packages": {"zoo": {"version": "1.9-1", "library_index": "1"}},
    }
    with pytest.raises(ValueError):
        evidence.parse_r_runtime("BIORAG\trversion\t4.6.1\n")


def test_g07_recognises_the_project_library_when_the_repository_is_given_as_a_relative_path(
    world, monkeypatch
):
    """Regression: `--repo .` made the active-library check compare a relative with an absolute path."""
    monkeypatch.chdir(world.repo)
    library = str((world.repo / "analysis/renv/library/x").resolve())
    out = R_OUTPUT.format(lib=library)
    ctx = world.probes(r_runtime=lambda d, p: out)
    ctx = replace(ctx, repo=pathlib.Path("."))
    g = evidence.r_environment(ctx)
    assert g.reason == "analysis_host_not_designated", g.detail
