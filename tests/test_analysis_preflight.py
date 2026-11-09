# ruff: noqa: F811
"""The Python preflight record: what Python verified before R, and nothing more.

A preflight is built only from the objects `read_handoff` returns. These tests pin that it is
immutable, never replaces anything, refuses unauthorized origins, and that no stronger claim can
be placed in it by editing JSON, whether or not the bytes are re-hashed.
"""

import json

import pytest
from test_analysis_cli_handoff import published_rehearsal
from test_analysis_handoff import CODE, official_table
from test_generation_official import recognised, view  # noqa: F401  (fixtures)
from test_generation_prompt import needs_tokenizer

from biorag_repro.core.hashing import HashMismatchError, canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import publish_bytes, publish_manifest
from biorag_repro.evaluation import analysis_handoff as ah
from biorag_repro.evaluation import analysis_preflight as ap
from biorag_repro.evaluation.analysis_cli import main

FIXED_ID = "0123456789abcdef0123456789abcdef"


def preflight_of(tmp_path, **options):
    table, handoff, ref, built = published_rehearsal(tmp_path)
    file_ref, build = ap.publish_preflight(built, tmp_path, code=CODE, **options)
    return table, handoff, ref, built, file_ref, build


def stored_json(root, ref):
    return json.loads((root / ref.logical_ref).read_text(encoding="utf-8"))


def republish(
    root,
    file_ref,
    edit,
    *,
    swap_inputs=False,
    kind="analysis_preflight",
    transformation=ap.TRANSFORMATION,
    settings=None,
):
    """Store an edited preflight with a new hash, a new id and a consistent manifest.

    This is what a forger with write access could do; only the model and the re-derivation
    from the handoff stand in the way.
    """
    data = stored_json(root, file_ref)
    edit(data)
    raw = (json.dumps(data, sort_keys=True, indent=2) + "\n").encode("utf-8")
    digest = sha256_bytes(raw)
    inputs = [
        BuildInput(role=role, artifact=ref)
        for role, ref in (
            ("analysis_handoff_build", ap.ArtifactRef.model_validate(data["handoff_build"])),
            ("analysis_handoff", ap.ArtifactRef.model_validate(data["handoff"])),
        )
    ]
    if swap_inputs:
        inputs.reverse()
    settings = settings or ap.settings_sha256()
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs],
        transformation_id=transformation,
        transformation_settings_sha256=settings,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    ref = ap.ArtifactRef(
        artifact_id=artifact_id, logical_ref=f"analysis/preflight/{artifact_id}.json", sha256=digest
    )
    publish_bytes(root / ref.logical_ref, raw, digest)
    manifest = BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=transformation,
        transformation_settings_sha256=settings,
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    build_ref = f"analysis/preflight/{sha256_bytes(canonical_json_bytes([artifact_id]))}.build.json"
    sha = publish_manifest(manifest, root / build_ref, {ref.logical_ref: root / ref.logical_ref})
    return ap.ArtifactRef(artifact_id="build_" + sha, logical_ref=build_ref, sha256=sha)


# ---- a valid preflight --------------------------------------------------------------------


def test_a_preflight_is_built_from_the_verified_handoff_and_reads_back(tmp_path):
    table, handoff, ref, built, file_ref, build = preflight_of(tmp_path)
    checked = ap.read_preflight(build, tmp_path)
    pre = checked.preflight
    assert pre.input_origin == "synthetic_rehearsal" and pre.scientific_authority == "none"
    assert pre.handoff.sha256 == ref.sha256 and pre.handoff_build == built
    assert pre.csv == handoff.csv and pre.csv_rows == len(table.rows)
    assert pre.table_build == handoff.table_build
    assert [(a.outcome, a.n_eligible, a.n_complete) for a in pre.accounting] == [
        (o.outcome, len(o.eligible_items), len(o.complete_item_ids)) for o in handoff.accounting
    ]
    assert pre.verifier.git == CODE and pre.verifier.tool == "analysis_preflight_v1"
    assert checked.handoff.handoff == handoff
    assert file_ref.sha256 == sha256_bytes((tmp_path / file_ref.logical_ref).read_bytes())


def test_the_claims_are_exactly_what_python_established(tmp_path):
    *_, build = preflight_of(tmp_path)
    pre = ap.read_preflight(build, tmp_path).preflight
    status = {c.name: c.status for c in pre.checks}
    assert status["source_run_lineage"] == "not_established"
    assert status["seed_approval"] == "not_established"
    assert status["handoff_bytes_sha256"] == "verified_by_python"
    assert status["handoff_contract"] == "verified_by_python"
    assert status["origin_authorization"] == "verified_by_python"
    scope = next(c.scope for c in pre.checks if c.name == "origin_authorization")
    assert "rehearsal only" in scope
    text = " ".join(pre.not_established)
    assert "source-run lineage" in text and "production" in text and "authorship" in text
    assert "not cryptographic proof" in text and "R re-hashes" in text


def test_a_second_preflight_is_a_new_artifact_and_replaces_nothing(tmp_path):
    _, _, _, built, first, first_build = preflight_of(tmp_path)
    before = (tmp_path / first.logical_ref).read_bytes()
    second, second_build = ap.publish_preflight(built, tmp_path, code=CODE)
    assert second != first and second_build != first_build
    assert (tmp_path / first.logical_ref).read_bytes() == before
    ap.read_preflight(first_build, tmp_path)
    ap.read_preflight(second_build, tmp_path)


def test_publication_never_replaces_an_existing_preflight(tmp_path):
    _, _, _, built, first, _ = preflight_of(tmp_path, now=lambda: "2026-11-06T10:00:00.000000Z")
    # Same nonce and same clock reading give the same bytes; the store still refuses to write.
    before = (tmp_path / first.logical_ref).read_bytes()
    with pytest.raises(FileExistsError):
        ap.publish_preflight(
            built,
            tmp_path,
            code=CODE,
            now=lambda: "2026-11-06T10:00:00.000000Z",
            new_id=lambda: stored_json(tmp_path, first)["preflight_id"],
        )
    assert (tmp_path / first.logical_ref).read_bytes() == before


def test_the_nonce_and_the_clock_are_recorded_but_not_trusted_as_ordering(tmp_path):
    *_, build = preflight_of(
        tmp_path, now=lambda: "2026-11-06T10:00:00.000000Z", new_id=lambda: FIXED_ID
    )
    pre = ap.read_preflight(build, tmp_path).preflight
    assert pre.preflight_id == FIXED_ID and pre.created_utc == "2026-11-06T10:00:00.000000Z"
    assert any("created_utc is a clock reading" in line for line in pre.not_established)


# ---- inputs that must not produce a preflight ------------------------------------------------


def test_a_handoff_claiming_an_official_origin_gets_no_preflight(tmp_path):
    _, _, ref, built = published_rehearsal(tmp_path)
    stored = tmp_path / ref.logical_ref
    stored.chmod(0o644)
    stored.write_bytes(
        stored.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run")
    )
    with pytest.raises(ValueError):
        ap.publish_preflight(built, tmp_path, code=CODE)
    assert not (tmp_path / "analysis" / "preflight").exists()


@needs_tokenizer
def test_a_verified_official_handoff_still_gets_no_preflight(tmp_path, view, tiny, recognised):
    verified, root, build = official_table(tmp_path, view, tiny)
    handoff = ah.build_handoff(build, root, code=CODE, official_runs=[verified])
    _, built = ah.publish_handoff(handoff, root, official_runs=[verified])
    with pytest.raises(ah.OfficialProvenanceNotVerified):
        ap.publish_preflight(built, root, code=CODE)
    assert not (root / "analysis" / "preflight").exists()


def test_tampered_handoff_table_or_manifest_gets_no_preflight(tmp_path):
    _, handoff, _, built = published_rehearsal(tmp_path)
    with pytest.raises(HashMismatchError):
        ap.publish_preflight(built.model_copy(update={"sha256": "0" * 64}), tmp_path, code=CODE)
    csv = tmp_path / handoff.csv.logical_ref
    csv.chmod(0o644)
    csv.write_bytes(csv.read_bytes().replace(b"TRUE", b"FALSE", 1))
    with pytest.raises(ValueError):
        ap.publish_preflight(built, tmp_path, code=CODE)
    assert not (tmp_path / "analysis" / "preflight").exists()


def test_a_table_build_is_not_a_handoff_reference(tmp_path):
    _, handoff, *_ = published_rehearsal(tmp_path)
    with pytest.raises(ValueError, match="does not describe an analysis handoff"):
        ap.publish_preflight(handoff.table_build, tmp_path, code=CODE)


# ---- tampering after publication -----------------------------------------------------------


def test_changed_preflight_bytes_fail_the_hash(tmp_path):
    _, _, _, _, file_ref, build = preflight_of(tmp_path)
    path = tmp_path / file_ref.logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes().replace(b"synthetic_rehearsal", b"synthetic_rehearsal ", 1))
    with pytest.raises(HashMismatchError):
        ap.read_preflight(build, tmp_path)


def test_a_missing_preflight_file_or_manifest_is_an_error(tmp_path):
    _, _, _, _, file_ref, build = preflight_of(tmp_path)
    (tmp_path / file_ref.logical_ref).unlink()
    with pytest.raises(FileNotFoundError):
        ap.read_preflight(build, tmp_path)
    with pytest.raises(FileNotFoundError):
        ap.read_preflight(
            build.model_copy(update={"logical_ref": "analysis/preflight/x"}), tmp_path
        )


def test_a_wrong_manifest_hash_is_an_error(tmp_path):
    *_, build = preflight_of(tmp_path)
    with pytest.raises(HashMismatchError):
        ap.read_preflight(build.model_copy(update={"sha256": "f" * 64}), tmp_path)


def test_a_handoff_edited_after_the_preflight_is_noticed(tmp_path):
    _, _, ref, _, _, build = preflight_of(tmp_path)
    stored = tmp_path / ref.logical_ref
    stored.chmod(0o644)
    stored.write_bytes(stored.read_bytes().replace(b'"phase"', b'"phase" ', 1))
    with pytest.raises(HashMismatchError):
        ap.read_preflight(build, tmp_path)


# ---- no stronger claim by editing JSON ---------------------------------------------------------


def claim_lineage_verified(data):
    next(c for c in data["checks"] if c["name"] == "source_run_lineage")["status"] = (
        "verified_by_python"
    )


def claim_official_origin(data):
    data["input_origin"] = "verified_official_run"


def claim_authority(data):
    data["scientific_authority"] = "development"


def drop_the_limits(data):
    data["not_established"] = []


def invent_a_check(data):
    data["checks"].append({"name": "official_runs", "status": "verified_by_python", "scope": "x"})


@pytest.mark.parametrize(
    "edit",
    [
        claim_lineage_verified,
        claim_official_origin,
        claim_authority,
        drop_the_limits,
        invent_a_check,
    ],
)
def test_a_stronger_claim_is_rejected_even_when_hashes_are_made_consistent(tmp_path, edit):
    _, _, _, _, file_ref, _ = preflight_of(tmp_path)
    forged = republish(tmp_path, file_ref, edit)
    with pytest.raises(ValueError):
        ap.read_preflight(forged, tmp_path)


def test_the_model_itself_refuses_the_stronger_claims(tmp_path):
    _, _, _, _, file_ref, _ = preflight_of(tmp_path)
    for edit in (claim_lineage_verified, claim_official_origin, claim_authority, drop_the_limits):
        data = stored_json(tmp_path, file_ref)
        edit(data)
        with pytest.raises(ValueError):
            ap.AnalysisPreflight.model_validate(data, strict=True)


def test_facts_that_python_did_not_verify_cannot_be_swapped_in(tmp_path):
    # A consistent, well-formed preflight that names another csv hash or other counts is not what
    # read_handoff returns for this handoff, so reading it back fails.
    _, _, _, _, file_ref, _ = preflight_of(tmp_path)
    for edit in (
        lambda d: d["csv"].update(sha256="1" * 64),
        lambda d: d["accounting"][0].update(n_complete=d["accounting"][0]["n_complete"] + 1),
        lambda d: d.update(csv_rows=d["csv_rows"] + 1),
        lambda d: d.update(analysis_spec_sha256="2" * 64),
        lambda d: d["handoff"].update(sha256="3" * 64),
    ):
        forged = republish(tmp_path, file_ref, edit)
        with pytest.raises(ValueError):
            ap.read_preflight(forged, tmp_path)


def test_a_preflight_whose_handoff_reference_is_not_a_handoff_is_rejected(tmp_path):
    _, handoff, _, _, file_ref, _ = preflight_of(tmp_path)
    forged = republish(
        tmp_path,
        file_ref,
        lambda d: d.update(handoff_build=handoff.table_build.model_dump(mode="json")),
    )
    with pytest.raises(ValueError, match="does not describe an analysis handoff"):
        ap.read_preflight(forged, tmp_path)


def test_a_preflight_naming_a_handoff_that_is_not_in_the_store_is_an_error(tmp_path):
    _, _, _, _, file_ref, _ = preflight_of(tmp_path)
    missing = {
        "artifact_id": "build_" + "4" * 64,
        "logical_ref": "analysis/handoff/absent.build.json",
        "sha256": "4" * 64,
    }
    forged = republish(tmp_path, file_ref, lambda d: d.update(handoff_build=missing))
    with pytest.raises(FileNotFoundError):
        ap.read_preflight(forged, tmp_path)


# ---- command line --------------------------------------------------------------------------------


def cli_preflight(root, built, *extra):
    return main(
        [
            "preflight",
            "--root",
            str(root),
            "--handoff-build",
            built.logical_ref,
            "--handoff-sha256",
            built.sha256,
            *extra,
        ]
    )


def test_the_cli_publishes_a_preflight_and_prints_the_references_r_needs(tmp_path, capsys):
    _, handoff, ref, built = published_rehearsal(tmp_path)
    assert cli_preflight(tmp_path, built) == 0
    out = capsys.readouterr().out
    pre_build = next(line for line in out.splitlines() if line.startswith("preflight build:"))
    pre_file = next(line for line in out.splitlines() if line.startswith("preflight file:"))
    assert str(tmp_path / ref.logical_ref) in out and ref.sha256 in out
    assert str(tmp_path / handoff.csv.logical_ref) in out
    assert "not cryptographic proof" in out and "official source-run lineage" in out
    assert "--preflight" in out and "--preflight-sha256" in out
    build_sha = pre_build.split()[-1]
    assert len(build_sha) == 64 and pre_file.split()[-1] != build_sha


def test_the_cli_verifies_a_preflight_and_rejects_a_changed_one(tmp_path, capsys):
    *_, file_ref, build = preflight_of(tmp_path)
    args = [
        "verify-preflight",
        "--root",
        str(tmp_path),
        "--preflight-build",
        build.logical_ref,
        "--preflight-sha256",
        build.sha256,
    ]
    assert main(args) == 0
    assert "valid preflight" in capsys.readouterr().out
    path = tmp_path / file_ref.logical_ref
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    assert main(args) == 1
    assert "not valid" in capsys.readouterr().err


def test_the_cli_publishes_nothing_for_a_handoff_edited_to_claim_official_origin(tmp_path, capsys):
    _, _, ref, built = published_rehearsal(tmp_path)
    stored = tmp_path / ref.logical_ref
    stored.chmod(0o644)
    stored.write_bytes(
        stored.read_bytes().replace(b"synthetic_rehearsal", b"verified_official_run")
    )
    assert cli_preflight(tmp_path, built) == 1
    assert "preflight not written" in capsys.readouterr().err
    assert not (tmp_path / "analysis" / "preflight").exists()


def test_the_cli_exits_with_2_when_the_handoff_needs_verified_official_runs(
    tmp_path, capsys, monkeypatch
):
    _, _, _, built = published_rehearsal(tmp_path)

    def refuse(*_args, **_kwargs):
        raise ah.OfficialProvenanceNotVerified("official")

    monkeypatch.setattr("biorag_repro.evaluation.analysis_cli.publish_preflight", refuse)
    assert cli_preflight(tmp_path, built) == 2
    captured = capsys.readouterr()
    assert "synthetic rehearsal only" in captured.err and captured.out == ""


def test_a_store_that_wrote_something_else_is_caught_by_the_read_back(tmp_path, monkeypatch):
    _, _, _, built = published_rehearsal(tmp_path)
    real_store = ap._store

    def store_with_other_counts(preflight, root):
        return real_store(preflight.model_copy(update={"csv_rows": preflight.csv_rows + 1}), root)

    monkeypatch.setattr(ap, "_store", store_with_other_counts)
    with pytest.raises(ap.PreflightError, match="csv_rows"):
        ap.publish_preflight(built, tmp_path, code=CODE)


def test_a_build_manifest_that_is_not_this_kind_of_record_is_rejected(tmp_path):
    _, _, _, _, file_ref, _ = preflight_of(tmp_path)

    def same(_data):
        pass

    cases = {
        "does not describe an analysis preflight": {"kind": "analysis_handoff"},
        "built under other settings": {"settings": "0" * 64},
        "built under other settings ": {"transformation": "analysis_preflight_v0"},
        "lineage differs": {"swap_inputs": True},
    }
    for message, options in cases.items():
        forged = republish(tmp_path, file_ref, same, **options)
        with pytest.raises(ap.PreflightError, match=message.strip()):
            ap.read_preflight(forged, tmp_path)


def test_two_preflights_at_the_same_clock_reading_still_differ(tmp_path):
    _, _, _, built, first, _ = preflight_of(tmp_path, now=lambda: "2026-11-06T10:00:00.000000Z")
    second, _ = ap.publish_preflight(
        built, tmp_path, code=CODE, now=lambda: "2026-11-06T10:00:00.000000Z"
    )
    assert second.sha256 != first.sha256
    assert (
        stored_json(tmp_path, first)["preflight_id"]
        != stored_json(tmp_path, second)["preflight_id"]
    )
