"""Publication attacks use internally consistent, synthetic scientific artifacts."""

import json
from types import SimpleNamespace as Item

import pytest
from test_documents_ingestion import _paper
from test_documents_table_fidelity import _native_data, _typed_native

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput
from biorag_repro.core.manifest import BuildInput
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.documents.ingestion import (
    JatsBuild,
    Phase4GateIndex,
    Phase4Validation,
    build_arm,
    publish_gate_index,
    publish_validation,
    validate_build,
)
from biorag_repro.documents.jats import ingest_jats
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.documents.pdf_ingest import (
    ConversionAttempt,
    _manifest,
    _ref,
    publish_pdf_result,
)


@pytest.fixture
def pair(tmp_path):
    corpus, output = tmp_path / "corpus", tmp_path / "output"
    paper = _paper(corpus)
    jats = build_arm(paper, "P001", "JATS", corpus, output, "d" * 64)
    data = _native_data()

    class Native:
        def save_as_json(self, path, **kwargs):
            path.write_text(json.dumps(data))

    result = Item(status=Item(value="success"), errors=[], pages=[], document=Native())
    from biorag_repro.core.manifest import ArtifactRef

    pdf_source = ArtifactRef(
        **{k: getattr(paper.pdf, k) for k in ("artifact_id", "logical_ref", "sha256")}
    )
    pdf = publish_pdf_result(
        result,
        corpus / "corpus/raw" / pdf_source.logical_ref,
        pdf_source,
        "P001",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        output,
        load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
    )
    assert pdf.build_status == "complete"
    jr = validate_build(jats, output, corpus_root=corpus)
    pr = validate_build(pdf, output, corpus_root=corpus)
    index = Phase4GateIndex(
        corpus_version="corpus_v1",
        paper_id="P001",
        corpus_release_sha256="e" * 64,
        implementation_git_sha="f" * 40,
        jats_validation=publish_validation(jr, output),
        pdf_validation=publish_validation(pr, output),
    )
    return Item(
        corpus=corpus, output=output, paper=paper, jats=jats, pdf=pdf, jr=jr, pr=pr, index=index
    )


def attack(pair, case):
    index = pair.index.model_copy(deep=True)
    if case in ("zero_revision", "wrong_revision"):
        index.implementation_git_sha = (
            "0" * 40 if case == "zero_revision" else "330364326b204cc328628f67155681604ac20d08"
        )
    elif case == "release":
        index.corpus_release_sha256 = "0" * 64
    elif case == "version":
        index.corpus_version = "corpus_v2"
        index.jats_validation = publish_validation(
            pair.jr.model_copy(update={"corpus_version": "corpus_v2"}), pair.output
        )
        index.pdf_validation = publish_validation(
            pair.pr.model_copy(update={"corpus_version": "corpus_v2"}), pair.output
        )
    elif case == "source":
        from test_documents_ingestion import _source

        src = _source(
            "P001",
            "JATS",
            b"<article><body><p>substitute</p></body></article>",
            pair.corpus / "substitute",
        )
        target = pair.corpus / "corpus/raw" / src.logical_ref
        target.write_bytes(b"<article><body><p>substitute</p></body></article>")
        doc = ingest_jats(
            target.read_bytes(),
            paper_id="P001",
            source_artifact_id=src.artifact_id,
            source_sha256=src.sha256,
            source_logical_ref=src.logical_ref,
            configuration_sha256="d" * 64,
        )
        from biorag_repro.documents.storage import publish_canonical_jats

        ref, manifest = publish_canonical_jats(doc, pair.output, target.read_bytes())
        built = JatsBuild(
            paper_id="P001",
            source=src,
            configuration_sha256="d" * 64,
            canonical_artifact=ref,
            build_manifest=manifest,
        )
        index.jats_validation = publish_validation(
            validate_build(built, pair.output, corpus_root=pair.corpus), pair.output
        )
    elif case == "role":
        doc = CanonicalDocument.model_validate_json(
            (pair.output / pair.jats.canonical_artifact.logical_ref).read_bytes()
        )
        content = serialize_project_json(doc, CanonicalDocument)
        src = pair.jats.source
        ref = _ref(
            "canonical_jats",
            [ArtifactInput("unrelated_input", src.artifact_id, src.sha256)],
            doc.transformation_id,
            doc.configuration_sha256,
            content,
            "4",
            "canonical/JATS",
        )
        (pair.output / ref.logical_ref).write_bytes(content)
        manifest = _manifest(
            [BuildInput(role="unrelated_input", artifact=src)],
            doc.transformation_id,
            doc.configuration_sha256,
            ref,
            "canonical_jats",
            "4",
            pair.output,
        )
        index.jats_validation = publish_validation(
            pair.jr.model_copy(update={"canonical_artifact": ref, "build_manifest": manifest}),
            pair.output,
        )
    elif case == "configuration":
        bad = pair.pdf.model_copy(update={"configuration_sha256": "0" * 64})
        content = serialize_project_json(bad, ConversionAttempt)
        path = pair.output / "forged_attempt.json"
        path.write_bytes(content)
        ref = pair.pr.attempt_artifact.model_copy(
            update={"logical_ref": "forged_attempt.json", "sha256": sha256_bytes(content)}
        )
        index.pdf_validation = publish_validation(
            pair.pr.model_copy(update={"attempt_artifact": ref}), pair.output
        )
    elif case == "validation_id":
        index.pdf_validation.artifact_id = "drv_" + "0" * 64
    elif case == "warnings":
        assert pair.pr.warnings
        index.pdf_validation = publish_validation(
            pair.pr.model_copy(update={"warnings": []}), pair.output
        )
    return index


CASES = [
    "zero_revision",
    "wrong_revision",
    "release",
    "version",
    "source",
    "role",
    "configuration",
    "validation_id",
    "warnings",
]


@pytest.mark.parametrize("case", CASES)
def test_historical_attacks_rejected(pair, case):
    index = attack(pair, case)
    with pytest.raises(ValueError):
        publish_gate_index(index, pair.output, pair.corpus)


@pytest.fixture
def authority(pair, monkeypatch):
    from test_corpus import paper as metadata

    from biorag_repro.core.manifest import ArtifactRef
    from biorag_repro.corpus.manifest import (
        CorpusManifest,
        PaperRecord,
        ReleaseRecord,
        corpus_content_hash,
        manifest_semantic_hash,
        serialize_csv,
    )
    from biorag_repro.documents.authorization import ScientificAuthority

    rows = []
    for i in range(1, 51):
        row = metadata(i)
        row["corpus_version"] = "corpus_v1"
        if i == 1:
            for kind in ("jats", "pdf"):
                row[kind].update(
                    {
                        k: getattr(getattr(pair.paper, kind), k)
                        for k in ("artifact_id", "logical_ref", "sha256")
                    }
                )
        rows.append(PaperRecord.model_validate(row))
    manifest = CorpusManifest(papers=rows)
    content = serialize_csv(manifest)
    path = pair.corpus / "manifest.csv"
    path.write_bytes(content)
    manifest_ref = ArtifactRef(
        artifact_id="manifest_csv", logical_ref="manifest.csv", sha256=sha256_bytes(content)
    )
    release = ReleaseRecord(
        record_schema_version="1",
        schema_version="1",
        corpus_version="corpus_v1",
        manifest_revision="test-revision",
        csv_sha256=manifest_ref.sha256,
        corpus_content_sha256=corpus_content_hash(manifest),
        manifest_semantic_sha256=manifest_semantic_hash(manifest),
        manifest_artifact=manifest_ref,
        validation_report_artifact=manifest_ref,
        git_sha="f" * 40,
    )
    content = serialize_project_json(release, ReleaseRecord)
    release_path = pair.corpus / "release.json"
    release_path.write_bytes(content)
    digest = sha256_bytes(content)
    pair.index.corpus_release_sha256 = digest
    from biorag_repro.documents import authorization

    monkeypatch.setattr(authorization, "PHASE2H_SHA256", digest)
    return ScientificAuthority(
        release_path, digest, pair.pr.attempt_artifact, "d" * 64, "a" * 64, "b" * 64, "c" * 64
    )


@pytest.mark.parametrize("case", CASES)
def test_scientific_attacks_rejected_before_approval(pair, authority, case):
    from biorag_repro.documents.authorization import validate_scientific_pair

    index = attack(pair, case)
    messages = {
        "zero_revision": "revision",
        "wrong_revision": "revision",
        "release": "release",
        "version": "version",
        "source": "SHA-256|source",
        "role": "contract",
        "configuration": "native lineage",
        "validation_id": "identity",
        "warnings": "warnings",
    }
    with pytest.raises(ValueError, match=messages[case]):
        validate_scientific_pair(index, pair.output, pair.corpus, authority, "f" * 40)


def candidate(pair):
    from biorag_repro.core.manifest import ArtifactRef
    from biorag_repro.documents.authorization import ReproductionCandidate

    content = b'{"python":"synthetic test runtime"}\n'
    path = pair.output / "runtime.json"
    path.write_bytes(content)
    runtime = ArtifactRef(
        artifact_id="runtime_" + sha256_bytes(content),
        logical_ref="runtime.json",
        sha256=sha256_bytes(content),
    )
    return ReproductionCandidate(
        reproduction_git_sha="f" * 40,
        started_utc="2026-11-03T00:00:00+00:00",
        ended_utc="2026-11-03T00:00:01+00:00",
        runtime_inventory=runtime,
        corpus_version=pair.index.corpus_version,
        paper_id="P001",
        corpus_release_sha256=pair.index.corpus_release_sha256,
        jats_validation=pair.index.jats_validation,
        pdf_validation=pair.index.pdf_validation,
    )


def test_candidate_science_and_production_authorization_are_distinct(pair, authority):
    from biorag_repro.documents.authorization import (
        candidate_ref,
        publish_candidate,
        publish_complete_gate,
        validate_candidate,
    )

    value = candidate(pair)
    validate_candidate(value, pair.output, pair.corpus, authority, "f" * 40)
    ref = publish_candidate(value, pair.output, pair.corpus, authority, "f" * 40)
    assert ref == candidate_ref(value)
    with pytest.raises(FileExistsError):
        publish_candidate(value, pair.output, pair.corpus, authority, "f" * 40)
    with pytest.raises(ValueError, match="independent human authorization"):
        publish_complete_gate(value, pair.output, pair.corpus, authority, "f" * 40)
    with pytest.raises(ValueError):
        validate_candidate(
            value.model_copy(update={"human_authorized": True}),
            pair.output,
            pair.corpus,
            authority,
            "f" * 40,
        )
    # The trusted authority and its publication are deliberately confined to this test.
    approved = frozenset([ref.sha256])

    def test_only_publish(item):
        validate_candidate(item, pair.output, pair.corpus, authority, "f" * 40)
        actual = candidate_ref(item)
        if actual.sha256 not in approved:
            raise ValueError("test authority did not approve this digest")
        from biorag_repro.core.storage import publish_bytes

        return publish_bytes(
            pair.output / "test_only_approved.json",
            serialize_project_json(item, type(item)),
            actual.sha256,
        )

    assert test_only_publish(value) == ref.sha256
    with pytest.raises(FileExistsError):
        test_only_publish(value)
    changed = value.model_copy(update={"ended_utc": "2026-11-03T00:00:02+00:00"})
    with pytest.raises(ValueError, match="digest"):
        test_only_publish(changed)
    # The same approved digest cannot authenticate changed bytes behind its references.
    bound = pair.output / pair.jats.canonical_artifact.logical_ref
    bound.write_bytes(bound.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="SHA-256"):
        test_only_publish(value)


def test_combined_corruption_retains_independent_checks(pair, authority):
    from biorag_repro.documents.authorization import validate_scientific_pair

    index = attack(pair, "warnings")
    index.corpus_release_sha256 = "0" * 64
    with pytest.raises(ValueError, match="release"):
        validate_scientific_pair(index, pair.output, pair.corpus, authority, "f" * 40)
    index.corpus_release_sha256 = authority.release_sha256
    with pytest.raises(ValueError, match="warnings"):
        validate_scientific_pair(index, pair.output, pair.corpus, authority, "f" * 40)


def test_reproduction_requires_observed_clean_revision(tmp_path):
    import subprocess

    from biorag_repro.documents.authorization import observed_clean_revision

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Initial",
        ],
        cwd=tmp_path,
        check=True,
    )
    assert len(observed_clean_revision(tmp_path)) == 40
    (tmp_path / "untracked.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="untracked"):
        observed_clean_revision(tmp_path)


@pytest.mark.parametrize("script", ["ingest_paper.py", "reproduce_pair.py"])
def test_cli_cannot_request_complete_gate(tmp_path, script):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "scripts/" + script, "--publication", "complete"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "independent human authorization is unavailable" in result.stderr


def test_fresh_reproduction_preserves_content_ids_and_supports_delay(
    pair, authority, tmp_path, monkeypatch
):
    import subprocess

    from biorag_repro.core.storage import read_project_json
    from biorag_repro.documents.authorization import (
        ReproductionCandidate,
        candidate_ref,
        publish_complete_gate,
        validate_candidate,
    )
    from biorag_repro.documents.reproduction import reproduce_pair

    repo = tmp_path / "repository"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Initial",
        ],
        cwd=repo,
        check=True,
    )
    output = tmp_path / "reproduced"
    ref = reproduce_pair(
        pair.index,
        pair.output,
        output,
        pair.corpus,
        authority,
        repo,
        load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
    )
    value = read_project_json(output / ref.logical_ref, ref.sha256, ReproductionCandidate)
    assert candidate_ref(value) == ref
    assert value.status == "validated_candidate_pending_human_review"
    assert not value.human_authorized and not value.historical_producer_authenticated
    assert value.reproduction_git_sha != pair.index.implementation_git_sha
    pr = read_project_json(
        output / value.pdf_validation.logical_ref,
        value.pdf_validation.sha256,
        Phase4Validation,
    )
    assert pr.canonical_artifact == pair.pr.canonical_artifact
    assert pr.warnings == pair.pr.warnings
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Later",
        ],
        cwd=repo,
        check=True,
    )
    validate_candidate(value, output, pair.corpus, authority, value.reproduction_git_sha)
    with pytest.raises(ValueError, match="independent human authorization"):
        publish_complete_gate(value, output, pair.corpus, authority, value.reproduction_git_sha)
    with pytest.raises(FileExistsError):
        reproduce_pair(
            pair.index,
            pair.output,
            output,
            pair.corpus,
            authority,
            repo,
            load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
        )


def test_candidate_rejects_false_runtime_identity(pair, authority):
    from biorag_repro.documents.authorization import validate_candidate

    value = candidate(pair)
    value.runtime_inventory.artifact_id = "runtime_" + "0" * 64
    with pytest.raises(ValueError, match="runtime identity"):
        validate_candidate(value, pair.output, pair.corpus, authority, "f" * 40)


def test_candidate_fidelity_check_rejects_rehashed_native_table_corruption(pair, authority):
    from biorag_repro.documents.authorization import validate_scientific_pair
    from biorag_repro.documents.pdf import TRANSFORMATION_ID

    doc = CanonicalDocument.model_validate_json(
        (pair.output / pair.pdf.canonical_artifact.logical_ref).read_bytes()
    )
    table = next(node.table for node in doc.nodes if node.kind == "table")
    table.cells[0].native_row_header = False
    table.cells[0].is_header = False
    content = serialize_project_json(doc, CanonicalDocument)
    settings = json.loads((pair.output / pair.pdf.build_manifest.logical_ref).read_bytes())[
        "transformation_settings_sha256"
    ]
    src, native = pair.pdf.source, pair.pdf.native_artifact
    ref = _ref(
        "canonical_pdf_docling",
        [
            ArtifactInput("raw_pdf", src.artifact_id, src.sha256),
            ArtifactInput("docling_native", native.artifact_id, native.sha256),
        ],
        TRANSFORMATION_ID,
        settings,
        content,
        "4",
        "canonical/PDF_DOCLING",
    )
    (pair.output / ref.logical_ref).write_bytes(content)
    manifest = _manifest(
        [
            BuildInput(role="raw_pdf", artifact=src),
            BuildInput(role="docling_native", artifact=native),
        ],
        TRANSFORMATION_ID,
        settings,
        ref,
        "canonical_pdf_docling",
        "4",
        pair.output,
    )
    from biorag_repro.documents.pdf_ingest import PdfCanonicalBuild, canonical_build_ref

    built = PdfCanonicalBuild(
        paper_id="P001",
        source=src,
        configuration_sha256="a" * 64,
        environment_sha256="b" * 64,
        asset_inventory_sha256="c" * 64,
        original_attempt_artifact=authority.native_attempt,
        native_artifact=native,
        native_build_manifest=pair.pdf.native_build_manifest,
        canonical_artifact=ref,
        build_manifest=manifest,
    )
    record = canonical_build_ref(built)
    (pair.output / record.logical_ref).write_bytes(serialize_project_json(built, PdfCanonicalBuild))
    report = pair.pr.model_copy(
        update={
            "canonical_artifact": ref,
            "build_manifest": manifest,
            "canonical_build_artifact": record,
        }
    )
    index = pair.index.model_copy(
        update={"pdf_validation": publish_validation(report, pair.output)}
    )
    with pytest.raises(ValueError, match="native table cell"):
        validate_scientific_pair(index, pair.output, pair.corpus, authority, "f" * 40)


def test_low_level_evidence_writer_cannot_publish_a_gate(pair):
    from biorag_repro.documents.ingestion import _publish_evidence

    with pytest.raises(ValueError, match="authorization"):
        _publish_evidence(
            pair.index,
            Phase4GateIndex,
            pair.output,
            "phase4_gate_index",
            [
                ArtifactInput(
                    "jats_validation",
                    pair.index.jats_validation.artifact_id,
                    pair.index.jats_validation.sha256,
                )
            ],
        )


def test_fully_rehashed_converter_configuration_is_not_authority(pair, authority):
    from biorag_repro.documents.authorization import validate_scientific_pair

    data = _native_data()

    class Native:
        def save_as_json(self, path, **kwargs):
            path.write_text(json.dumps(data))

    result = Item(status=Item(value="success"), errors=[], pages=[], document=Native())
    forged = publish_pdf_result(
        result,
        pair.corpus / "corpus/raw" / pair.pdf.source.logical_ref,
        pair.pdf.source,
        "P001",
        "0" * 64,
        "b" * 64,
        "c" * 64,
        pair.output,
        load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
    )
    assert forged.status == "success" and forged.build_status == "complete"
    assert forged.native_artifact.sha256 == pair.pdf.native_artifact.sha256
    assert forged.native_artifact.artifact_id != pair.pdf.native_artifact.artifact_id
    report = validate_build(forged, pair.output, corpus_root=pair.corpus)
    index = pair.index.model_copy(
        update={"pdf_validation": publish_validation(report, pair.output)}
    )
    with pytest.raises(ValueError, match="authorized configuration"):
        validate_scientific_pair(index, pair.output, pair.corpus, authority, "f" * 40)


def test_cli_candidate_uses_scientific_reproduction_without_final_publication(
    pair, authority, tmp_path, monkeypatch, capsys
):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    from biorag_repro.documents.reproduction import reproduce_pair

    spec = importlib.util.spec_from_file_location(
        "reproduce_pair_cli", Path("scripts/reproduce_pair.py")
    )
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    repo = tmp_path / "clean_repository"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "Initial",
        ],
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(cli, "PHASE2H_SHA256", authority.release_sha256)
    monkeypatch.setattr(cli, "QUALIFIED_ENVIRONMENT", authority.environment_sha256)
    monkeypatch.setattr(
        cli,
        "resolve_config",
        lambda content, model: Item(
            settings=None,
            configuration_sha256=authority.jats_configuration_sha256
            if model is cli.JatsConfiguration
            else authority.pdf_configuration_sha256,
        ),
    )
    monkeypatch.setattr(cli, "verify_assets", lambda *args: authority.asset_inventory_sha256)
    monkeypatch.setattr(cli, "verify_qualified_lock", lambda *args: None)
    monkeypatch.setattr(
        cli,
        "reproduce_pair",
        lambda reference, reference_root, output, corpus, selected, repository: reproduce_pair(
            reference,
            reference_root,
            output,
            corpus,
            selected,
            repo,
            load_native=lambda path: _typed_native(json.loads(path.read_bytes())),
        ),
    )
    content = serialize_project_json(pair.index, Phase4GateIndex)
    index = tmp_path / "reference.json"
    index.write_bytes(content)
    output = tmp_path / "cli_candidate"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_paper.py",
            "--corpus-root",
            str(pair.corpus),
            "--release-record",
            str(authority.release_path),
            "--reference-root",
            str(pair.output),
            "--reference-index",
            str(index),
            "--reference-sha256",
            sha256_bytes(content),
            "--native-attempt-id",
            pair.pdf.attempt_id,
            "--native-attempt-sha256",
            authority.native_attempt.sha256,
            "--data-root",
            str(output),
            "--assets-root",
            str(tmp_path),
            "--lock-path",
            str(tmp_path / "lock"),
        ],
    )
    assert cli.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "validated_candidate_pending_human_review"
    assert "gate_index" not in result
    assert (output / result["candidate_attestation"]["logical_ref"]).is_file()
