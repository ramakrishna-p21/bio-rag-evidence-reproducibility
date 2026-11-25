"""Synthetic checks for the frozen source-manifest contract."""

import json
from copy import deepcopy
from datetime import date
from hashlib import sha256

import pytest
from pydantic import ValidationError

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.storage import read_project_json, serialize_project_json
from biorag_repro.corpus.manifest import (
    CorpusManifest,
    PaperRecord,
    ReleaseRecord,
    ValidationReport,
    corpus_content_hash,
    csv_byte_hash,
    manifest_semantic_hash,
    parse_csv,
    serialize_csv,
    validate_corpus,
    verify_source_bytes,
)


def source(paper_id: str, kind: str) -> dict:
    content = b"<article/>" if kind == "jats" else b"%PDF-1.4\n%%EOF\n"
    digest = sha256_bytes(content)
    return {
        "artifact_id": raw_source_artifact_id(
            paper_id=paper_id, representation=kind.upper(), sha256=digest
        ),
        "sha256": digest,
        "provider": "synthetic provider",
        "acquisition_reference": "fixture-only",
        "acquired_at_utc": None,
        "acquired_at_status": "not_checked",
        "media_type": "application/xml" if kind == "jats" else "application/pdf",
        "logical_ref": f"fixture:{paper_id}:{kind}",
        "source_status": "frozen_local",
        "license_id": None,
        "license_status": "unknown",
        "license_evidence_ref": None,
        "analysis_access": "available",
        "redistribution_status": "unknown",
    }


def paper(i: int) -> dict:
    paper_id = f"P{i:03}"
    preprint = i in (1, 2)
    return {
        "schema_version": "test-schema",
        "corpus_version": "test-corpus",
        "manifest_revision": "test-revision",
        "paper_id": paper_id,
        "provenance_group": "original_29" if i <= 29 else "backfill_21",
        "title": f"Synthetic study {i}",
        "venue": "Synthetic journal" if not preprint else "bioRxiv",
        "publication_year": 2024,
        "publication_year_basis": "preprint_posted" if preprint else "online",
        "publication_state": "preprint" if preprint else "final_journal_article",
        "version_label": "bioRxiv v1 hosted on PMC" if preprint else None,
        "doi": f"10.0000/test.{i}",
        "doi_status": "known",
        "pmid": None,
        "pmid_status": "not_checked",
        "pmcid": None,
        "pmcid_status": "not_checked",
        "primary_domain": "single_cell",
        "secondary_domain": None,
        "publication_date": None,
        "publication_date_status": "not_checked",
        "authors_compact": None,
        "article_type": None,
        "bibliographic_note": None,
        "jats": source(paper_id, "jats"),
        "pdf": source(paper_id, "pdf"),
        "pair": {
            "pair_status": "verified_exact_pair",
            "pair_verification_basis": [
                "title_match",
                "identifier_match",
                "publication_state_version_review",
            ]
            if preprint
            else ["identifier_match", "title_match"],
            "pair_verified_at_utc": None,
            "pair_verified_at_status": "not_checked",
            "pair_verified_by": None,
            "pair_audit_ref": None,
            "pair_exception_note": None,
        },
    }


def corpus() -> CorpusManifest:
    rows = [paper(i) for i in range(1, 51)]
    for row in rows:
        row["corpus_version"] = "corpus_v1"
    return CorpusManifest(papers=[PaperRecord.model_validate(row) for row in rows])


def codes(report: ValidationReport, severity: str = "hard_error") -> set[str]:
    return {item.rule for item in report.findings if item.severity == severity}


def test_valid_synthetic_corpus_and_sources(tmp_path):
    manifest = corpus()
    assert not codes(validate_corpus(manifest))
    for kind, content in (("jats", b"<article/>"), ("pdf", b"%PDF-1.4\n%%EOF\n")):
        path = tmp_path / kind
        path.write_bytes(content)
        verify_source_bytes(manifest.papers[0], kind, path)
        path.write_bytes(content + b"changed")
        with pytest.raises(ValueError):
            verify_source_bytes(manifest.papers[0], kind, path)


def test_unqualified_jats_publication_date_has_explicit_year_basis():
    row = paper(3)
    row["publication_year_basis"] = "unqualified_source_date"
    manifest = CorpusManifest(papers=[PaperRecord.model_validate(row)])
    assert parse_csv(serialize_csv(manifest)) == manifest


@pytest.mark.parametrize(
    "path,value",
    [
        (("paper_id",), ""),
        (("paper_id",), "test_001"),
        (("provenance_group",), "other"),
        (("primary_domain",), "not_a_domain"),
        (("doi_status",), "wrong"),
        (("jats", "sha256"), "A" * 64),
        (("jats", "media_type"), "application/pdf"),
        (("pdf", "logical_ref"), "/tmp/test.pdf"),
        (("pdf", "logical_ref"), "../outside.pdf"),
        (("pdf", "analysis_access"), "forbidden"),
        (("pair", "pair_verification_basis"), []),
        (("pair", "pair_verification_basis"), ["title_match", "title_match"]),
        (("pair", "pair_verification_basis"), ["unsupported"]),
        (("pair", "pair_status"), "unverified"),
    ],
)
def test_record_rejects_invalid_values(path, value):
    row = paper(1)
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)


def test_unknown_fields_and_null_status_pairs_rejected():
    row = paper(1)
    row["benchmark_split"] = "final"
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)
    for value, status in ((None, "known"), ("10.0000/x", "unknown"), ("unknown", "known")):
        row = paper(1)
        row["doi"], row["doi_status"] = value, status
        with pytest.raises(ValidationError):
            PaperRecord.model_validate(row)


def test_pair_audit_and_timestamp_rules():
    row = paper(1)
    row["pair"]["pair_verification_basis"] = ["prior_audit_record"]
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)


@pytest.mark.parametrize(
    "status", ["known", "unknown", "unavailable", "not_applicable", "not_checked"]
)
def test_typed_null_status_vocabulary(status):
    row = paper(3)
    row["pmid_status"] = status
    row["pmid"] = "123" if status == "known" else None
    assert PaperRecord.model_validate(row).pmid == row["pmid"]
    row["pmid"] = None if status == "known" else "123"
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)


def test_known_dates_timestamps_and_license_round_trip():
    row = paper(3)
    row["publication_date"] = date(2024, 2, 29)
    row["publication_date_status"] = "known"
    row["jats"]["acquired_at_utc"] = "2024-03-01T01:02:03Z"
    row["jats"]["acquired_at_status"] = "known"
    row["jats"]["license_id"] = "CC-BY-4.0"
    row["jats"]["license_status"] = "known"
    row["jats"]["license_evidence_ref"] = "fixture:license"
    manifest = CorpusManifest(papers=[PaperRecord.model_validate(row)])
    assert parse_csv(serialize_csv(manifest)) == manifest
    row["jats"]["license_evidence_ref"] = None
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)
    row = paper(1)
    row["jats"]["acquired_at_utc"] = "2024-01-01T00:00:00Z"
    with pytest.raises(ValidationError):
        PaperRecord.model_validate(row)


def test_counts_versions_and_identity_collisions():
    baseline = corpus().papers
    assert "paper_count" in codes(validate_corpus(CorpusManifest(papers=baseline[:-1])))
    assert "source_count" in codes(validate_corpus(CorpusManifest(papers=baseline[:-1])))
    assert "paper_count" in codes(validate_corpus(CorpusManifest(papers=baseline + [baseline[-1]])))
    altered = deepcopy(baseline)
    altered[1].paper_id = altered[0].paper_id
    altered[1].jats.artifact_id = altered[0].jats.artifact_id
    assert {"paper_id_unique", "artifact_id_unique"} <= codes(
        validate_corpus(CorpusManifest(papers=altered))
    )
    altered = deepcopy(baseline)
    altered[-1].provenance_group = "original_29"
    assert "provenance_counts" in codes(validate_corpus(CorpusManifest(papers=altered)))
    altered = deepcopy(baseline)
    altered[-1].schema_version = "other"
    assert "version_consistency" in codes(validate_corpus(CorpusManifest(papers=altered)))
    altered = deepcopy(baseline)
    altered[0].jats.analysis_access = "unavailable"
    assert "analysis_access" in codes(validate_corpus(CorpusManifest(papers=altered)))


def test_frozen_paper_universe_and_version_scope():
    baseline = corpus()
    assert not codes(validate_corpus(baseline))
    replaced = deepcopy(baseline)
    replaced.papers[-1].paper_id = "P051"
    for representation, source in (
        ("JATS", replaced.papers[-1].jats),
        ("PDF", replaced.papers[-1].pdf),
    ):
        source.artifact_id = raw_source_artifact_id(
            paper_id="P051", representation=representation, sha256=source.sha256
        )
    assert {"paper_id_missing", "paper_id_unexpected"} <= codes(validate_corpus(replaced))
    assert "paper_id_missing" in codes(validate_corpus(CorpusManifest(papers=baseline.papers[:-1])))
    extra = deepcopy(baseline)
    extra.papers.append(deepcopy(replaced.papers[-1]))
    assert "paper_id_unexpected" in codes(validate_corpus(extra))
    duplicate = deepcopy(baseline)
    duplicate.papers[-1] = deepcopy(duplicate.papers[-2])
    assert "paper_id_unique" in codes(validate_corpus(duplicate))
    mixed = deepcopy(baseline)
    mixed.papers[-1].corpus_version = "other"
    assert "version_consistency" in codes(validate_corpus(mixed))
    substituted = deepcopy(baseline)
    for item in substituted.papers:
        item.corpus_version = "other"
    assert "corpus_version" in codes(validate_corpus(substituted))
    generic = validate_corpus(substituted, expected_corpus_version=None)
    assert not codes(generic)
    assert "unverified_corpus_version" in codes(generic, "review_warning")
    with pytest.raises(ValueError, match="unsupported frozen corpus version"):
        validate_corpus(substituted, expected_corpus_version="other")


def test_raw_source_id_must_bind_paper_representation_and_sha():
    baseline = corpus()
    assert "raw_artifact_id_mismatch" not in codes(validate_corpus(baseline))
    source = baseline.papers[0].jats
    for wrong_id in (
        "raw_" + "f" * 64,
        raw_source_artifact_id(paper_id="P002", representation="JATS", sha256=source.sha256),
        raw_source_artifact_id(paper_id="P001", representation="PDF", sha256=source.sha256),
        raw_source_artifact_id(paper_id="P001", representation="JATS", sha256="a" * 64),
    ):
        altered = deepcopy(baseline)
        altered.papers[0].jats.artifact_id = wrong_id
        assert any(
            finding.rule == "raw_artifact_id_mismatch"
            and finding.severity == "hard_error"
            and finding.paper_id == "P001"
            and finding.representation == "JATS"
            and finding.artifact_id == wrong_id
            for finding in validate_corpus(altered).findings
        )


def test_preprint_and_identifier_collision_rules():
    altered = deepcopy(corpus().papers)
    altered[2].publication_state = "preprint"
    altered[2].version_label = "v1"
    assert "preprint_count" in codes(validate_corpus(CorpusManifest(papers=altered)))
    altered = deepcopy(corpus().papers)
    altered[-1].publication_state = "preprint"
    altered[-1].version_label = "v1"
    assert "backfill_preprint" in codes(validate_corpus(CorpusManifest(papers=altered)))
    for field in ("doi", "pmid", "pmcid"):
        altered = deepcopy(corpus().papers)
        setattr(altered[1], field, getattr(altered[0], field) if field == "doi" else "12345")
        if field != "doi":
            setattr(altered[0], field, "12345")
            setattr(altered[0], field + "_status", "known")
            setattr(altered[1], field + "_status", "known")
        assert f"{field}_collision" in codes(validate_corpus(CorpusManifest(papers=altered)))
    altered = deepcopy(corpus().papers)
    altered[1].title = altered[0].title.upper() + "!"
    assert "strong_title_collision" in codes(validate_corpus(CorpusManifest(papers=altered)))


def test_csv_is_deterministic_and_round_trips():
    manifest = corpus()
    reversed_manifest = CorpusManifest(papers=list(reversed(manifest.papers)))
    data = serialize_csv(manifest)
    assert data == serialize_csv(reversed_manifest)
    assert data.startswith(b"schema_version,corpus_version,manifest_revision,paper_id,")
    assert b"\r\n" in data and data.count(b"\n") == data.count(b"\r\n")
    assert b"identifier_match|publication_state_version_review|title_match" in data
    restored = parse_csv(data)
    assert restored == manifest
    changed = deepcopy(manifest)
    changed.papers[0].title = 'Unicode β, "quoted"'
    assert parse_csv(serialize_csv(changed)) == changed
    with pytest.raises(ValueError, match="BOM"):
        parse_csv(b"\xef\xbb\xbf" + data)
    with pytest.raises(ValueError, match="columns"):
        parse_csv(data.replace(b"schema_version,", b"wrong_column,", 1))


def test_hashes_have_distinct_contracts():
    baseline = corpus()
    content = corpus_content_hash(baseline)
    semantic = manifest_semantic_hash(baseline)
    csv_hash = csv_byte_hash(serialize_csv(baseline))
    changed = deepcopy(baseline)
    changed.papers[0].title += " revised"
    assert corpus_content_hash(changed) == content
    assert manifest_semantic_hash(changed) != semantic
    for field, value in (("paper_id", "P051"),):
        changed = deepcopy(baseline)
        setattr(changed.papers[0], field, value)
        assert corpus_content_hash(changed) != content
    changed = deepcopy(baseline)
    changed.papers[0].jats.artifact_id += "x"
    assert corpus_content_hash(changed) != content
    changed = deepcopy(baseline)
    changed.papers[0].pdf.sha256 = "a" * 64
    assert corpus_content_hash(changed) != content
    alternate = serialize_csv(baseline).replace(b"\r\n", b"\n")
    assert manifest_semantic_hash(parse_csv(alternate)) == semantic
    assert corpus_content_hash(parse_csv(alternate)) == content
    assert csv_byte_hash(alternate) != csv_hash


def test_semantic_hash_and_csv_reject_mutated_invalid_record():
    manifest = corpus()
    manifest.papers[0].doi_status = "unknown"
    with pytest.raises(ValidationError):
        manifest_semantic_hash(manifest)
    with pytest.raises(ValidationError):
        serialize_csv(manifest)
    manifest = corpus()
    manifest.papers[0].manifest_revision = "other"
    with pytest.raises(ValueError, match="version"):
        manifest_semantic_hash(manifest)
    with pytest.raises(ValueError, match="version"):
        serialize_csv(manifest)


def test_content_hash_matches_independent_one_paper_oracle():
    item = PaperRecord.model_validate(paper(1))
    payload = {
        "corpus_version": "test-corpus",
        "papers": [
            {
                "paper_id": "P001",
                "representations": {
                    "JATS": {"artifact_id": item.jats.artifact_id, "sha256": item.jats.sha256},
                    "PDF": {"artifact_id": item.pdf.artifact_id, "sha256": item.pdf.sha256},
                },
            }
        ],
    }
    independent = sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert corpus_content_hash(CorpusManifest(papers=[item])) == independent


def test_malformed_xml_rejected_even_with_matching_sha(tmp_path):
    record = PaperRecord.model_validate(paper(1))
    invalid = b"<article>"
    record.jats.sha256 = sha256_bytes(invalid)
    path = tmp_path / "broken.xml"
    path.write_bytes(invalid)
    with pytest.raises(ValueError, match="XML"):
        verify_source_bytes(record, "jats", path)


def test_validation_report_and_release_record_are_typed():
    report = validate_corpus(corpus())
    assert ValidationReport.model_validate_json(report.model_dump_json()) == report
    release = ReleaseRecord.model_validate(
        {
            "record_schema_version": "test-record",
            "corpus_version": "test-corpus",
            "manifest_revision": "test-revision",
            "schema_version": "test-schema",
            "corpus_content_sha256": "a" * 64,
            "manifest_semantic_sha256": "b" * 64,
            "csv_sha256": "c" * 64,
            "manifest_artifact": {
                "artifact_id": "test_manifest",
                "logical_ref": "fixture:manifest",
                "sha256": "c" * 64,
            },
            "validation_report_artifact": {
                "artifact_id": "test_report",
                "logical_ref": "fixture:report",
                "sha256": "d" * 64,
            },
            "git_sha": "e" * 40,
        }
    )
    assert ReleaseRecord.model_validate_json(release.model_dump_json()) == release
    with pytest.raises(ValidationError):
        ReleaseRecord.model_validate({**release.model_dump(), "csv_sha256": "bad"})
    encoded = serialize_project_json(report, ValidationReport)
    assert encoded.endswith(b"\n") and b"\n  " in encoded
    assert json.loads(encoded)["schema_version"] == "1"
    assert serialize_project_json(report, ValidationReport) == encoded
    assert any(f.severity == "review_warning" and f.paper_id == "P001" for f in report.findings)
    changed = deepcopy(corpus())
    changed.papers[1].doi = changed.papers[0].doi
    bad_report = validate_corpus(changed)
    assert any(
        f.rule == "doi_collision" and f.severity == "hard_error" and f.paper_id == "P002"
        for f in bad_report.findings
    )
    assert b'"csv_sha256"' in serialize_project_json(release, ReleaseRecord)


def test_release_records_require_explicit_path_and_hash(tmp_path):
    from biorag_repro.core.hashing import HashMismatchError, sha256_bytes

    fields = {
        "record_schema_version": "1",
        "corpus_version": "corpus_v1",
        "manifest_revision": "1",
        "schema_version": "1",
        "corpus_content_sha256": "a" * 64,
        "manifest_semantic_sha256": "b" * 64,
        "csv_sha256": "c" * 64,
        "manifest_artifact": {
            "artifact_id": "manifest_csv",
            "logical_ref": "corpus_manifest.csv",
            "sha256": "c" * 64,
        },
        "validation_report_artifact": {
            "artifact_id": "validation_report",
            "logical_ref": "validation_report.json",
            "sha256": "d" * 64,
        },
    }
    old = ReleaseRecord.model_validate({**fields, "git_sha": "e" * 40})
    final = ReleaseRecord.model_validate({**fields, "git_sha": "f" * 40})
    old_path = tmp_path / "release_record.json"
    final_path = tmp_path / "release_record_phase2h_final.json"
    old_bytes = serialize_project_json(old, ReleaseRecord)
    final_bytes = serialize_project_json(final, ReleaseRecord)
    old_path.write_bytes(old_bytes)
    final_path.write_bytes(final_bytes)
    old_hash = sha256_bytes(old_bytes)
    final_hash = sha256_bytes(final_bytes)

    assert read_project_json(final_path, final_hash, ReleaseRecord) == final
    assert read_project_json(old_path, old_hash, ReleaseRecord) == old
    with pytest.raises(HashMismatchError):
        read_project_json(old_path, final_hash, ReleaseRecord)
    with pytest.raises(HashMismatchError):
        read_project_json(final_path, old_hash, ReleaseRecord)


def test_persistent_json_exact_hash_readback(tmp_path):
    report = validate_corpus(corpus())
    target = tmp_path / "report.json"
    content = serialize_project_json(report, ValidationReport)
    target.write_bytes(content)
    assert read_project_json(target, sha256_bytes(content), ValidationReport) == report
    with pytest.raises(ValueError):
        read_project_json(target, "a" * 64, ValidationReport)


def test_source_magic_does_not_override_declared_hash(tmp_path):
    record = PaperRecord.model_validate(paper(1))
    path = tmp_path / "wrong.pdf"
    path.write_bytes(b"<article/>")
    with pytest.raises(ValueError, match="PDF media type"):
        verify_source_bytes(record, "pdf", path)
