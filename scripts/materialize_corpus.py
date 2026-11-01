"""Materialize the approved first corpus from retained source authority."""

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from xml.etree import ElementTree

from biorag_repro.core.hashing import sha256_bytes, sha256_file
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.corpus.manifest import (
    CorpusManifest,
    Finding,
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

AUTHORITY_HASHES = {
    "selected_manifest.csv": "3e006841ad69cb6ab2abebf6cba3e3c493954334be3392e66832bcc669c5e03f",
    "PMC_JATS_LOCAL_VERIFICATION.csv": "ae820d2421c1bd081c7b138f9a5a72a8747e0422622c81aa18492533b614072b",
    "CORPUS_SOURCE_FEASIBILITY_AUDIT.csv": "77caf428070ce0ef2e4540c54797d5ba11d45575cd8ef353214d4ec5227e1421",
    "PMC_JATS_FOUR_PAPER_RESOLUTION.csv": "3d71d70f07980057e25092999355fde361a7a29e0f5db34e64cd030102b4d6b5",
}
DOMAIN = {
    "GWAS / population genetics": "population_genetics_gwas",
    "biomarker discovery": "biomarker_discovery",
    "computational biology / biological ML": "computational_biology_ml",
    "microbiome / metagenomics": "microbiome_metagenomics",
    "multi-omics": "multi_omics",
    "protein modeling": "protein_modeling",
    "single-cell": "single_cell",
    "spatial omics": "spatial_omics",
    "transcriptomics": "transcriptomics",
    "genomics": "other",
}


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def jsonl_rows(path: Path) -> list[dict[str, object]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def utc_text(value: str) -> str:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("acquisition time has no timezone")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def source_dates(article: ElementTree.Element) -> list[tuple[str, date | None, int]]:
    result = []
    for element in article.findall("pub-date"):
        year_text = element.findtext("year")
        if not year_text:
            continue
        year = int(year_text)
        month = element.findtext("month")
        day = element.findtext("day")
        full = date(year, int(month), int(day)) if month and day else None
        kind = element.get("pub-type") or (
            "epub"
            if element.get("publication-format") == "electronic"
            and element.get("date-type") == "pub"
            else element.get("date-type") or "unqualified"
        )
        result.append((kind, full, year))
    return result


def publication_basis(
    paper_id: str, year: int, state: str, dates: list[tuple[str, date | None, int]]
) -> tuple[str, date | None]:
    if state == "preprint":
        matching = [entry for entry in dates if entry[2] == year]
        if not matching:
            raise ValueError(f"preprint year unsupported: {paper_id}")
        return "preprint_posted", next((entry[1] for entry in matching if entry[1]), None)
    if paper_id in {"P015", "P023"}:
        if paper_id == "P015" and not any(k == "collection" and y == year for k, _, y in dates):
            raise ValueError("P015 issue year unsupported")
        return "issue", None
    for kind, full, observed_year in dates:
        if observed_year == year and kind in {"epub", "electronic"}:
            return "online", full
    for kind, full, observed_year in dates:
        if observed_year == year and kind in {"ppub", "collection"}:
            return "issue", full
    for kind, full, observed_year in dates:
        if observed_year == year and kind == "unqualified":
            return "unqualified_source_date", full
    raise ValueError(f"publication year has no source date: {paper_id}")


def jats_license(
    article: ElementTree.Element, logical_ref: str
) -> tuple[str | None, str, str | None]:
    license_element = article.find("./permissions/license")
    if license_element is None:
        return None, "unknown", None
    match = re.search(
        r"creativecommons\.org/licenses/(by(?:-nc)?(?:-nd)?)/4\.0",
        ElementTree.tostring(license_element, encoding="unicode"),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, "unknown", None
    return (
        "CC-" + match.group(1).upper() + "-4.0",
        "known",
        logical_ref + "#front/article-meta/permissions/license",
    )


def accepted_sources(authority: Path, attempts: Path) -> dict[tuple[str, str], tuple[str, str]]:
    result = {}
    for row in csv_rows(authority / "selected_manifest.csv"):
        for rep in ("JATS", "PDF"):
            result[row["pmcid"], rep] = (row[rep.lower() + "_sha256"], "historical_backfill")
    for row in jsonl_rows(attempts / "phase2e-20261031T064802Z_original_pdf_checks.jsonl"):
        assert row["computed_sha256"] == row["expected_historical_sha256"]
        result[str(row["pmcid"]), "PDF"] = (
            str(row["expected_historical_sha256"]),
            "historical_original_pdf",
        )
    for row in jsonl_rows(attempts / "phase2f-20261031T070000Z_attempts.jsonl"):
        result[str(row["pmcid"]), "JATS"] = (
            str(row["computed_sha256"]),
            "new_initial_original_jats",
        )
    adjudications = jsonl_rows(attempts / "phase2f-20261031T070000Z_adjudications.jsonl")
    assert len(adjudications) == 1 and adjudications[0]["pmcid"] == "PMC13008253"
    assert adjudications[0]["final_disposition"] == "INITIAL_CORPUS_BYTE_IDENTITY_FROZEN"
    assert adjudications[0]["durable_readback_sha256"] == result["PMC13008253", "JATS"][0]
    assert len(result) == 100 and Counter(value[1] for value in result.values()) == {
        "historical_backfill": 42,
        "historical_original_pdf": 29,
        "new_initial_original_jats": 29,
    }
    return result


def acquisition_times(
    attempts: Path, accepted: dict[tuple[str, str], tuple[str, str]]
) -> dict[tuple[str, str], str]:
    result = {}
    for name in (
        "phase2d-20261031_attempts.jsonl",
        "batch/attempts.jsonl",
        "pilot/retry_attempts.jsonl",
        "phase2f-20261031T070000Z_attempts.jsonl",
    ):
        for row in jsonl_rows(attempts / name):
            key = str(row["pmcid"]), str(row["manifestation"])
            if key in accepted and row.get("computed_sha256") == accepted[key][0]:
                timestamp = row.get("attempted_at_utc") or row.get("attempt_time_utc")
                if isinstance(timestamp, str):
                    result[key] = utc_text(timestamp)
    assert len(result) == 71, (
        f"expected 42 backfill and 29 original JATS acquisition times, found {len(result)}"
    )
    return result


def build_candidate(authority: Path, data: Path) -> CorpusManifest:
    for name, digest in AUTHORITY_HASHES.items():
        assert sha256_file(authority / name) == digest, f"authority changed: {name}"
    mapping = csv_rows(data / "corpus/paper_identity_map.csv")
    assert len(mapping) == 50 and {row["paper_id"] for row in mapping} == {
        f"P{i:03d}" for i in range(1, 51)
    }
    assert len({row["pmcid"] for row in mapping}) == 50
    original = {
        row["paper_id"]: row for row in csv_rows(authority / "PMC_JATS_LOCAL_VERIFICATION.csv")
    }
    backfill = {row["pmcid"]: row for row in csv_rows(authority / "selected_manifest.csv")}
    audit = {
        row["paper_id"]: row for row in csv_rows(authority / "CORPUS_SOURCE_FEASIBILITY_AUDIT.csv")
    }
    four = {
        row["paper_id"]: row for row in csv_rows(authority / "PMC_JATS_FOUR_PAPER_RESOLUTION.csv")
    }
    assert len(original) == 29 and len(backfill) == 21 and len(four) == 4
    assert {row["pmcid"] for row in mapping if row["provenance_group"] == "backfill_21"} == set(
        backfill
    )
    assert {
        (row["paper_id"], row["pmcid"])
        for row in mapping
        if row["provenance_group"] == "original_29"
    } == {(p, row["pmcid"]) for p, row in original.items()}
    accepted = accepted_sources(authority, data / "staging/attempts")
    acquired = acquisition_times(data / "staging/attempts", accepted)
    papers = []
    for item in sorted(mapping, key=lambda row: row["paper_id"]):
        paper_id, pmcid, group = item["paper_id"], item["pmcid"], item["provenance_group"]
        is_original = group == "original_29"
        evidence = audit[paper_id] if is_original else backfill[pmcid]
        xml_path = data / "staging/exact" / pmcid / "JATS/source.xml"
        xml_root = ElementTree.parse(xml_path).getroot()
        article = xml_root.find("./front/article-meta")
        journal = xml_root.find("./front/journal-meta/journal-title-group/journal-title")
        title_element = article.find("title-group/article-title") if article is not None else None
        assert article is not None and journal is not None and title_element is not None
        title = "".join(title_element.itertext()).strip()
        venue = "".join(journal.itertext()).strip()
        assert title and venue
        doi = evidence["doi"]
        pmid = original[paper_id]["xml_pmid"] if paper_id == "P036" else evidence["pmid"]
        ids = {
            (element.get("pub-id-type") or element.get("id-type")): (element.text or "").strip()
            for element in article.findall("article-id")
        }
        assert ids.get("doi", "").casefold() == doi.casefold(), paper_id
        assert (ids.get("pmcid") or ids.get("pmc") or "").removeprefix("PMC") == pmcid.removeprefix(
            "PMC"
        ), paper_id
        assert not ids.get("pmid") or ids["pmid"] == pmid, paper_id
        state = "preprint" if paper_id in {"P021", "P028"} else "final_journal_article"
        assert (
            not is_original
            or (original[paper_id]["article_state"] == "preprint") == (state == "preprint")
            or paper_id == "P024"
        )
        year = int(evidence["year"])
        basis, publication_date = publication_basis(paper_id, year, state, source_dates(article))
        notes = []
        if is_original:
            notes.append(
                "JATS byte identity newly frozen for initial corpus_v1; equality with historically inspected JATS bytes unknown"
            )
        if paper_id == "P023":
            notes.append("Online 2022-12-14; 2023 issue year, per four-paper adjudication")
        elif paper_id == "P015":
            notes.append("Online 2025-12-22; 2026 collection/issue year")
        elif paper_id == "P024":
            notes.append("PLOS version of record update; prior proof classification was false")
        elif paper_id == "P001":
            notes.append(
                "Retained PDF extracted-title mismatch was adjudicated by direct first-page DOI/title inspection"
            )
        if (
            not is_original
            and evidence["date"]
            and publication_date
            and evidence["date"] != publication_date.isoformat()
        ):
            notes.append(
                f"Retained selected-record date {evidence['date']} differs from JATS publication date {publication_date.isoformat()}"
            )
        domain = DOMAIN[evidence["domain"] if is_original else evidence["primary_domain"]]
        if is_original and evidence["domain"] == "genomics":
            notes.append("Retained broad domain genomics is represented as other in schema 1")
        sources = {}
        for rep in ("JATS", "PDF"):
            digest, _provenance = accepted[pmcid, rep]
            artifact_id = raw_source_artifact_id(
                paper_id=paper_id, representation=rep, sha256=digest
            )
            logical_ref = f"{paper_id}/{rep}/{artifact_id}.{'xml' if rep == 'JATS' else 'pdf'}"
            if rep == "JATS":
                provider = "Europe PMC" if is_original else "PMC Open Access S3"
                reference = (
                    str(
                        next(
                            row
                            for row in jsonl_rows(
                                data / "staging/attempts/phase2f-20261031T070000Z_attempts.jsonl"
                            )
                            if row["pmcid"] == pmcid
                        )["retained_endpoint"]
                    )
                    if is_original
                    else evidence["jats_endpoint"]
                )
                license_id, license_status, license_ref = jats_license(article, logical_ref)
            else:
                provider = "historical local PDF" if is_original else "PMC Open Access S3"
                reference = (
                    f"CORPUS_SOURCE_FEASIBILITY_AUDIT.csv#paper_id={paper_id}:existing_pdf"
                    if is_original
                    else evidence["pdf_endpoint"]
                )
                license_id, license_status, license_ref = None, "unknown", None
            sources[rep.lower()] = {
                "artifact_id": artifact_id,
                "sha256": digest,
                "provider": provider,
                "acquisition_reference": reference,
                "acquired_at_utc": acquired.get((pmcid, rep)),
                "acquired_at_status": "known" if (pmcid, rep) in acquired else "unknown",
                "media_type": "application/xml" if rep == "JATS" else "application/pdf",
                "logical_ref": logical_ref,
                "source_status": "frozen_local",
                "license_id": license_id,
                "license_status": license_status,
                "license_evidence_ref": license_ref,
                "analysis_access": "available",
                "redistribution_status": "unknown",
            }
            staged = (
                data
                / "staging/exact"
                / pmcid
                / rep
                / ("source.xml" if rep == "JATS" else "source.pdf")
            )
            assert staged.is_file() and not staged.is_symlink()
            assert sha256_file(staged) == digest, staged
        pair_ref = (
            f"PMC_JATS_FOUR_PAPER_RESOLUTION.csv#paper_id={paper_id}; PMC_JATS_LOCAL_VERIFICATION.csv#paper_id={paper_id}"
            if paper_id in four
            else f"PMC_JATS_LOCAL_VERIFICATION.csv#paper_id={paper_id}"
            if is_original
            else f"selected_manifest.csv#pmcid={pmcid}; FINAL_CORPUS_CONSISTENCY_CHECK.md"
        )
        pair_basis = ["identifier_match", "prior_audit_record"]
        if is_original:
            if paper_id in four:
                assert four[paper_id]["final_pair_status"] == "VERIFIED"
                pair_basis.append("publication_state_version_review")
            elif original[paper_id]["title_match"] == "True":
                pair_basis.append("title_match")
        else:
            assert evidence["pdf_doi_match"] == "True"
            pair_basis.append("pdf_doi_inspection")
            pair_basis.append(
                "title_match" if evidence["pdf_title_match"] == "True" else "source_inspection"
            )
        pair_note = None
        if paper_id in four:
            pair_note = four[paper_id]["notes"]
        elif paper_id == "P001":
            pair_note = "PDF extracted-title flag false; final audit records direct first-page DOI/title adjudication"
        verified_at = utc_text(evidence["verified_at_utc"]) if not is_original else None
        papers.append(
            PaperRecord.model_validate(
                {
                    "schema_version": "1",
                    "corpus_version": "corpus_v1",
                    "manifest_revision": "1",
                    "paper_id": paper_id,
                    "provenance_group": group,
                    "title": title,
                    "venue": venue,
                    "publication_year": year,
                    "publication_year_basis": basis,
                    "publication_state": state,
                    "version_label": f"bioRxiv {'v2' if paper_id == 'P021' else 'v1'} hosted on PMC"
                    if state == "preprint"
                    else four[paper_id]["explicit_source_version"]
                    if paper_id in {"P023", "P024"}
                    else None,
                    "doi": doi,
                    "doi_status": "known",
                    "pmid": pmid,
                    "pmid_status": "known",
                    "pmcid": pmcid,
                    "pmcid_status": "known",
                    "primary_domain": domain,
                    "secondary_domain": None,
                    "publication_date": publication_date,
                    "publication_date_status": "known" if publication_date else "unknown",
                    "authors_compact": evidence.get("authors") or None,
                    "article_type": xml_root.get("article-type") or None,
                    "bibliographic_note": "; ".join(notes) if notes else None,
                    "jats": sources["jats"],
                    "pdf": sources["pdf"],
                    "pair": {
                        "pair_status": "verified_exact_pair",
                        "pair_verification_basis": pair_basis,
                        "pair_verified_at_utc": verified_at,
                        "pair_verified_at_status": "known" if verified_at else "unknown",
                        "pair_verified_by": None,
                        "pair_audit_ref": pair_ref,
                        "pair_exception_note": pair_note,
                    },
                }
            )
        )
    manifest = CorpusManifest(papers=papers)
    assert len(manifest.papers) == 50
    assert {paper.paper_id for paper in manifest.papers} == {f"P{i:03d}" for i in range(1, 51)}
    assert (
        len({source.artifact_id for paper in papers for source in (paper.jats, paper.pdf)}) == 100
    )
    for paper in papers:
        for rep, source in (("JATS", paper.jats), ("PDF", paper.pdf)):
            assert source.artifact_id == raw_source_artifact_id(
                paper_id=paper.paper_id, representation=rep, sha256=source.sha256
            )
    report = validate_corpus(manifest)
    assert not any(f.severity == "hard_error" for f in report.findings), report
    return manifest


def verify_tree(manifest: CorpusManifest, root: Path) -> None:
    expected = {
        root / source.logical_ref for paper in manifest.papers for source in (paper.jats, paper.pdf)
    }
    actual = {path for path in root.rglob("*") if not path.is_dir()}
    assert len(expected) == len(actual) == 100 and expected == actual
    for paper in manifest.papers:
        verify_source_bytes(paper, "jats", root / paper.jats.logical_ref)
        verify_source_bytes(paper, "pdf", root / paper.pdf.logical_ref)


def main() -> None:
    if not __debug__:
        raise RuntimeError("corpus materialization requires Python assertion checks")
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--backup-source", required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    manifest = build_candidate(args.authority_root, args.data_root)
    staging = args.data_root / "staging/exact"
    for paper in manifest.papers:
        for rep in ("JATS", "PDF"):
            verify_source_bytes(
                paper,
                rep.lower(),
                staging / paper.pmcid / rep / ("source.xml" if rep == "JATS" else "source.pdf"),
            )
    backup_proof = json.loads((args.backup_root / "backup_verification.json").read_text())
    assert backup_proof["status"] == "verified" and backup_proof["source_count"] == 100
    assert backup_proof["mount_source"] == args.backup_source
    backup_records = {(r["paper_id"], r["representation"]): r for r in backup_proof["records"]}
    assert len(backup_records) == 100
    for paper in manifest.papers:
        for rep in ("JATS", "PDF"):
            source = paper.jats if rep == "JATS" else paper.pdf
            backup = backup_records[paper.paper_id, rep]
            assert (
                backup["accepted_sha256"] == source.sha256
                and backup["artifact_id"] == source.artifact_id
            )
            assert sha256_file(args.backup_root / backup["backup_ref"]) == source.sha256
    if args.check_only:
        assert parse_csv(serialize_csv(manifest)) == manifest
        print("CANDIDATE_VALID: 50 papers, 100 staged sources, 100 verified backup sources")
        return
    mount = (
        subprocess.check_output(
            ["findmnt", "-T", str(args.backup_root), "-n", "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
            text=True,
        )
        .strip()
        .split()
    )
    assert mount[1] == args.backup_source and "rw" in mount[3].split(",")
    raw_root = args.data_root / "corpus/raw"
    for paper in manifest.papers:
        for rep in ("JATS", "PDF"):
            source = paper.jats if rep == "JATS" else paper.pdf
            staged = staging / paper.pmcid / rep / ("source.xml" if rep == "JATS" else "source.pdf")
            target = raw_root / source.logical_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                verify_source_bytes(paper, rep.lower(), target)
            else:
                publish_bytes(target, staged.read_bytes(), source.sha256)
    verify_tree(manifest, raw_root)
    release_dir = args.data_root / "corpus/releases/corpus_v1/manifest_revision_1"
    release_dir.mkdir(parents=True, exist_ok=True)
    csv_content = serialize_csv(manifest)
    csv_target = release_dir / "corpus_manifest.csv"
    csv_hash = csv_byte_hash(csv_content)
    if csv_target.exists():
        assert sha256_file(csv_target) == csv_hash
    else:
        publish_bytes(csv_target, csv_content, csv_hash)
    loaded = parse_csv(csv_target.read_bytes())
    assert loaded == manifest
    content_hash = corpus_content_hash(loaded)
    semantic_hash = manifest_semantic_hash(loaded)
    assert csv_byte_hash(csv_target.read_bytes()) == csv_hash
    findings = list(validate_corpus(loaded).findings)
    assert not any(f.severity == "hard_error" for f in findings)
    assert Counter(p.provenance_group for p in loaded.papers) == {
        "original_29": 29,
        "backfill_21": 21,
    }
    assert {p.paper_id for p in loaded.papers if p.publication_state == "preprint"} == {
        "P021",
        "P028",
    }
    assert len({s.artifact_id for p in loaded.papers for s in (p.jats, p.pdf)}) == 100
    for paper in loaded.papers:
        for rep, source in (("JATS", paper.jats), ("PDF", paper.pdf)):
            assert source.artifact_id == raw_source_artifact_id(
                paper_id=paper.paper_id, representation=rep, sha256=source.sha256
            )
    for rule, evidence in (
        ("paper_ids_complete", "paper_identity_map.csv"),
        ("provenance_29_21", "corpus_manifest.csv"),
        ("two_original_preprints", "corpus_manifest.csv"),
        ("raw_artifact_ids_100_unique", "corpus_manifest.csv"),
        ("typed_fields_and_pair_evidence", "corpus_manifest.csv"),
        ("identifier_collisions_absent", "corpus_manifest.csv"),
        ("backup_100_exact", "backup_verification.json"),
        ("final_100_exact", "corpus/raw"),
        ("corpus_content_hash", content_hash),
        ("manifest_semantic_hash", semantic_hash),
        ("csv_byte_hash", csv_hash),
    ):
        findings.append(
            Finding(rule=rule, severity="pass", evidence_ref=evidence, disposition="verified")
        )
    for paper in loaded.papers:
        for rep in ("JATS", "PDF"):
            source = paper.jats if rep == "JATS" else paper.pdf
            findings.append(
                Finding(
                    rule="source_exact_hash",
                    severity="pass",
                    paper_id=paper.paper_id,
                    representation=rep,
                    artifact_id=source.artifact_id,
                    evidence_ref=source.logical_ref,
                    disposition="verified",
                )
            )
    for finding in findings:
        if finding.severity == "review_warning" and finding.evidence_ref is None:
            finding.evidence_ref = "corpus_manifest.csv"
            finding.disposition = "retained as typed missingness for review"
    report = ValidationReport(schema_version="1", findings=findings)
    assert not any(f.severity == "hard_error" for f in report.findings)
    report_content = serialize_project_json(report, ValidationReport)
    report_target = release_dir / "validation_report.json"
    report_hash = sha256_bytes(report_content)
    if report_target.exists():
        assert sha256_file(report_target) == report_hash
    else:
        publish_bytes(report_target, report_content, report_hash)
    assert read_project_json(report_target, report_hash, ValidationReport) == report
    git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    release = ReleaseRecord(
        record_schema_version="1",
        corpus_version="corpus_v1",
        manifest_revision="1",
        schema_version="1",
        corpus_content_sha256=content_hash,
        manifest_semantic_sha256=semantic_hash,
        csv_sha256=csv_hash,
        manifest_artifact=ArtifactRef(
            artifact_id="manifest_csv_" + csv_hash,
            logical_ref="corpus/releases/corpus_v1/manifest_revision_1/corpus_manifest.csv",
            sha256=csv_hash,
        ),
        validation_report_artifact=ArtifactRef(
            artifact_id="validation_report_" + report_hash,
            logical_ref="corpus/releases/corpus_v1/manifest_revision_1/validation_report.json",
            sha256=report_hash,
        ),
        git_sha=git_sha,
    )
    release_content = serialize_project_json(release, ReleaseRecord)
    release_target = release_dir / "release_record.json"
    release_hash = sha256_bytes(release_content)
    if release_target.exists():
        assert sha256_file(release_target) == release_hash
    else:
        publish_bytes(release_target, release_content, release_hash)
    assert read_project_json(release_target, release_hash, ReleaseRecord) == release
    print(
        json.dumps(
            {
                "status": "materialized",
                "paper_count": 50,
                "source_count": 100,
                "corpus_content_sha256": content_hash,
                "manifest_semantic_sha256": semantic_hash,
                "csv_sha256": csv_hash,
                "validation_report_sha256": report_hash,
                "release_record_sha256": release_hash,
                "warning_count": sum(f.severity == "review_warning" for f in report.findings),
            }
        )
    )


if __name__ == "__main__":
    main()
