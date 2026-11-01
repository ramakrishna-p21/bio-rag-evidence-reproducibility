"""Typed frozen-source records, CSV identities, and synthetic-ready validation."""

import csv
import io
import re
import unicodedata
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Self
from xml.etree import ElementTree

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from biorag_repro.core.hashing import (
    canonical_json_bytes,
    sha256_bytes,
    validate_sha256,
    verify_file_sha256,
)
from biorag_repro.core.manifest import ArtifactRef

Status = Literal["known", "unknown", "unavailable", "not_applicable", "not_checked"]
Domain = Literal[
    "population_genetics_gwas",
    "biomarker_discovery",
    "computational_biology_ml",
    "microbiome_metagenomics",
    "multi_omics",
    "protein_modeling",
    "single_cell",
    "spatial_omics",
    "transcriptomics",
    "other",
]
Basis = Literal[
    "identifier_match",
    "title_match",
    "publication_state_version_review",
    "pdf_doi_inspection",
    "source_inspection",
    "prior_audit_record",
]
Representation = Literal["JATS", "PDF"]


class StrictRecord(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


def _nonblank(value: str) -> str:
    if not value.strip():
        raise ValueError("required text must be nonblank")
    return value


def _status(value: object, status: Status) -> None:
    if (status == "known") != (value is not None):
        raise ValueError("known requires a value; other statuses require null")


def _utc_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z", value) is None:
        raise ValueError("timestamp must be UTC RFC 3339 with Z suffix")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo != UTC:
        raise ValueError("timestamp must be UTC")
    return value


class SourceRecord(StrictRecord):
    artifact_id: str
    sha256: str
    provider: str
    acquisition_reference: str
    acquired_at_utc: str | None
    acquired_at_status: Status
    media_type: Literal["application/xml", "application/pdf"]
    logical_ref: str
    source_status: Literal["frozen_local"]
    license_id: str | None
    license_status: Status
    license_evidence_ref: str | None
    analysis_access: Literal["available", "unavailable"]
    redistribution_status: Literal["permitted", "prohibited", "unknown"]

    @field_validator("artifact_id", "provider", "acquisition_reference")
    @classmethod
    def required_text(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @field_validator("logical_ref")
    @classmethod
    def reference(cls, value: str) -> str:
        ArtifactRef(artifact_id="reference-check", logical_ref=value, sha256="0" * 64)
        if ".." in re.split(r"[/\\]", value):
            raise ValueError("logical_ref must not traverse outside its data root")
        return value

    @field_validator("acquired_at_utc")
    @classmethod
    def timestamp(cls, value: str | None) -> str | None:
        return _utc_timestamp(value)

    @model_validator(mode="after")
    def paired_statuses(self) -> Self:
        _status(self.acquired_at_utc, self.acquired_at_status)
        _status(self.license_id, self.license_status)
        if self.license_id is not None and not self.license_evidence_ref:
            raise ValueError("known license requires evidence reference")
        if self.license_evidence_ref is not None and not self.license_evidence_ref.strip():
            raise ValueError("license evidence reference must be nonblank")
        return self


class PairVerification(StrictRecord):
    pair_status: Literal["verified_exact_pair"]
    pair_verification_basis: list[Basis] = Field(min_length=1)
    pair_verified_at_utc: str | None
    pair_verified_at_status: Status
    pair_verified_by: str | None
    pair_audit_ref: str | None
    pair_exception_note: str | None

    @field_validator("pair_verified_at_utc")
    @classmethod
    def timestamp(cls, value: str | None) -> str | None:
        return _utc_timestamp(value)

    @model_validator(mode="after")
    def validate_basis(self) -> Self:
        _status(self.pair_verified_at_utc, self.pair_verified_at_status)
        if len(self.pair_verification_basis) != len(set(self.pair_verification_basis)):
            raise ValueError("pair verification basis must be distinct")
        self.pair_verification_basis.sort()
        if "prior_audit_record" in self.pair_verification_basis and not self.pair_audit_ref:
            raise ValueError("prior audit basis requires audit reference")
        for value in (self.pair_verified_by, self.pair_audit_ref, self.pair_exception_note):
            if value is not None:
                _nonblank(value)
        return self


class PaperRecord(StrictRecord):
    schema_version: str
    corpus_version: str
    manifest_revision: str
    paper_id: str
    provenance_group: Literal["original_29", "backfill_21"]
    title: str
    venue: str
    publication_year: int
    publication_year_basis: Literal["online", "issue", "preprint_posted", "unqualified_source_date"]
    publication_state: Literal["final_journal_article", "preprint"]
    version_label: str | None
    doi: str | None
    doi_status: Status
    pmid: str | None
    pmid_status: Status
    pmcid: str | None
    pmcid_status: Status
    primary_domain: Domain
    secondary_domain: Domain | None
    publication_date: date | None
    publication_date_status: Status
    authors_compact: str | None
    article_type: str | None
    bibliographic_note: str | None
    jats: SourceRecord
    pdf: SourceRecord
    pair: PairVerification

    @field_validator(
        "schema_version", "corpus_version", "manifest_revision", "paper_id", "title", "venue"
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("paper_id")
    @classmethod
    def paper_identity_shape(cls, value: str) -> str:
        if re.fullmatch(r"P\d{3}", value) is None:
            raise ValueError("paper_id must use the frozen P001–P050 form")
        return value

    @model_validator(mode="after")
    def validate_paper(self) -> Self:
        for name in ("doi", "pmid", "pmcid", "publication_date"):
            _status(getattr(self, name), getattr(self, name + "_status"))
        for name in ("doi", "pmid", "pmcid"):
            value = getattr(self, name)
            if value is not None:
                _nonblank(value)
                if value.lower() in {"unknown", "unavailable", "not_applicable", "not_checked"}:
                    raise ValueError(f"{name} cannot contain a status sentinel")
        if self.publication_year < 1 or self.publication_year > 9999:
            raise ValueError("publication_year must be a calendar year")
        if self.publication_state == "preprint" and not self.version_label:
            raise ValueError("preprint requires version label")
        for value in (
            self.version_label,
            self.authors_compact,
            self.article_type,
            self.bibliographic_note,
        ):
            if value is not None:
                _nonblank(value)
        if self.jats.media_type != "application/xml" or self.pdf.media_type != "application/pdf":
            raise ValueError("JATS/PDF source media types do not match their manifestations")
        return self


class CorpusManifest(StrictRecord):
    papers: list[PaperRecord]


class Finding(StrictRecord):
    rule: str
    severity: Literal["pass", "hard_error", "review_warning"]
    paper_id: str | None = None
    representation: Representation | None = None
    artifact_id: str | None = None
    evidence_ref: str | None = None
    disposition: str | None = None


class ValidationReport(StrictRecord):
    schema_version: str
    findings: list[Finding]


class ReleaseRecord(StrictRecord):
    record_schema_version: str
    corpus_version: str
    manifest_revision: str
    schema_version: str
    corpus_content_sha256: str
    manifest_semantic_sha256: str
    csv_sha256: str
    manifest_artifact: ArtifactRef
    validation_report_artifact: ArtifactRef
    git_sha: str

    @field_validator(
        "record_schema_version", "corpus_version", "manifest_revision", "schema_version"
    )
    @classmethod
    def required_text(cls, value: str) -> str:
        return _nonblank(value)

    @field_validator("corpus_content_sha256", "manifest_semantic_sha256", "csv_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @field_validator("git_sha")
    @classmethod
    def git_digest(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("git_sha must be a full lowercase Git SHA-1")
        return value

    @model_validator(mode="after")
    def matching_manifest_hash(self) -> Self:
        if self.manifest_artifact.sha256 != self.csv_sha256:
            raise ValueError("manifest artifact hash must equal CSV byte hash")
        return self


_PAPER_FIELDS = tuple(
    name for name in PaperRecord.model_fields if name not in {"jats", "pdf", "pair"}
)
_SOURCE_FIELDS = tuple(SourceRecord.model_fields)
_PAIR_FIELDS = tuple(PairVerification.model_fields)
CSV_COLUMNS = (
    *_PAPER_FIELDS,
    *("jats_" + name for name in _SOURCE_FIELDS),
    *("pdf_" + name for name in _SOURCE_FIELDS),
    *_PAIR_FIELDS,
)


def _semantic_papers(manifest: CorpusManifest) -> list[dict[str, Any]]:
    return [
        paper.model_dump(mode="json") for paper in sorted(manifest.papers, key=lambda p: p.paper_id)
    ]


def _validated_manifest(manifest: CorpusManifest) -> CorpusManifest:
    validated = CorpusManifest.model_validate(
        manifest.model_dump(mode="python", warnings="error"), strict=True
    )
    for field in ("schema_version", "corpus_version", "manifest_revision"):
        if len({getattr(paper, field) for paper in validated.papers}) != 1:
            raise ValueError(f"{field} version must be consistent across papers")
    return validated


def corpus_content_hash(manifest: CorpusManifest) -> str:
    manifest = _validated_manifest(manifest)
    versions = {paper.corpus_version for paper in manifest.papers}
    if len(versions) != 1:
        raise ValueError("corpus_version must be consistent and nonempty")
    papers = [
        {
            "paper_id": paper.paper_id,
            "representations": {
                "JATS": {"artifact_id": paper.jats.artifact_id, "sha256": paper.jats.sha256},
                "PDF": {"artifact_id": paper.pdf.artifact_id, "sha256": paper.pdf.sha256},
            },
        }
        for paper in sorted(manifest.papers, key=lambda p: p.paper_id)
    ]
    return sha256_bytes(canonical_json_bytes({"corpus_version": versions.pop(), "papers": papers}))


def manifest_semantic_hash(manifest: CorpusManifest) -> str:
    manifest = _validated_manifest(manifest)
    papers = _semantic_papers(manifest)
    for paper in papers:
        paper["pair"]["pair_verification_basis"] = sorted(paper["pair"]["pair_verification_basis"])
    return sha256_bytes(canonical_json_bytes({"papers": papers}))


def csv_byte_hash(content: bytes) -> str:
    return sha256_bytes(content)


def serialize_csv(manifest: CorpusManifest) -> bytes:
    manifest = _validated_manifest(manifest)
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\r\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    for paper in sorted(manifest.papers, key=lambda item: item.paper_id):
        data = paper.model_dump(mode="json")
        row = {name: data[name] for name in _PAPER_FIELDS}
        for prefix in ("jats", "pdf"):
            row.update({prefix + "_" + name: data[prefix][name] for name in _SOURCE_FIELDS})
        row.update(data["pair"])
        row["pair_verification_basis"] = "|".join(sorted(row["pair_verification_basis"]))
        writer.writerow(["" if row[name] is None else row[name] for name in CSV_COLUMNS])
    return stream.getvalue().encode("utf-8")


def parse_csv(content: bytes) -> CorpusManifest:
    decoded = content.decode("utf-8-sig")
    if content.startswith(b"\xef\xbb\xbf"):
        raise ValueError("CSV must not have a UTF-8 BOM")
    reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
    if reader.fieldnames != list(CSV_COLUMNS):
        raise ValueError("CSV columns do not match the frozen order")
    papers = []
    nullable = {
        "version_label",
        "doi",
        "pmid",
        "pmcid",
        "secondary_domain",
        "publication_date",
        "authors_compact",
        "article_type",
        "bibliographic_note",
        *(
            prefix + "_" + name
            for prefix in ("jats", "pdf")
            for name in ("acquired_at_utc", "license_id", "license_evidence_ref")
        ),
        "pair_verified_at_utc",
        "pair_verified_by",
        "pair_audit_ref",
        "pair_exception_note",
    }
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("CSV row has incorrect column count")
        if any(row[name] == "" for name in CSV_COLUMNS if name not in nullable):
            raise ValueError("required CSV cell is empty")
        values = {
            name: (None if name in nullable and value == "" else value)
            for name, value in row.items()
        }
        year = values["publication_year"]
        basis = values["pair_verification_basis"]
        if not isinstance(year, str) or not isinstance(basis, str):
            raise TypeError("required CSV fields are missing")
        values["publication_year"] = int(year)
        publication_date = values["publication_date"]
        if isinstance(publication_date, str):
            values["publication_date"] = date.fromisoformat(publication_date)
        values["pair_verification_basis"] = basis.split("|")
        for prefix in ("jats", "pdf"):
            values[prefix] = {name: values.pop(prefix + "_" + name) for name in _SOURCE_FIELDS}
        values["pair"] = {name: values.pop(name) for name in _PAIR_FIELDS}
        papers.append(PaperRecord.model_validate(values))
    return CorpusManifest(papers=papers)


def _normalized_identifier(value: str, kind: str) -> str:
    compact = re.sub(r"\s+", "", value).casefold()
    if kind == "doi":
        return re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:)", "", compact)
    if kind == "pmcid":
        return compact.removeprefix("pmc")
    return compact


def _normalized_title(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return "".join(char for char in value if char.isalnum())


def validate_corpus(manifest: CorpusManifest) -> ValidationReport:
    findings: list[Finding] = []

    def add(
        rule: str,
        severity: Literal["hard_error", "review_warning"],
        paper: PaperRecord | None = None,
        representation: Representation | None = None,
        artifact_id: str | None = None,
    ) -> None:
        findings.append(
            Finding(
                rule=rule,
                severity=severity,
                paper_id=paper.paper_id if paper else None,
                representation=representation,
                artifact_id=artifact_id,
            )
        )

    papers = manifest.papers
    if len(papers) != 50:
        add("paper_count", "hard_error")
    if len({p.paper_id for p in papers}) != len(papers):
        add("paper_id_unique", "hard_error")
    if any(
        len({getattr(p, field) for p in papers}) != 1
        for field in ("schema_version", "corpus_version", "manifest_revision")
    ):
        add("version_consistency", "hard_error")
    groups = Counter(p.provenance_group for p in papers)
    if groups != {"original_29": 29, "backfill_21": 21}:
        add("provenance_counts", "hard_error")
    preprints = [p for p in papers if p.publication_state == "preprint"]
    if len(preprints) != 2 or any(p.provenance_group != "original_29" for p in preprints):
        add("preprint_count", "hard_error")
    for paper in papers:
        if paper.provenance_group == "backfill_21" and paper.publication_state == "preprint":
            add("backfill_preprint", "hard_error", paper)
        if paper.publication_state == "preprint":
            context = (paper.version_label or "") + " " + (paper.bibliographic_note or "")
            if (
                "biorxiv" not in context.casefold()
                or "pmc" not in context.casefold()
                or "publication_state_version_review" not in paper.pair.pair_verification_basis
            ):
                add("hosted_preprint_version", "hard_error", paper)
    artifacts = [source.artifact_id for p in papers for source in (p.jats, p.pdf)]
    if len(artifacts) != 100:
        add("source_count", "hard_error")
    if len(set(artifacts)) != len(artifacts):
        add("artifact_id_unique", "hard_error")
    for kind in ("doi", "pmid", "pmcid"):
        seen: dict[str, str] = {}
        for paper in papers:
            raw = getattr(paper, kind)
            if raw is None:
                continue
            normalized = _normalized_identifier(raw, kind)
            if normalized in seen and seen[normalized] != paper.paper_id:
                add(kind + "_collision", "hard_error", paper)
            seen[normalized] = paper.paper_id
    titles: dict[str, str] = {}
    for paper in papers:
        normalized = _normalized_title(paper.title)
        if normalized in titles and titles[normalized] != paper.paper_id:
            add("strong_title_collision", "hard_error", paper)
        titles[normalized] = paper.paper_id
        sources: tuple[tuple[Representation, SourceRecord], ...] = (
            ("JATS", paper.jats),
            ("PDF", paper.pdf),
        )
        for representation, source in sources:
            if source.analysis_access != "available":
                add("analysis_access", "hard_error", paper, representation, source.artifact_id)
            if source.acquired_at_status in {"unknown", "unavailable", "not_checked"}:
                add(
                    "acquisition_time_unavailable",
                    "review_warning",
                    paper,
                    representation,
                    source.artifact_id,
                )
            if source.license_status != "known":
                add("license_unknown", "review_warning", paper, representation, source.artifact_id)
            if source.redistribution_status == "unknown":
                add(
                    "redistribution_unknown",
                    "review_warning",
                    paper,
                    representation,
                    source.artifact_id,
                )
        if paper.pair.pair_verified_at_status in {"unknown", "unavailable", "not_checked"}:
            add("verification_time_unavailable", "review_warning", paper)
    return ValidationReport(schema_version="1", findings=findings)


def verify_source_bytes(paper: PaperRecord, kind: Literal["jats", "pdf"], path: Path) -> None:
    source = paper.jats if kind == "jats" else paper.pdf
    if path.is_symlink() or not path.is_file():
        raise ValueError("source must be a regular file")
    with path.open("rb") as stream:
        prefix = stream.read(16)
    if kind == "jats" and not prefix.lstrip().startswith((b"<", b"<?xml")):
        raise ValueError("JATS media type does not match bytes")
    if kind == "pdf" and not prefix.startswith(b"%PDF-"):
        raise ValueError("PDF media type does not match bytes")
    verify_file_sha256(path, source.sha256)
    if kind == "jats":
        try:
            ElementTree.parse(path)
        except ElementTree.ParseError as error:
            raise ValueError("JATS source is not well-formed XML") from error
