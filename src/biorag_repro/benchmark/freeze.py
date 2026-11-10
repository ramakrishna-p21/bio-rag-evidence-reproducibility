"""Final benchmark publication checks, called by the release process after the code commit."""

import subprocess
from pathlib import Path

from biorag_repro.benchmark.audit import AuditReport
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.benchmark.review import DecisionEvent, FlagDisposition
from biorag_repro.benchmark.seeds import BenchmarkSeedRegistry
from biorag_repro.benchmark.split import PaperMeta, PaperSplit, RelatednessRecord, make_split
from biorag_repro.benchmark.storage import read_jsonl
from biorag_repro.benchmark.validation import ValidationReport, validate_items
from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.corpus.manifest import (
    ReleaseRecord,
    corpus_content_hash,
    manifest_semantic_hash,
    parse_csv,
    validate_corpus,
    verify_source_bytes,
)


def _path(root: Path, ref: ArtifactRef) -> Path:
    path = (root / ref.logical_ref).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("release artifact reference leaves data root")
    return path


def verify_implementation(repo: Path, expected_sha: str) -> None:
    """Require the exact reviewed commit and a clean relevant tracked tree."""
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != expected_sha:
        raise ValueError("implementation HEAD differs from release record")
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if status.strip():
        raise ValueError("repository contains uncommitted relevant changes")


def verify_release_inputs(record: BenchmarkRelease, data_root: Path) -> None:
    """Reopen every proposed final artifact and rerun source-grounded validation."""
    corpus_release = read_project_json(
        _path(data_root, record.corpus_release), record.corpus_release.sha256, ReleaseRecord
    )
    if (
        corpus_release.corpus_version != record.corpus_version
        or corpus_release.corpus_content_sha256 != record.corpus_content_sha256
    ):
        raise ValueError("benchmark release differs from frozen corpus release")
    manifest_path = _path(data_root, corpus_release.manifest_artifact)
    verify_file_sha256(manifest_path, corpus_release.csv_sha256)
    manifest = parse_csv(manifest_path.read_bytes())
    if (
        corpus_content_hash(manifest) != corpus_release.corpus_content_sha256
        or manifest_semantic_hash(manifest) != corpus_release.manifest_semantic_sha256
        or any(f.severity == "hard_error" for f in validate_corpus(manifest).findings)
    ):
        raise ValueError("frozen corpus manifest validation failed")
    for paper in manifest.papers:
        verify_source_bytes(paper, "jats", data_root / "corpus/raw" / paper.jats.logical_ref)
        verify_source_bytes(paper, "pdf", data_root / "corpus/raw" / paper.pdf.logical_ref)
    split = read_project_json(_path(data_root, record.split), record.split.sha256, PaperSplit)
    relatedness = read_project_json(
        _path(data_root, record.relatedness), record.relatedness.sha256, RelatednessRecord
    )
    seeds = read_project_json(
        _path(data_root, record.seed_registry), record.seed_registry.sha256, BenchmarkSeedRegistry
    )
    items = read_jsonl(_path(data_root, record.items), record.items.sha256, BenchmarkItem)
    decisions = read_jsonl(
        _path(data_root, record.decisions), record.decisions.sha256, DecisionEvent
    )
    dispositions = read_jsonl(
        _path(data_root, record.dispositions), record.dispositions.sha256, FlagDisposition
    )
    pre_audit_items = read_jsonl(
        _path(data_root, record.pre_audit_population),
        record.pre_audit_population.sha256,
        BenchmarkItem,
    )
    audit = read_project_json(_path(data_root, record.audit), record.audit.sha256, AuditReport)
    saved_validation = read_project_json(
        _path(data_root, record.validation_report),
        record.validation_report.sha256,
        ValidationReport,
    )
    if split.corpus_content_sha256 != record.corpus_content_sha256:
        raise ValueError("paper split differs from frozen corpus")
    expected_papers = [
        PaperMeta(paper_id=p.paper_id, primary_domain=p.primary_domain)
        for p in sorted(manifest.papers, key=lambda item: item.paper_id)
    ]
    expected_preprints = {p.paper_id for p in manifest.papers if p.publication_state == "preprint"}
    if split.papers != expected_papers or set(split.preprint_assignments) != expected_preprints:
        raise ValueError("paper split metadata differs from frozen manifest")
    if relatedness.corpus_content_sha256 != record.corpus_content_sha256:
        raise ValueError("relatedness screen differs from frozen corpus")
    replayed_split = make_split(
        split.papers,
        relatedness.groups,
        seed=split.seed,
        corpus_version=split.corpus_version,
        corpus_content_sha256=split.corpus_content_sha256,
        preprint_ids=tuple(sorted(split.preprint_assignments)),
    )
    if replayed_split != split:
        raise ValueError("paper split does not replay from the relatedness record and seed")
    named_seeds = {entry.name: entry.seed for entry in seeds.entries}
    if (
        named_seeds.get("paper_split") != split.seed
        or named_seeds.get("benchmark_audit") != audit.draw.seed
    ):
        raise ValueError("split or audit seed differs from prospective registry")
    if len(items) != record.item_count or saved_validation.hard_errors:
        raise ValueError("item count or saved validation does not pass")
    if [item.item_id for item in pre_audit_items] != audit.proposed_item_ids:
        raise ValueError("pre-audit population differs from recorded audit population")
    if (
        saved_validation.item_count != record.item_count
        or saved_validation.items_sha256 != record.items.sha256
        or saved_validation.split_sha256 != record.split.sha256
        or saved_validation.corpus_content_sha256 != record.corpus_content_sha256
    ):
        raise ValueError("saved validation does not bind final benchmark inputs")
    accepted = {item.item_id for item in items if item.candidate_status == "accepted"}
    if len(accepted) != len(items) or set(audit.draw.selected_ids) != {
        entry.item_id for entry in audit.audited_items
    }:
        raise ValueError("release contains unaccepted or unaudited selected items")
    if not accepted <= set(audit.proposed_item_ids):
        raise ValueError("final items were not in the pre-audit proposed population")
    decision_ids = {event.event_id for event in decisions}
    audit_event_ids = {entry.decision_event_id for entry in audit.audited_items}
    repair_event_ids = {
        entry.repair_event_id for entry in audit.audited_items if entry.repair_event_id
    }
    expanded_event_ids = {
        event_id for review in audit.expanded_reviews for event_id in review.decision_event_ids
    }
    if not (audit_event_ids | repair_event_ids | expanded_event_ids) <= decision_ids:
        raise ValueError("audit or expanded review cites a missing decision event")
    report = validate_items(
        items,
        paper_ids={paper.paper_id for paper in manifest.papers},
        corpus_version=record.corpus_version,
        accepted_only=True,
        manifest=manifest,
        data_root=data_root,
        split=split,
        events=decisions,
        dispositions=dispositions,
    )
    if report.hard_errors or report.review_warnings:
        raise ValueError("final benchmark validation has open findings")


def publish_benchmark_release(
    record: BenchmarkRelease, target: Path, data_root: Path, repo: Path
) -> str:
    """Publish no-clobber final record only after committed-code and artifact verification."""
    verify_implementation(repo, record.implementation_git_sha)
    verify_release_inputs(record, data_root)
    content = serialize_project_json(record, BenchmarkRelease)
    digest = sha256_bytes(content)
    publish_bytes(target, content, digest)
    read_project_json(target, digest, BenchmarkRelease)
    return digest
