"""Source-only benchmark construction commands."""

import argparse
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from biorag_repro.benchmark.audit import sample_audit
from biorag_repro.benchmark.discovery import CandidateLead, discover_jats
from biorag_repro.benchmark.flags import flag_leads
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.benchmark.pdf import locate_pdf
from biorag_repro.benchmark.review import (
    DecisionEvent,
    FlagDisposition,
    review_queue,
    worksheet_rows,
)
from biorag_repro.benchmark.split import PaperMeta, PaperSplit, RelatednessRecord, make_split
from biorag_repro.benchmark.storage import encode_jsonl, read_jsonl
from biorag_repro.benchmark.validation import validate_items
from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.corpus.manifest import (
    CorpusManifest,
    ReleaseRecord,
    parse_csv,
    verify_source_bytes,
)

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_corpus(
    root: Path, release_path: Path, release_sha256: str
) -> tuple[ReleaseRecord, CorpusManifest]:
    release = read_project_json(release_path, release_sha256, ReleaseRecord)
    manifest_path = root / release.manifest_artifact.logical_ref
    verify_file_sha256(manifest_path, release.csv_sha256)
    manifest = parse_csv(manifest_path.read_bytes())
    return release, manifest


def publish_json(target: Path, model: ModelT) -> None:
    content = serialize_project_json(model, type(model))
    publish_bytes(target, content, sha256_bytes(content))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Construct source-grounded benchmark working artifacts"
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--corpus-release", type=Path, required=True)
    parser.add_argument("--corpus-release-sha256", required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    split = commands.add_parser("split")
    split.add_argument("--relatedness", type=Path, required=True)
    split.add_argument("--relatedness-sha256", required=True)
    split.add_argument("--output", type=Path, required=True)
    discover = commands.add_parser("discover")
    discover.add_argument("--paper-id", required=True)
    discover.add_argument("--split", type=Path, required=True)
    discover.add_argument("--split-sha256", required=True)
    discover.add_argument("--output-dir", type=Path, required=True)
    packet = commands.add_parser("review-pack")
    packet.add_argument("--paper-id", required=True)
    packet.add_argument("--leads", type=Path, required=True)
    packet.add_argument("--leads-sha256", required=True)
    packet.add_argument("--lead-id", action="append", required=True)
    packet.add_argument("--term", action="append")
    packet.add_argument("--output", type=Path, required=True)
    search = commands.add_parser("pdf-search")
    search.add_argument("--paper-id", required=True)
    search.add_argument("--term", action="append", required=True)
    search.add_argument("--output", type=Path, required=True)
    audit = commands.add_parser("audit-draw")
    audit.add_argument("--items", type=Path, required=True)
    audit.add_argument("--items-sha256", required=True)
    audit.add_argument("--output", type=Path, required=True)
    validation = commands.add_parser("validate")
    validation.add_argument("--items", type=Path, required=True)
    validation.add_argument("--items-sha256", required=True)
    validation.add_argument("--split", type=Path, required=True)
    validation.add_argument("--split-sha256", required=True)
    validation.add_argument("--output", type=Path, required=True)
    validation.add_argument("--accepted-only", action="store_true")
    validation.add_argument("--decisions", type=Path)
    validation.add_argument("--decisions-sha256")
    validation.add_argument("--dispositions", type=Path)
    validation.add_argument("--dispositions-sha256")
    args = parser.parse_args()
    release, manifest = load_corpus(args.data_root, args.corpus_release, args.corpus_release_sha256)
    by_id = {paper.paper_id: paper for paper in manifest.papers}
    if args.command == "split":
        related = read_project_json(args.relatedness, args.relatedness_sha256, RelatednessRecord)
        if related.corpus_content_sha256 != release.corpus_content_sha256:
            raise ValueError("relatedness record differs from frozen corpus")
        result = make_split(
            [
                PaperMeta(paper_id=p.paper_id, primary_domain=p.primary_domain)
                for p in manifest.papers
            ],
            related.groups,
            seed=3102026,
            corpus_version=release.corpus_version,
            corpus_content_sha256=release.corpus_content_sha256,
            preprint_ids=tuple(
                p.paper_id for p in manifest.papers if p.publication_state == "preprint"
            ),
        )
        publish_json(args.output, result)
        print(f"split_sha256={sha256_bytes(serialize_project_json(result, type(result)))}")
        print(f"development={','.join(result.development_ids)}")
        print(f"final_test={','.join(result.final_test_ids)}")
        print(f"preprints={result.preprint_assignments}")
    elif args.command == "discover":
        split_record = read_project_json(args.split, args.split_sha256, PaperSplit)
        if args.paper_id not in split_record.development_ids + split_record.final_test_ids:
            raise ValueError("discovery paper must belong to frozen split")
        paper = by_id[args.paper_id]
        source_path = args.data_root / "corpus/raw" / paper.jats.logical_ref
        verify_source_bytes(paper, "jats", source_path)
        leads: list[CandidateLead] = discover_jats(source_path.read_bytes(), paper.paper_id)
        flags = flag_leads(leads)
        worksheets = worksheet_rows(review_queue(leads), flags)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, records in (("leads", leads), ("worksheet", worksheets), ("flags", flags)):
            content = encode_jsonl(records)
            publish_bytes(
                args.output_dir / f"{args.paper_id}_{name}.jsonl", content, sha256_bytes(content)
            )
        print(
            f"paper={paper.paper_id} leads={len(leads)} worksheet={len(worksheets)} flags={len(flags)}"
        )
    elif args.command == "review-pack":
        paper = by_id[args.paper_id]
        leads = read_jsonl(args.leads, args.leads_sha256, CandidateLead)
        requested = set(args.lead_id)
        selected = [lead for lead in leads if lead.lead_id in requested]
        if len(selected) != len(requested) or any(
            lead.paper_id != paper.paper_id for lead in selected
        ):
            raise ValueError("review lead IDs must resolve once within the requested paper")
        rows = worksheet_rows(selected, flag_leads(leads))
        source_path = args.data_root / "corpus/raw" / paper.pdf.logical_ref
        verify_source_bytes(paper, "pdf", source_path)
        for row in rows:
            row.pdf_hits = [
                hit.model_dump(mode="json")
                for hit in locate_pdf(
                    source_path,
                    paper.pdf.sha256,
                    args.term if args.term else row.pdf_search_terms[:3],
                )
            ]
        content = encode_jsonl(rows)
        publish_bytes(args.output, content, sha256_bytes(content))
        print(f"paper={paper.paper_id} review_rows={len(rows)}")
    elif args.command == "pdf-search":
        paper = by_id[args.paper_id]
        source_path = args.data_root / "corpus/raw" / paper.pdf.logical_ref
        verify_source_bytes(paper, "pdf", source_path)
        hits = locate_pdf(source_path, paper.pdf.sha256, args.term)
        content = encode_jsonl(hits)
        publish_bytes(args.output, content, sha256_bytes(content))
        print(f"paper={paper.paper_id} hits={len(hits)}")
    elif args.command == "audit-draw":
        items = read_jsonl(args.items, args.items_sha256, BenchmarkItem)
        if any(item.candidate_status != "accepted" for item in items):
            raise ValueError("audit population must contain only proposed accepted items")
        draw = sample_audit([item.item_id for item in items], seed=3102027)
        publish_json(args.output, draw)
        print(f"eligible={draw.eligible_count} selected={len(draw.selected_ids)}")
    else:
        items = read_jsonl(args.items, args.items_sha256, BenchmarkItem)
        split_record = read_project_json(args.split, args.split_sha256, PaperSplit)
        if bool(args.decisions) != bool(args.decisions_sha256):
            raise ValueError("decision path and hash must be supplied together")
        if args.accepted_only and not args.decisions:
            raise ValueError("accepted-only validation requires reviewed decisions")
        events = (
            read_jsonl(args.decisions, args.decisions_sha256, DecisionEvent)
            if args.decisions
            else None
        )
        if bool(args.dispositions) != bool(args.dispositions_sha256):
            raise ValueError("disposition path and hash must be supplied together")
        dispositions = (
            read_jsonl(args.dispositions, args.dispositions_sha256, FlagDisposition)
            if args.dispositions
            else None
        )
        report = validate_items(
            items,
            paper_ids=set(by_id),
            corpus_version=release.corpus_version,
            accepted_only=args.accepted_only,
            manifest=manifest,
            data_root=args.data_root,
            split=split_record,
            events=events,
            dispositions=dispositions,
        )
        publish_json(args.output, report)
        print(f"hard_errors={report.hard_errors} review_warnings={report.review_warnings}")
        if report.hard_errors:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
