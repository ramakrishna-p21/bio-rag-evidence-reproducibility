"""Record the approved retrospective Stage A package; publish only after the full gate passes."""

import argparse
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes
from biorag_repro.evaluation import stage_a_retrospective as retrospective
from biorag_repro.evaluation.stage_a import stage_a_ref
from biorag_repro.evaluation.stage_a_storage import publish_candidate, read_candidate
from biorag_repro.generation.environment import utc_now
from biorag_repro.readiness.files import loads_strict
from scripts import stage_a_review as prospective

EXPECTED_COUNTS = {
    "preserved": 150,
    "altered_but_sufficient": 3,
    "scientifically_corrupted": 1,
    "unresolved": 2,
}


def source_ref(data: bytes) -> ArtifactRef:
    sha = sha256_bytes(data)
    return ArtifactRef(
        artifact_id="source_" + sha,
        logical_ref=f"evaluation/stage_a/retrospective_sources/{sha}.json",
        sha256=sha,
    )


def transcribe(args: argparse.Namespace) -> dict:
    # The independent prospective workflow and its files are read, never edited.
    args.out.mkdir(parents=True, exist_ok=False)
    if args.publish and args.inventory.exists():
        raise FileExistsError(
            "retrospective inventory already exists; retain its original recording"
        )
    manifest, coordinates = prospective.load_provenance(args.queue)
    inputs = {
        "approved_decisions": args.decisions.read_bytes(),
        "session_manifest": (args.queue / "provenance/session_manifest.json").read_bytes(),
        "coordinate_map": (args.queue / "provenance/unmasked_evidence.json").read_bytes(),
        "authorization": args.authorization.read_bytes(),
    }
    refs = {name: source_ref(data) for name, data in inputs.items()}
    if (
        refs["approved_decisions"].sha256,
        refs["session_manifest"].sha256,
        refs["authorization"].sha256,
    ) != (
        retrospective.APPROVED_DECISIONS_SHA256,
        retrospective.SESSION_SHA256,
        retrospective.AUTHORIZATION_SHA256,
    ):
        raise ValueError("transcription inputs differ from the approved exact package")
    rows = loads_strict(inputs["approved_decisions"])["records"]
    handles = [f"R{i:03d}" for i in range(1, 157)]
    if [r["handle"] for r in rows] != handles or list(coordinates) != handles:
        raise ValueError("transcription requires exactly the approved 156 handles in order")
    counts = Counter(r["researcher_concurred_label"] for r in rows)
    if counts != EXPECTED_COUNTS:
        raise ValueError("approved classifications differ from the frozen transcription")
    for name, data in inputs.items():
        target = args.out / refs[name].logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_bytes(target, data, refs[name].sha256)
    resolutions = (
        loads_strict(args.component_resolutions.read_bytes())
        if getattr(args, "component_resolutions", None) is not None
        else {}
    )
    if set(resolutions) - set(handles):
        raise ValueError("focused resolutions name handles outside the approved 156")
    extra_sources = {}
    pending, drafts, analyses = [], {}, {}
    for row in rows:
        handle, label = row["handle"], row["researcher_concurred_label"]
        analysis = prospective.load_analysis(handle, manifest["seed"], coordinates[handle])
        item, document = analysis.source.item, analysis.source.document
        data = {
            "benchmark_version": item.benchmark_version,
            "corpus_version": item.corpus_version,
            "item_id": item.item_id,
            "item_sha256": coordinates[handle]["item_sha256"],
            "paper_id": item.paper_id,
            "representation": document.representation,
            "canonical": analysis.source.canonical,
            "handle": handle,
            "label": label,
            "candidate": stage_a_ref(analysis.candidate),
            "sets": analysis.candidate.sets
            if label in {"preserved", "altered_but_sufficient"}
            else [],
            "rationale": row["researcher_concurred_rationale"],
            "alteration": row["researcher_concurred_rationale"]
            if label in {"altered_but_sufficient", "scientifically_corrupted"}
            else None,
            "recorded_at_utc": utc_now(),
            **refs,
        }
        if handle in resolutions:
            raw = Path(resolutions[handle]).read_bytes()
            resolution = loads_strict(raw)
            ref = source_ref(raw)
            target = args.out / ref.logical_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(target, raw, ref.sha256, reuse_identical=True)
            extra_sources[handle] = (ref, raw)
            data.update(component_resolution=ref, sets=resolution["sets"])
        try:
            value = retrospective.RetrospectiveJudgment(**data)
            retrospective.validate_retrospective(
                value, item, document, analysis.source.canonical, args.out
            )
        except ValueError as error:
            reason = (
                "; ".join(e["msg"] for e in error.errors(include_input=False))
                if isinstance(error, ValidationError)
                else str(error)
            )
            pending.append(
                {
                    "handle": handle,
                    "approved_label": label,
                    "blocker": reason,
                    "component_mapping_gaps": [
                        s.model_dump(mode="json")
                        for s in analysis.candidate.suggestions
                        if s.kind
                        in {"component_not_found", "split_across_contexts", "association_mismatch"}
                    ],
                    "approved_evidence_handles": row["supporting_evidence_handles"],
                }
            )
        else:
            drafts[handle] = value
            analyses[handle] = analysis
        if len(drafts) + len(pending) in (40, 80, 120, 156):
            print(f"validated {len(drafts) + len(pending)}/156; pending={len(pending)}", flush=True)
    report = {
        "protocol": retrospective.PROTOCOL,
        "classification_counts": dict(counts),
        "source_classifications_preserved": 156,
        "mechanically_valid_records": len(drafts),
        "publication": "blocked" if pending else "eligible_not_published",
        "pending": pending,
        "new_scientific_reviews": 0,
        "historical_metadata": "unavailable; not reconstructed",
        "partial_official_inventory_created": False,
    }
    # Drafts are not publications; they cannot be read as a submitted 156-record inventory.
    (args.out / "drafts").mkdir()
    for handle, value in drafts.items():
        (args.out / "drafts" / f"{handle}.json").write_bytes(
            prospective.dumps(value.model_dump(mode="json"))
        )
    if args.publish and not pending:
        for name, raw in inputs.items():
            target = args.destination / refs[name].logical_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(target, raw, refs[name].sha256, reuse_identical=True)
        for ref, raw in extra_sources.values():
            target = args.destination / ref.logical_ref
            target.parent.mkdir(parents=True, exist_ok=True)
            publish_bytes(target, raw, ref.sha256, reuse_identical=True)
        published = {}
        for handle, value in drafts.items():
            a = analyses[handle]
            candidate_ref = stage_a_ref(a.candidate)
            if (args.destination / candidate_ref.logical_ref).exists():
                read_candidate(
                    candidate_ref,
                    args.destination,
                    a.source.item,
                    a.source.document,
                    a.source.canonical,
                )
            else:
                publish_candidate(
                    a.candidate,
                    a.source.item,
                    a.source.document,
                    a.source.canonical,
                    args.destination,
                )
            ref, _ = retrospective.publish_judgment(
                value, a.source.item, a.source.document, a.source.canonical, args.destination
            )
            retrospective.read_judgment(
                ref, args.destination, a.source.item, a.source.document, a.source.canonical
            )
            published[f"{value.item_id}/{value.representation}"] = ref.model_dump(mode="json")
        inventory = {
            "status": "human_judgments_submitted",
            "review_mode": retrospective.PROTOCOL,
            "historical_metadata": "unavailable",
            "judgments": published,
            "stage_b_judgments": {},
            "reviewer": "Rama Krishna Pudota",
        }
        raw = prospective.dumps(inventory)
        publish_bytes(args.inventory, raw, sha256_bytes(raw))
        report["publication"] = "published"
    (args.out / "preflight.json").write_bytes(prospective.dumps(report))
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("queue", "decisions", "authorization", "destination", "inventory", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--publish", action="store_true")
    p.add_argument(
        "--component-resolutions",
        type=Path,
        help="handle-to-file map of separately approved focused component records",
    )
    args = p.parse_args()
    report = transcribe(args)
    print(f"{report['publication']}: {report['mechanically_valid_records']}/156 records pass")
    return 2 if report["pending"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
