"""Evaluate an approved, sealed development batch after human Stage A review.

This entry point never runs a generator. Source selection and generation are reverified
before any analysis publication. Missing reviewer judgments are reported, never invented.
"""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.benchmark.storage import read_jsonl
from biorag_repro.core.hashing import sha256_bytes, sha256_file, verify_file_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import (
    publish_bytes,
    read_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.documents.ingestion import Phase4Validation
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation import analysis_handoff, stage_b, stage_c
from biorag_repro.evaluation.analysis_models import (
    FROZEN_MATRIX,
    GeneratorInput,
    PlannedRoster,
    RosterArm,
)
from biorag_repro.evaluation.analysis_storage import publish_table, read_table
from biorag_repro.evaluation.analysis_table import (
    Evidence,
    Observation,
    build_table,
    roster_items_from_development_view,
    serialize_csv,
)
from biorag_repro.evaluation.seeds import APPROVED_REGISTRY_SHA256, read_analysis_seed
from biorag_repro.evaluation.stage_a_storage import read_judgment
from biorag_repro.evaluation.stage_b_storage import (
    publish_candidate,
    publish_judgment,
    read_candidate,
)
from biorag_repro.evaluation.stage_b_storage import read_judgment as read_b_judgment
from biorag_repro.evaluation.stage_c_storage import publish_score, read_score
from biorag_repro.generation.environment import GitState
from biorag_repro.retrieval.models import (
    Bm25Index,
    Chunk,
    ChunkSet,
    ContextRecord,
    DenseIndex,
    RankedRetrieval,
)
from biorag_repro.retrieval.storage import record_ref

if __package__:
    from . import run_generation as generation
else:
    import run_generation as generation


def evaluate(args: argparse.Namespace, *, publish: bool = True) -> dict:
    if (
        args.analysis_root.resolve() != args.published_context_root.resolve()
        or args.review_root.resolve() != args.analysis_root.resolve()
    ):
        raise generation.Refused(
            "evaluation, published contexts and human judgments require one common artifact root"
        )
    args.out.resolve().relative_to(args.analysis_root.resolve())
    results = generation.verify_official_batch(args)
    selection = json.loads(args.selection.read_bytes())
    view, items = read_development_view(Path(selection["view_dir"]), selection["view_sha256"])
    items = {i.item_id: i for i in items}
    package, plan, _, _ = generation.load_inputs(args)
    records = {(r["item_id"], r["matrix_arm"], r["condition"]): r for r in package["records"]}
    reviewers = json.loads(args.reviewer_inventory.read_bytes())
    if reviewers.get("status") != "human_judgments_submitted":
        raise generation.Refused("Stage A inventory is not a submitted human review")
    observations, pending, b_cache, indexes = [], [], {}, {}
    if publish:
        args.analysis_root.mkdir(parents=True, exist_ok=True)

    def retain(ref, data):
        path = args.analysis_root / ref.logical_ref
        if publish:
            path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            verify_file_sha256(path, ref.sha256)
        elif publish:
            publish_bytes(path, data, ref.sha256)
        else:
            raise generation.Refused("a required published evaluation dependency is missing")

    for result in results:
        item = items[result.item_id]
        score = stage_c.score(item, result)
        for role, ref in stage_c.dependencies(score):
            if role == "benchmark_item":
                retain(ref, serialize_project_json(item, type(item)))
            elif role == "generation_result":
                retain(ref, serialize_project_json(result, type(result)))
            elif role == "generator_output":
                retain(ref, (args.out / ref.logical_ref).read_bytes())
        score_ref = stage_c.score_ref(score)
        if (args.analysis_root / score_ref.logical_ref).exists():
            if read_score(score_ref, args.analysis_root, item, result) != score:
                raise generation.Refused("published Stage C differs from frozen scoring")
        elif publish:
            publish_score(score, item, result, args.analysis_root)
        else:
            raise generation.Refused("published Stage C score is missing")
        record = records[result.item_id, result.matrix_arm, result.representation_or_condition]
        evidence = None
        link = None
        if result.matrix_arm != "closed_book":
            key = f"{result.item_id}/{result.representation_or_condition}"
            if key not in reviewers.get("judgments", {}):
                pending.append({"item_representation": key, "stage": "A"})
                continue
            entry = selection["canonical"][item.paper_id + "/" + result.representation_or_condition]
            root = Path(entry["root"])
            validation = read_project_json(
                root / entry["logical_ref"], entry["sha256"], Phase4Validation
            )
            canonical = validation.canonical_artifact
            document = read_project_json(
                root / canonical.logical_ref, canonical.sha256, CanonicalDocument
            )
            a = read_judgment(
                ArtifactRef(**reviewers["judgments"][key]),
                args.review_root,
                item,
                document,
                canonical,
            )
            cref = ArtifactRef(**record["context_record"])
            context = read_project_json(
                args.published_context_root / cref.logical_ref, cref.sha256, ContextRecord
            )
            ranked = read_project_json(
                args.published_context_root / context.retrieval.logical_ref,
                context.retrieval.sha256,
                RankedRetrieval,
            )
            if ranked.index.artifact_id not in indexes:
                kind = DenseIndex if ranked.retriever == "bge" else Bm25Index
                index = read_project_json(
                    args.published_context_root / ranked.index.logical_ref,
                    ranked.index.sha256,
                    kind,
                )
                chunks, sets = {}, {}
                for ref in index.chunk_sets:
                    chunk_set = read_project_json(
                        args.published_context_root / ref.logical_ref, ref.sha256, ChunkSet
                    )
                    loaded = read_jsonl(
                        args.published_context_root / chunk_set.chunks.logical_ref,
                        chunk_set.chunks.sha256,
                        Chunk,
                    )
                    chunks.update({record_ref(c).artifact_id: c for c in loaded})
                    sets[chunk_set.input.paper_id] = chunk_set
                indexes[ranked.index.artifact_id] = chunks, sets
            chunks, sets = indexes[ranked.index.artifact_id]
            b_inputs = stage_b.Inputs(
                item, document, a, context, ranked, sets[item.paper_id], chunks, canonical, cref
            )
            b_key = (result.item_id, result.matrix_arm, result.representation_or_condition)
            if b_key in b_cache and b_cache[b_key] is None:
                continue
            if b_key not in b_cache:
                candidate = stage_b.evaluate(b_inputs)
                candidate_ref = stage_b.stage_b_ref(candidate)
                if (args.analysis_root / candidate_ref.logical_ref).exists():
                    if read_candidate(candidate_ref, args.analysis_root, b_inputs) != candidate:
                        raise generation.Refused("published Stage B candidate differs")
                elif publish:
                    publish_candidate(candidate, b_inputs, args.analysis_root)
                else:
                    raise generation.Refused("published Stage B candidate is missing")
                if candidate.determination == "needs_review":
                    review_key = (
                        f"{result.item_id}/{result.matrix_arm}/{result.representation_or_condition}"
                    )
                    review_ref = reviewers.get("stage_b_judgments", {}).get(review_key)
                    if review_ref is None:
                        b_cache[b_key] = None
                        pending.append(
                            {
                                "item_representation": key,
                                "arm": result.matrix_arm,
                                "stage": "B",
                                "candidate": candidate_ref.model_dump(),
                            }
                        )
                        continue
                    b = read_b_judgment(ArtifactRef(**review_ref), args.review_root, b_inputs)
                else:
                    b = stage_b.deterministic_judgment(candidate, inputs=b_inputs)
                    b_ref = stage_b.stage_b_ref(b)
                    if (args.analysis_root / b_ref.logical_ref).exists():
                        if read_b_judgment(b_ref, args.analysis_root, b_inputs) != b:
                            raise generation.Refused("published Stage B judgment differs")
                    elif publish:
                        publish_judgment(b, b_inputs, args.analysis_root)
                    else:
                        raise generation.Refused("published Stage B judgment is missing")
                b_cache[b_key] = b
            evidence = Evidence(document, canonical, a, b_cache[b_key], b_inputs)
            if result.status == "completed":
                stored = json.loads(
                    (args.out / "results" / f"{result.observation_key}.json").read_bytes()
                )
                input_ref = ArtifactRef(**stored["generator_input"])
                retain(input_ref, (args.out / input_ref.logical_ref).read_bytes())
                link = GeneratorInput(
                    observation_key=result.observation_key,
                    attempt_id=result.attempt_id,
                    context=cref,
                    exact_context_sha256=sha256_bytes(context.exact_context.encode()),
                    input=ArtifactRef(**stored["generator_input"]),
                    provenance="recorded",
                )
        observations.append(
            Observation(
                item,
                "development",
                result.matrix_arm,
                result.representation_or_condition,
                result.repeat_index,
                result.resolved_config_sha256,
                record["retrieval_config_sha256"],
                result,
                score,
                evidence,
                link,
            )
        )
    if pending:
        # Do not publish an incomplete table as the completed development matrix.
        return {
            "status": "human_review_pending",
            "stage_c_scores": len(results),
            "pending": pending,
        }
    roster = PlannedRoster(
        scope="complete_matrix",
        phase="development",
        benchmark_version=view.benchmark_version,
        corpus_version="corpus_v1",
        declared_by="Rama Krishna Pudota",
        basis="Frozen approved 86-item development view and five-arm matrix",
        arms=[
            RosterArm(matrix_arm=a, conditions=list(c), repeats=list(r))
            for a, (c, r) in FROZEN_MATRIX.items()
        ],
        items=roster_items_from_development_view(view, list(items.values())),
    )
    table = build_table(observations, roster)
    if publish:
        _, _, table_build = publish_table(table, args.analysis_root)
    else:
        handoff_build = ArtifactRef(**args.handoff_build)
        manifest = read_manifest(
            args.analysis_root / handoff_build.logical_ref, handoff_build.sha256
        )
        href = manifest.outputs[0].artifact
        actual_handoff = read_project_json(
            args.analysis_root / href.logical_ref, href.sha256, analysis_handoff.AnalysisHandoff
        )
        table_build = actual_handoff.table_build
        rows = read_table(table_build, args.analysis_root)
        if serialize_csv(rows) != serialize_csv(table.rows):
            raise generation.Refused("analysis rows differ from verified Stage A/B/C evaluation")
        table_manifest = read_manifest(
            args.analysis_root / table_build.logical_ref, table_build.sha256
        )
        if table_manifest.inputs != table.inputs:
            raise generation.Refused("analysis table has different scientific dependencies")
        roster_ref = next(
            o.artifact for o in table_manifest.outputs if o.artifact_kind == "planned_roster"
        )
        if (
            read_project_json(
                args.analysis_root / roster_ref.logical_ref, roster_ref.sha256, PlannedRoster
            )
            != roster
        ):
            raise generation.Refused("analysis roster differs from the frozen development roster")
    batch = json.loads((args.out / "batch.json").read_bytes())
    code = GitState(
        status="ok", commit=batch["identity"]["git"]["head"], dirty=False, dirty_paths=0
    )
    template, _ = analysis_handoff._describe(table_build, args.analysis_root, code)
    analysis_seed = read_analysis_seed(Path(__file__).resolve().parents[1])
    sources = []
    for arm in FROZEN_MATRIX:
        config = next(p.config for p in plan if p.record["matrix_arm"] == arm)
        sources.append(
            analysis_handoff.SourceRun(
                run_id=batch["run_id"],
                completion=ArtifactRef(
                    artifact_id="batch_anchor_" + args.anchor_sha256,
                    logical_ref=args.anchor.resolve()
                    .relative_to(args.analysis_root.resolve())
                    .as_posix(),
                    sha256=args.anchor_sha256,
                ),
                run_record_sha256=sha256_file(args.out / "batch.json"),
                resolved_config_sha256=config.configuration_sha256,
                raw_config_sha256=config.raw_toml_sha256,
                roster_sha256=batch["identity"]["planned_keys_sha256"],
                matrix_arm=arm,
                backend_kind="official",
                observations=sum(r.matrix_arm == arm for r in results),
            )
        )
    handoff = analysis_handoff.AnalysisHandoff.model_validate(
        template.model_dump()
        | {
            "input_origin": "verified_official_run",
            "source_runs": [s.model_dump() for s in sources],
            "scoring_reference": template.scoring_reference
            | {"primary_bootstrap_registry_sha256": APPROVED_REGISTRY_SHA256},
            "awaiting_approval": [],
        }
    )
    # Publication is reachable only after fresh batch verification and scientific table construction.
    if publish:
        href, hbuild = analysis_handoff._store(handoff, args.analysis_root)
    else:
        if actual_handoff != handoff or manifest.inputs != analysis_handoff._inputs(handoff):
            raise generation.Refused(
                "official handoff differs from freshly verified source evidence"
            )
        hbuild = handoff_build
    return {
        "status": "verified_official_development_handoff",
        "handoff": href.model_dump(),
        "build": hbuild.model_dump(),
        "observations": len(results),
        "csv_sha256": handoff.csv.sha256,
        "source_verification": "fresh_python_verification",
        "scientific_bootstrap_seed": {
            "registry_sha256": APPROVED_REGISTRY_SHA256,
            "record": analysis_seed.model_dump(mode="json"),
        },
    }


def verify_request(path: Path) -> dict:
    request = json.loads(path.read_bytes())
    names = (
        "contexts",
        "configs",
        "prompt",
        "seeds",
        "tokenizer_root",
        "out",
        "published_context_root",
        "environment_lock",
        "selection",
        "anchor",
        "reviewer_inventory",
        "review_root",
        "analysis_root",
    )
    args = argparse.Namespace(
        **{name: Path(request[name]) for name in names},
        contexts_sha256=request["contexts_sha256"],
        anchor_sha256=request["anchor_sha256"],
        handoff_build=request["handoff_build"],
        phase="development",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        result = evaluate(args, publish=False)
    if result["status"] != "verified_official_development_handoff":
        raise generation.Refused("human review is still pending")
    return result


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-request":
        print(json.dumps(verify_request(Path(sys.argv[2])), sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "contexts",
        "configs",
        "prompt",
        "seeds",
        "tokenizer-root",
        "out",
        "published-context-root",
        "environment-lock",
        "selection",
        "anchor",
        "reviewer-inventory",
        "review-root",
        "analysis-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--contexts-sha256", required=True)
    parser.add_argument("--anchor-sha256", required=True)
    args = parser.parse_args()
    args.phase = "development"
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()
