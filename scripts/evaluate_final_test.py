"""Evaluate the preserved final-test batch: Stage B and Stage C review queues, reviewer-record
ingestion, the analysis table and the official-origin handoff for R.

Nothing runs without the researcher's evaluation authorization
(`approvals/final_test_evaluation.json`), which binds the evaluation protocol, the custody
exception and the exact batch. Every command re-verifies what it reads. The batch is never
regenerated or changed; review records and evaluation outputs are written separately and
write-once. Stage A packets are built by `scripts/stage_a_review.py final-queue`.

All paths come from one request file (JSON):
  authorization, authorization_sha256, data_root, contexts, contexts_sha256, configs, prompt,
  seeds, tokenizer_root, published_context_root, selection, batch, environment_lock,
  analysis_root, stage_a_inventory, and, once they exist, stage_b_inventory, stage_c_review
  and handoff_build.

Commands, in the order of the protocol:
  stage-b-queue  --request R --out DIR --seed N
  stage-b-ingest --request R --out DIR --reviewer NAME
  stage-c-queue  --request R --out DIR --seed N
  evaluate       --request R
  --verify-request R          (fresh verification for R; writes nothing)
"""

import argparse
import contextlib
import io
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from biorag_repro.benchmark.release import read_final_test_items
from biorag_repro.benchmark.storage import read_jsonl
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import (
    publish_bytes,
    read_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.documents.ingestion import Phase4Validation
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation import analysis_handoff, final_test, stage_b, stage_c
from biorag_repro.evaluation.analysis_models import (
    FROZEN_MATRIX,
    GeneratorInput,
    PlannedRoster,
    RosterArm,
)
from biorag_repro.evaluation.analysis_storage import publish_table
from biorag_repro.evaluation.analysis_table import (
    Evidence,
    Observation,
    build_table,
    roster_items_from_final_test,
)
from biorag_repro.evaluation.seeds import APPROVED_REGISTRY_SHA256, read_analysis_seed
from biorag_repro.evaluation.stage_a import final_test_scope
from biorag_repro.evaluation.stage_a_storage import read_judgment as read_a_judgment
from biorag_repro.evaluation.stage_b_storage import (
    publish_candidate,
    publish_judgment,
    read_candidate,
)
from biorag_repro.evaluation.stage_b_storage import read_judgment as read_b_judgment
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.evaluation.stage_c_storage import publish_score, read_score
from biorag_repro.finaltest.drive_publication import DrivePublicationError, verify_drive_batch
from biorag_repro.finaltest.history_v2 import HistoryRefused, _verify_internal_seals
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

REPO = Path(__file__).resolve().parents[1]
RULES = "v2"  # stage_c_rules_v2 (G03 freeze) and Stage B rules v2 (evaluation protocol)
PATHS = (
    "authorization", "data_root", "contexts", "configs", "prompt", "seeds", "tokenizer_root",
    "published_context_root", "selection", "batch", "environment_lock", "analysis_root",
    "stage_a_inventory", "stage_b_inventory", "stage_c_review",
)  # fmt: skip
Refused = generation.Refused


def dumps(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def write_once(path: Path, data: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256_bytes(data)
    if path.exists():
        if sha256_file(path) != digest:
            raise Refused(f"{path.name} already exists with other content")
    else:
        publish_bytes(path, data, digest)
    return digest


# ---- inputs ----------------------------------------------------------------------------------


@dataclass
class Inputs:
    request: dict[str, Any]
    authorization: final_test.EvaluationAuthorization
    authorization_ref: ArtifactRef
    items: dict[str, Any]
    package: dict[str, Any]
    plan: list[Any]
    seeds: Any
    prompt: Any

    def path(self, name: str) -> Path:
        value = self.request.get(name)
        if value is None:
            raise Refused(f"the request names no {name}")
        return Path(value)


def check_source(revision: str) -> None:
    try:
        final_test.check_source(REPO, revision)
    except final_test.EvaluationRefused as error:
        raise Refused(str(error)) from None


def load(request_path: Path) -> Inputs:
    """Authorization first; protected items, contexts and plan only after it verifies."""
    request = json.loads(request_path.read_bytes())
    try:
        authorization, ref = final_test.read_authorization(
            Path(request["authorization"]), request["authorization_sha256"], REPO
        )
    except final_test.EvaluationRefused as error:
        raise Refused(str(error)) from None
    check_source(authorization.evaluation_source_revision)
    args = argparse.Namespace(
        phase="final_test",
        contexts=Path(request["contexts"]),
        contexts_sha256=request["contexts_sha256"],
        configs=Path(request["configs"]),
        prompt=Path(request["prompt"]),
        seeds=Path(request["seeds"]),
        published_context_root=Path(request["published_context_root"]),
    )
    package, plan, seeds, prompt = generation._load_inputs(args)
    generation.check_authorized_final_test_roster(Path(request["data_root"]), package)
    _, items = read_final_test_items(Path(request["data_root"]))
    return Inputs(
        request, authorization, ref, {i.item_id: i for i in items}, package, plan, seeds, prompt
    )


def official_identity(ctx: Inputs, identity: dict, configurations: dict) -> None:
    """Official execution on the approved checkpoint and environment lock. Synthetic batches
    cannot carry this evidence; nothing else in batch verification depends on it."""
    from biorag_repro.generation import config as policy

    lock = sha256_bytes(ctx.path("environment_lock").read_bytes())
    model = configurations["confirmatory"].settings.model
    checkpoint = policy.APPROVED_CHECKPOINTS.get((model.repository, model.revision))
    if (
        identity.get("official_execution") is not True
        or identity.get("environment_lock_sha256") != lock
        or lock not in policy.APPROVED_ENVIRONMENT_LOCKS
        or checkpoint is None
        or identity.get("model_manifest_sha256") != checkpoint
    ):
        raise Refused("batch is not official execution on the approved checkpoint and lock")


def verify_batch(ctx: Inputs) -> list[GenerationResult]:
    """Fresh, fail-closed verification of the preserved batch under the custody exception."""
    from biorag_repro.generation.inputs import InputRenderer

    auth = ctx.authorization
    out = ctx.path("batch")
    if out.name != auth.run_id:
        raise Refused("batch directory is not the authorized run")
    try:
        files = verify_drive_batch(out)
        _verify_internal_seals(files, out)
    except (DrivePublicationError, HistoryRefused) as error:
        raise Refused(f"batch integrity: {error}") from None
    if sha256_file(out / "drive_manifest.json") != auth.drive_manifest_sha256:
        raise Refused("Drive manifest differs from the custody exception")
    batch = json.loads((out / "batch.json").read_bytes())
    identity = batch["identity"]
    configurations = {p.record["matrix_arm"]: p.config for p in ctx.plan}
    expected = {
        "phase": "final_test",
        "contexts_sha256": ctx.request["contexts_sha256"],
        "planned_keys_sha256": sha256_bytes(canonical_json_bytes(sorted(p.key for p in ctx.plan))),
        "generator_configurations": {
            a: c.configuration_sha256 for a, c in sorted(configurations.items())
        },
        "prompt_sha256": generation.prompt_sha256(ctx.prompt),
        "seed_registry_sha256": generation.seed_registry_sha256(ctx.seeds),
    }
    if batch["run_id"] != auth.run_id or any(identity.get(k) != v for k, v in expected.items()):
        changed = sorted(k for k, v in expected.items() if identity.get(k) != v)
        raise Refused(f"batch identity differs from the frozen plan: {changed}")
    official_identity(ctx, identity, configurations)
    if len(ctx.plan) != final_test.PLANNED_OBSERVATIONS:
        raise Refused("plan is not the 4,575-observation matrix")
    renderers = {
        arm: InputRenderer(ctx.path("tokenizer_root"), c, ctx.prompt)
        for arm, c in configurations.items()
    }
    if generation.verify_batch_results(out, ctx.plan, renderers, batch["run_id"]) != len(ctx.plan):
        raise Refused("batch does not hold every planned observation")
    results = []
    for planned in ctx.plan:
        value = json.loads((out / "results" / f"{planned.key}.json").read_bytes())
        result = GenerationResult.model_validate(value["generation_result"])
        if result.phase != "final_test":
            raise Refused("observation belongs to another phase")
        results.append(result)
    return results


# ---- Stage A and the Stage B inputs ---------------------------------------------------------


def stage_a(ctx: Inputs) -> dict[str, tuple[Any, Any, Any]]:
    """Every final-test Stage A judgment, read back through the Stage A validator."""
    inventory = json.loads(ctx.path("stage_a_inventory").read_bytes())
    if inventory.get("status") != "human_judgments_submitted":
        raise Refused("Stage A inventory is not a submitted human review")
    selection = json.loads(ctx.path("selection").read_bytes())
    root = ctx.path("analysis_root")
    expected = {f"{i}/{r}" for i in ctx.items for r in ("JATS", "PDF_DOCLING")}
    if set(inventory["judgments"]) != expected:
        raise Refused("Stage A inventory does not cover the 610 final-test units exactly")
    documents: dict[str, tuple[Any, Any]] = {}
    judged = {}
    for key, ref in sorted(inventory["judgments"].items()):
        item_id, representation = key.split("/")
        item = ctx.items[item_id]
        name = f"{item.paper_id}/{representation}"
        if name not in documents:
            entry = selection["canonical"][name]
            source = Path(entry["root"])
            validation = read_project_json(
                source / entry["logical_ref"], entry["sha256"], Phase4Validation
            )
            canonical = validation.canonical_artifact
            documents[name] = (
                read_project_json(
                    source / canonical.logical_ref, canonical.sha256, CanonicalDocument
                ),
                canonical,
            )
        document, canonical = documents[name]
        judgment = read_a_judgment(ArtifactRef(**ref), root, item, document, canonical)
        judged[key] = (document, canonical, judgment)
    return judged


def contexts(ctx: Inputs, judged: dict[str, tuple[Any, Any, Any]]) -> Any:
    """(coordinate, package record, Stage B inputs) for every retrieval-arm context, by arm."""
    root = ctx.path("published_context_root")
    indexes: dict[str, tuple[dict, dict]] = {}
    active = None
    records = sorted(
        (r for r in ctx.package["records"] if r["matrix_arm"] != "closed_book"),
        key=lambda r: (r["matrix_arm"], r["item_id"], r["condition"]),
    )
    for record in records:
        if record["matrix_arm"] != active:
            indexes.clear()
            active = record["matrix_arm"]
        item = ctx.items[record["item_id"]]
        document, canonical, judgment = judged[f"{item.item_id}/{record['condition']}"]
        cref = ArtifactRef(**record["context_record"])
        context = read_project_json(root / cref.logical_ref, cref.sha256, ContextRecord)
        ranked = read_project_json(
            root / context.retrieval.logical_ref, context.retrieval.sha256, RankedRetrieval
        )
        if ranked.index.artifact_id not in indexes:
            kind = DenseIndex if ranked.retriever == "bge" else Bm25Index
            index = read_project_json(root / ranked.index.logical_ref, ranked.index.sha256, kind)
            chunks, sets = {}, {}
            for ref in index.chunk_sets:
                chunk_set = read_project_json(root / ref.logical_ref, ref.sha256, ChunkSet)
                loaded = read_jsonl(
                    root / chunk_set.chunks.logical_ref, chunk_set.chunks.sha256, Chunk
                )
                chunks.update({record_ref(c).artifact_id: c for c in loaded})
                sets[chunk_set.input.paper_id] = chunk_set
            indexes[ranked.index.artifact_id] = chunks, sets
        chunks, sets = indexes[ranked.index.artifact_id]
        inputs = stage_b.Inputs(
            item, document, judgment, context, ranked, sets[item.paper_id], chunks, canonical,
            cref, ctx.path("analysis_root"),
        )  # fmt: skip
        yield f"{item.item_id}/{record['matrix_arm']}/{record['condition']}", record, inputs


# ---- Stage B ---------------------------------------------------------------------------------


def stage_b_queue(ctx: Inputs, out: Path, seed: int) -> dict[str, Any]:
    groups: dict[tuple[str, ...], list[str]] = {}
    views: dict[tuple[str, ...], dict] = {}
    deterministic = Counter()
    for coordinate, _, inputs in contexts(ctx, stage_a(ctx)):
        candidate = stage_b.evaluate(inputs)
        if not stage_b.needs_review_v2(candidate, inputs):
            deterministic[candidate.determination] += 1
            continue
        key = final_test.reuse_key(inputs)
        if key not in groups:
            views[key], _ = final_test.stage_b_view(inputs)
        groups.setdefault(key, []).append(coordinate)
    keyed = {json.dumps(list(k)): k for k in groups}
    entries, mapping, template = [], {}, []
    for n, text in enumerate(final_test.review_order(keyed, seed), 1):
        handle = f"B{n:04d}"
        entries.append({"handle": handle, **views[keyed[text]]})
        mapping[handle] = sorted(groups[keyed[text]])
        template.append(
            {"handle": handle, "status": None, "reason": None, "citations": [],
             "missing_components": [], "unresolved_reason": None, "flags": [],
             "conflict_citations": [], "rationale": None, "at_utc": None}
        )  # fmt: skip
    manifest = {
        "status": "final_test_stage_b_pending_human_review",
        "rules": RULES,
        "seed": seed,
        "authorization_sha256": ctx.authorization_ref.sha256,
        "queue_sha256": write_once(out / "reviewer" / "queue.json", dumps(entries)),
        "mapping_sha256": write_once(out / "provenance" / "mapping.json", dumps(mapping)),
        "reviewed_contexts": sum(len(v) for v in mapping.values()),
        "deterministic_contexts": dict(deterministic),
    }
    write_once(out / "reviewer" / "review.json", dumps(template))
    write_once(out / "provenance" / "manifest.json", dumps(manifest))
    return {"entries": len(entries), **manifest}


def stage_b_ingest(ctx: Inputs, out: Path, reviewer: str) -> dict[str, Any]:
    """Validate every completed review record, apply it to its identical contexts, publish
    every v2 judgment and write the Stage B inventory. Nothing is published if one fails."""
    manifest_bytes = (out / "provenance" / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    mapping_bytes = (out / "provenance" / "mapping.json").read_bytes()
    if sha256_bytes(mapping_bytes) != manifest["mapping_sha256"]:
        raise Refused("Stage B mapping differs from its queue manifest")
    mapping = json.loads(mapping_bytes)
    records = {r["handle"]: r for r in json.loads((out / "reviewer" / "review.json").read_bytes())}
    if set(records) != set(mapping) or any(r["status"] is None for r in records.values()):
        raise Refused("every Stage B queue entry needs a completed review record")
    handle_of = {c: h for h, cs in mapping.items() for c in cs}
    root = ctx.path("analysis_root")
    reviewed: dict[str, tuple[Any, Any]] = {}
    pending, errors = [], []
    for coordinate, _, inputs in contexts(ctx, stage_a(ctx)):
        candidate = stage_b.evaluate(inputs)
        try:
            handle = handle_of.get(coordinate)
            if handle is None:
                judged = stage_b.deterministic_judgment(candidate, RULES, inputs)
            elif handle not in reviewed:
                _, handles = final_test.stage_b_view(inputs)
                judged = final_test.judgment_from_record(records[handle], handles, inputs, reviewer)
                reviewed[handle] = (judged, inputs)
            else:
                judged = final_test.apply_review(reviewed[handle][0], reviewed[handle][1], inputs)
        except (ValueError, KeyError) as error:
            errors.append(f"{coordinate}: {error}")
            continue
        pending.append((coordinate, candidate, judged, inputs))
    if errors:
        raise Refused("Stage B review records do not validate:\n" + "\n".join(errors[:20]))
    refs = {}
    for coordinate, candidate, judged, inputs in pending:
        cref = stage_b.stage_b_ref(candidate)
        if not (root / cref.logical_ref).exists():
            publish_candidate(candidate, inputs, root)
        ref = stage_b.stage_b_ref(judged)
        if (root / ref.logical_ref).exists():
            if read_b_judgment(ref, root, inputs) != judged:
                raise Refused("a published Stage B judgment differs")
        else:
            publish_judgment(judged, inputs, root)
        refs[coordinate] = ref.model_dump(mode="json")
    inventory = {
        "status": "stage_b_judgments_submitted",
        "rules": RULES,
        "reviewer": reviewer,
        "queue_manifest_sha256": sha256_bytes(manifest_bytes),
        "stage_b_judgments": refs,
    }
    target = out / "stage_b_inventory.json"
    write_once(target, dumps(inventory))
    return {"contexts": len(refs), "reviewed_entries": len(reviewed), "inventory": str(target)}


# ---- Stage C ---------------------------------------------------------------------------------


def automatic_scores(ctx: Inputs, results: list[GenerationResult]) -> list[Any]:
    return [stage_c.score(ctx.items[r.item_id], r, RULES) for r in results]


def stage_c_queue(ctx: Inputs, out: Path, seed: int) -> dict[str, Any]:
    results = verify_batch(ctx)
    scores = automatic_scores(ctx, results)
    answers = {r.observation_key: r.answer for r in results}
    entries, mapping = final_test.stage_c_queue(scores, ctx.items, seed, answers)
    manifest = {
        "status": "final_test_stage_c_pending_adjudication",
        "rules": "stage_c_rules_v2",
        "seed": seed,
        "authorization_sha256": ctx.authorization_ref.sha256,
        "queue_sha256": write_once(out / "reviewer" / "queue.json", dumps(entries)),
        "mapping_sha256": write_once(out / "provenance" / "mapping.json", dumps(mapping)),
    }
    template = {e["handle"]: {"status": None, "rationale": None, "at_utc": None} for e in entries}
    write_once(out / "reviewer" / "decisions.json", dumps(template))
    write_once(out / "provenance" / "manifest.json", dumps(manifest))
    return {"entries": len(entries), "observations": sum(map(len, mapping.values())), **manifest}


def adjudicated_scores(ctx: Inputs, results: list[GenerationResult]) -> list[Any]:
    review = ctx.path("stage_c_review")
    manifest = json.loads((review / "provenance" / "manifest.json").read_bytes())
    mapping_bytes = (review / "provenance" / "mapping.json").read_bytes()
    if sha256_bytes(mapping_bytes) != manifest["mapping_sha256"]:
        raise Refused("Stage C mapping differs from its queue manifest")
    mapping = json.loads(mapping_bytes)
    decisions = json.loads((review / "reviewer" / "decisions.json").read_bytes())
    reviewer = ctx.request.get("stage_c_reviewer", final_test.RESEARCHER)
    if set(decisions) != set(mapping) or any(
        d.get("status") not in ("correct", "incorrect", "non_evaluable") for d in decisions.values()
    ):
        raise Refused("every Stage C queue entry needs a recorded decision")
    scores = automatic_scores(ctx, results)
    answers = {r.observation_key: r.answer for r in results}
    _, fresh = final_test.stage_c_queue(scores, ctx.items, manifest["seed"], answers)
    if fresh != mapping:
        raise Refused("Stage C queue differs from the freshly scored batch")
    try:
        return final_test.apply_decisions(scores, mapping, decisions, reviewer)
    except ValueError as error:
        raise Refused(f"Stage C decisions: {error}") from None


# ---- table and handoff -----------------------------------------------------------------------


def evaluate(ctx: Inputs, *, publish: bool = True) -> dict[str, Any]:
    root = ctx.path("analysis_root")
    out = ctx.path("batch")
    results = verify_batch(ctx)
    scores = {s.observation_key: s for s in adjudicated_scores(ctx, results)}
    judged = stage_a(ctx)
    b_inventory = json.loads(ctx.path("stage_b_inventory").read_bytes())
    if b_inventory.get("status") != "stage_b_judgments_submitted" or b_inventory["rules"] != RULES:
        raise Refused("Stage B inventory is not a submitted v2 review")
    evidence = {}
    for coordinate, record, inputs in contexts(ctx, judged):
        ref = b_inventory["stage_b_judgments"].get(coordinate)
        if ref is None:
            raise Refused(f"Stage B judgment missing for {coordinate}")
        b = read_b_judgment(ArtifactRef(**ref), root, inputs)
        candidate = stage_b.evaluate(inputs)
        read_candidate(stage_b.stage_b_ref(candidate), root, inputs)
        document, canonical, a = judged[f"{inputs.item.item_id}/{record['condition']}"]
        evidence[coordinate] = (Evidence(document, canonical, a, b, inputs, root), record)
    records = {(r["item_id"], r["matrix_arm"], r["condition"]): r for r in ctx.package["records"]}
    observations = []
    for result in results:
        item = ctx.items[result.item_id]
        score = scores[result.observation_key]
        if publish:
            # The bytes each score depends on, as in the development evaluator.
            for role, ref in stage_c.dependencies(score):
                if role == "benchmark_item":
                    data = canonical_json_bytes(item.model_dump(mode="json"))
                elif role == "generation_result":
                    data = serialize_project_json(result, GenerationResult)
                elif role == "generator_output":
                    data = (out / ref.logical_ref).read_bytes()
                else:
                    continue
                if sha256_bytes(data) != ref.sha256:
                    raise Refused(f"{role} bytes differ from the score's dependency")
                write_once(root / ref.logical_ref, data)
            if score.automatic is not None:
                auto = stage_c.score(item, result, RULES)
                if not (root / score.automatic.logical_ref).exists():
                    publish_score(auto, item, result, root)
            ref = stage_c.score_ref(score)
            if (root / ref.logical_ref).exists():
                if read_score(ref, root, item, result) != score:
                    raise Refused("published Stage C score differs")
            else:
                publish_score(score, item, result, root)
        arm, condition = result.matrix_arm, result.representation_or_condition
        record = records[result.item_id, arm, condition]
        found, link = None, None
        if arm != "closed_book":
            found, _ = evidence[f"{result.item_id}/{arm}/{condition}"]
            if result.status == "completed":
                stored = json.loads(
                    (out / "results" / f"{result.observation_key}.json").read_bytes()
                )
                input_ref = ArtifactRef(**stored["generator_input"])
                if publish:
                    write_once(
                        root / input_ref.logical_ref, (out / input_ref.logical_ref).read_bytes()
                    )
                link = GeneratorInput(
                    observation_key=result.observation_key,
                    attempt_id=result.attempt_id,
                    context=found.b_inputs.context_ref,
                    exact_context_sha256=sha256_bytes(
                        found.b_inputs.context.exact_context.encode()
                    ),
                    input=input_ref,
                    provenance="recorded",
                )
        observations.append(
            Observation(
                item,
                "final_test",
                arm,
                condition,
                result.repeat_index,
                result.resolved_config_sha256,
                record["retrieval_config_sha256"],
                result,
                score,
                found,
                link,
            )
        )
    roster = PlannedRoster(
        scope="complete_matrix",
        phase="final_test",
        benchmark_version=next(iter(ctx.items.values())).benchmark_version,
        corpus_version="corpus_v1",
        declared_by=final_test.RESEARCHER,
        basis="Frozen 305-item final-test roster and five-arm matrix (EXPERIMENT_PLAN v2)",
        arms=[
            RosterArm(matrix_arm=a, conditions=list(c), repeats=list(r))
            for a, (c, r) in FROZEN_MATRIX.items()
        ],
        items=roster_items_from_final_test(list(ctx.items.values())),
    )
    table = build_table(observations, roster)
    if publish:
        _, _, table_build = publish_table(table, root)
    else:
        table_build = published_handoff(ctx).table_build
    batch = json.loads((out / "batch.json").read_bytes())
    code = GitState(
        status="ok", commit=batch["identity"]["git"]["head"], dirty=False, dirty_paths=0
    )
    template, rows = analysis_handoff._describe(table_build, root, code)
    if [r.model_dump() for r in rows] != [r.model_dump() for r in table.rows]:
        raise Refused("published analysis rows differ from the verified evaluation")
    sources = []
    head = ArtifactRef(
        artifact_id="drive_manifest_" + ctx.authorization.drive_manifest_sha256,
        logical_ref=ctx.authorization_ref.logical_ref,
        sha256=ctx.authorization.drive_manifest_sha256,
    )
    for arm in FROZEN_MATRIX:
        config = next(p.config for p in ctx.plan if p.record["matrix_arm"] == arm)
        sources.append(
            analysis_handoff.SourceRun(
                run_id=batch["run_id"],
                completion=head,
                run_record_sha256=sha256_file(out / "batch.json"),
                resolved_config_sha256=config.configuration_sha256,
                raw_config_sha256=config.raw_toml_sha256,
                roster_sha256=batch["identity"]["planned_keys_sha256"],
                matrix_arm=arm,
                backend_kind="official",
                observations=sum(r.matrix_arm == arm for r in results),
            )
        )
    auth = ctx.authorization
    handoff = analysis_handoff.AnalysisHandoff.model_validate(
        template.model_dump()
        | {
            "input_origin": "verified_official_run",
            "source_runs": [s.model_dump() for s in sources],
            "scoring_reference": template.scoring_reference
            | {"primary_bootstrap_registry_sha256": APPROVED_REGISTRY_SHA256},
            "awaiting_approval": [],
            "custody_exception": {
                "status": "custody_gate_not_established_exception_v1",
                "authorization": ctx.authorization_ref.model_dump(),
                "amendment_sha256": auth.custody_exception_sha256,
                "run_id": auth.run_id,
                "drive_manifest_sha256": auth.drive_manifest_sha256,
            },
        }
    )
    if publish:
        href, hbuild = analysis_handoff._store(handoff, root)
    else:
        if published_handoff(ctx) != handoff:
            raise Refused("official handoff differs from freshly verified evaluation")
        hbuild = ArtifactRef(**ctx.request["handoff_build"])
        href = read_manifest(root / hbuild.logical_ref, hbuild.sha256).outputs[0].artifact
    return {
        "status": "verified_official_final_test_handoff",
        "handoff": href.model_dump(),
        "build": hbuild.model_dump(),
        "observations": len(results),
        "csv_sha256": handoff.csv.sha256,
        "source_verification": "fresh_python_verification",
        "custody": "custody_gate_not_established_exception_v1",
        "scientific_bootstrap_seed": {
            "registry_sha256": APPROVED_REGISTRY_SHA256,
            "record": read_analysis_seed(REPO).model_dump(mode="json"),
        },
    }


def published_handoff(ctx: Inputs) -> analysis_handoff.AnalysisHandoff:
    root = ctx.path("analysis_root")
    build = ArtifactRef(**ctx.request["handoff_build"])
    ref = read_manifest(root / build.logical_ref, build.sha256).outputs[0].artifact
    return read_project_json(root / ref.logical_ref, ref.sha256, analysis_handoff.AnalysisHandoff)


def verify_request(path: Path) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()), final_test_scope():
        return evaluate(load(path), publish=False)


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-request":
        print(json.dumps(verify_request(Path(sys.argv[2])), sort_keys=True))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("stage-b-queue", "stage-b-ingest", "stage-c-queue", "evaluate"):
        c = commands.add_parser(name)
        c.add_argument("--request", type=Path, required=True)
        if name != "evaluate":
            c.add_argument("--out", type=Path, required=True)
        if name.endswith("queue"):
            c.add_argument("--seed", type=int, required=True)
        if name == "stage-b-ingest":
            c.add_argument("--reviewer", required=True)
    args = parser.parse_args()
    with final_test_scope(), stage_b.reuse_checks():
        ctx = load(args.request)
        if args.command == "stage-b-queue":
            result = stage_b_queue(ctx, args.out, args.seed)
        elif args.command == "stage-b-ingest":
            result = stage_b_ingest(ctx, args.out, args.reviewer)
        elif args.command == "stage-c-queue":
            result = stage_c_queue(ctx, args.out, args.seed)
        else:
            result = evaluate(ctx)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
