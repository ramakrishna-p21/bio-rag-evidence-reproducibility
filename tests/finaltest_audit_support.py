"""Hand-built, final-test-shaped records. Nothing here executes a run or calls a backend.

Runs are written file by file: the run record, the configuration copies, hash-chained events,
environment records, raw outputs and seals. All identities are synthetic; `CANARY` stands for
protected question text and may only ever be fed to a digest.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from finaltest_campaign_support import CANARY, FINAL_ITEMS, oracle_key, population, sha
from test_finaltest_citation import initial_citation
from test_generation_attempts import GOOD_OUTPUT, Clock, environment
from test_generation_config import resolve, synthetic_settings

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_manifest, serialize_project_json
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest.audit import AuditContext, open_view
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    BatchManifest,
    citation_sha256,
    make_manifest,
)
from biorag_repro.finaltest.campaign import build_plan, plan_sha256
from biorag_repro.finaltest.inputs import (
    BatchInputBinding,
    ConditionRoster,
    RosterIdentityRow,
    binding_sha256,
    build_binding,
    ordered_digest,
    retry_policy_sha256,
)
from biorag_repro.finaltest.runrecord import FinalTestRunRecord, run_record_sha256
from biorag_repro.generation.attempts import AttemptEvent, Coordinates, retry_policy_of
from biorag_repro.generation.environment import environment_sha256
from biorag_repro.generation.parsing import parse_output

ARM = "dense_robustness"  # JATS and PDF_DOCLING, one repeat
ARMS = tuple(FROZEN_MATRIX)
NOW = "2026-11-08T00:00:00Z"
ROOT_ID = sha("rehearsal-root")
RUN_ID = sha("synthetic-run")[:32]


def settings_for(arm: str, **changes: object) -> dict:
    settings = synthetic_settings() | {"matrix_arm": arm, "purpose": "synthetic_test"}
    if arm == "closed_book":
        settings.pop("retrieval_config_sha256")
    return settings | changes


CONFIGS = {arm: resolve(settings_for(arm)) for arm in ARMS}
PLAN = build_plan(
    population(),
    {arm: config.configuration_sha256 for arm, config in CONFIGS.items()},
    CONFIGS["confirmatory"].settings.seed_registry_sha256,
)
ITEMS = sorted(FINAL_ITEMS)


def row_for(arm: str, condition: str, item: str, repeat: int, **changes: object):
    config = CONFIGS[arm].configuration_sha256
    closed = arm == "closed_book"
    values = {
        "item_id": item,
        "paper_id": FINAL_ITEMS[item],
        "item_sha256": sha(CANARY + item),  # content only ever enters as a digest
        "matrix_arm": arm,
        "condition": condition,
        "repeat_index": repeat,
        "resolved_config_sha256": config,
        "seed": None,
        "source_identity_sha256": None if closed else sha(f"source-{item}-{condition}"),
        "context_sha256": None if closed else sha(f"context-{item}-{condition}"),
        "evidence_sequence_sha256": None
        if closed
        else ordered_digest([f"chunk-{item}-{condition}-{n}" for n in (1, 2)]),
        "rendered_input_sha256": sha(f"rendered-{item}-{condition}-{repeat}"),
        "observation_key": oracle_key(item, arm, condition, repeat, config),
    }
    return RosterIdentityRow(**(values | changes))


def roster_for(arm: str, condition: str, rows=None, **changes: object) -> ConditionRoster:
    repeats = FROZEN_MATRIX[arm][1]
    manifest = make_manifest(ROOT_ID, PLAN, arm, NOW)
    values = {
        "batch_id": manifest.batch_id,
        "matrix_arm": arm,
        "condition": condition,
        "benchmark_version": PLAN.benchmark_version,
        "resolved_config_sha256": CONFIGS[arm].configuration_sha256,
        "rows": rows
        if rows is not None
        else [row_for(arm, condition, i, r) for i in ITEMS for r in repeats],
    }
    return ConditionRoster(**(values | changes))


def rosters_for(arm: str = ARM) -> dict[str, ConditionRoster]:
    return {c: roster_for(arm, c) for c in FROZEN_MATRIX[arm][0]}


def manifest_for(arm: str = ARM) -> BatchManifest:
    return make_manifest(ROOT_ID, PLAN, arm, NOW)


def binding_for(arm: str = ARM, rosters=None, seeds=None, config=None) -> BatchInputBinding:
    config = config or CONFIGS[arm]
    return build_binding(
        manifest=manifest_for(arm),
        plan=PLAN,
        resolved=config,
        retry=retry_policy_of(config.settings),
        seeds=seeds,
        rosters=list((rosters or rosters_for(arm)).values()),
    )


@dataclass
class Built:
    base: Path
    run_dir: Path
    record: FinalTestRunRecord
    run_sha: str
    view: object
    context: AuditContext
    keys: dict[tuple[str, str, int], str]

    def event_path(self, key: str, sequence: int) -> Path:
        return self.run_dir / "observations" / key / "events" / f"{sequence:06d}.json"


def write_observation(built_view, run_dir, record, run_sha, row, status, clock) -> None:
    key = row.observation_key
    events = run_dir / "observations" / key / "events"
    events.mkdir(parents=True)
    attempt = sha(key)[:32]
    state = {"previous": None, "sequence": 0}

    def put(**fields) -> Path:
        state["sequence"] += 1
        event = AttemptEvent(
            run_id=record.run_id,
            run_record_sha256=run_sha,
            observation_key=key,
            sequence=state["sequence"],
            previous_event_sha256=state["previous"],
            at_utc=clock(),
            **fields,
        )
        data = serialize_project_json(event, AttemptEvent)
        target = events / f"{state['sequence']:06d}.json"
        target.write_bytes(data)
        state["previous"] = sha256_bytes(data)
        return target

    put(
        state="planned",
        coordinates=Coordinates(
            item_id=row.item_id,
            paper_id=row.paper_id,
            matrix_arm=row.matrix_arm,
            representation_or_condition=row.condition,
            repeat_index=row.repeat_index,
            seed=row.seed,
        ),
    )
    if status == "planned":
        return
    env = environment("mock")
    digest = environment_sha256(env)
    env_dir = run_dir / "environments"
    env_dir.mkdir(exist_ok=True)
    (env_dir / f"{digest}.json").write_bytes(serialize_project_json(env, type(env)))
    running = {"attempt_id": attempt, "attempt_number": 1}
    put(
        state="running",
        backend_kind="mock",
        environment_sha256=digest,
        rendered_input_sha256=row.rendered_input_sha256,
        seed=row.seed,
        **running,
    )
    if status == "interrupted":
        return
    if status == "terminal_failure":
        put(state="terminal_failure", failure_kind="synthetic", failure_detail="x", **running)
    else:
        raw = GOOD_OUTPUT.encode()
        ref = built_view._raw_ref(key, attempt, row.rendered_input_sha256, sha256_bytes(raw))
        target = run_dir.parent.parent / ref.logical_ref
        target.parent.mkdir(parents=True)
        target.write_bytes(raw)
        parsed = parse_output(GOOD_OUTPUT, "short_answer_json_v1")
        put(
            state="complete",
            raw_output=ref,
            parse_status=parsed.parse_status,
            answer=parsed.answer,
            abstained=parsed.abstained,
            parse_reason=parsed.reason,
            **running,
        )
    if status != "complete_unsealed":
        loaded = built_view.observation(key)
        manifest = built_view._seal_manifest(loaded)
        logical = manifest.outputs[0].artifact.logical_ref
        last = events / f"{state['sequence']:06d}.json"
        publish_manifest(manifest, built_view._seal_path(key), {logical: last})


def build_run(
    base: Path,
    arm: str = ARM,
    statuses: dict[str, str] | None = None,
    *,
    default: str = "complete",
    extra_rows: tuple = (),
    root_kind: str = "rehearsal",
    citations: list[AuthorizationCitation] | None = None,
    **record_changes: object,
) -> Built:
    """Write a final-test-shaped run. `statuses` maps an item id to a status for that item."""
    config = CONFIGS[arm]
    rosters = rosters_for(arm)
    binding = binding_for(arm, rosters)
    manifest = manifest_for(arm)
    citation = initial_citation(
        batch_id=manifest.batch_id, inputs_set_sha256=binding_sha256(binding)
    )
    citations = [citation] if citations is None else citations
    values = {
        "run_id": RUN_ID,
        "created_utc": NOW,
        "root_id": ROOT_ID,
        "root_kind": root_kind,
        "campaign_plan_sha256": plan_sha256(PLAN),
        "batch_id": manifest.batch_id,
        "benchmark_version": PLAN.benchmark_version,
        "corpus_content_sha256": PLAN.corpus_content_sha256,
        "benchmark_release_sha256": PLAN.benchmark_release_sha256,
        "matrix_arm": arm,
        "resolved_config_sha256": config.configuration_sha256,
        "raw_config_sha256": config.raw_toml_sha256,
        "purpose": "synthetic_test",
        "backend_kind": "mock",
        "retry": retry_policy_of(config.settings),
        "retry_policy_sha256": retry_policy_sha256(retry_policy_of(config.settings)),
        "inputs_set_sha256": binding_sha256(binding),
        "authorization_citation_sha256": citation_sha256(citation),
    }
    record = FinalTestRunRecord(**(values | record_changes))
    run_dir = base / "runs" / record.run_id
    run_dir.mkdir(parents=True)
    run_sha = run_record_sha256(record)
    (run_dir / "run.json").write_bytes(serialize_project_json(record, FinalTestRunRecord))
    (run_dir / "config.canonical.json").write_bytes(config.canonical_json)
    (run_dir / "config.toml").write_bytes(config.raw_toml)
    view = open_view(base, record, run_sha, config)
    clock = Clock()
    keys = {}
    for roster in rosters.values():
        for row in roster.rows:
            keys[(row.item_id, row.condition, row.repeat_index)] = row.observation_key
            status = (statuses or {}).get(row.item_id, default)
            write_observation(view, run_dir, record, run_sha, row, status, clock)
    for row in extra_rows:
        write_observation(view, run_dir, record, run_sha, row, default, clock)
    context = AuditContext(
        root_id=ROOT_ID,
        root_kind=root_kind,
        plan=PLAN,
        item_papers=dict(FINAL_ITEMS),
        manifest=manifest,
        binding=binding,
        rosters=rosters,
        citations=citations,
    )
    return Built(base, run_dir, record, run_sha, view, context, keys)


def snapshot(base: Path) -> dict[str, tuple]:
    """Path, kind, mode, mtime and content hash of everything below `base`, directories too."""
    found: dict[str, tuple] = {}
    for current, directories, files in os.walk(base, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            info = path.lstat()
            digest = (
                sha256_bytes(path.read_bytes())
                if path.is_file() and not path.is_symlink()
                else None
            )
            found[str(path.relative_to(base))] = (
                info.st_mode,
                info.st_mtime_ns,
                info.st_size,
                digest,
            )
    found["."] = (base.lstat().st_mode, base.lstat().st_mtime_ns, 0, None)
    return found
