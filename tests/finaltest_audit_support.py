"""Hand-built, final-test-shaped plan, roster and binding records for the input-contract tests.

Nothing here executes a run or calls a backend. All identities are synthetic; `CANARY` stands
for protected question text and may only ever be fed to a digest.
"""

from finaltest_campaign_support import CANARY, FINAL_ITEMS, oracle_key, population, sha
from generation_support import resolve, synthetic_settings

from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest.batches import (
    BatchManifest,
    make_manifest,
)
from biorag_repro.finaltest.campaign import build_plan
from biorag_repro.finaltest.inputs import (
    BatchInputBinding,
    ConditionRoster,
    RosterIdentityRow,
    build_binding,
    ordered_digest,
)
from biorag_repro.generation.attempts import retry_policy_of

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
