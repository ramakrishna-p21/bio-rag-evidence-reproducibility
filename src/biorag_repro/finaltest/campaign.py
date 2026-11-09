"""The planned final-test campaign: which observations the frozen matrix requires.

The plan is derived from the verified benchmark release and the frozen matrix
(`FROZEN_MATRIX`); no count is an input. Observation keys come from
`core.identity.observation_key`. Scientific repeat indices are 1, 2 and 3, exactly as in the
matrix, the seed registry and the roster; they are passed to the key function unchanged.

A plan is structural. It names candidate configuration identities per matrix arm and cannot
authorize anything: authorization exists only as the Phase 9 guard's result at execution time.
Only identifiers and split metadata are read from the benchmark items; question text, gold
answers and evidence are never parsed into a record here.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, model_validator

from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.benchmark.split import PaperSplit
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.storage import read_project_json, serialize_project_json
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX, MATRIX_ID
from biorag_repro.finaltest.layout import FrozenRecord
from biorag_repro.readiness.files import loads_strict, read_regular

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
PHASE: Final = "final_test"
ARMS: Final = tuple(FROZEN_MATRIX)  # matrix order


class PlanError(ValueError):
    """The population or plan contradicts the frozen design."""


@dataclass(frozen=True)
class Population:
    """Identity and split metadata of the benchmark, without any item content."""

    benchmark_version: str
    corpus_version: str
    corpus_content_sha256: str
    benchmark_release_sha256: str
    split_sha256: str
    final_test_paper_ids: tuple[str, ...]
    development_paper_ids: tuple[str, ...]
    final_test_item_ids: tuple[str, ...]  # sorted
    item_papers: Mapping[str, str]  # final-test item id -> paper id


def digest_lines(values: Sequence[str]) -> str:
    """SHA-256 of the sorted values joined by newlines (an order-independent set digest)."""
    return sha256_bytes("\n".join(sorted(values)).encode())


def _read_items(data: bytes) -> list[dict[str, str]]:
    """Identity fields only. The other fields are parsed by the JSON decoder and dropped."""
    text = data.decode("utf-8")
    kept = []
    for line in text.splitlines():
        row = loads_strict(line.encode())
        kept.append({k: row[k] for k in ("item_id", "paper_id", "split", "candidate_status")})
    return kept


def load_population(data_root: Path, release_bytes: bytes) -> Population:
    """Read the final-test population from the immutable release and its verified split."""
    release = BenchmarkRelease.model_validate(loads_strict(release_bytes), strict=True)
    split_data = read_regular(data_root, release.split.logical_ref)
    if sha256_bytes(split_data) != release.split.sha256:
        raise PlanError("the split differs from the release record")
    split = read_project_json(
        data_root / release.split.logical_ref, release.split.sha256, PaperSplit
    )
    items_data = read_regular(data_root, release.items.logical_ref)
    if sha256_bytes(items_data) != release.items.sha256:
        raise PlanError("the items differ from the release record")
    return population_from(
        release=release,
        release_sha256=sha256_bytes(release_bytes),
        split=split,
        split_sha256=release.split.sha256,
        items=_read_items(items_data),
    )


def population_from(
    *,
    release: BenchmarkRelease,
    release_sha256: str,
    split: PaperSplit,
    split_sha256: str,
    items: Sequence[Mapping[str, str]],
) -> Population:
    final_papers, dev_papers = set(split.final_test_ids), set(split.development_ids)
    if len(final_papers) != release.final_test_paper_count or len(dev_papers) != (
        release.development_paper_count
    ):
        raise PlanError("the split does not match the release paper counts")
    if split.corpus_version != release.corpus_version:
        raise PlanError("the split names another corpus version")
    if len(items) != release.item_count:
        raise PlanError("the item count differs from the release record")
    final: dict[str, str] = {}
    for item in items:
        if item["candidate_status"] != "accepted":
            raise PlanError(f"item {item['item_id']} is not accepted")
        paper = item["paper_id"]
        if paper in final_papers and item["split"] == "final_test":
            if item["item_id"] in final:
                raise PlanError(f"duplicate item {item['item_id']}")
            final[item["item_id"]] = paper
        elif paper in dev_papers and item["split"] == "development":
            continue
        else:
            raise PlanError(f"item {item['item_id']} disagrees with the validated split")
    if not final:
        raise PlanError("the final-test population is empty")
    return Population(
        benchmark_version=release.benchmark_version,
        corpus_version=release.corpus_version,
        corpus_content_sha256=release.corpus_content_sha256,
        benchmark_release_sha256=release_sha256,
        split_sha256=split_sha256,
        final_test_paper_ids=tuple(sorted(final_papers)),
        development_paper_ids=tuple(sorted(dev_papers)),
        final_test_item_ids=tuple(sorted(final)),
        item_papers=final,
    )


class PlannedCell(FrozenRecord):
    condition: Nonblank
    repeats: list[int]
    observation_count: int


class PlannedBatch(FrozenRecord):
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256  # a candidate identity; approval is not recorded here
    cells: list[PlannedCell]
    expected_count: int
    key_set_sha256: Sha256


class CampaignPlan(FrozenRecord):
    schema_version: Literal["1"] = "1"
    matrix_id: Literal["experiment_plan_v1"] = MATRIX_ID
    execution_phase: Literal["final_test"] = PHASE
    status: Literal["structural"] = "structural"
    benchmark_version: Nonblank
    corpus_version: Nonblank
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256
    split_sha256: Sha256
    final_test_paper_count: int
    final_test_item_count: int
    final_test_item_ids_sha256: Sha256
    generation_seed_registry_sha256: Sha256
    total_expected: int
    batches: list[PlannedBatch]

    @model_validator(mode="after")
    def matches_frozen_matrix(self) -> Self:
        if [b.matrix_arm for b in self.batches] != list(ARMS):
            raise ValueError("a plan has exactly one batch per matrix arm, in matrix order")
        for batch in self.batches:
            conditions, repeats = FROZEN_MATRIX[batch.matrix_arm]
            cells = [(c.condition, tuple(c.repeats), c.observation_count) for c in batch.cells]
            wanted = [(c, repeats, len(repeats) * self.final_test_item_count) for c in conditions]
            if cells != wanted:
                raise ValueError(f"{batch.matrix_arm} cells differ from the frozen matrix")
            if batch.expected_count != sum(c.observation_count for c in batch.cells):
                raise ValueError(f"{batch.matrix_arm} expected count differs from its cells")
        if len({b.resolved_config_sha256 for b in self.batches}) != len(self.batches):
            raise ValueError("every matrix arm needs its own configuration identity")
        if self.total_expected != sum(b.expected_count for b in self.batches):
            raise ValueError("total differs from the sum of the batches")
        return self


def plan_sha256(plan: CampaignPlan) -> str:
    return sha256_bytes(serialize_project_json(plan, CampaignPlan))


def expected_keys(plan: CampaignPlan, item_ids: Sequence[str], matrix_arm: str) -> list[str]:
    """Every observation key of one batch, sorted. Duplicate coordinates are an error."""
    batch = next(b for b in plan.batches if b.matrix_arm == matrix_arm)
    keys = [
        observation_key(
            phase=PHASE,
            benchmark_version=plan.benchmark_version,
            item_id=item_id,
            matrix_arm=matrix_arm,
            representation_or_condition=cell.condition,
            repeat_index=repeat,
            resolved_config_sha256=batch.resolved_config_sha256,
        )
        for cell in batch.cells
        for item_id in item_ids
        for repeat in cell.repeats
    ]
    if len(set(keys)) != len(keys):
        raise PlanError("duplicate scientific coordinates")
    return sorted(keys)


def build_plan(
    population: Population,
    config_sha256_by_arm: Mapping[str, str],
    generation_seed_registry_sha256: str,
) -> CampaignPlan:
    if set(config_sha256_by_arm) != set(ARMS):
        raise PlanError("a configuration identity is needed for each matrix arm, and no other")
    count = len(population.final_test_item_ids)
    if set(population.final_test_paper_ids) & set(population.development_paper_ids):
        raise PlanError("development papers leak into the final-test population")
    if not set(population.item_papers.values()) <= set(population.final_test_paper_ids):
        raise PlanError("a final-test item belongs to a non-final-test paper")
    batches = []
    for arm in ARMS:
        conditions, repeats = FROZEN_MATRIX[arm]
        cells = [
            PlannedCell(condition=c, repeats=list(repeats), observation_count=len(repeats) * count)
            for c in conditions
        ]
        batch = PlannedBatch(
            matrix_arm=arm,
            resolved_config_sha256=config_sha256_by_arm[arm],
            cells=cells,
            expected_count=sum(c.observation_count for c in cells),
            key_set_sha256="0" * 64,  # replaced below, once the keys exist
        )
        keys = _keys_for(population, batch)
        batches.append(batch.model_copy(update={"key_set_sha256": digest_lines(keys)}))
    return CampaignPlan(
        benchmark_version=population.benchmark_version,
        corpus_version=population.corpus_version,
        corpus_content_sha256=population.corpus_content_sha256,
        benchmark_release_sha256=population.benchmark_release_sha256,
        split_sha256=population.split_sha256,
        final_test_paper_count=len(population.final_test_paper_ids),
        final_test_item_count=count,
        final_test_item_ids_sha256=digest_lines(population.final_test_item_ids),
        generation_seed_registry_sha256=generation_seed_registry_sha256,
        total_expected=sum(b.expected_count for b in batches),
        batches=batches,
    )


def _keys_for(population: Population, batch: PlannedBatch) -> list[str]:
    keys = [
        observation_key(
            phase=PHASE,
            benchmark_version=population.benchmark_version,
            item_id=item_id,
            matrix_arm=batch.matrix_arm,
            representation_or_condition=cell.condition,
            repeat_index=repeat,
            resolved_config_sha256=batch.resolved_config_sha256,
        )
        for cell in batch.cells
        for item_id in population.final_test_item_ids
        for repeat in cell.repeats
    ]
    if len(set(keys)) != len(keys):
        raise PlanError("duplicate scientific coordinates")
    return keys


def check_plan(plan: CampaignPlan, population: Population) -> None:
    """The plan must be exactly the one this population and these configurations produce."""
    rebuilt = build_plan(
        population,
        {b.matrix_arm: b.resolved_config_sha256 for b in plan.batches},
        plan.generation_seed_registry_sha256,
    )
    if plan_sha256(rebuilt) != plan_sha256(plan):
        raise PlanError("the plan does not match the benchmark release and split")


__all__ = [
    "ARMS",
    "CampaignPlan",
    "PlanError",
    "Population",
    "build_plan",
    "check_plan",
    "digest_lines",
    "expected_keys",
    "load_population",
    "plan_sha256",
    "population_from",
]
