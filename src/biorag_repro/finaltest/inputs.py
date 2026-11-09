"""Digest-only scientific input contracts of one final-test batch.

A `ConditionRoster` names, for one condition of one matrix arm, every expected observation by
identities only: item and paper ids, the item content digest, the selected seed, the
representation-specific source identity, the context and ordered-evidence digests, the rendered
generator-input digest and the observation key. It holds no question text, gold answer,
reference evidence or retrieval context, so it can be stored and audited without opening
protected content. Real rosters are built from protected content in a later pass.

`BatchInputBinding` fixes the rosters, configuration, seeds and key set of one batch. It is a
description of what an authorized batch must contain, not an authorization: nothing here
publishes a registry event or starts anything.

Content identity is the SHA-256 of the stored canonical bytes (`layout.serialize`). Neither
record carries a timestamp, so equal inputs give an equal identity. Scientific coordinates are
checked against `FROZEN_MATRIX`, the verified campaign plan and `core.identity.observation_key`.
Problems are returned as `Finding` values naming ids and digests only.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest.batches import BatchManifest, Nonblank, Sha256, batch_id_for
from biorag_repro.finaltest.campaign import (
    PHASE,
    CampaignPlan,
    digest_lines,
    expected_keys,
    plan_sha256,
)
from biorag_repro.finaltest.layout import LAYOUT_VERSION, FrozenRecord, serialize
from biorag_repro.generation.attempts import RetryPolicy
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.seeds import MAX_SEED, GenerationSeedRegistry, repeat_seed

ObservationKey = Annotated[str, StringConstraints(pattern=r"^obs_[0-9a-f]{64}$")]
CLOSED_BOOK = "closed_book"

FindingKind = Literal["structure", "claim", "incomplete", "note"]


@dataclass(frozen=True)
class Finding:
    """One typed result of a verification. `subject` and `detail` carry ids and digests only."""

    kind: FindingKind
    code: str
    subject: str
    detail: str = ""


def ordered_digest(values: Sequence[str]) -> str:
    """Digest of an ordered list of identifiers (retrieval rank order matters)."""
    return sha256_bytes(canonical_json_bytes(list(values)))


def retry_policy_sha256(policy: RetryPolicy) -> str:
    return sha256_bytes(canonical_json_bytes(policy.model_dump(mode="json")))


class RosterIdentityRow(FrozenRecord):
    item_id: Nonblank
    paper_id: Nonblank
    item_sha256: Sha256
    matrix_arm: Nonblank
    condition: Nonblank
    repeat_index: int = Field(ge=1)  # scientific repeats are 1-based; 0 is never valid here
    resolved_config_sha256: Sha256
    seed: int | None = Field(default=None, ge=0, le=MAX_SEED)
    source_identity_sha256: Sha256 | None  # raw source of the representation; None: closed book
    context_sha256: Sha256 | None
    evidence_sequence_sha256: Sha256 | None  # ordered_digest of the retrieved evidence ids
    rendered_input_sha256: Sha256
    observation_key: ObservationKey

    @model_validator(mode="after")
    def inside_matrix(self) -> Self:
        if self.matrix_arm not in FROZEN_MATRIX:
            raise ValueError(f"{self.matrix_arm} is not a matrix arm")
        conditions, repeats = FROZEN_MATRIX[self.matrix_arm]
        if self.condition not in conditions:
            raise ValueError(f"{self.condition} is not a condition of {self.matrix_arm}")
        if self.repeat_index not in repeats:
            raise ValueError(
                f"repeat {self.repeat_index} is outside the matrix of {self.matrix_arm}"
            )
        retrieval = (
            self.source_identity_sha256,
            self.context_sha256,
            self.evidence_sequence_sha256,
        )
        if self.matrix_arm == CLOSED_BOOK:
            if any(v is not None for v in retrieval):
                raise ValueError("a closed-book observation has no source, context or evidence")
        elif any(v is None for v in retrieval):
            raise ValueError("a retrieval observation needs source, context and evidence identity")
        return self


class ConditionRoster(FrozenRecord):
    schema_version: Literal["1"] = "1"
    batch_id: Sha256
    matrix_arm: Nonblank
    condition: Nonblank
    benchmark_version: Nonblank
    resolved_config_sha256: Sha256
    rows: list[RosterIdentityRow]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if not self.rows:
            raise ValueError("a roster has at least one row")
        for row in self.rows:
            if (row.matrix_arm, row.condition, row.resolved_config_sha256) != (
                self.matrix_arm,
                self.condition,
                self.resolved_config_sha256,
            ):
                raise ValueError(
                    f"row {row.observation_key} is not of this arm, condition and configuration"
                )
        names = [(r.item_id, r.repeat_index) for r in self.rows]
        if names != sorted(set(names)):
            raise ValueError("roster rows must be unique and sorted by item and repeat")
        return self


def roster_sha256(roster: ConditionRoster) -> str:
    return sha256_bytes(serialize(roster))


class RosterRef(FrozenRecord):
    condition: Nonblank
    roster_sha256: Sha256
    row_count: int = Field(ge=1)


class RepeatSeed(FrozenRecord):
    repeat_index: int = Field(ge=1)
    seed: int = Field(ge=0, le=MAX_SEED)


class BatchInputBinding(FrozenRecord):
    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"] = LAYOUT_VERSION
    root_id: Sha256
    campaign_plan_sha256: Sha256
    manifest_sha256: Sha256
    batch_id: Sha256
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256
    benchmark_version: Nonblank
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256
    split_sha256: Sha256
    model_repository: Nonblank
    model_revision: Nonblank
    tokenizer_repository: Nonblank
    tokenizer_revision: Nonblank
    prompt_sha256: Sha256
    retrieval_config_sha256: Sha256 | None
    generation_seed_registry_sha256: Sha256
    seeds: list[RepeatSeed]  # empty unless decoding samples
    retry_policy_sha256: Sha256
    rosters: list[RosterRef]
    key_set_sha256: Sha256
    expected_count: int = Field(ge=1)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.matrix_arm not in FROZEN_MATRIX:
            raise ValueError(f"{self.matrix_arm} is not a matrix arm")
        conditions, repeats = FROZEN_MATRIX[self.matrix_arm]
        if [r.condition for r in self.rosters] != list(conditions):
            raise ValueError(
                "rosters must be the conditions of the arm, once each, in matrix order"
            )
        if sum(r.row_count for r in self.rosters) != self.expected_count:
            raise ValueError("row counts do not add up to the expected count")
        marks = [s.repeat_index for s in self.seeds]
        if marks != sorted(set(marks)) or not set(marks) <= set(repeats):
            raise ValueError("seeds are unique, ascending and inside the matrix repeats")
        if (self.matrix_arm == CLOSED_BOOK) != (self.retrieval_config_sha256 is None):
            raise ValueError(
                "a retrieval arm names its retrieval configuration; closed book has none"
            )
        if self.batch_id != batch_id_for(self.root_id, self.campaign_plan_sha256, self.matrix_arm):
            raise ValueError("batch_id does not match root, plan and arm")
        return self


def binding_sha256(binding: BatchInputBinding) -> str:
    """Identity of a binding: what a later citation records as `inputs_set_sha256`."""
    return sha256_bytes(serialize(binding))


def manifest_sha256(manifest: BatchManifest) -> str:
    return sha256_bytes(serialize(manifest))


def build_binding(
    *,
    manifest: BatchManifest,
    plan: CampaignPlan,
    resolved: ResolvedConfiguration[GeneratorConfiguration],
    retry: RetryPolicy,
    seeds: GenerationSeedRegistry | None,
    rosters: Sequence[ConditionRoster],
) -> BatchInputBinding:
    """Derive the binding of a batch from validated inputs. Pure; publishes nothing."""
    config = resolved.settings
    if resolved.configuration_sha256 != manifest.resolved_config_sha256:
        raise ValueError("the configuration is not the one the manifest names")
    repeats = FROZEN_MATRIX[manifest.matrix_arm][1]
    chosen = (
        [RepeatSeed(repeat_index=r, seed=repeat_seed(seeds, r)) for r in repeats]
        if config.decoding.do_sample and seeds is not None
        else []
    )
    keys = [row.observation_key for roster in rosters for row in roster.rows]
    return BatchInputBinding(
        root_id=manifest.root_id,
        campaign_plan_sha256=manifest.campaign_plan_sha256,
        manifest_sha256=manifest_sha256(manifest),
        batch_id=manifest.batch_id,
        matrix_arm=manifest.matrix_arm,
        resolved_config_sha256=manifest.resolved_config_sha256,
        benchmark_version=manifest.benchmark_version,
        corpus_content_sha256=manifest.corpus_content_sha256,
        benchmark_release_sha256=manifest.benchmark_release_sha256,
        split_sha256=plan.split_sha256,
        model_repository=config.model.repository,
        model_revision=config.model.revision,
        tokenizer_repository=config.tokenizer.repository,
        tokenizer_revision=config.tokenizer.revision,
        prompt_sha256=config.prompt_sha256,
        retrieval_config_sha256=config.retrieval_config_sha256,
        generation_seed_registry_sha256=manifest.generation_seed_registry_sha256,
        seeds=chosen,
        retry_policy_sha256=retry_policy_sha256(retry),
        rosters=[
            RosterRef(condition=r.condition, roster_sha256=roster_sha256(r), row_count=len(r.rows))
            for r in rosters
        ],
        key_set_sha256=digest_lines(keys),
        expected_count=len(keys),
    )


def _short(value: str) -> str:
    return value[:16]


def verify_binding(
    binding: BatchInputBinding,
    *,
    plan: CampaignPlan,
    manifest: BatchManifest,
    item_papers: Mapping[str, str],
    rosters: Mapping[str, ConditionRoster],
    resolved: ResolvedConfiguration[GeneratorConfiguration],
    retry: RetryPolicy,
    seeds: GenerationSeedRegistry | None,
) -> tuple[Finding, ...]:
    """Compare a binding and its rosters with the plan, manifest, configuration and seeds.

    An empty result means the records agree with each other and with the frozen matrix; it
    does not mean the batch is authorized or that the digests describe the real content.
    """
    out: list[Finding] = []

    def add(kind: FindingKind, code: str, subject: str, detail: str = "") -> None:
        out.append(Finding(kind, code, subject, detail))

    config = resolved.settings
    batch = binding.batch_id
    plan_batch = next((b for b in plan.batches if b.matrix_arm == binding.matrix_arm), None)
    if plan_batch is None or manifest.campaign_plan_sha256 != plan_sha256(plan):
        add("structure", "manifest_plan_mismatch", batch)
    elif (manifest.resolved_config_sha256, manifest.key_set_sha256, manifest.expected_count) != (
        plan_batch.resolved_config_sha256,
        plan_batch.key_set_sha256,
        plan_batch.expected_count,
    ):
        add("structure", "manifest_plan_mismatch", batch, "configuration, key set or count")
    pairs = {
        "root_id": (binding.root_id, manifest.root_id),
        "campaign_plan_sha256": (binding.campaign_plan_sha256, manifest.campaign_plan_sha256),
        "manifest_sha256": (binding.manifest_sha256, manifest_sha256(manifest)),
        "batch_id": (binding.batch_id, manifest.batch_id),
        "matrix_arm": (binding.matrix_arm, manifest.matrix_arm),
        "resolved_config_sha256": (binding.resolved_config_sha256, manifest.resolved_config_sha256),
        "benchmark_version": (binding.benchmark_version, manifest.benchmark_version),
        "corpus_content_sha256": (binding.corpus_content_sha256, manifest.corpus_content_sha256),
        "benchmark_release_sha256": (
            binding.benchmark_release_sha256,
            manifest.benchmark_release_sha256,
        ),
        "split_sha256": (binding.split_sha256, plan.split_sha256),
        "generation_seed_registry_sha256": (
            binding.generation_seed_registry_sha256,
            manifest.generation_seed_registry_sha256,
        ),
        "key_set_sha256": (binding.key_set_sha256, manifest.key_set_sha256),
        "expected_count": (binding.expected_count, manifest.expected_count),
        "model_repository": (binding.model_repository, config.model.repository),
        "model_revision": (binding.model_revision, config.model.revision),
        "tokenizer_repository": (binding.tokenizer_repository, config.tokenizer.repository),
        "tokenizer_revision": (binding.tokenizer_revision, config.tokenizer.revision),
        "prompt_sha256": (binding.prompt_sha256, config.prompt_sha256),
        "retrieval_config_sha256": (
            binding.retrieval_config_sha256,
            config.retrieval_config_sha256,
        ),
        "retry_policy_sha256": (binding.retry_policy_sha256, retry_policy_sha256(retry)),
        "matrix_arm_of_configuration": (binding.matrix_arm, config.matrix_arm),
        "configuration_identity": (binding.resolved_config_sha256, resolved.configuration_sha256),
    }
    for name, (bound, actual) in pairs.items():
        if bound != actual:
            add("structure", "binding_mismatch", batch, name)

    # selected seeds: the registry decides, never the roster
    if config.decoding.do_sample:
        try:
            wanted = (
                [(r, repeat_seed(seeds, r)) for r in FROZEN_MATRIX[binding.matrix_arm][1]]
                if seeds is not None
                else None
            )
        except ValueError:
            wanted = None
        if wanted is None or [(s.repeat_index, s.seed) for s in binding.seeds] != wanted:
            add("structure", "seed_mapping_mismatch", batch)
    elif binding.seeds:
        add("structure", "seed_mapping_mismatch", batch, "seeds bound for greedy decoding")
    seed_of = {s.repeat_index: s.seed for s in binding.seeds}

    for ref in binding.rosters:
        if ref.condition not in rosters:
            add("structure", "roster_missing", ref.condition)
    for condition in sorted(set(rosters) - {r.condition for r in binding.rosters}):
        add("structure", "roster_unreferenced", condition)

    keys: list[str] = []
    for ref in binding.rosters:
        roster = rosters.get(ref.condition)
        if roster is None:
            continue
        if roster_sha256(roster) != ref.roster_sha256:
            add("structure", "roster_hash_mismatch", ref.condition, _short(ref.roster_sha256))
        if len(roster.rows) != ref.row_count:
            add("structure", "roster_count_mismatch", ref.condition)
        if (
            roster.batch_id,
            roster.matrix_arm,
            roster.condition,
            roster.resolved_config_sha256,
        ) != (
            binding.batch_id,
            binding.matrix_arm,
            ref.condition,
            binding.resolved_config_sha256,
        ):
            add("structure", "roster_identity_mismatch", ref.condition)
        if roster.benchmark_version != binding.benchmark_version:
            add("structure", "roster_identity_mismatch", ref.condition, "benchmark_version")
        for row in roster.rows:
            keys.append(row.observation_key)
            paper = item_papers.get(row.item_id)
            if paper is None:
                add("structure", "foreign_item", row.item_id)
            elif paper != row.paper_id:
                add("structure", "paper_mismatch", row.item_id)
            recomputed = observation_key(
                phase=PHASE,
                benchmark_version=binding.benchmark_version,
                item_id=row.item_id,
                matrix_arm=row.matrix_arm,
                representation_or_condition=row.condition,
                repeat_index=row.repeat_index,
                resolved_config_sha256=binding.resolved_config_sha256,
            )
            if recomputed != row.observation_key:
                add("structure", "observation_key_mismatch", row.observation_key)
            if row.seed != seed_of.get(row.repeat_index):
                add("structure", "seed_mapping_mismatch", row.observation_key, "row seed")

    for key, count in sorted(Counter(keys).items()):
        if count > 1:
            add("structure", "duplicate_observation_key", key)
    expected = set(expected_keys(plan, sorted(item_papers), binding.matrix_arm))
    for key in sorted(set(keys) - expected):
        add("structure", "foreign_observation_key", key)
    for key in sorted(expected - set(keys)):
        add("structure", "missing_observation", key)
    if digest_lines(keys) != binding.key_set_sha256:
        add("structure", "key_set_digest_mismatch", batch)

    # the two representations of one item must stay distinguishable
    by_cell: dict[tuple[str, int], list[RosterIdentityRow]] = {}
    for roster in rosters.values():
        for row in roster.rows:
            by_cell.setdefault((row.item_id, row.repeat_index), []).append(row)
    for (item, repeat), cell in sorted(by_cell.items()):
        if len(cell) < 2:
            continue
        for field in ("source_identity_sha256", "context_sha256"):
            values = [getattr(r, field) for r in cell]
            if len(set(values)) != len(values):
                add("structure", "representations_not_distinguishable", f"{item}/{repeat}", field)
    return tuple(out)
