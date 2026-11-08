"""Structural batch records of the final-test root.

A batch is one matrix arm: the resolved generator configuration names exactly one arm and
carries no condition (`generation.config.GeneratorConfiguration`), so the JATS and PDF_DOCLING
conditions of an arm share one configuration identity and one future RunStore.

`BatchManifest` is immutable and structural: it records what the batch is expected to contain
and nothing that happens to it later. Roster and context binding, authorization and resume
assessments are separate append-only records written by later passes; none of them is a field
here, so no later step needs to modify a published manifest. A reservation is not an
authorization, and `AuthorizationCitation` is only the shape of the evidence a later pass cites.
"""

from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.finaltest.campaign import CampaignPlan, plan_sha256
from biorag_repro.finaltest.layout import LAYOUT_VERSION, FrozenRecord
from biorag_repro.generation.environment import check_utc

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
GUARD_SCOPE: Final = "final_test_batch_start"  # readiness.guard.SCOPE


def batch_id_for(root_id: str, campaign_plan_sha256: str, matrix_arm: str) -> str:
    """One batch per matrix arm per campaign, so the identity has no ordinal or retry part."""
    return sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "final_test_batch",
                "root_id": root_id,
                "campaign_plan_sha256": campaign_plan_sha256,
                "matrix_arm": matrix_arm,
            }
        )
    )


def ownership_sha256(benchmark_release_sha256: str, matrix_arm: str, config_sha256: str) -> str:
    """Claim on a set of observation keys.

    Keys are a function of benchmark, arm and configuration over the whole final-test
    population, so two batches with the same claim would own the same keys.
    """
    return sha256_bytes(
        canonical_json_bytes(
            {
                "kind": "final_test_key_ownership",
                "benchmark_release_sha256": benchmark_release_sha256,
                "matrix_arm": matrix_arm,
                "resolved_config_sha256": config_sha256,
            }
        )
    )


def observed_state_sha256(manifest_sha256: str | None, materialized: bool) -> str:
    return sha256_bytes(
        canonical_json_bytes({"manifest_sha256": manifest_sha256, "materialized": materialized})
    )


class BatchManifest(FrozenRecord):
    schema_version: Literal["1"] = "1"
    layout_version: Literal["final_test_v1"] = LAYOUT_VERSION
    root_id: Sha256
    campaign_plan_sha256: Sha256
    batch_id: Sha256
    matrix_arm: Nonblank
    resolved_config_sha256: Sha256
    corpus_content_sha256: Sha256
    benchmark_release_sha256: Sha256
    benchmark_version: Nonblank
    generation_seed_registry_sha256: Sha256
    expected_count: int
    key_set_sha256: Sha256
    created_utc: str  # informational

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def identity(self) -> Self:
        if self.batch_id != batch_id_for(self.root_id, self.campaign_plan_sha256, self.matrix_arm):
            raise ValueError("batch_id does not match root, plan and arm")
        if self.expected_count < 1:
            raise ValueError("a batch expects at least one observation")
        return self

    @property
    def ownership(self) -> str:
        return ownership_sha256(
            self.benchmark_release_sha256, self.matrix_arm, self.resolved_config_sha256
        )


def make_manifest(
    root_id: str, plan: CampaignPlan, matrix_arm: str, created_utc: str
) -> BatchManifest:
    batch = next(b for b in plan.batches if b.matrix_arm == matrix_arm)
    plan_sha = plan_sha256(plan)
    return BatchManifest(
        root_id=root_id,
        campaign_plan_sha256=plan_sha,
        batch_id=batch_id_for(root_id, plan_sha, matrix_arm),
        matrix_arm=matrix_arm,
        resolved_config_sha256=batch.resolved_config_sha256,
        corpus_content_sha256=plan.corpus_content_sha256,
        benchmark_release_sha256=plan.benchmark_release_sha256,
        benchmark_version=plan.benchmark_version,
        generation_seed_registry_sha256=plan.generation_seed_registry_sha256,
        expected_count=batch.expected_count,
        key_set_sha256=batch.key_set_sha256,
        created_utc=created_utc,
    )


class AuthorizationCitation(FrozenRecord):
    """Shape of one authorization assessment that a later pass will record separately.

    Each assessment is its own immutable record; a resume cites a fresh readiness report and
    the previous citation and never rewrites it. The hash of a readiness report is evidence of
    an assessment at some time, not a reusable credential, and nothing in P10-B can publish or
    accept a citation as permission.
    """

    schema_version: Literal["1"] = "1"
    kind: Literal["initial", "resume"]
    batch_id: Sha256
    readiness_report_sha256: Sha256
    guard_scope: Literal["final_test_batch_start"] = GUARD_SCOPE
    registry_head_sequence: int  # head the assessment was made against
    registry_head_event_sha256: Sha256
    previous_citation_sha256: Sha256 | None
    created_utc: str

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def chain(self) -> Self:
        if (self.kind == "initial") != (self.previous_citation_sha256 is None):
            raise ValueError("an initial citation has no predecessor and a resume has one")
        if self.registry_head_sequence < 1:
            raise ValueError("registry head sequence starts at 1")
        return self
