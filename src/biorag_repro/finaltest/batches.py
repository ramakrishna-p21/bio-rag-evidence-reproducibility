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

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Annotated, Final, Literal, Self

from pydantic import StringConstraints, field_validator, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.finaltest.campaign import CampaignPlan, plan_sha256
from biorag_repro.finaltest.continuity import ContinuityAssessment, continuity_assessment_sha256
from biorag_repro.finaltest.layout import LAYOUT_VERSION, FrozenRecord, serialize
from biorag_repro.generation.environment import check_utc
from biorag_repro.readiness.models import GitRevision

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


class CitationChainError(ValueError):
    """A sequence of authorization citations is not a valid history of one batch."""


class AuthorizationCitation(FrozenRecord):
    """One authorization assessment of a batch, as a historical record.

    Each assessment is its own immutable record. A resume cites a fresh readiness report, the
    previous citation and the continuity assessment that compared it with the first one; it
    never rewrites anything. The hash of a readiness report is evidence of an assessment at
    some time, not a reusable credential, so a report can be cited once, and nothing in this
    package publishes or accepts a citation as permission. `head_authority` records what was
    established about the registry head at that time and has no value for independent authority:
    none exists. Nothing in the repository publishes a citation yet.
    """

    schema_version: Literal["2"] = "2"
    kind: Literal["initial", "resume"]
    batch_id: Sha256
    initial_report_sha256: Sha256
    readiness_report_sha256: Sha256
    guard_scope: Literal["final_test_batch_start"] = GUARD_SCOPE
    policy_sha256: Sha256
    evaluator_code_sha256: Sha256
    report_binding_sha256: Sha256
    head_sha: GitRevision
    scientific_source_identity_sha256: Sha256
    scientific_identity_sha256: Sha256
    registry_head_sequence: int  # head the assessment was made against
    registry_head_event_sha256: Sha256
    head_authority: Literal["genesis_anchored", "not_established"]
    inputs_set_sha256: Sha256 | None  # None until the batch inputs are bound
    previous_citation_sha256: Sha256 | None
    continuity_assessment_sha256: Sha256 | None
    created_utc: str

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def chain(self) -> Self:
        initial = self.kind == "initial"
        if initial != (self.previous_citation_sha256 is None):
            raise ValueError("an initial citation has no predecessor and a resume has one")
        if initial != (self.continuity_assessment_sha256 is None):
            raise ValueError("only a resume cites a continuity assessment, and it must")
        if initial != (self.initial_report_sha256 == self.readiness_report_sha256):
            raise ValueError("an initial citation cites the original report; a resume its own")
        if self.registry_head_sequence < 1:
            raise ValueError("registry head sequence starts at 1")
        if (self.head_authority == "genesis_anchored") != (self.registry_head_sequence == 1):
            raise ValueError("only the genesis head is anchored")
        return self


def citation_sha256(citation: AuthorizationCitation) -> str:
    """Identity of a citation: SHA-256 of its stored canonical bytes."""
    return sha256_bytes(serialize(citation))


def verify_citation_chain(
    citations: Sequence[AuthorizationCitation],
    assessments: Mapping[str, ContinuityAssessment],
) -> None:
    """Check that `citations` are, in order, the authorization history of one batch.

    `assessments` maps the cited identity of each continuity assessment to the stored record.
    Raises `CitationChainError`. This verifies a record; it grants nothing.
    """
    if not citations or citations[0].kind != "initial":
        raise CitationChainError("a history starts with the initial citation")
    first = citations[0]
    reports = {first.readiness_report_sha256}
    for before, citation in pairwise(citations):
        if citation.kind != "resume":
            raise CitationChainError("only the first citation is initial")
        if citation.previous_citation_sha256 != citation_sha256(before):
            raise CitationChainError("a citation does not name its predecessor")
        if citation.batch_id != first.batch_id:
            raise CitationChainError("a citation belongs to another batch")
        if citation.initial_report_sha256 != first.initial_report_sha256:
            raise CitationChainError("the original report identity changed")
        if citation.readiness_report_sha256 in reports:
            raise CitationChainError("a readiness report is cited twice")
        reports.add(citation.readiness_report_sha256)
        if citation.registry_head_sequence <= before.registry_head_sequence:
            raise CitationChainError("the registry head did not move forward")
        if citation.scientific_identity_sha256 != first.scientific_identity_sha256:
            raise CitationChainError("the scientific identity changed")
        assessment = assessments.get(citation.continuity_assessment_sha256 or "")
        if assessment is None or continuity_assessment_sha256(assessment) != (
            citation.continuity_assessment_sha256
        ):
            raise CitationChainError("the cited continuity assessment is missing or differs")
        if assessment.verdict != "continuous":
            raise CitationChainError("the cited continuity assessment is not continuous")
        # a continuous assessment has equal identities, and the citation's equals the first's
        if (
            assessment.later_scientific_identity_sha256 != citation.scientific_identity_sha256
            or assessment.later_binding_sha256 != citation.report_binding_sha256
        ):
            raise CitationChainError("the cited continuity assessment is about other reports")
