"""Typed Stage B (sufficient evidence delivery) candidate and judgment records."""

from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.retrieval.models import ConfigurationRef, Representation

# Which papers contributed evidence to the final supplied context. This says
# nothing about why delivery failed.
Composition = Literal["designated_only", "other_only", "mixed", "empty"]

# Earliest pipeline stage at which every alternative evidence set was lost.
Reason = Literal[
    "excluded_in_chunking",
    "not_ranked",
    "lost_in_packing",
    "partial_components",
    "other_paper_only",
    "empty_context",
]
Determination = Literal[
    "automatic_delivered",
    "automatic_not_delivered",
    "needs_review",
    "not_reachable_stage_a",
    "non_evaluable_stage_a",
]
Status = Literal["delivered", "not_delivered", "not_reachable_stage_a", "non_evaluable"]
Outcome = Literal["delivered", "review", "lost"]


def composition_of(designated: int, other: int) -> Composition:
    if designated and other:
        return "mixed"
    if designated:
        return "designated_only"
    return "other_only" if other else "empty"


class ComponentSpan(Record):
    """Where a component's text sits in the exact supplied context."""

    role: str
    text: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class Delivered(Record):
    """One placement whose evidence is inspectable in the supplied context."""

    set_index: int = Field(ge=0)
    container_node_id: str
    chunk_id: str
    rank: int = Field(ge=1, le=20)
    spans: list[ComponentSpan]


class Diagnosis(Record):
    """Per-placement outcome; alternative sets and placements are kept separate."""

    set_index: int = Field(ge=0)
    container_node_id: str | None = None
    outcome: Outcome
    reason: Reason | None = None
    detail: str

    @model_validator(mode="after")
    def valid_outcome(self) -> Self:
        if (self.outcome == "lost") != (self.reason is not None):
            raise ValueError("only a lost placement carries a loss reason")
        return self


class StageBIdentity(Record):
    schema_version: Literal["1"] = "1"
    benchmark_version: str
    corpus_version: str
    item_id: str
    item_sha256: str
    paper_id: str
    representation: Representation
    stage_a: ArtifactRef
    context: ArtifactRef
    retrieval: ArtifactRef
    chunk_set: ArtifactRef
    configuration: ConfigurationRef
    composition: Composition
    designated_chunks_included: int = Field(ge=0)
    other_chunks_included: int = Field(ge=0)

    @field_validator("item_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @model_validator(mode="after")
    def consistent_composition(self) -> Self:
        if self.composition != composition_of(
            self.designated_chunks_included, self.other_chunks_included
        ):
            raise ValueError("composition differs from the included chunk counts")
        return self


class StageBCandidate(StageBIdentity):
    """Inspectable result of the evaluator; scientific uncertainty stays `needs_review`."""

    determination: Determination
    reason: Reason | None = None
    delivered: list[Delivered]
    diagnoses: list[Diagnosis]

    @model_validator(mode="after")
    def valid_determination(self) -> Self:
        if (self.determination == "automatic_delivered") != bool(self.delivered):
            raise ValueError("automatic delivery requires, and only allows, delivered evidence")
        if (self.determination == "automatic_not_delivered") != (self.reason is not None):
            raise ValueError("a loss reason belongs to automatic non-delivery only")
        if self.determination == "automatic_delivered" and not any(
            d.outcome == "delivered" for d in self.diagnoses
        ):
            raise ValueError("automatic delivery requires a delivered diagnosis")
        return self


class StageBJudgment(StageBIdentity):
    status: Status
    basis: Literal["deterministic", "reviewer"]
    candidate: ArtifactRef
    reason: Reason | None = None
    delivered: list[Delivered]
    rationale: str | None = None
    reviewer: str | None = None
    at_utc: str | None = None

    @model_validator(mode="after")
    def valid_judgment(self) -> Self:
        if (self.status == "delivered") != bool(self.delivered):
            raise ValueError("delivered status requires, and only allows, cited evidence")
        if (self.status == "not_delivered") != (self.reason is not None):
            raise ValueError("a loss reason belongs to the not_delivered status only")
        if self.basis == "deterministic":
            if any((self.reviewer, self.at_utc, self.rationale)):
                raise ValueError("deterministic basis carries no reviewer fields")
            return self
        if self.status not in {"delivered", "not_delivered"}:
            raise ValueError("a reviewer decides delivery only; Stage A states are not reviewable")
        if not (self.reviewer and self.reviewer.strip()):
            raise ValueError("reviewer judgment requires a reviewer")
        if not (self.rationale and self.rationale.strip()):
            raise ValueError("reviewer judgment requires a rationale")
        if self.at_utc is None or not self.at_utc.endswith("Z"):
            raise ValueError("review timestamp must be UTC with Z suffix")
        if datetime.fromisoformat(self.at_utc).utcoffset() != timedelta(0):
            raise ValueError("review timestamp must be UTC with Z suffix")
        return self
