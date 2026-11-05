"""Typed Stage A candidate and judgment records."""

from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.retrieval.models import Representation

Label = Literal[
    "preserved",
    "altered_but_sufficient",
    "scientifically_corrupted",
    "absent",
    "unresolved",
]
SuggestionKind = Literal[
    "component_not_found",
    "split_across_contexts",
    "association_mismatch",
    "qualified_mapping",
    "contained_in_longer_token",
    "sign_difference",
    "numeric_digit_sequence_match",
    "incomplete_canonical",
]


class ComponentLocus(Record):
    """Canonical text occurrences carrying one benchmark evidence component."""

    role: str
    text: str
    occurrence_ids: list[str] = Field(min_length=1)
    observed_text: str | None = None  # None: the benchmark text occurs verbatim

    @model_validator(mode="after")
    def valid_observation(self) -> Self:
        if self.observed_text is not None and (
            not self.observed_text.strip() or self.observed_text == self.text
        ):
            raise ValueError("observed text must be nonblank and differ from the component")
        if len(set(self.occurrence_ids)) != len(self.occurrence_ids):
            raise ValueError("duplicate occurrence in component locus")
        return self


class Placement(Record):
    """One canonical context (block or table) that holds a complete evidence set."""

    container_node_id: str
    components: list[ComponentLocus] = Field(min_length=1)


class SetLocus(Record):
    """Placements of one benchmark evidence set; alternative sets stay separate."""

    set_index: int = Field(ge=0)
    placements: list[Placement] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_containers(self) -> Self:
        ids = [p.container_node_id for p in self.placements]
        if len(set(ids)) != len(ids):
            raise ValueError("one container cannot hold two placements of a set")
        return self


class Suggestion(Record):
    """Advisory evidence for a reviewer; never a Stage A label."""

    kind: SuggestionKind
    set_index: int | None = Field(default=None, ge=0)
    role: str | None = None
    detail: str
    occurrence_ids: list[str] = Field(default_factory=list)


class StageAIdentity(Record):
    schema_version: Literal["1"] = "1"
    benchmark_version: str
    corpus_version: str
    item_id: str
    item_sha256: str
    paper_id: str
    representation: Representation
    canonical: ArtifactRef

    @field_validator("item_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value


def _unique_sets(sets: list[SetLocus]) -> None:
    indexes = [s.set_index for s in sets]
    if len(set(indexes)) != len(indexes):
        raise ValueError("duplicate evidence-set locus")


class StageACandidate(StageAIdentity):
    """Inspectable output of the locator: pending review unless strictly preserved."""

    determination: Literal["automatic_preserved", "needs_review"]
    sets: list[SetLocus]
    suggestions: list[Suggestion]
    qualifications: list[str]

    @model_validator(mode="after")
    def valid_determination(self) -> Self:
        _unique_sets(self.sets)
        if (self.determination == "automatic_preserved") != bool(self.sets):
            raise ValueError("automatic preservation requires located sets and nothing else")
        if self.determination == "automatic_preserved" and any(
            c.observed_text is not None
            for s in self.sets
            for p in s.placements
            for c in p.components
        ):
            raise ValueError("automatic preservation requires verbatim components")
        return self


class StageAJudgment(StageAIdentity):
    label: Label
    basis: Literal["deterministic", "reviewer"]
    candidate: ArtifactRef | None = None
    sets: list[SetLocus]
    alteration: str | None = None
    rationale: str | None = None
    reviewer: str | None = None
    at_utc: str | None = None

    @model_validator(mode="after")
    def valid_judgment(self) -> Self:
        _unique_sets(self.sets)
        if self.basis == "deterministic":
            if (
                self.label != "preserved"
                or self.candidate is None
                or any((self.reviewer, self.at_utc, self.rationale, self.alteration))
            ):
                raise ValueError(
                    "deterministic basis is only a candidate-backed preserved label "
                    "without reviewer fields"
                )
        else:
            if not (self.reviewer and self.reviewer.strip()):
                raise ValueError("reviewer judgment requires a reviewer")
            if not (self.rationale and self.rationale.strip()):
                raise ValueError("reviewer judgment requires a rationale")
            if self.at_utc is None or not self.at_utc.endswith("Z"):
                raise ValueError("review timestamp must be UTC with Z suffix")
            if datetime.fromisoformat(self.at_utc).utcoffset() != timedelta(0):
                raise ValueError("review timestamp must be UTC with Z suffix")
        if self.label in {"altered_but_sufficient", "scientifically_corrupted"} and not (
            self.alteration and self.alteration.strip()
        ):
            raise ValueError("altered or corrupted evidence requires the observed alteration")
        if self.label in {"preserved", "altered_but_sufficient"} and not self.sets:
            raise ValueError("a usable fact requires located evidence")
        if self.label == "absent" and self.sets:
            raise ValueError("absent evidence has no located occurrences")
        return self
