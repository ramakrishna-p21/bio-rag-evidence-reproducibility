"""Typed generation-result input and Stage C score records."""

from datetime import datetime, timedelta
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import AnswerClass, Record
from biorag_repro.core.hashing import sha256_bytes, validate_sha256
from biorag_repro.core.manifest import ArtifactRef

Status = Literal["correct", "incorrect", "missing", "non_evaluable"]

# Every reason belongs to exactly one status; a correct score has none.
REASONS: dict[str, set[str]] = {
    "incorrect": {
        "abstained",
        "unparseable",
        "empty_answer",
        "answer_shape_mismatch",
        "no_numeric_value",
        "value_mismatch",
        "comparator_mismatch",
        "incompatible_units",
        "entity_not_approved",
        "set_incomplete",
        "set_extra_member",
        "adjudicated_incorrect",
    },
    "missing": {"operational_failure"},
    "non_evaluable": {
        "unknown_unit_conversion",
        "unit_missing",
        "unit_not_in_gold",
        "uninterpretable_numeric_text",
        "case_only_difference",
        "approved_name_in_longer_text",
        "wording_not_approved",
        "set_segmentation_ambiguous",
        "gold_unusable",
    },
}
# A reviewer may settle a scientific scoring ambiguity, never an unusable gold.
ADJUDICABLE = REASONS["non_evaluable"] - {"gold_unusable"}


def utc_stamp(value: str | None) -> str:
    if value is None or not value.endswith("Z"):
        raise ValueError("review timestamp must be UTC with Z suffix")
    if datetime.fromisoformat(value).utcoffset() != timedelta(0):
        raise ValueError("review timestamp must be UTC with Z suffix")
    return value


class GenerationResult(Record):
    """Minimal Stage C view of one generator attempt; Phase 7 owns the full output schema."""

    schema_version: Literal["1"] = "1"
    phase: str = Field(min_length=1)
    benchmark_version: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    matrix_arm: str = Field(min_length=1)
    representation_or_condition: str = Field(min_length=1)
    repeat_index: int = Field(ge=0)
    resolved_config_sha256: str
    observation_key: str
    run_id: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    status: Literal["completed", "operational_failure"]
    output: ArtifactRef | None = None
    raw_output: str | None = None
    parse_status: Literal["parsed", "unparseable"] | None = None
    answer: str | list[str] | None = None
    abstained: bool | None = None
    failure: str | None = None

    @field_validator("resolved_config_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @model_validator(mode="after")
    def valid_status(self) -> Self:
        outcome = (self.output, self.raw_output, self.parse_status, self.answer, self.abstained)
        if self.status == "operational_failure":
            if any(v is not None for v in outcome):
                raise ValueError("an operational failure has no output; partial output is not kept")
            if not (self.failure and self.failure.strip()):
                raise ValueError("an operational failure needs a description")
            return self
        if self.failure is not None:
            raise ValueError("a completed generation has no failure description")
        if self.output is None or self.raw_output is None or self.parse_status is None:
            raise ValueError("a completed generation needs its output artifact and raw output")
        if sha256_bytes(self.raw_output.encode("utf-8")) != self.output.sha256:
            raise ValueError("raw output does not match the generator output artifact hash")
        if self.parse_status == "unparseable":
            if self.answer is not None or self.abstained is not None:
                raise ValueError("an unparseable output carries no parsed answer")
        elif self.abstained is None or (not self.abstained and self.answer is None):
            raise ValueError(
                "a parsed output needs an abstention indicator and, unless abstaining, an answer"
            )
        return self


class StageCScore(Record):
    schema_version: Literal["1"] = "1"
    benchmark_version: str
    corpus_version: str
    item_id: str
    item_sha256: str
    paper_id: str
    answer_class: AnswerClass
    observation_key: str
    repeat_index: int = Field(ge=0)
    attempt_id: str
    generation_sha256: str
    output: ArtifactRef | None = None
    # Omitted when "v1" so that v1 records keep their published bytes and identities.
    scoring_rules: Literal["v1", "v2"] = Field("v1", exclude_if=lambda rules: rules == "v1")
    status: Status
    basis: Literal["deterministic", "reviewer"]
    reason: str | None = None
    detail: str
    interpreted: list[str] = Field(default_factory=list)
    automatic: ArtifactRef | None = None
    rationale: str | None = None
    reviewer: str | None = None
    at_utc: str | None = None

    @field_validator("item_sha256", "generation_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @model_validator(mode="after")
    def valid_score(self) -> Self:
        if (self.status == "correct") != (self.reason is None) or (
            self.reason is not None and self.reason not in REASONS[self.status]
        ):
            raise ValueError("reason does not belong to the status")
        if self.basis == "deterministic":
            if any((self.automatic, self.rationale, self.reviewer, self.at_utc)):
                raise ValueError("deterministic basis carries no reviewer fields")
            if self.reason == "adjudicated_incorrect":
                raise ValueError("only a reviewer can adjudicate")
            return self
        if self.status not in {"correct", "incorrect"}:
            raise ValueError("a reviewer decides correct or incorrect only")
        if self.automatic is None:
            raise ValueError("reviewer judgment cites the automatic score it resolves")
        if not (self.reviewer and self.reviewer.strip()):
            raise ValueError("reviewer judgment requires a reviewer")
        if not (self.rationale and self.rationale.strip()):
            raise ValueError("reviewer judgment requires a rationale")
        utc_stamp(self.at_utc)
        if self.status == "incorrect" and self.reason != "adjudicated_incorrect":
            raise ValueError("an adjudicated incorrect score carries its own reason")
        return self
