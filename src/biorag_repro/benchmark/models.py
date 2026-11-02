"""Typed source-grounded benchmark and adjudication records."""

import re
from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from biorag_repro.core.hashing import validate_sha256

AnswerClass = Literal[
    "quantitative_value",
    "biological_entity",
    "dataset_cohort",
    "bounded_entity_set",
    "categorical_conclusion",
]
SourceStatus = Literal["present_equivalent", "absent", "materially_different", "unresolved"]


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)


class QuantityGold(Record):
    answer_class: Literal["quantitative_value"] = "quantitative_value"
    value: str
    comparator: Literal["eq", "lt", "le", "gt", "ge"]
    unit: str | None = None
    reported_precision: str | None = None
    rounding_interval: list[str] | None = None
    approved_equivalents: list[str] = Field(
        default_factory=list, exclude_if=lambda value: not value
    )

    @field_validator("value")
    @classmethod
    def decimal_value(cls, value: str) -> str:
        try:
            number = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError("quantity must be an ASCII decimal or scientific notation") from exc
        if not number.is_finite() or not re.fullmatch(
            r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value
        ):
            raise ValueError("quantity must be a finite ASCII decimal or scientific notation")
        return value

    @model_validator(mode="after")
    def valid_rounding(self) -> Self:
        if self.approved_equivalents:
            if self.comparator != "eq":
                raise ValueError("numeric equivalents require equality")
            values = [self.value, *self.approved_equivalents]
            for value in self.approved_equivalents:
                self.decimal_value(value)
            if len({Decimal(value) for value in values}) != len(values):
                raise ValueError("numeric equivalents must differ from primary value")
        if self.rounding_interval is not None:
            if len(self.rounding_interval) != 2:
                raise ValueError("rounding interval needs two endpoints")
            if self.comparator != "eq" or not self.reported_precision:
                raise ValueError("rounding interval requires equality and reported precision")
            low, high = (Decimal(value) for value in self.rounding_interval)
            if not low.is_finite() or not high.is_finite() or low > high:
                raise ValueError("invalid rounding interval")
        return self


class SetMember(Record):
    answer_class: Literal["biological_entity", "dataset_cohort", "feature_or_process"]
    canonical: str
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_names(self) -> Self:
        names = [self.canonical, *self.aliases]
        if any(not name.strip() for name in names) or len(
            {name.casefold() for name in names}
        ) != len(names):
            raise ValueError("canonical referent and aliases must be nonblank and distinct")
        return self


class EntityGold(SetMember):
    answer_class: Literal["biological_entity", "dataset_cohort"]


class SetGold(Record):
    answer_class: Literal["bounded_entity_set"] = "bounded_entity_set"
    members: list[SetMember] = Field(min_length=1)
    inclusion_criterion: str | None = None

    @model_validator(mode="after")
    def distinct_members(self) -> Self:
        names = [
            name.casefold() for item in self.members for name in [item.canonical, *item.aliases]
        ]
        if len(set(names)) != len(names):
            raise ValueError("bounded set member names and aliases must be distinct")
        return self

    def resolve_member(self, name: str) -> str | None:
        for member in self.members:
            if name == member.canonical or name in member.aliases:
                return member.canonical
        return None


class ConclusionGold(Record):
    answer_class: Literal["categorical_conclusion"] = "categorical_conclusion"
    subject: str
    relation: str
    direction: str
    comparison: str | None = None
    condition: str | None = None
    equivalent_wording: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_proposition(self) -> Self:
        if any(not value.strip() for value in (self.subject, self.relation, self.direction)):
            raise ValueError("conclusion requires subject, relation, and direction")
        return self


Gold = Annotated[
    QuantityGold | EntityGold | SetGold | ConclusionGold, Field(discriminator="answer_class")
]


class EvidenceComponent(Record):
    role: Literal[
        "subject",
        "relation",
        "answer",
        "condition",
        "comparison",
        "unit",
        "direction",
        "qualifier",
        "row",
        "column",
    ]
    text: str

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence component must be nonblank")
        return value


class EvidenceSet(Record):
    components: list[EvidenceComponent] = Field(min_length=3)

    @model_validator(mode="after")
    def complete_core(self) -> Self:
        roles = {component.role for component in self.components}
        if not {"subject", "relation", "answer"} <= roles:
            raise ValueError("evidence requires subject, relation, and answer")
        return self


class SourceRef(Record):
    artifact_id: str
    source_sha256: str
    paper_id: str
    source_text: str
    section: str | None = None
    table_id: str | None = None
    row_label: str | None = None
    column_label: str | None = None

    @field_validator("source_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @model_validator(mode="after")
    def valid_context(self) -> Self:
        if (
            not self.artifact_id.strip()
            or not self.paper_id.strip()
            or not self.source_text.strip()
        ):
            raise ValueError("source reference requires artifact, paper, and text")
        if self.table_id and (not self.row_label or not self.column_label):
            raise ValueError("table reference requires row and column labels")
        return self


class JatsRef(SourceRef):
    xml_path: str

    @field_validator("xml_path")
    @classmethod
    def path_shape(cls, value: str) -> str:
        if not value.startswith("/article/"):
            raise ValueError("JATS path must start within article")
        return value


class PdfRef(SourceRef):
    page: int = Field(ge=1)
    region: list[float] | None = None

    @model_validator(mode="after")
    def valid_region(self) -> Self:
        if self.region is not None:
            if len(self.region) != 4:
                raise ValueError("PDF region needs four coordinates")
            x0, y0, x1, y1 = self.region
            if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
                raise ValueError("PDF region must have positive bounds")
        return self


class BenchmarkItem(Record):
    schema_version: Literal["1"]
    benchmark_version: str
    corpus_version: str
    item_id: str
    paper_id: str
    question: str
    answer_class: AnswerClass
    subtype: str | None = None
    gold: Gold
    evidence_sets: list[EvidenceSet] = Field(min_length=1)
    jats_refs: list[JatsRef] = Field(min_length=1)
    pdf_refs: list[PdfRef] = Field(min_length=1)
    jats_status: SourceStatus
    pdf_status: SourceStatus
    split: Literal["development", "final_test"] | None = None
    candidate_status: Literal["lead", "proposed", "accepted", "excluded"]
    review_status: Literal["pending", "reviewed"]
    audit_status: Literal["not_selected", "selected_pending", "audited"]
    decision_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def item_consistency(self) -> Self:
        if self.gold.answer_class != self.answer_class:
            raise ValueError("gold class differs from item class")
        if any(not value.strip() for value in (self.item_id, self.paper_id, self.question)):
            raise ValueError("item identity and question must be nonblank")
        if any(ref.paper_id != self.paper_id for ref in [*self.jats_refs, *self.pdf_refs]):
            raise ValueError("source reference paper differs from item")
        if self.candidate_status == "accepted":
            if self.jats_status != "present_equivalent" or self.pdf_status != "present_equivalent":
                raise ValueError("accepted item requires paired equivalent sources")
            if self.review_status != "reviewed" or self.split is None or not self.decision_ids:
                raise ValueError("accepted item requires review, split, and decision provenance")
        if self.candidate_status == "excluded" and (
            self.review_status != "reviewed" or not self.decision_ids
        ):
            raise ValueError("excluded item requires review and decision provenance")
        return self
