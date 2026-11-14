"""Typed records for the item-level analysis table handed from Python to R.

Row grain: one planned observation, identified by phase, matrix arm, item, condition
and repeat. Each row carries both co-primary outcomes (evidence delivered, answer
correct) with separate evaluability, the earliest failure, and references to the
validated Stage A/B/C records it was derived from.
"""

from typing import Annotated, Final, Literal, Self, get_args, get_origin

from pydantic import Field, StringConstraints, field_validator, model_validator

from biorag_repro.benchmark.models import AnswerClass
from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.evaluation.attribution import (
    A_FAILED,
    Correct,
    Delivered,
    Earliest,
    EvidenceReason,
    Execution,
    Tri,
    attribute,
)

SCHEMA_VERSION = "1"
PRIMARY_REPEAT_INDEX = 1  # EXPERIMENT_PLAN section 6: repeat 1 is the sole primary observation
MISSING = "NA"  # the CSV missing token; in nullable columns only. The non-nullable tri-state
# column correct_without_evidence also has NA as a data value ("not evaluable"); it is never null.

Phase = Literal["development", "qualification", "final_test"]
MatrixArm = Literal[
    "confirmatory", "dense_robustness", "chunk_robustness", "scope_robustness", "closed_book"
]
Condition = Literal["JATS", "PDF_DOCLING", "CLOSED_BOOK"]
Role = Literal["development", "confirmatory", "secondary"]
ContextProvenance = Literal[
    "not_applicable", "assembled_only", "generator_input_simulated", "generator_input_recorded"
]
Completion = Literal["answer", "abstained", "unparseable"]
BStatus = Literal[
    "delivered", "not_delivered", "not_reachable_stage_a", "non_evaluable", "unresolved"
]
Token = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:@+\-]*$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class GeneratorInput(Record):
    """Claim that one attempt received the exact context of a given ContextRecord.

    Assembling a context (Phase 5) is not proof it was sent to a generator. This link
    is the only thing that upgrades the provenance of a row, and a simulated link is
    never recorded as real delivery.
    """

    schema_version: Literal["1"] = "1"
    observation_key: str = Field(min_length=1)
    attempt_id: str = Field(min_length=1)
    context: ArtifactRef
    exact_context_sha256: str
    input: ArtifactRef
    provenance: Literal["recorded", "simulated"]

    @field_validator("exact_context_sha256")
    @classmethod
    def hash_shape(cls, value: str) -> str:
        validate_sha256(value)
        return value


class AnalysisRow(Record):
    benchmark_version: Token
    corpus_version: Token
    item_id: Token
    item_sha256: Sha256
    paper_id: Token
    answer_class: AnswerClass
    split: Literal["development", "final_test"]
    phase: Phase
    analysis_role: Role
    matrix_arm: MatrixArm
    condition: Condition
    repeat_index: int = Field(ge=1, le=3)
    primary_row: bool
    observation_key: Token
    config_sha256: Sha256
    retrieval_config_sha256: Sha256 | None
    run_id: Token | None
    attempt_id: Token | None
    execution_status: Execution
    completion_kind: Completion | None
    stage_a_label: (
        Literal[
            "preserved",
            "altered_but_sufficient",
            "scientifically_corrupted",
            "absent",
            "unresolved",
        ]
        | None
    )
    stage_a_artifact: Token | None
    stage_b_status: BStatus | None
    stage_b_artifact: Token | None
    context_artifact: Token | None
    context_composition: Literal["designated_only", "other_only", "mixed", "empty"] | None
    context_provenance: ContextProvenance
    generator_input_artifact: Token | None
    evidence_delivered: Delivered
    evidence_reason: EvidenceReason | None
    stage_c_status: Literal["correct", "incorrect", "missing", "non_evaluable"] | None
    stage_c_reason: Token | None
    stage_c_basis: Literal["deterministic", "reviewer"] | None
    stage_c_artifact: Token | None
    answer_correct: Correct
    earliest_failure: Earliest
    correct_without_evidence: Tri

    @field_validator("run_id", "attempt_id", "stage_c_reason", mode="after")
    @classmethod
    def not_the_missing_token(cls, value: str | None) -> str | None:
        if value == MISSING:
            raise ValueError("NA is reserved for missing values")
        return value

    @model_validator(mode="after")
    def valid_row(self) -> Self:
        closed = self.matrix_arm == "closed_book"
        if closed != (self.condition == "CLOSED_BOOK"):
            raise ValueError("closed-book arm and condition must coincide")
        if self.repeat_index > 1 and self.matrix_arm not in {"confirmatory", "closed_book"}:
            raise ValueError("robustness arms have one generation")
        if (self.retrieval_config_sha256 is None) != closed:
            raise ValueError(
                "a retrieval row names its retrieval configuration; closed-book has none"
            )
        if self.primary_row != (self.repeat_index == PRIMARY_REPEAT_INDEX):
            raise ValueError("only the designated repeat is primary")
        expected_split = "final_test" if self.phase == "final_test" else "development"
        if self.split != expected_split:
            raise ValueError("phase and benchmark split disagree")
        if self.analysis_role != analysis_role(self.phase, self.matrix_arm):
            raise ValueError("analysis role does not follow phase and matrix arm")

        started = self.execution_status != "planned_not_executed"
        if (self.run_id is not None) != started or (self.attempt_id is not None) != started:
            raise ValueError("an attempt exists exactly when the observation was executed")
        if (self.completion_kind is not None) != (self.execution_status == "completed"):
            raise ValueError("completion kind belongs to completed executions only")
        if (self.stage_c_status is not None) != started:
            raise ValueError("an answer score exists exactly when the observation was executed")
        scored = self.stage_c_status is not None
        if (self.stage_c_artifact is not None, self.stage_c_basis is not None) != (scored, scored):
            raise ValueError("score reference and basis belong to a scored observation")
        if (self.stage_c_status == "correct") != (scored and self.stage_c_reason is None):
            raise ValueError("every score except correct carries a reason")

        self._check_evidence_fields(closed, started)
        out = attribute(
            closed_book=closed,
            stage_a=self.stage_a_label,
            stage_b=None if self.stage_b_artifact is None else self.stage_b_status,
            stage_c=self.stage_c_status,
            execution=self.execution_status,
        )
        derived = (
            out.evidence_delivered,
            out.evidence_reason,
            out.answer_correct,
            out.earliest_failure,
            out.correct_without_evidence,
        )
        stated = (
            self.evidence_delivered,
            self.evidence_reason,
            self.answer_correct,
            self.earliest_failure,
            self.correct_without_evidence,
        )
        if derived != stated:
            raise ValueError("outcome columns differ from their derivation from the stage columns")
        return self

    def _check_evidence_fields(self, closed: bool, started: bool) -> None:
        recorded = (self.stage_b_artifact, self.context_artifact, self.context_composition)
        if len({v is None for v in recorded}) != 1:
            raise ValueError("Stage B artifact, context and composition belong together")
        has_b = self.stage_b_artifact is not None
        if closed:
            if any(
                v is not None
                for v in (self.stage_a_label, self.stage_a_artifact, self.stage_b_status, *recorded)
            ):
                raise ValueError("closed-book rows carry no Stage A/B/context fields")
            if self.context_provenance != "not_applicable" or self.generator_input_artifact:
                raise ValueError("closed-book rows have no context provenance")
            return
        if (self.stage_a_label is None) != (self.stage_a_artifact is None):
            raise ValueError("Stage A label and artifact belong together")
        if self.stage_a_label is None and (has_b or self.stage_b_status is not None):
            raise ValueError("Stage B needs its Stage A judgment")
        if self.stage_a_label in A_FAILED:
            if self.stage_b_status != "not_reachable_stage_a":
                raise ValueError("Stage B is not reachable after a Stage A loss")
        elif has_b != (self.stage_b_status is not None):
            raise ValueError("a Stage B status needs its judgment artifact")
        if self.context_provenance == "not_applicable":
            raise ValueError("a retrieval row has a context")
        linked = self.context_provenance != "assembled_only"
        if linked != (self.generator_input_artifact is not None):
            raise ValueError("generator input artifact and provenance disagree")
        if linked and not (started and has_b):
            raise ValueError("a generator input needs an attempt and an assembled context")
        if (
            self.phase == "final_test"
            and self.execution_status == "completed"
            and self.context_provenance != "generator_input_recorded"
        ):
            raise ValueError("final-test execution needs the recorded generator input")


# EXPERIMENT_PLAN sections 2 and 6: representation arms are JATS and PDF_DOCLING; confirmatory
# and closed-book run three repeats, the three robustness cells one.
REPRESENTATIONS: tuple[str, ...] = ("JATS", "PDF_DOCLING")
FROZEN_MATRIX: dict[str, tuple[tuple[str, ...], tuple[int, ...]]] = {
    "confirmatory": (REPRESENTATIONS, (1, 2, 3)),
    "dense_robustness": (REPRESENTATIONS, (1,)),
    "chunk_robustness": (REPRESENTATIONS, (1,)),
    "scope_robustness": (REPRESENTATIONS, (1,)),
    "closed_book": (("CLOSED_BOOK",), (1, 2, 3)),
}
MATRIX_ID: Final = "experiment_plan_v1"
# The only rows that form the primary paired comparison. primary_row alone is a repeat
# designation and is TRUE for repeat 1 of every arm.
PRIMARY_SELECTION = (
    "matrix_arm = confirmatory AND condition IN (JATS, PDF_DOCLING) AND primary_row = TRUE "
    "AND analysis_role = confirmatory; paired on item_id; development tables carry "
    "analysis_role = development and hold no confirmatory result; robustness arms are "
    "secondary, closed_book is secondary and unpaired, repeats 2 and 3 are not primary"
)


def analysis_role(phase: str, arm: str) -> Role:
    if phase != "final_test":
        return "development"
    return "confirmatory" if arm == "confirmatory" else "secondary"


COLUMNS: list[str] = list(AnalysisRow.model_fields)


def column_type(annotation: object) -> str:
    """CSV type of a column: integer, logical or string (labels are strings)."""
    if get_origin(annotation) is Literal:
        return "string"
    members = [a for a in get_args(annotation) if a is not type(None)] or [annotation]
    base = members[0]
    return "integer" if base is int else "logical" if base is bool else "string"


class ColumnSpec(Record):
    name: str
    type: Literal["integer", "logical", "string"]
    nullable: bool


ItemState = Literal["included", "missing", "unresolved", "excluded"]
Scope = Literal["complete_matrix", "partial_development"]


class RosterItem(Record):
    """One declared candidate item. Only included items generate planned observations."""

    item_id: Token
    state: ItemState
    paper_id: Token | None = None
    item_sha256: Sha256 | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def valid_item(self) -> Self:
        if self.state == "included":
            if self.paper_id is None or self.item_sha256 is None or self.reason is not None:
                raise ValueError("an included item carries its paper and item hash, no reason")
        elif self.paper_id is not None or self.item_sha256 is not None:
            raise ValueError("only included items carry paper and item hash")
        elif not (self.reason and self.reason.strip()):
            raise ValueError(f"a {self.state} item needs a reason")
        return self


class RosterArm(Record):
    matrix_arm: MatrixArm
    conditions: list[Condition] = Field(min_length=1)
    repeats: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def inside_frozen_matrix(self) -> Self:
        conditions, repeats = FROZEN_MATRIX[self.matrix_arm]
        if self.conditions != [c for c in conditions if c in self.conditions]:
            raise ValueError("conditions must be unique, in matrix order and inside the matrix")
        if self.repeats != [n for n in repeats if n in self.repeats]:
            raise ValueError("repeats must be unique, ascending and inside the matrix")
        return self


class PlannedRoster(Record):
    """The declared scope of one table: which observations are planned and why.

    Expected coordinates are the product of included items, declared arms, their
    conditions and repeats, all inside the frozen matrix. A complete_matrix roster
    declares the whole matrix; anything narrower is partial_development.
    """

    schema_version: Literal["1"] = "1"
    matrix: Literal["experiment_plan_v1"] = MATRIX_ID
    scope: Scope
    phase: Phase
    benchmark_version: Token
    corpus_version: Token
    declared_by: str = Field(min_length=1)
    basis: str = Field(min_length=1)
    arms: list[RosterArm] = Field(min_length=1)
    items: list[RosterItem] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_roster(self) -> Self:
        if not (self.declared_by.strip() and self.basis.strip()):
            raise ValueError("a roster names who declared it and on what basis")
        names = [a.matrix_arm for a in self.arms]
        if names != sorted(set(names), key=list(FROZEN_MATRIX).index):
            raise ValueError("arms must be unique and in matrix order")
        ids = [i.item_id for i in self.items]
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("roster items must be unique and sorted")
        if not any(i.state == "included" for i in self.items):
            raise ValueError("a roster includes at least one item")
        if self.scope == "complete_matrix":
            full = [
                RosterArm(matrix_arm=a, conditions=list(c), repeats=list(r))  # type: ignore[arg-type]
                for a, (c, r) in FROZEN_MATRIX.items()
            ]
            if self.arms != full:
                raise ValueError("a complete_matrix roster declares every arm in full")
            if any(i.state in {"missing", "unresolved"} for i in self.items):
                raise ValueError("a roster with missing or unresolved items is partial")
        return self

    def coordinates(self) -> set[tuple[str, str, str, int]]:
        included = [i.item_id for i in self.items if i.state == "included"]
        return {
            (a.matrix_arm, item, c, n)
            for a in self.arms
            for item in included
            for c in a.conditions
            for n in a.repeats
        }


class OpenObservation(Record):
    """An observation that does not contribute a clean evaluable pair of outcomes."""

    observation_key: str
    item_id: str
    paper_id: str
    condition: Condition
    matrix_arm: MatrixArm
    repeat_index: int
    execution_status: Execution
    evidence_delivered: Delivered
    evidence_reason: str | None = None
    answer_correct: Correct
    stage_c_reason: str | None = None


class AnalysisInventory(Record):
    schema_version: Literal["1"] = "1"
    table_schema_version: Literal["1"] = "1"
    row_grain: str
    pairing: str
    missing_token: str
    primary_repeat_index: int
    scoring_reference: dict[str, str]
    columns: list[ColumnSpec]
    primary_selection: str
    scope: Scope
    roster: ArtifactRef
    items: list[RosterItem]
    planned_observations: int = Field(ge=0)
    executed_observations: int = Field(ge=0)
    row_count: int = Field(ge=0)
    csv: ArtifactRef
    counts: dict[str, int]
    not_evaluable: list[OpenObservation]
