"""Build the item-level analysis table from validated Stage A/B/C records.

Every input is re-validated against the authoritative evaluators; a mismatched or
counterfeit record is rejected, never reconciled. A planned observation without an
execution record is kept as planned, not turned into a failure.
"""

import csv
import io
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, get_args

from biorag_repro.benchmark.development_view import DevelopmentView, check_development_items
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.manifest import ArtifactRef, BuildInput
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.analysis_models import (
    COLUMNS,
    MISSING,
    PRIMARY_REPEAT_INDEX,
    PRIMARY_SELECTION,
    SCHEMA_VERSION,
    AnalysisInventory,
    AnalysisRow,
    BStatus,
    ColumnSpec,
    Completion,
    Condition,
    ContextProvenance,
    GeneratorInput,
    MatrixArm,
    OpenObservation,
    Phase,
    PlannedRoster,
    Role,
    RosterItem,
    analysis_role,
    column_type,
)
from biorag_repro.evaluation.attribution import A_FAILED, Execution, attribute
from biorag_repro.evaluation.models import StageAJudgment
from biorag_repro.evaluation.stage_a import item_digest, stage_a_ref
from biorag_repro.evaluation.stage_a import validate_judgment as validate_stage_a
from biorag_repro.evaluation.stage_b import Inputs, stage_b_ref
from biorag_repro.evaluation.stage_b import validate_judgment as validate_stage_b
from biorag_repro.evaluation.stage_b_models import StageBJudgment
from biorag_repro.evaluation.stage_c import check_result, score_ref, validate_score
from biorag_repro.evaluation.stage_c_models import GenerationResult, StageCScore

# Retrieval scope and retriever that each retrieval arm of the frozen matrix uses.
ARM_RETRIEVAL = {
    "confirmatory": ("designated_paper", "bm25"),
    "dense_robustness": ("designated_paper", "bge"),
    "chunk_robustness": ("designated_paper", "bm25"),
    "scope_robustness": ("full_corpus", "bm25"),
}
SCORING_REFERENCE = {
    "stage_a": "stage_a_judgment_v1",
    "stage_b": "stage_b_judgment_v1",
    "stage_c": "stage_c_score_v1",
    "table": "analysis_table_v1",
}
ROW_GRAIN = "one planned observation: phase, matrix_arm, item_id, condition, repeat_index"
PAIRING = (
    "JATS and PDF_DOCLING rows pair on item_id, matrix_arm and repeat_index; primary_row marks "
    "repeat 1 of every arm; "
    "CLOSED_BOOK rows are outside the paired representation comparison; the primary "
    "comparison is selected by primary_selection, not by primary_row alone"
)


@dataclass(frozen=True)
class Evidence:
    """Validated Stage A judgment and, once the context exists, its Stage B judgment."""

    document: CanonicalDocument
    canonical: ArtifactRef
    stage_a: StageAJudgment
    stage_b: StageBJudgment | None = None
    b_inputs: Inputs | None = None  # the records Stage B was judged from


@dataclass(frozen=True)
class Observation:
    """One planned observation with whatever has been recorded about it so far."""

    item: BenchmarkItem
    phase: Phase
    matrix_arm: MatrixArm
    condition: Condition
    repeat_index: int
    resolved_config_sha256: str  # full observation identity (REPRODUCIBILITY_SPEC section 2)
    retrieval_config_sha256: str | None = None  # Phase 5 retrieval/context identity it contains
    result: GenerationResult | None = None
    score: StageCScore | None = None
    evidence: Evidence | None = None
    generator_input: GeneratorInput | None = None


@dataclass
class Table:
    rows: list[AnalysisRow]
    roster: PlannedRoster
    inputs: list[BuildInput] = field(default_factory=list)


def _check_evidence(obs: Observation, evidence: Evidence) -> None:
    item = obs.item
    if evidence.document.representation != obs.condition:
        raise ValueError("evidence is for another representation than the observation's")
    validate_stage_a(evidence.stage_a, item, evidence.document, evidence.canonical)
    b, inputs = evidence.stage_b, evidence.b_inputs
    if b is None:
        if inputs is not None:
            raise ValueError("Stage B inputs without a Stage B judgment")
        return
    if inputs is None:
        raise ValueError("a Stage B judgment must come with the records it was judged from")
    if (
        inputs.item != item
        or inputs.document != evidence.document
        or inputs.canonical != evidence.canonical
        or inputs.stage_a != evidence.stage_a
    ):
        raise ValueError("Stage B was judged from other item, document or Stage A inputs")
    validate_stage_b(b, inputs)
    if b.configuration.resolved_sha256 != obs.retrieval_config_sha256:
        raise ValueError("retrieval and context configuration differs from the observation's")
    expected = ARM_RETRIEVAL.get(obs.matrix_arm)
    if expected is None or (inputs.ranked.scope, inputs.ranked.retriever) != expected:
        raise ValueError("retrieval does not match the matrix arm")


def _check_generator_input(obs: Observation, link: GeneratorInput) -> None:
    result, evidence = obs.result, obs.evidence
    if result is None or evidence is None or evidence.stage_b is None or not evidence.b_inputs:
        raise ValueError("a generator input needs an executed attempt and its assembled context")
    context = evidence.b_inputs.context
    if (link.observation_key, link.attempt_id) != (result.observation_key, result.attempt_id):
        raise ValueError("generator input belongs to another attempt")
    if link.context != evidence.stage_b.context:
        raise ValueError("generator input names another context than the one judged in Stage B")
    if link.exact_context_sha256 != sha256_bytes(context.exact_context.encode("utf-8")):
        raise ValueError("generator input does not carry the exact judged context")


def _execution(result: GenerationResult | None) -> Execution:
    if result is None:
        return "planned_not_executed"
    return "operational_failure" if result.status == "operational_failure" else "completed"


def _completion(result: GenerationResult | None) -> Completion | None:
    if result is None or result.status != "completed":
        return None
    if result.parse_status == "unparseable":
        return "unparseable"
    return "abstained" if result.abstained else "answer"


def build_row(obs: Observation) -> tuple[AnalysisRow, list[BuildInput]]:
    item, result, score = obs.item, obs.result, obs.score
    if item.candidate_status != "accepted" or item.split is None:
        raise ValueError("only accepted, split benchmark items enter the analysis table")
    if item.split == "final_test" or obs.phase == "final_test":
        # Stage A already refuses these items; the final-test readiness gate lifts this later.
        raise ValueError("final-test rows are not built before the final-test readiness gate")
    key = observation_key(
        phase=obs.phase,
        benchmark_version=item.benchmark_version,
        item_id=item.item_id,
        matrix_arm=obs.matrix_arm,
        representation_or_condition=obs.condition,
        repeat_index=obs.repeat_index,
        resolved_config_sha256=obs.resolved_config_sha256,
    )
    if result is None:
        if obs.generator_input is not None:
            raise ValueError("an unexecuted observation has no generator input")
    else:
        check_result(item, result)
        if result.observation_key != key:
            raise ValueError("generation result belongs to another planned observation")
        if score is None:
            raise ValueError("an executed observation needs its Stage C score")
        validate_score(score, item, result)

    closed = obs.matrix_arm == "closed_book"
    if (obs.retrieval_config_sha256 is None) != closed:
        raise ValueError(
            "a retrieval observation names its retrieval configuration; closed-book has none"
        )
    evidence = obs.evidence
    if evidence is not None:
        _check_evidence(obs, evidence)
    if obs.generator_input is not None:
        _check_generator_input(obs, obs.generator_input)

    a = evidence.stage_a if evidence else None
    b = evidence.stage_b if evidence else None
    link = obs.generator_input
    provenance: ContextProvenance
    if closed:
        provenance = "not_applicable"
    elif link is None:
        provenance = "assembled_only"
    else:
        provenance = (
            "generator_input_recorded"
            if link.provenance == "recorded"
            else "generator_input_simulated"
        )
    b_status: BStatus | None
    if a is not None and a.label in A_FAILED:
        b_status = "not_reachable_stage_a"
    else:
        b_status = b.status if b else None
    row = AnalysisRow(
        benchmark_version=item.benchmark_version,
        corpus_version=item.corpus_version,
        item_id=item.item_id,
        item_sha256=item_digest(item),
        paper_id=item.paper_id,
        answer_class=item.answer_class,
        split=item.split,
        phase=obs.phase,
        analysis_role=analysis_role(obs.phase, obs.matrix_arm),
        matrix_arm=obs.matrix_arm,
        condition=obs.condition,
        repeat_index=obs.repeat_index,
        primary_row=obs.repeat_index == PRIMARY_REPEAT_INDEX,
        observation_key=key,
        config_sha256=obs.resolved_config_sha256,
        retrieval_config_sha256=obs.retrieval_config_sha256,
        run_id=result.run_id if result else None,
        attempt_id=result.attempt_id if result else None,
        execution_status=_execution(result),
        completion_kind=_completion(result),
        stage_a_label=a.label if a else None,
        stage_a_artifact=stage_a_ref(a).artifact_id if a else None,
        stage_b_status=b_status,
        stage_b_artifact=stage_b_ref(b).artifact_id if b else None,
        context_artifact=b.context.artifact_id if b else None,
        context_composition=b.composition if b else None,
        context_provenance=provenance,
        generator_input_artifact=link.input.artifact_id if link else None,
        stage_c_status=score.status if score else None,
        stage_c_reason=score.reason if score else None,
        stage_c_basis=score.basis if score else None,
        stage_c_artifact=score_ref(score).artifact_id if score else None,
        **_outcomes(closed, a, b_status, b, score, _execution(result)),
    )
    return row, _inputs(item, a, b, score, link)


def _outcomes(
    closed: bool,
    a: StageAJudgment | None,
    b_status: str | None,
    b: StageBJudgment | None,
    score: StageCScore | None,
    execution: Execution,
) -> dict[str, Any]:
    out = attribute(
        closed_book=closed,
        stage_a=a.label if a else None,
        stage_b=b_status if b else None,
        stage_c=score.status if score else None,
        execution=execution,
    )
    return {
        "evidence_delivered": out.evidence_delivered,
        "evidence_reason": out.evidence_reason,
        "answer_correct": out.answer_correct,
        "earliest_failure": out.earliest_failure,
        "correct_without_evidence": out.correct_without_evidence,
    }


def _inputs(
    item: BenchmarkItem,
    a: StageAJudgment | None,
    b: StageBJudgment | None,
    score: StageCScore | None,
    link: GeneratorInput | None,
) -> list[BuildInput]:
    refs = [
        (
            "benchmark_item",
            ArtifactRef(
                artifact_id=item.item_id,
                logical_ref=f"benchmark/{item.benchmark_version}/items/{item.item_id}",
                sha256=item_digest(item),
            ),
        )
    ]
    if a:
        refs.append(("stage_a_judgment", stage_a_ref(a)))
    if b:
        refs += [("stage_b_judgment", stage_b_ref(b)), ("context", b.context)]
    if score:
        refs.append(("stage_c_score", score_ref(score)))
        refs.append(
            (
                "generation_result",
                ArtifactRef(
                    artifact_id=f"{score.observation_key}:{score.attempt_id}",
                    logical_ref=f"generation/{score.observation_key}/{score.attempt_id}",
                    sha256=score.generation_sha256,
                ),
            )
        )
    if link:
        refs.append(("generator_input", link.input))
    return [BuildInput(role=r, artifact=x) for r, x in refs]


# ---- table -------------------------------------------------------------------------------


def _order(row: AnalysisRow) -> tuple[str, str, str, int, str]:
    return (row.matrix_arm, row.item_id, row.condition, row.repeat_index, row.observation_key)


def build_table(observations: list[Observation], roster: PlannedRoster) -> Table:
    built = [build_row(o) for o in observations]
    rows = sorted((r for r, _ in built), key=_order)
    unique: dict[tuple[str, str], BuildInput] = {}
    for _, inputs in built:
        for entry in inputs:
            previous = unique.setdefault((entry.role, entry.artifact.artifact_id), entry)
            if previous != entry:
                raise ValueError("one artifact id names two different references")
    check_rows(rows)
    check_roster(rows, roster)
    return Table(rows, roster, [unique[k] for k in sorted(unique)])


def roster_items_from_development_view(
    view: DevelopmentView, items: list[BenchmarkItem]
) -> list[RosterItem]:
    """Roster entries for exactly the items of a verified development-only view.

    Membership comes from the published view, never from a caller-supplied list. A
    roster built from this covers the development subset only; it says nothing about
    the complete frozen benchmark or the final-test partition.
    """
    try:
        check_development_items(items, view.development_paper_ids, view.benchmark_version)
    except ValueError:
        raise ValueError("items are not the development-only view") from None
    if len(items) != view.development_item_count:
        raise ValueError("items differ from the development view membership")
    return [
        RosterItem(
            item_id=item.item_id,
            state="included",
            paper_id=item.paper_id,
            item_sha256=item_digest(item),
        )
        for item in sorted(items, key=lambda i: i.item_id)
    ]


def check_roster(rows: list[AnalysisRow], roster: PlannedRoster) -> None:
    """Rows are exactly the planned observations: none undeclared, none omitted.

    A planned observation that was not executed must still be present as a
    planned_not_executed row; this check never creates one.
    """
    for r in rows:
        if (r.phase, r.benchmark_version, r.corpus_version) != (
            roster.phase,
            roster.benchmark_version,
            roster.corpus_version,
        ):
            raise ValueError("table rows are outside the roster's phase, benchmark or corpus")
    declared = {i.item_id: i for i in roster.items if i.state == "included"}
    for r in rows:
        item = declared.get(r.item_id)
        if item is not None and (item.paper_id, item.item_sha256) != (r.paper_id, r.item_sha256):
            raise ValueError(f"item {r.item_id} differs from the roster's paper or item hash")
    seen = {(r.matrix_arm, r.item_id, r.condition, r.repeat_index) for r in rows}
    expected = roster.coordinates()
    if undeclared := sorted(seen - expected):
        raise ValueError(f"observations outside the roster: {undeclared[:3]}")
    if omitted := sorted(expected - seen):
        raise ValueError(f"planned observations without a row ({len(omitted)}): {omitted[:3]}")


def primary_pairs(
    rows: list[AnalysisRow], role: Role
) -> dict[str, tuple[AnalysisRow, AnalysisRow]]:
    """JATS/PDF_DOCLING repeat-1 pairs of the confirmatory arm, by item, for one analysis role.

    Robustness arms, closed-book rows and repeats 2-3 are never selected. Delivery and
    correctness evaluability stay in the rows; this selects the comparison only.
    """
    chosen: dict[str, dict[str, AnalysisRow]] = defaultdict(dict)
    for r in rows:
        if (
            r.matrix_arm == "confirmatory"
            and r.primary_row
            and r.analysis_role == role
            and r.condition in {"JATS", "PDF_DOCLING"}
        ):
            chosen[r.item_id][r.condition] = r
    return {
        item: (by["JATS"], by["PDF_DOCLING"]) for item, by in sorted(chosen.items()) if len(by) == 2
    }


def check_rows(rows: list[AnalysisRow]) -> None:
    """Cross-row rules: identity, uniqueness, one phase, shared evidence across repeats."""
    if [_order(r) for r in rows] != sorted(_order(r) for r in rows):
        raise ValueError("rows are not in canonical order")
    for name in ("phase", "benchmark_version", "corpus_version"):
        if len({getattr(r, name) for r in rows}) > 1:
            raise ValueError(f"one table holds one {name}")
    keys = [r.observation_key for r in rows]
    coordinates = [(r.matrix_arm, r.item_id, r.condition, r.repeat_index) for r in rows]
    if len(set(keys)) != len(keys) or len(set(coordinates)) != len(coordinates):
        raise ValueError("duplicate observation")
    items: dict[str, tuple[str, str, str, str]] = {}
    configs: dict[str, str] = {}
    retrievals: dict[str, str | None] = {}
    groups: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    for r in rows:
        identity = (r.item_sha256, r.paper_id, r.answer_class, r.split)
        if items.setdefault(r.item_id, identity) != identity:
            raise ValueError("one item appears with different identities")
        if configs.setdefault(r.matrix_arm, r.config_sha256) != r.config_sha256:
            raise ValueError("one matrix arm runs under one resolved configuration")
        if retrievals.setdefault(r.matrix_arm, r.retrieval_config_sha256) != (
            r.retrieval_config_sha256
        ):
            raise ValueError("one matrix arm runs under one retrieval configuration")
        shared = (
            r.stage_a_artifact,
            r.stage_b_artifact,
            r.context_artifact,
            r.context_composition,
            r.evidence_delivered,
            r.evidence_reason,
        )
        if groups.setdefault((r.matrix_arm, r.item_id, r.condition), shared) != shared:
            raise ValueError("repeats of one condition must share their evidence outcome")


def check_references(rows: list[AnalysisRow], inputs: list[BuildInput]) -> None:
    """Rows and published inputs must cite exactly the same upstream artifacts."""
    expected: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        expected["benchmark_item"].add(r.item_id)
        pairs = {
            "stage_a_judgment": r.stage_a_artifact,
            "stage_b_judgment": r.stage_b_artifact,
            "context": r.context_artifact,
            "stage_c_score": r.stage_c_artifact,
            "generator_input": r.generator_input_artifact,
            "generation_result": None
            if r.attempt_id is None
            else f"{r.observation_key}:{r.attempt_id}",
        }
        for role, artifact_id in pairs.items():
            if artifact_id is not None:
                expected[role].add(artifact_id)
    actual: dict[str, set[str]] = defaultdict(set)
    for entry in inputs:
        actual[entry.role].add(entry.artifact.artifact_id)
    if dict(expected) != dict(actual):
        raise ValueError("table rows and the listed source artifacts differ")
    shas = {r.item_id: r.item_sha256 for r in rows}
    for entry in inputs:
        if (
            entry.role == "benchmark_item"
            and entry.artifact.sha256 != shas[entry.artifact.artifact_id]
        ):
            raise ValueError("benchmark item hash differs from the rows")


# ---- CSV ---------------------------------------------------------------------------------

_TYPES = {name: column_type(f.annotation) for name, f in AnalysisRow.model_fields.items()}
_NULLABLE = {
    name for name, f in AnalysisRow.model_fields.items() if type(None) in get_args(f.annotation)
}
_INTEGER = re.compile(r"^(0|[1-9][0-9]*)$")


def _cell(value: object) -> str:
    if value is None:
        return MISSING
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    return str(value)


def serialize_csv(rows: list[AnalysisRow]) -> bytes:
    """Canonical UTF-8 CSV: fixed columns, LF endings, sorted rows, NA for missing."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(COLUMNS)
    for r in sorted(rows, key=_order):
        writer.writerow([_cell(getattr(r, name)) for name in COLUMNS])
    return buffer.getvalue().encode("utf-8")


def _value(name: str, text: str) -> object:
    if text == MISSING and name in _NULLABLE:
        return None
    kind = _TYPES[name]
    if kind == "logical":
        if text not in {"TRUE", "FALSE"}:
            raise ValueError(f"{name}: logical values are TRUE or FALSE")
        return text == "TRUE"
    if kind == "integer":
        if not _INTEGER.match(text):
            raise ValueError(f"{name}: not a canonical integer")
        return int(text)
    return text


def parse_csv(data: bytes) -> list[AnalysisRow]:
    """Independent re-read: strict types, row re-derivation, and canonical form."""
    records = list(csv.reader(io.StringIO(data.decode("utf-8"), newline="")))
    if not records or records[0] != COLUMNS:
        raise ValueError("analysis table header differs from the fixed column order")
    rows = []
    for number, record in enumerate(records[1:], start=2):
        if len(record) != len(COLUMNS):
            raise ValueError(f"line {number}: wrong number of cells")
        rows.append(
            AnalysisRow.model_validate(
                {name: _value(name, text) for name, text in zip(COLUMNS, record, strict=True)},
                strict=True,
            )
        )
    check_rows(rows)
    if serialize_csv(rows) != data:
        raise ValueError("analysis table is not in canonical form")
    return rows


# ---- inventory -----------------------------------------------------------------------------


def make_inventory(
    rows: list[AnalysisRow], csv_ref: ArtifactRef, roster: PlannedRoster, roster_ref: ArtifactRef
) -> AnalysisInventory:
    counts: dict[str, int] = defaultdict(int)
    open_rows = []
    for r in rows:
        state = (r.matrix_arm, r.condition, r.execution_status, r.evidence_delivered)
        counts["|".join((*state, r.answer_correct))] += 1
        clean = (
            r.execution_status == "completed"
            and r.answer_correct in {"yes", "no"}
            and r.evidence_delivered != "non_evaluable"
        )
        if not clean:
            open_rows.append(
                OpenObservation(
                    observation_key=r.observation_key,
                    item_id=r.item_id,
                    paper_id=r.paper_id,
                    condition=r.condition,
                    matrix_arm=r.matrix_arm,
                    repeat_index=r.repeat_index,
                    execution_status=r.execution_status,
                    evidence_delivered=r.evidence_delivered,
                    evidence_reason=r.evidence_reason,
                    answer_correct=r.answer_correct,
                    stage_c_reason=r.stage_c_reason,
                )
            )
    return AnalysisInventory(
        row_grain=ROW_GRAIN,
        pairing=PAIRING,
        missing_token=MISSING,
        primary_repeat_index=PRIMARY_REPEAT_INDEX,
        primary_selection=PRIMARY_SELECTION,
        scope=roster.scope,
        roster=roster_ref,
        items=roster.items,
        planned_observations=len(roster.coordinates()),
        executed_observations=sum(r.execution_status != "planned_not_executed" for r in rows),
        scoring_reference=SCORING_REFERENCE,
        columns=[
            ColumnSpec(name=n, type=_TYPES[n], nullable=n in _NULLABLE)  # type: ignore[arg-type]
            for n in COLUMNS
        ],
        row_count=len(rows),
        csv=csv_ref,
        counts=dict(sorted(counts.items())),
        not_evaluable=open_rows,
    )


__all__ = [
    "SCHEMA_VERSION",
    "Evidence",
    "Observation",
    "Table",
    "build_row",
    "build_table",
    "check_references",
    "check_roster",
    "check_rows",
    "make_inventory",
    "parse_csv",
    "primary_pairs",
    "serialize_csv",
]
