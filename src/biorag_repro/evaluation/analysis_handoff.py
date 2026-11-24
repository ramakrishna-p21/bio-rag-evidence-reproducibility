"""Analysis handoff: the Python-authored record that precedes any R execution.

It pins one published analysis table and states what R must check before it analyses:
the CSV reading contract, the frozen analysis method, the expected complete cases for
each co-primary outcome, and the provenance of the generation runs behind the rows. It
holds no value that exists only after R has run (R version, package inventory, output
hashes); a later result record cites this one by hash. Python stays the only source of
Stage A/B/C labels and answer correctness, and R must reproduce the accounting here
rather than re-derive evaluability.

Only an explicit synthetic rehearsal can be built. A rehearsal has no scientific authority and
cannot be read back as official. An official-origin handoff (`verified_official_run`) is
refused both when building and when reading: verifying official runs is an outstanding Phase 8
requirement, and its earlier implementation was removed in Phase 2.5. The schema keeps
`input_origin`, `source_runs` and `SourceRun` unchanged so that published handoffs and the R
contract keep their bytes. A final-test handoff is built only inside an authorized final-test
evaluation (`stage_a.final_test_scope`), always with official origin, and names the custody
exception under which the batch was accepted; it never claims that a custody gate passed.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self, get_args, get_origin

from pydantic import Field, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef, BuildInput
from biorag_repro.core.records import Record
from biorag_repro.core.storage import (
    read_manifest,
    read_project_json,
)
from biorag_repro.evaluation.analysis_models import (
    COLUMNS,
    MISSING,
    PRIMARY_SELECTION,
    REPRESENTATIONS,
    AnalysisInventory,
    AnalysisRow,
    ColumnSpec,
    PlannedRoster,
    RosterItem,
    Scope,
)
from biorag_repro.evaluation.analysis_storage import publish_derived_record, read_table
from biorag_repro.evaluation.analysis_table import primary_pairs
from biorag_repro.evaluation.attribution import Correct, Delivered, Execution
from biorag_repro.evaluation.stage_a import in_final_test_scope
from biorag_repro.generation.environment import GitState

SCHEMA_VERSION = "1"
TRANSFORMATION = "analysis_handoff_v1"
OutcomeName = Literal["evidence_delivered", "answer_correct"]
Origin = Literal["verified_official_run", "synthetic_rehearsal"]


class HandoffError(ValueError):
    """The handoff does not describe the table or runs it claims to."""


class OfficialProvenanceNotVerified(HandoffError):
    """An official-origin handoff was requested; official provenance cannot be verified here."""


OFFICIAL_UNAVAILABLE: Final = (
    "an official-origin handoff is unavailable: verifying official runs is an outstanding "
    "Phase 8 requirement"
)


# ---- frozen analysis method (EXPERIMENT_PLAN sections 6-9, REPRODUCIBILITY_SPEC section 8) ----


class OutcomeRule(Record):
    """How one co-primary outcome column maps to success, failure and non-evaluable."""

    name: OutcomeName
    column: str
    success: str
    failure: str
    non_evaluable: list[str]


class AnalysisSpec(Record):
    spec_version: Literal["analysis_spec_v1"] = "analysis_spec_v1"
    contrast: str
    direction: str
    estimand: str
    primary_scope: str
    pair_key: str
    cluster_key: str
    outcomes: list[OutcomeRule]
    bootstrap_method: str
    bootstrap_resamples: int
    interval_percent: int
    bootstrap_seed_name: str  # the name only; its value is a setting that still needs approval
    model: str
    variance: str
    test: str
    package: str
    multiplicity: str
    multiplicity_family: list[str]
    sensitivity: list[str]


def _rule(name: OutcomeName, states: object) -> OutcomeRule:
    """Success is 'yes' and failure 'no' in both vocabularies; everything else is non-evaluable."""
    vocabulary = get_args(states)
    if not {"yes", "no"} <= set(vocabulary):
        raise ValueError(f"{name} does not use yes/no")
    return OutcomeRule(
        name=name,
        column=name,
        success="yes",
        failure="no",
        non_evaluable=[s for s in vocabulary if s not in ("yes", "no")],
    )


ANALYSIS_SPEC: Final = AnalysisSpec(
    contrast="JATS - PDF_DOCLING",
    direction="JATS_minus_PDF_DOCLING",
    estimand="item_weighted_paired_risk_difference",
    primary_scope=PRIMARY_SELECTION,
    pair_key="item_id",
    cluster_key="paper_id",
    outcomes=[_rule("evidence_delivered", Delivered), _rule("answer_correct", Correct)],
    bootstrap_method="whole_paper_percentile",
    bootstrap_resamples=10000,
    interval_percent=95,
    bootstrap_seed_name="primary_bootstrap",
    model="intercept_only_paired_differences",
    variance="CR2",
    test="satterthwaite_wald_two_sided",
    package="clubSandwich",
    multiplicity="holm",
    multiplicity_family=["evidence_delivered", "answer_correct"],
    sensitivity=["equal_paper_weighted", "best_worst_case_bounds"],
)

# Scientific settings that still need approval before they can be part of the method.
AWAITING_APPROVAL: Final = ["primary_bootstrap_seed"]
# Execution provenance a later result record must supply once R has actually run.
EXECUTION_PROVENANCE_REQUIRED: Final = [
    "analysis_script_git_sha",
    "analysis_script_sha256",
    "applied_bootstrap_seed",
    "handoff_sha256",
    "installed_package_inventory_sha256",
    "output_hashes",
    "package_versions_used",
    "r_dependency_lock_sha256",
    "r_version",
]


# ---- CSV reading contract ------------------------------------------------------------------


class CsvContract(Record):
    """Everything an independent reader needs to parse the table exactly as Python wrote it."""

    encoding: Literal["utf-8"] = "utf-8"
    byte_order_mark: bool = False
    line_ending: Literal["\n"] = "\n"
    delimiter: Literal[","] = ","
    quote: Literal['"'] = '"'
    quote_escape: Literal["doubled"] = "doubled"
    header_row: bool = True
    missing_token: str
    logical_tokens: list[str]
    integer_pattern: str
    sort_key: list[str]
    sort_collation: Literal["unicode_code_point"] = "unicode_code_point"
    columns: list[ColumnSpec]
    columns_sha256: str
    row_count: int = Field(ge=0)
    na_value_columns: list[str]  # non-nullable columns whose data values include the literal NA


def columns_sha256(columns: Sequence[ColumnSpec]) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            [{"name": c.name, "type": c.type, "nullable": c.nullable} for c in columns]
        )
    )


def derive_csv_contract(inventory: AnalysisInventory) -> CsvContract:
    if [c.name for c in inventory.columns] != COLUMNS:
        raise HandoffError("inventory columns differ from the row model")
    na_values = [
        name
        for name, field in AnalysisRow.model_fields.items()
        if get_origin(field.annotation) is Literal and MISSING in get_args(field.annotation)
    ]
    return CsvContract(
        missing_token=inventory.missing_token,
        logical_tokens=["TRUE", "FALSE"],
        integer_pattern=r"^(0|[1-9][0-9]*)$",
        sort_key=["matrix_arm", "item_id", "condition", "repeat_index", "observation_key"],
        columns=inventory.columns,
        columns_sha256=columns_sha256(inventory.columns),
        row_count=inventory.row_count,
        na_value_columns=na_values,
    )


# ---- inclusion and missingness accounting ----------------------------------------------------


class RepOutcome(Record):
    """One representation's primary observation, with its underlying states kept apart."""

    observation_key: str
    state: str
    execution_status: Execution
    reason: str | None


class ItemAccounting(Record):
    item_id: str
    paper_id: str
    jats: RepOutcome | None  # None: the roster declares no primary row for this representation
    pdf_docling: RepOutcome | None
    complete_case: bool


class PaperCount(Record):
    paper_id: str
    eligible: int
    complete: int


class OutcomeAccounting(Record):
    outcome: OutcomeName
    eligible_items: list[ItemAccounting]
    complete_item_ids: list[str]
    state_counts: dict[str, int]
    papers: list[PaperCount]


NOT_IN_ROSTER = "not_in_roster"


def _outcome_cell(row: AnalysisRow | None, outcome: OutcomeName) -> RepOutcome | None:
    if row is None:
        return None
    if outcome == "evidence_delivered":
        return RepOutcome(
            observation_key=row.observation_key,
            state=row.evidence_delivered,
            execution_status=row.execution_status,
            reason=row.evidence_reason,
        )
    return RepOutcome(
        observation_key=row.observation_key,
        state=row.answer_correct,
        execution_status=row.execution_status,
        reason=row.stage_c_reason,
    )


def derive_accounting(rows: list[AnalysisRow], roster: PlannedRoster) -> list[OutcomeAccounting]:
    """Primary complete-case accounting: confirmatory repeat 1, one outcome at a time.

    Repeats 2-3, robustness arms and closed-book rows are never part of it. Every included
    roster item is eligible. Both representations' states and reasons are kept; a complete
    case needs a yes/no outcome in each.
    """
    primary: dict[tuple[str, str], AnalysisRow] = {}
    for row in rows:
        if (
            row.matrix_arm == "confirmatory"
            and row.repeat_index == 1
            and row.condition in REPRESENTATIONS
        ):
            primary[(row.item_id, row.condition)] = row
    included = [i for i in roster.items if i.state == "included"]
    known = {i.item_id for i in included}
    if stray := sorted({item for item, _ in primary} - known):
        raise HandoffError(f"primary rows for items outside the roster: {stray[:3]}")
    both = {i.item_id for i in included if all((i.item_id, c) in primary for c in REPRESENTATIONS)}
    role = "confirmatory" if roster.phase == "final_test" else "development"
    if both != set(primary_pairs(rows, role)):
        raise HandoffError("primary pairing differs from the published pair definition")

    result = []
    for rule in ANALYSIS_SPEC.outcomes:
        evaluable = {rule.success, rule.failure}
        items = []
        counts: dict[str, int] = defaultdict(int)
        papers: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for item in sorted(included, key=lambda i: i.item_id):
            jats_row = primary.get((item.item_id, "JATS"))
            pdf_row = primary.get((item.item_id, "PDF_DOCLING"))
            jats = _outcome_cell(jats_row, rule.name)
            pdf = _outcome_cell(pdf_row, rule.name)
            complete = (
                jats is not None
                and pdf is not None
                and jats.state in evaluable
                and pdf.state in evaluable
            )
            if item.paper_id is None:
                raise HandoffError(f"included item {item.item_id} does not name its paper")
            items.append(
                ItemAccounting(
                    item_id=item.item_id,
                    paper_id=item.paper_id,
                    jats=jats,
                    pdf_docling=pdf,
                    complete_case=complete,
                )
            )
            for condition, cell in (("JATS", jats), ("PDF_DOCLING", pdf)):
                counts[f"{condition}|{NOT_IN_ROSTER if cell is None else cell.state}"] += 1
            papers[item.paper_id][0] += 1
            papers[item.paper_id][1] += complete
        result.append(
            OutcomeAccounting(
                outcome=rule.name,
                eligible_items=items,
                complete_item_ids=[i.item_id for i in items if i.complete_case],
                state_counts=dict(sorted(counts.items())),
                papers=[
                    PaperCount(paper_id=p, eligible=e, complete=c)
                    for p, (e, c) in sorted(papers.items())
                ],
            )
        )
    return result


# ---- the handoff record --------------------------------------------------------------------


class ArmConfig(Record):
    matrix_arm: str
    resolved_config_sha256: str
    retrieval_config_sha256: str | None


class SourceRun(Record):
    """The completion anchor of a verified official run behind some rows."""

    run_id: str
    completion: ArtifactRef
    run_record_sha256: str
    resolved_config_sha256: str
    raw_config_sha256: str
    roster_sha256: str
    matrix_arm: str
    backend_kind: str
    observations: int = Field(ge=0)


class CustodyException(Record):
    """The researcher's exception under which a batch with unmet custody was evaluated."""

    status: Literal["custody_gate_not_established_exception_v1"]
    authorization: ArtifactRef  # the evaluation authorization record that approves it
    amendment_sha256: str
    run_id: str
    drive_manifest_sha256: str


class AnalysisHandoff(Record):
    schema_version: Literal["1"] = "1"
    kind: Literal["analysis_handoff"] = "analysis_handoff"
    input_origin: Origin
    phase: Literal["development", "final_test"]
    benchmark_version: str = Field(min_length=1)
    corpus_version: str = Field(min_length=1)
    scope: Scope
    table_build: ArtifactRef
    csv: ArtifactRef
    inventory: ArtifactRef
    roster: ArtifactRef
    scoring_reference: dict[str, str]
    stage_a_review_protocols: dict[str, list[str]] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    csv_contract: CsvContract
    arms: list[ArmConfig] = Field(min_length=1)
    source_runs: list[SourceRun]
    analysis_spec: AnalysisSpec
    accounting: list[OutcomeAccounting] = Field(min_length=2, max_length=2)
    non_included_items: list[RosterItem]
    awaiting_approval: list[str]
    execution_provenance_required: list[str]
    code: GitState
    custody_exception: CustodyException | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def origin_matches_runs(self) -> Self:
        if (self.input_origin == "verified_official_run") != bool(self.source_runs):
            raise ValueError(
                "a verified official origin needs its source runs; a rehearsal has none"
            )
        if self.phase == "final_test" and (
            self.input_origin != "verified_official_run" or self.custody_exception is None
        ):
            raise ValueError("a final-test handoff is official and names its custody exception")
        if self.phase == "development" and self.custody_exception is not None:
            raise ValueError("a custody exception belongs to the final-test campaign only")
        return self


# ---- construction --------------------------------------------------------------------------


def build_handoff(
    table_build: ArtifactRef,
    root: Path,
    *,
    code: GitState,
    rehearsal: bool = False,
) -> AnalysisHandoff:
    """Describe one published analysis table as an explicit synthetic rehearsal; nothing is written."""
    if not rehearsal:
        raise OfficialProvenanceNotVerified(OFFICIAL_UNAVAILABLE)
    return _describe(table_build, root, code)[0]


def _describe(
    table_build: ArtifactRef,
    root: Path,
    code: GitState,
) -> tuple[AnalysisHandoff, list[AnalysisRow]]:
    """The rehearsal handoff and the table rows it was built from (verified once, by `read_table`)."""
    rows = read_table(table_build, root)
    manifest = read_manifest(root / table_build.logical_ref, table_build.sha256)
    outputs = {o.artifact_kind: o.artifact for o in manifest.outputs}
    inventory = read_project_json(
        root / outputs["analysis_inventory"].logical_ref,
        outputs["analysis_inventory"].sha256,
        AnalysisInventory,
    )
    roster = read_project_json(
        root / outputs["planned_roster"].logical_ref,
        outputs["planned_roster"].sha256,
        PlannedRoster,
    )
    if roster.phase == "final_test" and not in_final_test_scope():
        raise HandoffError("a final-test handoff is built only inside the final-test evaluation")
    if roster.phase not in ("development", "final_test"):
        raise HandoffError("qualification tables have no analysis handoff")
    arms: dict[str, set[tuple[str, str | None]]] = defaultdict(set)
    for row in rows:
        arms[row.matrix_arm].add((row.config_sha256, row.retrieval_config_sha256))
    if any(len(configs) != 1 for configs in arms.values()):
        raise HandoffError("one matrix arm must run under one configuration")
    # A final-test template is completed with its verified source runs and custody exception
    # by the final-test evaluator, which validates the finished record; it is never valid alone.
    make = AnalysisHandoff.model_construct if roster.phase == "final_test" else AnalysisHandoff
    handoff = make(
        input_origin="synthetic_rehearsal",
        phase=roster.phase,
        benchmark_version=roster.benchmark_version,
        corpus_version=roster.corpus_version,
        scope=roster.scope,
        table_build=table_build,
        csv=outputs["analysis_table"],
        inventory=outputs["analysis_inventory"],
        roster=outputs["planned_roster"],
        scoring_reference=inventory.scoring_reference,
        stage_a_review_protocols=inventory.stage_a_review_protocols,
        csv_contract=derive_csv_contract(inventory),
        arms=[
            ArmConfig(
                matrix_arm=arm, resolved_config_sha256=cfg[0][0], retrieval_config_sha256=cfg[0][1]
            )
            for arm, cfg in ((a, list(c)) for a, c in sorted(arms.items()))
        ],
        source_runs=[],
        analysis_spec=ANALYSIS_SPEC,
        accounting=derive_accounting(rows, roster),
        non_included_items=[i for i in roster.items if i.state != "included"],
        awaiting_approval=list(AWAITING_APPROVAL),
        execution_provenance_required=list(EXECUTION_PROVENANCE_REQUIRED),
        code=code,
    )
    return handoff, rows


# ---- verification, publication and read-back ---------------------------------------------------


def _verify(handoff: AnalysisHandoff, root: Path) -> list[AnalysisRow]:
    """Rebuild the rehearsal handoff from the published table and require equality."""
    if handoff.input_origin != "synthetic_rehearsal" or handoff.source_runs:
        raise OfficialProvenanceNotVerified(OFFICIAL_UNAVAILABLE)
    expected, rows = _describe(handoff.table_build, root, handoff.code)
    if expected != handoff:
        fields = sorted(
            name
            for name in AnalysisHandoff.model_fields
            if getattr(expected, name) != getattr(handoff, name)
        )
        raise HandoffError(f"handoff differs from the table it describes: {fields}")
    return rows


def _settings_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"schema_version": SCHEMA_VERSION, "spec": ANALYSIS_SPEC.model_dump(mode="json")}
        )
    )


def _inputs(handoff: AnalysisHandoff) -> list[BuildInput]:
    return [
        BuildInput(role="analysis_table_build", artifact=handoff.table_build),
        *(BuildInput(role="run_completion", artifact=r.completion) for r in handoff.source_runs),
    ]


def _store(handoff: AnalysisHandoff, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    """Write a handoff and its build manifest without replacing anything or checking it."""
    return publish_derived_record(
        handoff,
        "analysis_handoff",
        "handoff",
        _inputs(handoff),
        TRANSFORMATION,
        _settings_sha256(),
        root,
    )


def publish_handoff(handoff: AnalysisHandoff, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    """Verify, then publish. A handoff that cannot be verified is never written."""
    _verify(handoff, root)
    return _store(handoff, root)


@dataclass(frozen=True)
class VerifiedHandoff:
    handoff: AnalysisHandoff
    rows: list[AnalysisRow]
    official: bool  # always False: no official origin can currently be verified


def read_handoff(
    build: ArtifactRef,
    root: Path,
    *,
    require_official: bool = False,
) -> VerifiedHandoff:
    """Check manifest, bytes, lineage and the table; return the rows.

    `require_official` always refuses: no official origin can currently be verified.
    """
    manifest = read_manifest(root / build.logical_ref, build.sha256)
    if len(manifest.outputs) != 1 or manifest.outputs[0].artifact_kind != "analysis_handoff":
        raise HandoffError("build manifest does not describe an analysis handoff")
    if manifest.transformation_id != TRANSFORMATION or (
        manifest.transformation_settings_sha256 != _settings_sha256()
    ):
        raise HandoffError("handoff was built under other settings")
    ref = manifest.outputs[0].artifact
    handoff = read_project_json(root / ref.logical_ref, ref.sha256, AnalysisHandoff)
    if manifest.inputs != _inputs(handoff):
        raise HandoffError("handoff lineage differs from its build manifest")
    if require_official and handoff.input_origin != "verified_official_run":
        raise HandoffError("a rehearsal handoff has no scientific authority")
    rows = _verify(handoff, root)
    return VerifiedHandoff(handoff, rows, False)
