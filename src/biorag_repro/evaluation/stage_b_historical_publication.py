"""Immutable publication of the approved historical development Stage B cohort."""

import hashlib
import json
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from biorag_repro.core.manifest import ArtifactRef, BuildInput
from biorag_repro.core.records import Record
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.historical_context_readback import read_case
from biorag_repro.evaluation.stage_b import (
    Inputs,
    stage_b_ref,
)
from biorag_repro.evaluation.stage_b import (
    dependencies as judgment_dependencies,
)
from biorag_repro.evaluation.stage_b import (
    transformation_id as judgment_transformation,
)
from biorag_repro.evaluation.stage_b_models import StageBJudgment
from biorag_repro.evaluation.stage_records import publish_record, read_record, stage_record_ref

PACKAGE_SHA256 = "e8754b85a0431f0ba11513cca5589153dd6fbee1d7b4710b2947b90c51a2a817"
APPROVAL_SHA256 = "08e67c1314826f78fc4da840714ee0cb3198b1d1240eec8861e0dab2b4be94ea"
CASES_SHA256 = "ab5f185cc6f5cb59c8274059eb4336929402dc2ad8f155e100e754fcc9650651"
PROVENANCE_AMENDMENT_SHA256 = "bc9f3c03d06d9f7fb91a38bc76f0029e73f10e024b8e63eb6255560ee044daca"
CONTEXT_AMENDMENT_SHA256 = "c2dddb18ee3e029b93480f29f1ddec956f6c349587ee3f6e3891af8ea143dc3a"
STAGE_C_OVERLAY_SHA256 = "ea7998ea3d890f4e8a125ad32d35f47a85ce467ce37746964d4cff1214f91bfa"
EXCEPTIONS = {14, 27, 86, 98, 175, 215, 228}
CORE = (
    "stage_b_accounting_v1.json",
    "stage_b_rationales_v1.json",
    "stage_c_rationales_v1.json",
    "stage_b_publication_drafts_v1.json",
    "stage_c_publication_drafts_v1.json",
    "researcher_review_v1.md",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _control_ref(name: str, digest: str) -> ArtifactRef:
    return ArtifactRef(
        artifact_id="control_" + digest,
        logical_ref="local_controls/" + name,
        sha256=digest,
    )


class ContributorFailure(Record):
    chunk_id: str
    occurrence_id: str
    declared_node_id: str
    owner_node_id: str | None
    canonical_sha256: str
    source_artifact_id: str
    source_sha256: str


class HistoricalStageBAccounting(Record):
    """A case-level accounting record; its type selects the historical reader explicitly."""

    publication_type: Literal["historical_supplied_context_v1"] = "historical_supplied_context_v1"
    case_number: int = Field(ge=1, le=283)
    judgment: ArtifactRef
    judgment_manifest: ArtifactRef
    observation_keys: list[str] = Field(min_length=1)
    item_id: str
    paper_id: str
    representation: Literal["JATS", "PDF_DOCLING"]
    source_sha256: str
    canonical_sha256: str
    context_sha256: str
    source_authenticity: Literal["PASS"]
    exact_context_delivery: Literal["PASS"]
    evidence_sufficiency: Literal["PASS"]
    selected_context_provenance: Literal["PASS", "FAILED"]
    complete_set_provenance: Literal["PASS", "FAILED"]
    complete_set_failure_count: int = Field(ge=0)
    selected_failures: list[ContributorFailure]
    researcher_disposition: Literal["APPROVED", "NOT_NEEDED"]
    approval_sha256: Literal[APPROVAL_SHA256]
    package_sha256: Literal[PACKAGE_SHA256]
    historical_readback_sha256: Literal[CASES_SHA256]
    provenance_amendment_sha256: Literal[PROVENANCE_AMENDMENT_SHA256]
    context_amendment_sha256: Literal[CONTEXT_AMENDMENT_SHA256]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if len(set(self.observation_keys)) != len(self.observation_keys):
            raise ValueError("duplicate historical observation identity")
        exception = self.case_number in EXCEPTIONS
        if (
            (self.selected_context_provenance == "FAILED") != exception
            or bool(self.selected_failures) != exception
            or (self.researcher_disposition == "APPROVED") != exception
        ):
            raise ValueError("historical selected-context accounting differs from approval")
        if (self.complete_set_provenance == "FAILED") != (self.complete_set_failure_count > 0):
            raise ValueError("complete-set accounting conflicts with its failure count")
        return self


class Authority:
    """Exact approved local controls, checked before a historical writer or reader is used."""

    def __init__(self, report_root: Path, frozen_root: Path) -> None:
        self.report_root = report_root
        self.frozen_root = frozen_root
        package = report_root / "stage_bc_publication_review_v1"
        lines = "".join(f"{name} {_sha(package / name)}\n" for name in CORE)
        if hashlib.sha256(lines.encode()).hexdigest() != PACKAGE_SHA256:
            raise ValueError("approved rationale package differs")
        for line in (package / "SHA256SUMS_V1").read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            if _sha(report_root / name) != digest:
                raise ValueError(f"approved package integrity differs: {name}")
        approval_path = report_root / "rationale_approval_record_v1.json"
        cases_path = report_root / "historical_context_readback_v1/stage_b_cases.json"
        if _sha(approval_path) != APPROVAL_SHA256 or _sha(cases_path) != CASES_SHA256:
            raise ValueError("researcher approval or historical readback differs")
        if (
            _sha(frozen_root / "AMENDMENT_STAGE_B_HISTORICAL_PROVENANCE_V1.md")
            != PROVENANCE_AMENDMENT_SHA256
            or _sha(frozen_root / "AMENDMENT_STAGE_B_HISTORICAL_CONTEXT_READBACK_V1.md")
            != CONTEXT_AMENDMENT_SHA256
        ):
            raise ValueError("governing historical amendment differs")
        approval = json.loads(approval_path.read_bytes())
        if (
            approval["review_package_sha256"] != PACKAGE_SHA256
            or approval["researcher"] != "Rama Krishna Pudota"
            or approval["decision"] != "approved_retrospective_rationale_wording"
            or approval["case_edits"]
            or approval["approval_scope"]["stage_b_rationale_records"] != 283
            or approval["approval_scope"]["stage_c_canonical_rationale_records"] != 133
            or approval["approval_scope"]["stage_c_legacy_ids"] != 134
            or set(approval["approval_scope"]["stage_b_provenance_failed_cases"]) != EXCEPTIONS
            or approval["approval_scope"]["stage_b_affected_observations"] != 11
        ):
            raise ValueError("researcher approval differs")
        self.reviewer = approval["researcher"]
        self.reviewed_utc = approval["reviewed_utc"]
        drafts = json.loads((package / "stage_b_publication_drafts_v1.json").read_bytes())
        cases = json.loads(cases_path.read_bytes())
        if [row["case_number"] for row in drafts] != list(range(1, 284)) or [
            row["case_number"] for row in cases
        ] != list(range(1, 284)):
            raise ValueError("historical cohort coverage differs")
        self.drafts = {row["case_number"]: row for row in drafts}
        self.cases = {row["case_number"]: row for row in cases}
        approved_cases = json.loads((package / "stage_b_accounting_v1.json").read_bytes())["cases"]
        if {row["case_number"] for row in approved_cases} != EXCEPTIONS:
            raise ValueError("approved provenance exception coverage differs")
        for approved_case in approved_cases:
            number = approved_case["case_number"]
            if (
                approved_case["observation_keys"] != self.cases[number]["observation_keys"]
                or approved_case["contributor_node_provenance_certification"] != "FAILED"
                or approved_case["delivery_decision"] != "YES"
            ):
                raise ValueError("approved exception observation accounting differs")
        observations = set()
        for number, case in self.cases.items():
            if (
                case["observation_keys"] != self.drafts[number]["observation_keys"]
                or any(
                    case[key] != "PASS"
                    for key in (
                        "source_authenticity",
                        "exact_context_delivery",
                        "evidence_sufficiency",
                    )
                )
                or (case["selected_context_provenance"] == "FAILED") != (number in EXCEPTIONS)
                or case["researcher_disposition"] == "RESEARCHER_DECISION_REQUIRED"
                or case["complete_set_provenance"] not in {"PASS", "FAILED"}
            ):
                raise ValueError(f"historical scientific accounting differs: {number}")
            for key in case["observation_keys"]:
                if key in observations:
                    raise ValueError("duplicate historical observation")
                observations.add(key)
        if (
            len(observations) != 449
            or sum(row["complete_set_provenance"] == "FAILED" for row in cases) != 117
        ):
            raise ValueError("historical observation or complete-set coverage differs")

    def judgment(self, number: int) -> StageBJudgment:
        return StageBJudgment.model_validate(
            {
                **self.drafts[number]["record"],
                "reviewer": self.reviewer,
                "at_utc": self.reviewed_utc,
            }
        )

    def accounting(
        self, number: int, judgment: ArtifactRef, manifest: ArtifactRef
    ) -> HistoricalStageBAccounting:
        case = self.cases[number]
        return HistoricalStageBAccounting.model_validate(
            {
                **{
                    key: case[key]
                    for key in (
                        "case_number",
                        "observation_keys",
                        "item_id",
                        "paper_id",
                        "representation",
                        "source_sha256",
                        "canonical_sha256",
                        "context_sha256",
                        "source_authenticity",
                        "exact_context_delivery",
                        "evidence_sufficiency",
                        "selected_context_provenance",
                        "complete_set_provenance",
                        "complete_set_failure_count",
                        "selected_failures",
                        "researcher_disposition",
                    )
                },
                "judgment": judgment,
                "judgment_manifest": manifest,
                "approval_sha256": APPROVAL_SHA256,
                "package_sha256": PACKAGE_SHA256,
                "historical_readback_sha256": CASES_SHA256,
                "provenance_amendment_sha256": PROVENANCE_AMENDMENT_SHA256,
                "context_amendment_sha256": CONTEXT_AMENDMENT_SHA256,
            }
        )


def _accounting_dependencies(value: HistoricalStageBAccounting) -> list[tuple[str, ArtifactRef]]:
    return [
        ("stage_b_judgment", value.judgment),
        ("judgment_manifest", value.judgment_manifest),
        ("researcher_approval", _control_ref("rationale_approval_record_v1.json", APPROVAL_SHA256)),
        ("historical_readback", _control_ref("stage_b_cases.json", CASES_SHA256)),
        (
            "provenance_amendment",
            _control_ref("historical_provenance_v1.md", PROVENANCE_AMENDMENT_SHA256),
        ),
        ("context_amendment", _control_ref("historical_context_v1.md", CONTEXT_AMENDMENT_SHA256)),
    ]


def accounting_ref(value: HistoricalStageBAccounting) -> ArtifactRef:
    return stage_record_ref(
        value,
        "stage_b_historical_v1",
        "stage_b",
        _accounting_dependencies(value),
        "stage_b_historical_supplied_context_v1",
    )


def _check_manifest(
    ref: ArtifactRef,
    manifest: ArtifactRef,
    root: Path,
    dependencies: list[tuple[str, ArtifactRef]],
    transformation: str,
) -> None:
    built = read_manifest(root / manifest.logical_ref, manifest.sha256)
    if (
        built.status != "complete"
        or [row.artifact for row in built.outputs] != [ref]
        or built.inputs
        != [BuildInput(role=role, artifact=artifact) for role, artifact in dependencies]
        or built.transformation_id != transformation
    ):
        raise ValueError("historical publication manifest does not seal its record")


def _verified_case(number: int, authority: Authority, case_inputs: dict) -> Inputs:
    if case_inputs.get("phase") != "historical_development":
        raise ValueError("historical publication cannot accept another phase")
    if case_inputs.get("packet", {}).get("number") != number:
        raise ValueError("historical publication case input differs")
    if case_inputs.get("draft") != authority.drafts[number]:
        raise ValueError("historical publication draft differs from approved package")
    result, inputs = read_case(**case_inputs, return_inputs=True)
    if result != authority.cases[number]:
        raise ValueError("historical scientific readback differs from pinned result")
    return inputs


def publish_case(
    number: int, authority: Authority, case_inputs: dict, root: Path, *, resume: bool = False
) -> tuple[ArtifactRef, ArtifactRef]:
    """Publish a verified judgment, then its case accounting; resume only identical bytes."""
    _verified_case(number, authority, case_inputs)
    judgment = authority.judgment(number)
    judgment_ref = stage_b_ref(judgment)
    judgment_ref, judgment_manifest = publish_record(
        judgment,
        judgment_ref,
        "stage_b_judgment",
        "stage_b",
        judgment_dependencies(judgment),
        judgment_transformation(judgment),
        root,
        reuse_identical=resume,
    )
    accounting = authority.accounting(number, judgment_ref, judgment_manifest)
    ref = accounting_ref(accounting)
    return publish_record(
        accounting,
        ref,
        "stage_b_historical_v1",
        "stage_b",
        _accounting_dependencies(accounting),
        "stage_b_historical_supplied_context_v1",
        root,
        reuse_identical=resume,
    )


def read_case_publication(
    ref: ArtifactRef,
    manifest: ArtifactRef,
    number: int,
    authority: Authority,
    case_inputs: dict,
    root: Path,
) -> tuple[StageBJudgment, HistoricalStageBAccounting, Inputs]:
    """Explicit historical readback; a missing or contradictory accounting record fails."""
    accounting = read_record(
        ref, root, HistoricalStageBAccounting, accounting_ref, "historical Stage B accounting"
    )
    _check_manifest(
        ref,
        manifest,
        root,
        _accounting_dependencies(accounting),
        "stage_b_historical_supplied_context_v1",
    )
    if accounting.case_number != number:
        raise ValueError("historical case number differs")
    expected = authority.accounting(number, accounting.judgment, accounting.judgment_manifest)
    if accounting != expected:
        raise ValueError("published historical accounting differs from approved readback")
    judgment = read_record(
        accounting.judgment, root, StageBJudgment, stage_b_ref, "historical Stage B judgment"
    )
    _check_manifest(
        accounting.judgment,
        accounting.judgment_manifest,
        root,
        judgment_dependencies(judgment),
        judgment_transformation(judgment),
    )
    if judgment != authority.judgment(number):
        raise ValueError("published historical scientific judgment differs from approval")
    inputs = _verified_case(number, authority, case_inputs)
    return judgment, accounting, inputs


class HistoricalCohortInventory(Record):
    """Final commit marker for the complete approved development adjudication cohort."""

    publication_type: Literal["historical_stage_bc_cohort_v1"] = "historical_stage_bc_cohort_v1"
    stage_b_accounting: list[ArtifactRef] = Field(min_length=283, max_length=283)
    stage_c_scores: list[ArtifactRef] = Field(min_length=483, max_length=483)
    stage_b_observation_count: Literal[449] = 449
    stage_b_selected_failed_cases: list[int]
    stage_b_complete_set_failed_count: Literal[117] = 117
    stage_c_canonical_count: Literal[133] = 133
    stage_c_legacy_count: Literal[134] = 134
    package_sha256: Literal[PACKAGE_SHA256] = PACKAGE_SHA256
    approval_sha256: Literal[APPROVAL_SHA256] = APPROVAL_SHA256
    stage_c_overlay_sha256: Literal[STAGE_C_OVERLAY_SHA256] = STAGE_C_OVERLAY_SHA256

    @model_validator(mode="after")
    def complete(self) -> Self:
        if self.stage_b_selected_failed_cases != sorted(EXCEPTIONS):
            raise ValueError("historical exception coverage differs")
        if len({ref.artifact_id for ref in self.stage_b_accounting}) != 283:
            raise ValueError("duplicate Stage B accounting identity")
        if len({ref.artifact_id for ref in self.stage_c_scores}) != 483:
            raise ValueError("duplicate Stage C score identity")
        return self


def _cohort_dependencies(value: HistoricalCohortInventory) -> list[tuple[str, ArtifactRef]]:
    return [
        *[("stage_b_accounting", ref) for ref in value.stage_b_accounting],
        *[("stage_c_score", ref) for ref in value.stage_c_scores],
        ("researcher_approval", _control_ref("rationale_approval_record_v1.json", APPROVAL_SHA256)),
        (
            "stage_c_overlay",
            _control_ref("stage_c_reference_overlay_v2.json", STAGE_C_OVERLAY_SHA256),
        ),
    ]


def cohort_ref(value: HistoricalCohortInventory) -> ArtifactRef:
    return stage_record_ref(
        value,
        "stage_bc_historical_cohort_v1",
        "stage_bc",
        _cohort_dependencies(value),
        "stage_bc_historical_cohort_v1",
    )


def publish_cohort_inventory(
    value: HistoricalCohortInventory, root: Path, *, resume: bool = False
) -> tuple[ArtifactRef, ArtifactRef]:
    for ref in [*value.stage_b_accounting, *value.stage_c_scores]:
        path = root / ref.logical_ref
        if not path.is_file() or _sha(path) != ref.sha256:
            raise ValueError("cohort output is missing or changed")
    ref = cohort_ref(value)
    return publish_record(
        value,
        ref,
        "stage_bc_historical_cohort_v1",
        "stage_bc",
        _cohort_dependencies(value),
        "stage_bc_historical_cohort_v1",
        root,
        reuse_identical=resume,
    )


def read_cohort_inventory(
    ref: ArtifactRef, manifest: ArtifactRef, expected: HistoricalCohortInventory, root: Path
) -> HistoricalCohortInventory:
    value = read_record(
        ref, root, HistoricalCohortInventory, cohort_ref, "historical Stage B/C cohort"
    )
    _check_manifest(
        ref, manifest, root, _cohort_dependencies(value), "stage_bc_historical_cohort_v1"
    )
    if value != expected:
        raise ValueError("historical cohort differs from validated complete inventory")
    for output in [*value.stage_b_accounting, *value.stage_c_scores]:
        path = root / output.logical_ref
        if not path.is_file() or _sha(path) != output.sha256:
            raise ValueError("historical cohort output is missing or changed")
    return value
