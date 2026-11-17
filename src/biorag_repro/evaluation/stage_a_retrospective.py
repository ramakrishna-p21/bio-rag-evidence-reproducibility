"""Bounded retrospective publication; the original prospective readers remain unchanged."""

from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import read_verified_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.models import Label, SetLocus, StageAIdentity, StageAJudgment
from biorag_repro.evaluation.stage_a import (
    _index,
    _scope_titles_for,
    _spans,
    check_item,
    stage_a_dependencies,
    stage_a_ref,
)
from biorag_repro.evaluation.stage_a import validate_judgment as validate_prospective
from biorag_repro.evaluation.stage_a_storage import read_judgment as read_prospective
from biorag_repro.evaluation.stage_records import publish_record, read_record, stage_record_ref
from biorag_repro.generation.environment import check_utc
from biorag_repro.readiness.files import loads_strict

PROTOCOL = "researcher_approved_retrospective_transcription_v1"
APPROVED_DECISIONS_SHA256 = "293f494a3f7a9cc14a68be1fbd9f6f800b8764450f9f983ef514e91b252e1b24"
SESSION_SHA256 = "3b5a1754747b954d79e2d3b7ee4140e94b732d32a24cefae2997a48b48d1c0de"
AUTHORIZATION_SHA256 = "375da084a719766c476093d096197884dbbdbeb603887961ba7ee29bd99ea521"
KIND = "stage_a_retrospective"
# A focused resolution needs separate, explicit researcher approval of its exact bytes.
APPROVED_COMPONENT_RESOLUTIONS: frozenset[str] = frozenset()


class RetrospectiveJudgment(StageAIdentity):
    schema_version: Literal["2"] = "2"
    basis: Literal["researcher_approved_retrospective_transcription"] = (
        "researcher_approved_retrospective_transcription"
    )
    protocol: Literal["researcher_approved_retrospective_transcription_v1"] = PROTOCOL
    handle: str = Field(pattern=r"R\d{3}")
    label: Label
    candidate: ArtifactRef
    sets: list[SetLocus]
    alteration: str | None = None
    rationale: str = Field(min_length=1)
    reviewer: Literal["Rama Krishna Pudota"] = "Rama Krishna Pudota"
    recorded_at_utc: str
    historical_review_at_utc: None = None
    started_utc: None = None
    finished_utc: None = None
    representation_guess: None = None
    guess_confidence: None = None
    guessed_utc: None = None
    consulted_original_source: None = None
    historical_metadata_status: Literal["unavailable"] = "unavailable"
    eligible_for_review_timing_or_guess_analysis: Literal[False] = False
    approved_decisions: ArtifactRef
    session_manifest: ArtifactRef
    coordinate_map: ArtifactRef
    authorization: ArtifactRef
    component_resolution: ArtifactRef | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    _utc = field_validator("recorded_at_utc")(check_utc)

    @model_validator(mode="after")
    def evidence_shape(self) -> Self:
        if len({s.set_index for s in self.sets}) != len(self.sets):
            raise ValueError("duplicate evidence-set locus")
        if self.label in {"preserved", "altered_but_sufficient"} and not self.sets:
            raise ValueError("sufficient judgment requires complete component placements")
        if (
            self.label in {"altered_but_sufficient", "scientifically_corrupted"}
            and not self.alteration
        ):
            raise ValueError("altered or corrupted evidence needs its approved description")
        if self.label == "absent" and self.sets:
            raise ValueError("absent evidence has no placements")
        for s in self.sets:
            for p in s.placements:
                if self.component_resolution is None and any(
                    c.equivalence_note is not None or c.answer_surface is not None
                    for c in p.components
                ):
                    raise ValueError("ambiguous equivalence requires focused researcher resolution")
        return self


Judgment = StageAJudgment | RetrospectiveJudgment


def dependencies(value: RetrospectiveJudgment) -> list[tuple[str, ArtifactRef]]:
    result = stage_a_dependencies(value) + [
        ("stage_a_candidate", value.candidate),
        ("approved_decisions", value.approved_decisions),
        ("session_manifest", value.session_manifest),
        ("coordinate_map", value.coordinate_map),
        ("retrospective_authorization", value.authorization),
    ]
    if value.component_resolution is not None:
        result.append(("focused_component_resolution", value.component_resolution))
    return result


def judgment_ref(value: Judgment) -> ArtifactRef:
    if isinstance(value, StageAJudgment):
        return stage_a_ref(value)
    return stage_record_ref(value, KIND, "stage_a", dependencies(value), PROTOCOL)


def _source(ref: ArtifactRef, root: Path) -> dict:
    return loads_strict(read_verified_bytes(root / ref.logical_ref, ref.sha256))


def validate_retrospective(
    value: RetrospectiveJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> None:
    value = RetrospectiveJudgment.model_validate(value.model_dump())
    check_item(item, document, canonical)
    if (
        value.approved_decisions.sha256,
        value.session_manifest.sha256,
        value.authorization.sha256,
    ) != (
        APPROVED_DECISIONS_SHA256,
        SESSION_SHA256,
        AUTHORIZATION_SHA256,
    ):
        raise ValueError("retrospective publication is outside the approved exact scope")
    approval = _source(value.approved_decisions, root)
    session = _source(value.session_manifest, root)
    authorization = _source(value.authorization, root)
    if value.coordinate_map.sha256 != session["unmasked_evidence_sha256"]:
        raise ValueError("coordinate map differs from the approved session")
    coordinates = _source(value.coordinate_map, root)
    rows = {r["handle"]: r for r in approval["records"]}
    if len(rows) != 156 or value.handle not in rows or value.handle not in coordinates:
        raise ValueError("judgment does not name one of the 156 approved handles")
    row, coordinate = rows[value.handle], coordinates[value.handle]
    if (
        row["researcher_concurred_label"] != value.label
        or row["researcher_concurred_rationale"] != value.rationale
        or coordinate["item_sha256"] != value.item_sha256
        or coordinate["representation"] != value.representation
        or ArtifactRef(**coordinate["canonical"]) != value.canonical
        or authorization["approved_decisions_sha256"] != value.approved_decisions.sha256
        or authorization["protocol"] != value.protocol
    ):
        raise ValueError("retrospective judgment changes approved science or coordinates")
    expected_alteration = (
        value.rationale
        if value.label in {"altered_but_sufficient", "scientifically_corrupted"}
        else None
    )
    if value.alteration != expected_alteration:
        raise ValueError("alteration description differs from the approved rationale")
    # Reuse scientific validation without constructing prospective review metadata.
    validate_prospective(value, item, document, canonical)
    if value.component_resolution is not None:
        if value.component_resolution.sha256 not in APPROVED_COMPONENT_RESOLUTIONS:
            raise ValueError("focused component resolution has no registered researcher approval")
        resolution = _source(value.component_resolution, root)
        if (
            resolution["protocol"] != PROTOCOL
            or resolution["handle"] != value.handle
            or resolution["approved_decisions_sha256"] != value.approved_decisions.sha256
            or resolution["item_sha256"] != value.item_sha256
            or resolution["canonical_sha256"] != value.canonical.sha256
            or resolution["sets"] != [s.model_dump(mode="json") for s in value.sets]
        ):
            raise ValueError("focused resolution names different evidence or coordinates")
        index = _index(document, canonical)
        for s in value.sets:
            for p in s.placements:
                for c in p.components:
                    matches = [
                        q
                        for q in resolution["quotes"]
                        if (q["set_index"], q["container_node_id"], q["role"], q["component_text"])
                        == (s.set_index, p.container_node_id, c.role, c.text)
                    ]
                    if len(matches) != 1 or not matches[0]["canonical_quote"]:
                        raise ValueError("focused resolution needs one genuine quote per component")
                    quote = matches[0]["canonical_quote"]
                    containers = {p.container_node_id} | _scope_titles_for(
                        index, p.container_node_id, item, s.set_index
                    )
                    anchored = {
                        i
                        for container in containers
                        for unit in index.units[container]
                        for i in unit.ids(_spans(unit.text, quote, bounded=False))
                    }
                    if not set(c.occurrence_ids) <= anchored:
                        raise ValueError(
                            "focused component quote does not carry its occurrence anchors"
                        )


def publish_judgment(
    value: RetrospectiveJudgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_retrospective(value, item, document, canonical, root)
    return publish_record(
        value, judgment_ref(value), KIND, "stage_a", dependencies(value), PROTOCOL, root
    )


def read_judgment(
    ref: ArtifactRef,
    root: Path,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
) -> Judgment:
    if not ref.logical_ref.startswith(f"evaluation/stage_a/{KIND}/"):
        return read_prospective(ref, root, item, document, canonical)
    value = read_record(ref, root, RetrospectiveJudgment, judgment_ref, "retrospective Stage A")
    validate_retrospective(value, item, document, canonical, root)
    return value


def validate_scientific_judgment(
    value: Judgment,
    item: BenchmarkItem,
    document: CanonicalDocument,
    canonical: ArtifactRef,
    root: Path | None = None,
) -> None:
    if isinstance(value, RetrospectiveJudgment):
        if root is None:
            raise ValueError("retrospective evidence requires its approval artifact root")
        checked = RetrospectiveJudgment.model_validate(value.model_dump())
        validate_retrospective(checked, item, document, canonical, root)
    else:
        validate_prospective(value, item, document, canonical)
