"""Python preflight record: what Python verified about the R inputs before R runs.

The record is built only from the objects `read_handoff` returns, never from caller-typed
hashes, and it states what was and was not established. It is immutable and is never replaced.

It is not an official certificate. A synthetic rehearsal is the only origin it can describe;
source-run lineage and the production seed are recorded as not established.

Ordering: this record has a nonce and a clock reading. A later R run record that cites the
SHA-256 of these bytes depends on them, which is operational sequencing. Neither the nonce, the
clock, nor the citation is cryptographic proof of when the verification happened.
"""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id, new_run_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.evaluation.analysis_handoff import VerifiedHandoff, read_handoff
from biorag_repro.generation.environment import GitState, check_utc, utc_now

SCHEMA_VERSION = "1"
TRANSFORMATION = "analysis_preflight_v1"
Status = Literal["verified_by_python", "not_established"]

# The only checks a preflight can hold, in order, with the status each can have in this schema
# version. A different status or an additional check is not representable.
CHECKS: Final[tuple[tuple[str, Status, str], ...]] = (
    (
        "handoff_bytes_sha256",
        "verified_by_python",
        "published handoff bytes hashed through its build manifest",
    ),
    ("table_csv_bytes_sha256", "verified_by_python", "published analysis CSV bytes hashed"),
    (
        "handoff_contract",
        "verified_by_python",
        "strict model, build lineage, settings hash and rebuild equality from the table (read_handoff)",
    ),
    (
        "table_contract",
        "verified_by_python",
        "rows, planned roster, inventory and references (read_table)",
    ),
    ("analysis_spec_frozen", "verified_by_python", "analysis_spec equals the frozen ANALYSIS_SPEC"),
    (
        "complete_case_accounting",
        "verified_by_python",
        "primary complete cases rederived from the table and equal to the handoff",
    ),
    (
        "origin_authorization",
        "verified_by_python",
        "origin is synthetic_rehearsal; authorized for rehearsal only",
    ),
    (
        "source_run_lineage",
        "not_established",
        "a synthetic handoff has no source runs; none were traversed",
    ),
    (
        "seed_approval",
        "not_established",
        "the production primary_bootstrap seed is awaiting approval",
    ),
)

NOT_ESTABLISHED: Final = (
    "official source-run lineage",
    "approval of the production primary_bootstrap seed",
    "authorship: no signature or key is involved",
    "that R will read these bytes: R re-hashes its own inputs and R re-hashes any cited preflight",
    (
        "when this verification happened: created_utc is a clock reading and the nonce is not a "
        "timestamp; sequencing before an R run is operational and not cryptographic proof"
    ),
    "any scientific authority: a synthetic rehearsal has none",
)


class PreflightCheck(Record):
    name: str
    status: Status
    scope: str


class PreflightOutcome(Record):
    outcome: str
    n_eligible: int = Field(ge=0)
    n_complete: int = Field(ge=0)


class PreflightVerifier(Record):
    tool: Literal["analysis_preflight_v1"]
    git: GitState


class AnalysisPreflight(Record):
    schema_version: Literal["1"] = "1"
    kind: Literal["analysis_preflight"] = "analysis_preflight"
    preflight_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_utc: str
    purpose: Literal["synthetic_rehearsal"]
    scientific_authority: Literal["none"]
    input_origin: Literal["synthetic_rehearsal"]
    handoff_build: ArtifactRef
    handoff: ArtifactRef
    table_build: ArtifactRef
    csv: ArtifactRef
    inventory: ArtifactRef
    roster: ArtifactRef
    csv_rows: int = Field(ge=0)
    analysis_spec_version: str
    analysis_spec_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accounting: list[PreflightOutcome] = Field(min_length=2, max_length=2)
    checks: list[PreflightCheck]
    not_established: list[str]
    verifier: PreflightVerifier

    _utc = field_validator("created_utc")(check_utc)

    @model_validator(mode="after")
    def claims_are_the_fixed_set(self) -> Self:
        if [(c.name, c.status, c.scope) for c in self.checks] != list(CHECKS):
            raise ValueError("checks differ from the claims this schema version can make")
        if self.not_established != list(NOT_ESTABLISHED):
            raise ValueError("the statement of what is not established was changed")
        return self


class PreflightError(ValueError):
    """A stored preflight differs from what the verified handoff yields."""


def settings_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {"schema_version": SCHEMA_VERSION, "checks": [list(c) for c in CHECKS]}
        )
    )


def _claims(handoff_build: ArtifactRef, root: Path) -> tuple[dict[str, Any], VerifiedHandoff]:
    """Everything a preflight asserts about the handoff, taken from the verified objects."""
    verified = read_handoff(handoff_build, root)
    handoff = verified.handoff
    stored = read_manifest(root / handoff_build.logical_ref, handoff_build.sha256)
    return (
        {
            "purpose": "synthetic_rehearsal",
            "scientific_authority": "none",
            "input_origin": handoff.input_origin,
            "handoff_build": handoff_build,
            "handoff": stored.outputs[0].artifact,
            "table_build": handoff.table_build,
            "csv": handoff.csv,
            "inventory": handoff.inventory,
            "roster": handoff.roster,
            "csv_rows": len(verified.rows),
            "analysis_spec_version": handoff.analysis_spec.spec_version,
            "analysis_spec_sha256": sha256_bytes(
                canonical_json_bytes(handoff.analysis_spec.model_dump(mode="json"))
            ),
            "accounting": [
                PreflightOutcome(
                    outcome=o.outcome,
                    n_eligible=len(o.eligible_items),
                    n_complete=len(o.complete_item_ids),
                )
                for o in handoff.accounting
            ],
            "checks": [PreflightCheck(name=n, status=s, scope=c) for n, s, c in CHECKS],
            "not_established": list(NOT_ESTABLISHED),
        },
        verified,
    )


def _inputs(preflight: AnalysisPreflight) -> list[BuildInput]:
    return [
        BuildInput(role="analysis_handoff_build", artifact=preflight.handoff_build),
        BuildInput(role="analysis_handoff", artifact=preflight.handoff),
    ]


def _store(preflight: AnalysisPreflight, root: Path) -> tuple[ArtifactRef, ArtifactRef]:
    """Write a preflight and its build manifest without replacing anything."""
    data = serialize_project_json(preflight, AnalysisPreflight)
    digest = sha256_bytes(data)
    inputs = _inputs(preflight)
    artifact_id = derived_artifact_id(
        artifact_kind="analysis_preflight",
        inputs=[ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs],
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings_sha256(),
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    ref = ArtifactRef(
        artifact_id=artifact_id, logical_ref=f"analysis/preflight/{artifact_id}.json", sha256=digest
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=settings_sha256(),
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind="analysis_preflight",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        data,
        digest,
        validate_temp=lambda p: read_project_json(p, digest, AnalysisPreflight),
    )
    build_ref = f"analysis/preflight/{sha256_bytes(canonical_json_bytes([artifact_id]))}.build.json"
    manifest_digest = publish_manifest(manifest, root / build_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(
        artifact_id="build_" + manifest_digest, logical_ref=build_ref, sha256=manifest_digest
    )


def publish_preflight(
    handoff_build: ArtifactRef,
    root: Path,
    *,
    code: GitState,
    now: Callable[[], str] = utc_now,
    new_id: Callable[[], str] = new_run_id,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Verify a rehearsal handoff, then publish the preflight; returns (file, build) references.

    Nothing is written unless `read_handoff` accepts the handoff, and nothing is replaced.
    """
    claims, _ = _claims(handoff_build, root)
    preflight = AnalysisPreflight(
        preflight_id=new_id(),
        created_utc=now(),
        verifier=PreflightVerifier(tool="analysis_preflight_v1", git=code),
        **claims,
    )
    read_back, build = _store(preflight, root)
    read_preflight(build, root)
    return read_back, build


@dataclass(frozen=True)
class VerifiedPreflight:
    preflight: AnalysisPreflight
    handoff: VerifiedHandoff


def read_preflight(build: ArtifactRef, root: Path) -> VerifiedPreflight:
    """Check bytes, lineage and every claim against a fresh `read_handoff` of its handoff."""
    manifest = read_manifest(root / build.logical_ref, build.sha256)
    if len(manifest.outputs) != 1 or manifest.outputs[0].artifact_kind != "analysis_preflight":
        raise PreflightError("build manifest does not describe an analysis preflight")
    if manifest.transformation_id != TRANSFORMATION or (
        manifest.transformation_settings_sha256 != settings_sha256()
    ):
        raise PreflightError("preflight was built under other settings")
    ref = manifest.outputs[0].artifact
    preflight = read_project_json(root / ref.logical_ref, ref.sha256, AnalysisPreflight)
    if manifest.inputs != _inputs(preflight):
        raise PreflightError("preflight lineage differs from its build manifest")
    claims, verified = _claims(preflight.handoff_build, root)
    differing = sorted(k for k, v in claims.items() if getattr(preflight, k) != v)
    if differing:
        raise PreflightError(f"preflight differs from the handoff it describes: {differing}")
    return VerifiedPreflight(preflight, verified)
