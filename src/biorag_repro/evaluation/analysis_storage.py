"""Immutable publication and verified read-back of the analysis table."""

from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, verify_file_sha256
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
)
from biorag_repro.evaluation.analysis_models import (
    COLUMNS,
    MATRIX_ID,
    PRIMARY_REPEAT_INDEX,
    PRIMARY_SELECTION,
    SCHEMA_VERSION,
    AnalysisInventory,
    AnalysisRow,
    PlannedRoster,
)
from biorag_repro.evaluation.analysis_table import (
    SCORING_REFERENCE,
    Table,
    check_references,
    check_roster,
    check_rows,
    make_inventory,
    parse_csv,
    serialize_csv,
)

TRANSFORMATION = "analysis_table_v1"


def _settings_sha256() -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": SCHEMA_VERSION,
                "columns": COLUMNS,
                "primary_repeat_index": PRIMARY_REPEAT_INDEX,
                "primary_selection": PRIMARY_SELECTION,
                "matrix": MATRIX_ID,
                "scoring_reference": SCORING_REFERENCE,
            }
        )
    )


def _ref(kind: str, suffix: str, digest: str, table: Table) -> ArtifactRef:
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[
            ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in table.inputs
        ],
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=_settings_sha256(),
        output_sha256=digest,
        output_format=suffix,
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"analysis/{kind}/{artifact_id}.{suffix}",
        sha256=digest,
    )


def publish_table(table: Table, root: Path) -> tuple[ArtifactRef, ArtifactRef, ArtifactRef]:
    """Publish the CSV, its inventory, the planned roster and the build manifest.

    Never replaces anything. Rows must be exactly the roster's planned observations.
    """
    rows = sorted(
        table.rows,
        key=lambda r: (r.matrix_arm, r.item_id, r.condition, r.repeat_index, r.observation_key),
    )
    check_rows(rows)
    check_references(rows, table.inputs)
    check_roster(rows, table.roster)
    canonical = Table(rows, table.roster, table.inputs)
    data = serialize_csv(rows)
    csv_ref = _ref("analysis_table", "csv", sha256_bytes(data), canonical)
    roster_bytes = serialize_project_json(table.roster, PlannedRoster)
    roster_ref = _ref("planned_roster", "json", sha256_bytes(roster_bytes), canonical)
    inventory = make_inventory(rows, csv_ref, table.roster, roster_ref)
    inventory_bytes = serialize_project_json(inventory, AnalysisInventory)
    inventory_ref = _ref("analysis_inventory", "json", sha256_bytes(inventory_bytes), canonical)
    manifest = BuildManifest(
        schema_version="1",
        inputs=table.inputs,
        transformation_id=TRANSFORMATION,
        transformation_settings_sha256=_settings_sha256(),
        outputs=[
            DerivedOutput(
                artifact=csv_ref,
                artifact_kind="analysis_table",
                output_format="csv",
                serialization_version="1",
                identity_version="1",
            ),
            DerivedOutput(
                artifact=inventory_ref,
                artifact_kind="analysis_inventory",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            ),
            DerivedOutput(
                artifact=roster_ref,
                artifact_kind="planned_roster",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            ),
        ],
        status="complete",
    )
    paths = {
        csv_ref.logical_ref: root / csv_ref.logical_ref,
        inventory_ref.logical_ref: root / inventory_ref.logical_ref,
        roster_ref.logical_ref: root / roster_ref.logical_ref,
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        paths[csv_ref.logical_ref],
        data,
        csv_ref.sha256,
        validate_temp=lambda p: parse_csv(p.read_bytes()),
    )
    publish_bytes(
        paths[roster_ref.logical_ref],
        roster_bytes,
        roster_ref.sha256,
        validate_temp=lambda p: read_project_json(p, roster_ref.sha256, PlannedRoster),
    )
    publish_bytes(
        paths[inventory_ref.logical_ref],
        inventory_bytes,
        inventory_ref.sha256,
        validate_temp=lambda p: read_project_json(p, inventory_ref.sha256, AnalysisInventory),
    )
    build_id = sha256_bytes(
        canonical_json_bytes(
            [csv_ref.artifact_id, inventory_ref.artifact_id, roster_ref.artifact_id]
        )
    )
    logical_ref = f"analysis/build/{build_id}.build.json"
    (root / logical_ref).parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(manifest, root / logical_ref, paths)
    return (
        csv_ref,
        inventory_ref,
        ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest),
    )


def read_table(build: ArtifactRef, root: Path) -> list[AnalysisRow]:
    """Verify manifest, bytes, rows, inventory and source citations; return the rows."""
    manifest = read_manifest(root / build.logical_ref, build.sha256)
    outputs = {o.artifact_kind: o.artifact for o in manifest.outputs}
    if set(outputs) != {"analysis_table", "analysis_inventory", "planned_roster"}:
        raise ValueError("build manifest does not describe an analysis table")
    if manifest.transformation_id != TRANSFORMATION or (
        manifest.transformation_settings_sha256 != _settings_sha256()
    ):
        raise ValueError("analysis table was built under other settings")
    csv_ref, inventory_ref = outputs["analysis_table"], outputs["analysis_inventory"]
    roster_ref = outputs["planned_roster"]
    verify_file_sha256(root / csv_ref.logical_ref, csv_ref.sha256)
    rows = parse_csv((root / csv_ref.logical_ref).read_bytes())
    check_references(rows, manifest.inputs)
    roster = read_project_json(root / roster_ref.logical_ref, roster_ref.sha256, PlannedRoster)
    check_roster(rows, roster)
    inventory = read_project_json(
        root / inventory_ref.logical_ref, inventory_ref.sha256, AnalysisInventory
    )
    if inventory != make_inventory(rows, csv_ref, roster, roster_ref):
        raise ValueError("inventory differs from the table it describes")
    return rows
