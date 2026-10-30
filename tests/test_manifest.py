import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildManifest


def manifest_data(status="complete"):
    source = {"artifact_id": "src_P001_JATS", "logical_ref": "source:P001:JATS", "sha256": "a" * 64}
    output_hash = "b" * 64
    output_id = derived_artifact_id(
        artifact_kind="canonical_document",
        inputs=[ArtifactInput("source", source["artifact_id"], source["sha256"])],
        transformation_id="canonicalize_jats@1",
        transformation_settings_sha256="c" * 64,
        output_sha256=output_hash,
        output_format="json",
        serialization_version="1",
    )
    return {
        "schema_version": "1",
        "inputs": [{"role": "source", "artifact": source}],
        "transformation_id": "canonicalize_jats@1",
        "transformation_settings_sha256": "c" * 64,
        "outputs": [
            {
                "artifact": {
                    "artifact_id": output_id,
                    "logical_ref": "derived:P001:JATS:canonical",
                    "sha256": output_hash,
                },
                "artifact_kind": "canonical_document",
                "output_format": "json",
                "serialization_version": "1",
                "identity_version": "1",
            }
        ],
        "status": status,
        "failure_reason": None
        if status == "complete"
        else "parser returned an incomplete document",
    }


def test_reference_accepts_abstract_logical_ref_and_rejects_unknown_fields():
    ref = ArtifactRef.model_validate(manifest_data()["inputs"][0]["artifact"])
    assert ref.logical_ref == "source:P001:JATS"
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate({**ref.model_dump(), "unexpected": True})


@pytest.mark.parametrize(
    "logical_ref", ["", "  ", "/home/user/file", "C:\\data\\file", "\\\\server\\share\\file"]
)
def test_reference_rejects_empty_or_machine_absolute_ref(logical_ref):
    data = {**manifest_data()["inputs"][0]["artifact"], "logical_ref": logical_ref}
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(data)


@pytest.mark.parametrize(
    "field,value", [("artifact_id", ""), ("sha256", "A" * 64), ("sha256", "bad")]
)
def test_reference_rejects_invalid_identifiers(field, value):
    data = {**manifest_data()["inputs"][0]["artifact"], field: value}
    with pytest.raises(ValidationError):
        ArtifactRef.model_validate(data)


def test_manifest_validates_lineage_and_round_trips_without_filesystem_access():
    with patch("pathlib.Path.open", side_effect=AssertionError("filesystem access")):
        manifest = BuildManifest.model_validate(manifest_data())
        restored = BuildManifest.model_validate_json(manifest.model_dump_json())
    assert restored == manifest
    assert [item.role for item in manifest.inputs] == ["source"]


def test_input_order_is_preserved_and_affects_derived_identity():
    data = manifest_data()
    second = {
        "role": "settings",
        "artifact": {"artifact_id": "settings_v1", "logical_ref": "config:1", "sha256": "d" * 64},
    }
    data["inputs"].append(second)
    inputs = [
        ArtifactInput(item["role"], item["artifact"]["artifact_id"], item["artifact"]["sha256"])
        for item in data["inputs"]
    ]
    output = data["outputs"][0]
    output["artifact"]["artifact_id"] = derived_artifact_id(
        artifact_kind=output["artifact_kind"],
        inputs=inputs,
        transformation_id=data["transformation_id"],
        transformation_settings_sha256=data["transformation_settings_sha256"],
        output_sha256=output["artifact"]["sha256"],
        output_format=output["output_format"],
        serialization_version=output["serialization_version"],
        identity_version=output["identity_version"],
    )
    assert [item.role for item in BuildManifest.model_validate(data).inputs] == [
        "source",
        "settings",
    ]
    data["inputs"].reverse()
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(data)


@pytest.mark.parametrize(
    "change",
    [
        {"schema_version": "2"},
        {"schema_version": None},
        {"inputs": []},
        {"transformation_id": ""},
        {"transformation_settings_sha256": "bad"},
        {"outputs": []},
        {"failure_reason": "unexpected"},
        {"unknown": 1},
    ],
)
def test_complete_manifest_rejects_invalid_structure(change):
    with pytest.raises(ValidationError):
        BuildManifest.model_validate({**manifest_data(), **change})


def test_manifest_requires_schema_version():
    data = manifest_data()
    data.pop("schema_version")
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(data)


def test_manifest_rejects_derived_id_or_output_hash_inconsistency():
    data = manifest_data()
    data["outputs"][0]["artifact"]["sha256"] = "d" * 64
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(data)


def test_optional_manifest_fields_can_be_omitted():
    data = manifest_data()
    data.pop("transformation_settings_sha256")
    data.pop("failure_reason")
    output = data["outputs"][0]
    source = data["inputs"][0]["artifact"]
    output["artifact"]["artifact_id"] = derived_artifact_id(
        artifact_kind=output["artifact_kind"],
        inputs=[ArtifactInput("source", source["artifact_id"], source["sha256"])],
        transformation_id=data["transformation_id"],
        transformation_settings_sha256=None,
        output_sha256=output["artifact"]["sha256"],
        output_format=output["output_format"],
        serialization_version=output["serialization_version"],
        identity_version=output["identity_version"],
    )
    manifest = BuildManifest.model_validate(data)
    assert manifest.transformation_settings_sha256 is None
    assert manifest.failure_reason is None


@pytest.mark.parametrize(
    "location",
    [
        "role",
        "transformation_id",
        "artifact_kind",
        "output_format",
        "serialization_version",
        "identity_version",
    ],
)
def test_manifest_rejects_blank_identity_fields(location):
    data = manifest_data()
    if location == "role":
        data["inputs"][0]["role"] = "  "
    elif location == "transformation_id":
        data[location] = "  "
    else:
        data["outputs"][0][location] = "  "
    with pytest.raises(ValidationError):
        BuildManifest.model_validate(data)


def test_failed_manifest_requires_reason_and_may_retain_partial_outputs():
    assert len(BuildManifest.model_validate(manifest_data("failed")).outputs) == 1
    for reason in (None, "", "  "):
        with pytest.raises(ValidationError):
            BuildManifest.model_validate({**manifest_data("failed"), "failure_reason": reason})
    assert BuildManifest.model_validate({**manifest_data("failed"), "outputs": []}).outputs == []


def test_generated_schema_has_nested_definitions_and_matches_model():
    path = Path(__file__).resolve().parents[1] / "schemas" / "build_manifest.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert {"ArtifactRef", "BuildInput", "DerivedOutput"} <= set(schema["$defs"])
    assert schema["properties"]["schema_version"]["const"] == "1"
    assert schema == BuildManifest.model_json_schema()


def test_schema_check_detects_drift_without_modifying_tracked_schema(tmp_path):
    from scripts.export_schemas import check_schema, schema_bytes

    target = tmp_path / "build_manifest.schema.json"
    target.write_bytes(schema_bytes())
    assert check_schema(target)
    target.write_bytes(b"{}\n")
    assert not check_schema(target)


def test_schema_check_cli_succeeds_when_current():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/export_schemas.py", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
