import re

import pytest

from biorag_repro.core.identity import (
    ArtifactInput,
    derived_artifact_id,
    new_attempt_id,
    new_run_id,
    observation_key,
)


def artifact_args():
    return {
        "artifact_kind": "canonical_document",
        "inputs": [ArtifactInput("source", "src_P001_JATS", "a" * 64)],
        "transformation_id": "canonicalize_jats@1",
        "transformation_settings_sha256": "c" * 64,
        "output_sha256": "b" * 64,
        "output_format": "json",
        "serialization_version": "1",
    }


def observation_args():
    return {
        "phase": "development",
        "benchmark_version": "bench_v1",
        "item_id": "Q001",
        "matrix_arm": "confirmatory",
        "representation_or_condition": "JATS",
        "repeat_index": 0,
        "resolved_config_sha256": "d" * 64,
    }


def test_derived_artifact_id_matches_fixed_canonical_payload_oracle():
    artifact_id = derived_artifact_id(
        artifact_kind="canonical_document",
        inputs=[ArtifactInput("source", "src_P001_JATS", "a" * 64)],
        transformation_id="canonicalize_jats@1",
        transformation_settings_sha256="c" * 64,
        output_sha256="b" * 64,
        output_format="json",
        serialization_version="1",
    )

    assert artifact_id == ("drv_b263d2de5015200b4ed0877a5147bc0b417497bcc3f587de3e11a9a3c6d462b5")


def test_derived_id_is_deterministic_and_has_expected_shape():
    first = derived_artifact_id(**artifact_args())
    assert first == derived_artifact_id(**artifact_args())
    assert re.fullmatch(r"drv_[0-9a-f]{64}", first)


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("artifact_kind", "chunk_set"),
        ("inputs", [ArtifactInput("other", "src_P001_JATS", "a" * 64)]),
        ("inputs", [ArtifactInput("source", "src_P002_JATS", "a" * 64)]),
        ("inputs", [ArtifactInput("source", "src_P001_JATS", "d" * 64)]),
        ("transformation_id", "canonicalize_jats@2"),
        ("transformation_settings_sha256", "d" * 64),
        ("transformation_settings_sha256", None),
        ("output_sha256", "d" * 64),
        ("output_format", "jsonl"),
        ("serialization_version", "2"),
        ("identity_version", "2"),
    ],
)
def test_changing_any_identity_coordinate_changes_derived_id(field, different):
    original = artifact_args()
    changed = {**original, field: different}
    assert derived_artifact_id(**changed) != derived_artifact_id(**original)


def test_derived_id_preserves_explicit_input_order():
    first = ArtifactInput("source", "src_P001_JATS", "a" * 64)
    second = ArtifactInput("settings", "settings_v1", "c" * 64)
    original = artifact_args()
    original["inputs"] = [first, second]
    reversed_inputs = {**original, "inputs": [second, first]}
    assert derived_artifact_id(**original) != derived_artifact_id(**reversed_inputs)


@pytest.mark.parametrize("bad_hash", ["abc", "A" * 64, "g" * 64, "a" * 63])
@pytest.mark.parametrize("location", ["input", "settings", "output"])
def test_derived_id_rejects_malformed_hashes(location, bad_hash):
    args = artifact_args()
    if location == "input":
        args["inputs"] = [ArtifactInput("source", "src_P001_JATS", bad_hash)]
    elif location == "settings":
        args["transformation_settings_sha256"] = bad_hash
    else:
        args["output_sha256"] = bad_hash
    with pytest.raises(ValueError, match="SHA-256"):
        derived_artifact_id(**args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_kind", ""),
        ("inputs", []),
        ("inputs", [ArtifactInput("", "src_P001_JATS", "a" * 64)]),
        ("inputs", [ArtifactInput("source", "", "a" * 64)]),
        ("transformation_id", ""),
        ("output_format", ""),
        ("serialization_version", ""),
        ("identity_version", ""),
    ],
)
def test_derived_id_rejects_missing_identity_fields(field, value):
    with pytest.raises(ValueError):
        derived_artifact_id(**{**artifact_args(), field: value})


@pytest.mark.parametrize(
    ("bad_inputs", "error"),
    [
        (["source"], TypeError),
        ([ArtifactInput(7, "src_P001_JATS", "a" * 64)], ValueError),
        ([ArtifactInput("source", 7, "a" * 64)], ValueError),
    ],
)
def test_derived_id_rejects_invalid_input_references(bad_inputs, error):
    with pytest.raises(error, match="input"):
        derived_artifact_id(**{**artifact_args(), "inputs": bad_inputs})


def test_observation_key_matches_fixed_canonical_payload_oracle():
    assert observation_key(**observation_args()) == (
        "obs_a3fbf7221ae128b2534505842fdb91d7ff11a9fb68a633981a5d815ea6c49471"
    )


@pytest.mark.parametrize(
    ("field", "different"),
    [
        ("phase", "final_test"),
        ("benchmark_version", "bench_v2"),
        ("item_id", "Q002"),
        ("matrix_arm", "closed_book"),
        ("representation_or_condition", "PDF_DOCLING"),
        ("repeat_index", 1),
        ("resolved_config_sha256", "e" * 64),
        ("identity_version", "2"),
    ],
)
def test_each_observation_coordinate_changes_key(field, different):
    original = observation_args()
    assert observation_key(**{**original, field: different}) != observation_key(**original)


def test_observation_key_is_deterministic_and_has_expected_shape():
    key = observation_key(**observation_args())
    assert key == observation_key(**observation_args())
    assert re.fullmatch(r"obs_[0-9a-f]{64}", key)


@pytest.mark.parametrize("bad_hash", ["abc", "D" * 64, "g" * 64])
def test_observation_key_rejects_malformed_config_hash(bad_hash):
    with pytest.raises(ValueError, match="SHA-256"):
        observation_key(**{**observation_args(), "resolved_config_sha256": bad_hash})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", ""),
        ("benchmark_version", ""),
        ("item_id", ""),
        ("matrix_arm", ""),
        ("representation_or_condition", ""),
        ("repeat_index", -1),
        ("repeat_index", True),
        ("identity_version", ""),
    ],
)
def test_observation_key_rejects_invalid_coordinates(field, value):
    with pytest.raises(ValueError):
        observation_key(**{**observation_args(), field: value})


@pytest.mark.parametrize("generate", [new_run_id, new_attempt_id])
def test_execution_ids_are_unique_uuid4_hex_locators(generate):
    values = [generate() for _ in range(20)]
    assert len(set(values)) == len(values)
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in values)
    assert all(value[12] == "4" and value[16] in "89ab" for value in values)
