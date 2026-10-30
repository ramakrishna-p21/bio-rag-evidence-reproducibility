import json
from datetime import date
from hashlib import sha256
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from biorag_repro.core.config import resolve_config


class SyntheticSettings(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)

    schema_version: Literal["1"] = "1"
    threshold: float = 0.25
    enabled: bool = True
    method: str = "exact"


def test_defaults_and_explicit_default_have_same_scientific_identity():
    implicit = resolve_config(b"", SyntheticSettings)
    explicit = resolve_config(
        b'schema_version = "1"\nthreshold = 0.25\nenabled = true\nmethod = "exact"\n',
        SyntheticSettings,
    )
    expected = b'{"enabled":true,"method":"exact","schema_version":"1","threshold":0.25}'

    assert implicit.settings == SyntheticSettings()
    assert implicit.canonical_json == expected
    assert implicit.configuration_sha256 == sha256(expected).hexdigest()
    assert explicit.configuration_sha256 == implicit.configuration_sha256
    assert explicit.raw_toml_sha256 != implicit.raw_toml_sha256


def test_comments_whitespace_and_key_order_do_not_change_configuration_identity():
    first = resolve_config(b'method = "bm25"\nthreshold = 0.5\n', SyntheticSettings)
    second = resolve_config(
        b'\n# same effective settings\nthreshold=0.5\n\nmethod = "bm25" # comment\n',
        SyntheticSettings,
    )

    assert first.configuration_sha256 == second.configuration_sha256
    assert first.raw_toml_sha256 != second.raw_toml_sha256


def test_scientific_setting_change_changes_configuration_identity():
    first = resolve_config(b"threshold = 0.25\n", SyntheticSettings)
    second = resolve_config(b"threshold = 0.5\n", SyntheticSettings)
    assert first.configuration_sha256 != second.configuration_sha256


def test_hashed_settings_match_returned_typed_settings():
    class TransformingSettings(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)
        schema_version: Literal["1"] = "1"
        threshold: float

        @field_validator("threshold")
        @classmethod
        def adjust_threshold(cls, value: float) -> float:
            return value + 1.0

    resolved = resolve_config(b"threshold = 0.25\n", TransformingSettings)
    assert json.loads(resolved.canonical_json)["threshold"] == resolved.settings.threshold


@pytest.mark.parametrize(
    "coordinate", ["item_id", "representation", "repeat_index", "actual_repeat_seed"]
)
def test_observation_coordinates_are_not_accepted_as_shared_settings(coordinate):
    with pytest.raises(ValidationError):
        resolve_config(f"{coordinate} = 1\n".encode(), SyntheticSettings)


def test_unknown_settings_and_wrong_types_are_rejected():
    with pytest.raises(ValidationError):
        resolve_config(b"unknown = 1\n", SyntheticSettings)
    with pytest.raises(ValidationError):
        resolve_config(b'threshold = "0.25"\n', SyntheticSettings)


def test_nonfinite_configuration_value_is_rejected():
    with pytest.raises((TypeError, ValueError)):
        resolve_config(b"threshold = nan\n", SyntheticSettings)


def test_invalid_declared_default_is_rejected():
    class BadDefaultSettings(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)
        schema_version: Literal["1"] = "1"
        threshold: float = "not a number"  # type: ignore[assignment]

    with pytest.raises(ValidationError):
        resolve_config(b"", BadDefaultSettings)


def test_model_must_validate_declared_defaults():
    class NonvalidatingDefaults(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid")
        schema_version: Literal["1"] = "1"
        threshold: float = Field(default=-1.0, ge=0)

    with pytest.raises(ValueError, match="validate_default"):
        resolve_config(b"", NonvalidatingDefaults)


def test_configuration_model_requires_schema_version():
    class UnversionedSettings(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)
        threshold: float = 0.25

    with pytest.raises(ValueError, match="schema_version"):
        resolve_config(b"", UnversionedSettings)


def test_non_json_toml_date_is_rejected_without_implicit_string_conversion():
    class DatedSettings(BaseModel):
        model_config = ConfigDict(strict=True, extra="forbid", validate_default=True)
        schema_version: Literal["1"] = "1"
        day: date

    with pytest.raises((TypeError, ValueError)):
        resolve_config(b"day = 2026-10-30\n", DatedSettings)


def test_raw_toml_hash_is_exact_byte_hash_not_configuration_identity():
    raw = b"# provenance only\n"
    resolved = resolve_config(raw, SyntheticSettings)
    assert resolved.raw_toml_sha256 == sha256(raw).hexdigest()
    assert resolved.raw_toml_sha256 != resolved.configuration_sha256
