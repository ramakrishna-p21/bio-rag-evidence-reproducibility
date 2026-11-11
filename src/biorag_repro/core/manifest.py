"""Versioned build lineage and artifact references."""

import ntpath
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.records import Record


class ArtifactRef(Record):
    artifact_id: str = Field(min_length=1)
    logical_ref: str = Field(min_length=1)
    sha256: str

    @field_validator("artifact_id", "logical_ref")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("reference fields must be nonblank")
        return value

    @field_validator("logical_ref")
    @classmethod
    def nonlocal_reference(cls, value: str) -> str:
        if value.startswith("/") or ntpath.isabs(value):
            raise ValueError("logical_ref must not be an absolute local path")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_hash(cls, value: str) -> str:
        validate_sha256(value)
        return value


class BuildInput(Record):
    role: str = Field(min_length=1)
    artifact: ArtifactRef

    @field_validator("role")
    @classmethod
    def nonblank_role(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("role must be nonblank")
        return value


class DerivedOutput(Record):
    artifact: ArtifactRef
    artifact_kind: str = Field(min_length=1)
    output_format: str = Field(min_length=1)
    serialization_version: str = Field(min_length=1)
    identity_version: str = Field(min_length=1)

    @field_validator("artifact_kind", "output_format", "serialization_version", "identity_version")
    @classmethod
    def nonblank_identity_field(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("output identity fields must be nonblank")
        return value


class BuildManifest(Record):
    schema_version: Literal["1"]
    inputs: list[BuildInput] = Field(min_length=1)
    transformation_id: str = Field(min_length=1)
    transformation_settings_sha256: str | None = None
    outputs: list[DerivedOutput]
    status: Literal["complete", "failed"]
    failure_reason: str | None = None

    @field_validator("transformation_id")
    @classmethod
    def nonblank_transformation(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("transformation_id must be nonblank")
        return value

    @field_validator("transformation_settings_sha256")
    @classmethod
    def valid_settings_hash(cls, value: str | None) -> str | None:
        if value is not None:
            validate_sha256(value)
        return value

    @model_validator(mode="after")
    def validate_lineage_and_status(self) -> Self:
        if self.status == "complete":
            if not self.outputs:
                raise ValueError("complete build requires an output")
            if self.failure_reason is not None:
                raise ValueError("complete build cannot have a failure reason")
        elif self.failure_reason is None or not self.failure_reason.strip():
            raise ValueError("failed build requires a failure reason")

        inputs = [
            ArtifactInput(item.role, item.artifact.artifact_id, item.artifact.sha256)
            for item in self.inputs
        ]
        for output in self.outputs:
            expected_id = derived_artifact_id(
                artifact_kind=output.artifact_kind,
                inputs=inputs,
                transformation_id=self.transformation_id,
                transformation_settings_sha256=self.transformation_settings_sha256,
                output_sha256=output.artifact.sha256,
                output_format=output.output_format,
                serialization_version=output.serialization_version,
                identity_version=output.identity_version,
            )
            if output.artifact.artifact_id != expected_id:
                raise ValueError("derived output artifact_id does not match its lineage")
        return self
