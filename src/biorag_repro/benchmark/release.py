"""Final benchmark release record; publication occurs only after implementation commit."""

import re
from typing import Self

from pydantic import field_validator, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record


class BenchmarkRelease(Record):
    schema_version: str = "1"
    benchmark_version: str
    corpus_version: str
    corpus_content_sha256: str
    corpus_release: ArtifactRef
    implementation_git_sha: str
    seed_registry: ArtifactRef
    relatedness: ArtifactRef
    split: ArtifactRef
    items: ArtifactRef
    decisions: ArtifactRef
    dispositions: ArtifactRef
    pre_audit_population: ArtifactRef
    audit: ArtifactRef
    validation_report: ArtifactRef
    item_count: int
    development_paper_count: int
    final_test_paper_count: int

    @field_validator("corpus_content_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        validate_sha256(value)
        return value

    @field_validator("implementation_git_sha")
    @classmethod
    def git_sha(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError("implementation Git SHA must be full lowercase SHA-1")
        return value

    @model_validator(mode="after")
    def counts(self) -> Self:
        if (
            self.item_count < 1
            or self.development_paper_count != 10
            or self.final_test_paper_count != 40
        ):
            raise ValueError("release requires items and exact 10/40 split")
        return self
