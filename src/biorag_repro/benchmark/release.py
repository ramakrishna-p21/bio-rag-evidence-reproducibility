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


def read_final_test_items(data_root) -> tuple[str, list]:
    """Return (release record SHA-256, the 305 frozen final-test items).

    Protected: call only after execution authorization. Errors never carry item content.
    """
    from biorag_repro.benchmark.models import BenchmarkItem
    from biorag_repro.benchmark.split import PaperSplit
    from biorag_repro.benchmark.storage import decode_jsonl
    from biorag_repro.core.hashing import sha256_bytes
    from biorag_repro.readiness import evidence
    from biorag_repro.readiness.files import read_regular

    record_bytes = read_regular(data_root, evidence.BENCHMARK_RELEASE)
    record = BenchmarkRelease.model_validate_json(record_bytes)
    data = read_regular(data_root, record.items.logical_ref)
    if sha256_bytes(data) != record.items.sha256:
        raise ValueError("frozen benchmark items differ from the release")
    split_bytes = read_regular(data_root, record.split.logical_ref)
    if sha256_bytes(split_bytes) != record.split.sha256:
        raise ValueError("frozen paper split differs from the release")
    final_papers = set(PaperSplit.model_validate_json(split_bytes).final_test_ids)
    items = [i for i in decode_jsonl(data, BenchmarkItem) if i.paper_id in final_papers]
    if len(items) != 305 or len({i.item_id for i in items}) != 305:
        raise ValueError("frozen final-test roster does not hold 305 distinct items")
    return sha256_bytes(record_bytes), items
