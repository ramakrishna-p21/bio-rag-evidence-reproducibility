"""Named prospective seeds for benchmark construction operations."""

from typing import Self

from pydantic import Field, model_validator

from biorag_repro.benchmark.models import Record


class SeedEntry(Record):
    name: str
    seed: int
    algorithm: str


class BenchmarkSeedRegistry(Record):
    schema_version: str = "1"
    registry_version: str = "benchmark_seeds_v1"
    entries: list[SeedEntry] = Field(min_length=2)

    @model_validator(mode="after")
    def required_names(self) -> Self:
        names = [entry.name for entry in self.entries]
        if len(names) != len(set(names)) or not {"paper_split", "benchmark_audit"} <= set(names):
            raise ValueError("seed registry requires distinct split and audit entries")
        return self


def prospective_seeds() -> BenchmarkSeedRegistry:
    return BenchmarkSeedRegistry(
        entries=[
            SeedEntry(name="paper_split", seed=3102026, algorithm="python_random_shuffle_dp_v1"),
            SeedEntry(name="benchmark_audit", seed=3102027, algorithm="python_random_sample_v1"),
        ]
    )
