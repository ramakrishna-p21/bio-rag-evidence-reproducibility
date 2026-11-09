"""Named generation seeds: explicit integers, never derived or implied.

Seeds are consumed only when decoding samples. Corresponding JATS and PDF_DOCLING calls
(and closed-book calls) use the same named repeat seed.
"""

import re
from typing import Annotated, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.generation.config import GeneratorConfiguration

_REPEAT_NAME = re.compile(r"generation_repeat_([1-3])")
MAX_SEED = 2**32 - 1  # the range accepted by the numpy-based seeding in transformers
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]


class GenerationSeed(Record):
    name: str
    seed: int
    consumer: Nonblank  # library call that consumes the seed

    @field_validator("name")
    @classmethod
    def repeat_name(cls, value: str) -> str:
        if _REPEAT_NAME.fullmatch(value) is None:
            raise ValueError("generation seeds are named generation_repeat_1 to _3")
        return value

    @field_validator("seed")
    @classmethod
    def seed_range(cls, value: int) -> int:
        if not 0 <= value <= MAX_SEED:
            raise ValueError(f"seed must lie in [0, {MAX_SEED}]")
        return value


class GenerationSeedRegistry(Record):
    schema_version: str = "1"
    registry_version: Nonblank
    entries: list[GenerationSeed] = Field(min_length=1)

    @model_validator(mode="after")
    def distinct(self) -> Self:
        names = [entry.name for entry in self.entries]
        seeds = [entry.seed for entry in self.entries]
        if len(set(names)) != len(names):
            raise ValueError("duplicate seed name")
        if len(set(seeds)) != len(seeds):
            raise ValueError("repeats must use distinct seeds")
        return self


def seed_registry_sha256(registry: GenerationSeedRegistry) -> str:
    return sha256_bytes(canonical_json_bytes(registry.model_dump(mode="json")))


def repeat_seed(registry: GenerationSeedRegistry, repeat_index: int) -> int:
    if type(repeat_index) is not int or repeat_index not in (1, 2, 3):
        raise ValueError("repeat_index must be 1, 2 or 3")
    name = f"generation_repeat_{repeat_index}"
    for entry in registry.entries:
        if entry.name == name:
            return entry.seed
    raise ValueError(f"seed registry has no entry {name}")


def check_seeds_for_configuration(
    config: GeneratorConfiguration, registry: GenerationSeedRegistry
) -> None:
    """The registry must be the one the configuration names and cover its repeats if sampling."""
    if (
        registry.registry_version != config.seed_registry_version
        or seed_registry_sha256(registry) != config.seed_registry_sha256
    ):
        raise ValueError("seed registry is not the one the configuration names")
    if not config.decoding.do_sample:
        return
    repeats = (1, 2, 3) if config.matrix_arm in ("confirmatory", "closed_book") else (1,)
    for repeat in repeats:
        repeat_seed(registry, repeat)
