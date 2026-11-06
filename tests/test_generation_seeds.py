"""Generation seed registry. Seed integers are arbitrary synthetic test values."""

import pytest
from pydantic import ValidationError
from test_generation_config import resolve, synthetic_settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.generation.seeds import (
    GenerationSeedRegistry,
    check_seeds_for_configuration,
    repeat_seed,
    seed_registry_sha256,
)

CONSUMER = "transformers.set_seed"


def registry(seeds=(11, 22, 33), version="synthetic-seeds-0"):
    return GenerationSeedRegistry(
        registry_version=version,
        entries=[
            {"name": f"generation_repeat_{i}", "seed": s, "consumer": CONSUMER}
            for i, s in enumerate(seeds, 1)
        ],
    )


def sampling_config(reg, arm="confirmatory", sample=True):
    settings = synthetic_settings()
    settings["matrix_arm"] = arm
    if arm == "closed_book":
        del settings["retrieval_config_sha256"]
    if sample:
        settings["decoding"] = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    settings["seed_registry_version"] = reg.registry_version
    settings["seed_registry_sha256"] = seed_registry_sha256(reg)
    return resolve(settings).settings


def test_registry_hash_is_canonical_json_of_the_registry():
    reg = registry()
    expected = sha256_bytes(canonical_json_bytes(reg.model_dump(mode="json")))
    assert seed_registry_sha256(reg) == expected
    assert seed_registry_sha256(registry()) == expected
    assert seed_registry_sha256(registry((11, 22, 34))) != expected


def test_same_named_seed_serves_every_representation():
    reg = registry()
    assert [repeat_seed(reg, i) for i in (1, 2, 3)] == [11, 22, 33]


@pytest.mark.parametrize("bad", [0, 4, -1, True])
def test_repeat_seed_rejects_unknown_repeats(bad):
    with pytest.raises(ValueError):
        repeat_seed(registry(), bad)


def test_missing_repeat_entry_is_an_error_not_a_default():
    reg = GenerationSeedRegistry(
        registry_version="v",
        entries=[{"name": "generation_repeat_1", "seed": 5, "consumer": CONSUMER}],
    )
    assert repeat_seed(reg, 1) == 5
    with pytest.raises(ValueError):
        repeat_seed(reg, 2)


@pytest.mark.parametrize(
    "entry",
    [
        {"name": "generation_repeat_1", "seed": -1, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": 2**32, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": 1.5, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": True, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": "7", "consumer": CONSUMER},
        {"name": "generation_repeat_4", "seed": 7, "consumer": CONSUMER},
        {"name": "primary_bootstrap", "seed": 7, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": 7, "consumer": " "},
        {"name": "generation_repeat_1", "seed": 7, "consumer": CONSUMER, "extra": 1},
        {"name": "generation_repeat_1", "seed": 7},
    ],
)
def test_invalid_entries_are_rejected(entry):
    with pytest.raises(ValidationError):
        GenerationSeedRegistry(registry_version="v", entries=[entry])


def test_duplicate_names_and_seeds_are_rejected():
    same_name = [
        {"name": "generation_repeat_1", "seed": 1, "consumer": CONSUMER},
        {"name": "generation_repeat_1", "seed": 2, "consumer": CONSUMER},
    ]
    with pytest.raises(ValidationError):
        GenerationSeedRegistry(registry_version="v", entries=same_name)
    with pytest.raises(ValidationError):
        registry((5, 5, 6))  # repeats must use distinct seeds


def test_empty_registry_and_blank_version_are_rejected():
    with pytest.raises(ValidationError):
        GenerationSeedRegistry(registry_version="v", entries=[])
    with pytest.raises(ValidationError):
        registry(version=" ")


def test_sampling_configuration_needs_all_three_confirmatory_seeds():
    reg = registry()
    check_seeds_for_configuration(sampling_config(reg), reg)
    check_seeds_for_configuration(sampling_config(reg, "closed_book"), reg)
    short = GenerationSeedRegistry(
        registry_version=reg.registry_version,
        entries=[e for e in reg.model_dump()["entries"][:2]],
    )
    bad = sampling_config(short)
    with pytest.raises(ValueError):
        check_seeds_for_configuration(bad, short)


def test_robustness_arm_only_needs_repeat_one():
    one = GenerationSeedRegistry(
        registry_version="v",
        entries=[{"name": "generation_repeat_1", "seed": 9, "consumer": CONSUMER}],
    )
    check_seeds_for_configuration(sampling_config(one, "dense_robustness"), one)


def test_registry_must_be_the_one_the_configuration_names():
    reg, other = registry(), registry((1, 2, 3))
    config = sampling_config(reg)
    with pytest.raises(ValueError):
        check_seeds_for_configuration(config, other)
    wrong_version = registry(version="another-version")
    with pytest.raises(ValueError):
        check_seeds_for_configuration(config, wrong_version)


def test_greedy_configuration_does_not_require_seeds_but_checks_identity():
    reg = registry()
    config = sampling_config(reg, sample=False)
    check_seeds_for_configuration(config, reg)
    with pytest.raises(ValueError):
        check_seeds_for_configuration(config, registry((1, 2, 3)))
