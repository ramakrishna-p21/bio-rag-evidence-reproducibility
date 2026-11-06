"""Generator configuration identity. Every value here is a synthetic test setting."""

import copy
import tomllib

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.generation.config import (
    GeneratorConfiguration,
    check_retrieval_reference,
    resolve_generator_config,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TOKENIZER_FILES = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def synthetic_settings() -> dict:
    return {
        "schema_version": "1",
        "purpose": "synthetic_test",
        "experiment_plan_version": "synthetic-plan-0",
        "matrix_arm": "confirmatory",
        "retrieval_config_sha256": SHA_A,
        "model": {"repository": "Qwen/Qwen3-8B", "revision": QWEN_REVISION, "dtype": "bfloat16"},
        "tokenizer": {
            "repository": "Qwen/Qwen3-8B",
            "revision": QWEN_REVISION,
            "file_sha256": dict(TOKENIZER_FILES),
        },
        "serving_path": "transformers",
        "thinking_mode": "non_thinking",
        "budgets": {
            "evidence_tokens": 2048,
            "max_input_tokens": 4096,
            "max_new_tokens": 128,
            "context_window_tokens": 8192,
        },
        "decoding": {"do_sample": False},
        "prompt_sha256": SHA_A,
        "output_schema_sha256": SHA_B,
        "parser_version": "short_answer_json_v1",
        "seed_registry_version": "synthetic-seeds-0",
        "seed_registry_sha256": SHA_B,
        "retry_max_attempts": 2,
        "retryable_failures": ["out_of_memory", "transport_error"],
        "scoring_version": "synthetic-scoring-0",
    }


def to_toml(settings: dict, prefix: str = "") -> str:
    scalars, tables = [], []
    for key, value in settings.items():
        name = f'"{key}"'
        if isinstance(value, dict):
            tables.append((name, value))
        elif isinstance(value, bool):
            scalars.append(f"{name} = {str(value).lower()}")
        elif isinstance(value, str):
            scalars.append(f'{name} = "{value}"')
        else:
            scalars.append(f"{name} = {value!r}")
    text = "\n".join(scalars) + "\n"
    for name, value in tables:
        path = f"{prefix}{name}"
        text += f"\n[{path}]\n" + to_toml(value, prefix=path + ".")
    return text


def resolve(settings: dict):
    return resolve_generator_config(to_toml(settings).encode())


def test_toml_helper_round_trips():
    assert tomllib.loads(to_toml(synthetic_settings())) == synthetic_settings()


def test_synthetic_configuration_resolves_with_stable_hash():
    first, second = resolve(synthetic_settings()), resolve(synthetic_settings())
    assert first.configuration_sha256 == second.configuration_sha256
    assert first.configuration_sha256 == sha256_bytes(first.canonical_json)
    assert isinstance(first.settings, GeneratorConfiguration)


def test_hash_ignores_toml_formatting_and_key_order():
    settings = synthetic_settings()
    reordered = dict(reversed(list(settings.items())))
    plain = resolve_generator_config(to_toml(settings).encode())
    noisy = resolve_generator_config(
        ("# a comment\n\n" + to_toml(reordered).replace(" = ", "   =   ")).encode()
    )
    assert plain.configuration_sha256 == noisy.configuration_sha256
    assert plain.raw_toml_sha256 != noisy.raw_toml_sha256


@pytest.mark.parametrize(
    "path,value",
    [
        (("purpose",), "development_candidate"),
        (("matrix_arm",), "dense_robustness"),
        (("retrieval_config_sha256",), SHA_B),
        (("model", "revision"), "c" * 40),
        (("budgets", "max_new_tokens"), 256),
        (("budgets", "max_input_tokens"), 4097),
        (("prompt_sha256",), SHA_B),
        (("output_schema_sha256",), SHA_A),
        (("seed_registry_sha256",), SHA_A),
        (("retry_max_attempts",), 3),
        (("retryable_failures",), ["transport_error"]),
        (("retryable_failures",), ["out_of_memory", "timeout", "transport_error"]),
        (("retryable_failures",), []),
        (("scoring_version",), "synthetic-scoring-1"),
        (("experiment_plan_version",), "synthetic-plan-1"),
    ],
)
def test_every_scientific_setting_changes_identity(path, value):
    changed = synthetic_settings()
    target = changed
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    if path == (("purpose",)):
        changed["model"]["repository"] = "Qwen/Qwen3-8B"
    assert (
        resolve(changed).configuration_sha256 != resolve(synthetic_settings()).configuration_sha256
    )


def test_sampling_decoding_is_explicit():
    settings = synthetic_settings()
    settings["decoding"] = {"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": 20}
    assert resolve(settings).settings.decoding.temperature == 0.7


def drop(settings: dict, *path: str) -> dict:
    out = copy.deepcopy(settings)
    target = out
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    return out


@pytest.mark.parametrize(
    "path",
    [
        ("purpose",),
        ("matrix_arm",),
        ("model",),
        ("model", "revision"),
        ("model", "dtype"),
        ("tokenizer", "file_sha256"),
        ("serving_path",),
        ("thinking_mode",),
        ("budgets", "evidence_tokens"),
        ("budgets", "max_new_tokens"),
        ("budgets", "context_window_tokens"),
        ("decoding",),
        ("decoding", "do_sample"),
        ("prompt_sha256",),
        ("output_schema_sha256",),
        ("parser_version",),
        ("seed_registry_sha256",),
        ("retry_max_attempts",),
        ("retryable_failures",),
        ("scoring_version",),
        ("schema_version",),
    ],
)
def test_missing_settings_are_rejected(path):
    with pytest.raises(ValueError):
        resolve(drop(synthetic_settings(), *path))


def test_unknown_fields_are_rejected_at_every_level():
    for where in ((), ("model",), ("budgets",), ("decoding",), ("tokenizer",)):
        settings = synthetic_settings()
        target = settings
        for key in where:
            target = target[key]
        target["quantization"] = "int4"
        with pytest.raises(ValueError):
            resolve(settings)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s: s["decoding"].update(temperature=0.7),  # greedy with a temperature
        lambda s: s["decoding"].update(top_k=5),
        lambda s: s.update(decoding={"do_sample": True}),  # sampling without its settings
        lambda s: s.update(
            decoding={"do_sample": True, "temperature": 0.0, "top_p": 0.9, "top_k": 20}
        ),
        lambda s: s.update(
            decoding={"do_sample": True, "temperature": 0.7, "top_p": 1.5, "top_k": 20}
        ),
        lambda s: s.update(
            decoding={"do_sample": True, "temperature": 0.7, "top_p": 0.9, "top_k": -1}
        ),
        lambda s: s["model"].update(revision="main"),
        lambda s: s["model"].update(revision=QWEN_REVISION.upper()),
        lambda s: s["model"].update(dtype="int8"),
        lambda s: s["tokenizer"].update(revision="main"),
        lambda s: s["tokenizer"].update(file_sha256={"tokenizer.json": "xyz"}),
        lambda s: s["tokenizer"].update(file_sha256={}),
        lambda s: s.update(serving_path="vllm"),
        lambda s: s.update(thinking_mode="thinking"),
        lambda s: s.update(parser_version="lenient_v9"),
        lambda s: s.update(prompt_sha256="abc"),
        lambda s: s.update(retry_max_attempts=0),
        lambda s: s.update(schema_version="2"),
        lambda s: s["budgets"].update(evidence_tokens=4096),  # frozen evidence cap
        lambda s: s["budgets"].update(evidence_tokens=1024),
        lambda s: s["budgets"].update(max_new_tokens=0),
        lambda s: s["budgets"].update(max_input_tokens=2048),  # no room beyond the evidence
        lambda s: s["budgets"].update(context_window_tokens=4200),  # input + output exceed window
        lambda s: s.pop("retrieval_config_sha256"),  # a retrieval arm must name one
        lambda s: s.update(matrix_arm="closed_book"),  # closed book names no retrieval config
        lambda s: s.update(matrix_arm="not_an_arm"),
    ],
)
def test_conflicting_or_invalid_values_are_rejected(mutate):
    settings = synthetic_settings()
    mutate(settings)
    with pytest.raises(ValueError):
        resolve(settings)


def test_closed_book_has_no_retrieval_reference():
    settings = synthetic_settings()
    settings["matrix_arm"] = "closed_book"
    del settings["retrieval_config_sha256"]
    assert resolve(settings).settings.retrieval_config_sha256 is None


def test_values_are_not_coerced():
    settings = synthetic_settings()
    settings["budgets"]["max_new_tokens"] = "128"
    with pytest.raises(ValueError):
        resolve(settings)
    settings = synthetic_settings()
    settings["retry_max_attempts"] = True
    with pytest.raises(ValueError):
        resolve(settings)


def test_development_candidate_requires_the_official_checkpoint():
    settings = synthetic_settings()
    settings["purpose"] = "development_candidate"
    assert resolve(settings).settings.purpose == "development_candidate"
    for section in ("model", "tokenizer"):
        broken = copy.deepcopy(settings)
        broken[section]["repository"] = "tiny-random/qwen3"
        with pytest.raises(ValueError):
            resolve(broken)


def test_synthetic_identity_never_equals_a_candidate_identity():
    candidate = synthetic_settings()
    candidate["purpose"] = "development_candidate"
    assert (
        resolve(candidate).configuration_sha256
        != resolve(synthetic_settings()).configuration_sha256
    )


def test_retrieval_reference_must_match_phase5_configuration():
    config = resolve(synthetic_settings()).settings
    check_retrieval_reference(config, SHA_A)
    with pytest.raises(ValueError):
        check_retrieval_reference(config, SHA_B)
    closed = synthetic_settings()
    closed["matrix_arm"] = "closed_book"
    del closed["retrieval_config_sha256"]
    with pytest.raises(ValueError):
        check_retrieval_reference(resolve(closed).settings, SHA_A)
    check_retrieval_reference(resolve(closed).settings, None)


# ---- retry policy is part of the configuration identity -----------------------------------


def test_retryable_failure_order_does_not_change_identity_and_is_stored_sorted():
    a = synthetic_settings() | {"retryable_failures": ["transport_error", "out_of_memory"]}
    b = synthetic_settings() | {"retryable_failures": ["out_of_memory", "transport_error"]}
    first, second = resolve(a), resolve(b)
    assert first.configuration_sha256 == second.configuration_sha256
    assert first.settings.retryable_failures == ["out_of_memory", "transport_error"]


@pytest.mark.parametrize("value", [["a", "a"], [" "], [""], "transport_error", [1]])
def test_invalid_retryable_failures_are_rejected(value):
    with pytest.raises(ValueError):
        resolve(synthetic_settings() | {"retryable_failures": value})


def test_an_empty_retry_category_list_is_explicit_and_distinct():
    none = resolve(synthetic_settings() | {"retryable_failures": []})
    some = resolve(synthetic_settings())
    assert none.settings.retryable_failures == []
    assert none.configuration_sha256 != some.configuration_sha256
