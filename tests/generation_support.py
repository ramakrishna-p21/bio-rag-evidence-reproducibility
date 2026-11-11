"""Shared synthetic generator settings, prompt artifact and the pinned tokenizer root.

Every value is a synthetic test setting; none is the project's scientific configuration.
"""

import os
from pathlib import Path

import pytest

from biorag_repro.generation.config import resolve_generator_config
from biorag_repro.generation.parsing import OUTPUT_SCHEMAS
from biorag_repro.generation.prompt import PromptArtifact

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


REVISION = "b968826d9c46dd6066d109eabc6255188de91218"


ROOT = Path(
    os.environ.get(
        "BIORAG_QWEN_ROOT", Path(__file__).parents[1] / "models" / "qwen3-8b-tokenizer" / REVISION
    )
)


needs_tokenizer = pytest.mark.skipif(not ROOT.is_dir(), reason="pinned Qwen tokenizer not present")


PARSER = "short_answer_json_v1"


WITH_CONTEXT = (
    "Answer the question using only the evidence below. If it is insufficient, abstain.\n\n"
    "{context}\n\nQuestion: {question}"
)


CLOSED_BOOK = "Answer the question from your own knowledge, or abstain.\n\nQuestion: {question}"


def artifact(**changes):
    fields = {
        "prompt_version": "synthetic-prompt-0",
        "with_context": WITH_CONTEXT,
        "closed_book": CLOSED_BOOK,
        "parser_version": PARSER,
        "output_schema": OUTPUT_SCHEMAS[PARSER],
    }
    return PromptArtifact(**{**fields, **changes})
