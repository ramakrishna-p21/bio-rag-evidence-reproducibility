"""Matching the qualified settings does not approve a model, environment or execution."""

from pathlib import Path

import pytest

from biorag_repro.generation.config import require_qualified_settings, resolve_generator_config


def settings():
    return resolve_generator_config(
        (Path(__file__).parent / "fixtures/qualified_generator.toml").read_bytes()
    ).settings


@pytest.mark.parametrize(
    "arm",
    ["confirmatory", "dense_robustness", "chunk_robustness", "scope_robustness", "closed_book"],
)
def test_only_matrix_coordinates_and_retrieval_reference_vary(arm):
    base = settings()
    changed = base.model_copy(
        update={
            "matrix_arm": arm,
            "retrieval_config_sha256": None if arm == "closed_book" else "a" * 64,
        }
    )
    require_qualified_settings(changed)


def test_changed_qualified_output_allowance_is_refused():
    base = settings()
    changed = base.model_copy(
        update={"budgets": base.budgets.model_copy(update={"max_new_tokens": 129})}
    )
    with pytest.raises(ValueError, match="qualified"):
        require_qualified_settings(changed)
