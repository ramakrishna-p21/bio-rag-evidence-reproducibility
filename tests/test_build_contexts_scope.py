"""A development-only paper pool cannot stand in for the frozen scope arm."""

import pytest

from scripts.build_contexts import check_scope_pool


def test_scope_arm_requires_exactly_the_frozen_fifty_paper_ids():
    frozen = [f"P{i:03d}" for i in range(1, 51)]
    check_scope_pool(["scope_robustness"], frozen)
    with pytest.raises(SystemExit, match="all 50"):
        check_scope_pool(["scope_robustness"], frozen[:10])
    with pytest.raises(SystemExit, match="all 50"):
        check_scope_pool(["scope_robustness"], frozen[:-1] + ["P051"])


def test_other_arms_may_use_a_development_pool_but_never_duplicate_a_paper():
    check_scope_pool(["confirmatory", "closed_book"], ["P006", "P007"])
    with pytest.raises(SystemExit, match="twice"):
        check_scope_pool(["confirmatory"], ["P006", "P006"])


def test_published_builder_requires_complete_sources_before_opening_any_source(tmp_path):
    from scripts.build_contexts import build_published

    selection = {
        "pool_papers": [f"P{n:03d}" for n in range(1, 51)],
        "arms": [
            "confirmatory",
            "dense_robustness",
            "chunk_robustness",
            "scope_robustness",
            "closed_book",
        ],
        "canonical": {},
    }
    with pytest.raises(ValueError, match="100 source selections"):
        build_published(selection, tmp_path / "contexts.json", tmp_path)
    assert not list(tmp_path.iterdir())


def test_published_designated_queries_still_require_full_corpus_statistics(tmp_path):
    from scripts.build_contexts import build_published

    with pytest.raises(SystemExit, match="all 50"):
        build_published(
            {"arms": ["confirmatory"], "pool_papers": ["P006"]},
            tmp_path / "contexts.json",
            tmp_path,
        )
