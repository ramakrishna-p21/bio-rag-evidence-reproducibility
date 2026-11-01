from collections import Counter

from biorag_repro.benchmark.split import PaperGroup, PaperMeta, make_split
from biorag_repro.core.storage import serialize_project_json


def papers() -> list[PaperMeta]:
    return [
        PaperMeta(paper_id=f"P{i:03d}", primary_domain=f"domain_{i % 10}") for i in range(1, 51)
    ]


def test_split_replay_and_exact_counts() -> None:
    first = make_split(papers(), [], seed=3102026)
    second = make_split(list(reversed(papers())), [], seed=3102026)
    assert first.development_ids == second.development_ids
    assert first.shuffled_groups == second.shuffled_groups
    assert len(first.development_ids) == 10
    assert len(first.final_test_ids) == 40
    assert len(first.represented_domains) == 10
    assert first.development_ids == [
        "P006",
        "P011",
        "P014",
        "P015",
        "P027",
        "P038",
        "P042",
        "P043",
        "P049",
        "P050",
    ]
    assert type(first).model_validate_json(serialize_project_json(first, type(first))) == first


def test_related_group_is_atomic() -> None:
    split = make_split(
        papers(), [PaperGroup(paper_ids=["P001", "P011"], basis="same version")], seed=3102026
    )
    assert ("P001" in split.development_ids) == ("P011" in split.development_ids)
    assert Counter(split.development_ids + split.final_test_ids) == Counter(
        p.paper_id for p in papers()
    )


def test_group_input_and_tie_break_are_stable() -> None:
    groups = [PaperGroup(paper_ids=["P011", "P001"], basis="same version")]
    split = make_split(papers(), groups, seed=3102026)
    assert split.ordered_groups[0].paper_ids == ["P001", "P011"]
    assert split.algorithm == "python_random_shuffle_dp_v1"
