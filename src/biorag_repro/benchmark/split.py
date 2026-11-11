"""Prospective paper-level split with atomic related-publication groups."""

import random
import sys
from typing import Self

from pydantic import Field, model_validator

from biorag_repro.core.records import Record


class PaperMeta(Record):
    paper_id: str
    primary_domain: str


class PaperGroup(Record):
    paper_ids: list[str]
    basis: str

    @model_validator(mode="after")
    def valid_group(self) -> Self:
        if not self.paper_ids or len(self.paper_ids) != len(set(self.paper_ids)):
            raise ValueError("paper group requires distinct IDs")
        if not self.basis.strip():
            raise ValueError("paper group requires evidence basis")
        return self


class RelatednessRecord(Record):
    schema_version: str = "1"
    corpus_content_sha256: str
    screening_basis: str
    groups: list[PaperGroup]
    reviewer: str | None = None


class PaperSplit(Record):
    schema_version: str = "1"
    corpus_version: str
    corpus_content_sha256: str | None = None
    seed_name: str = "paper_split"
    seed: int
    algorithm: str = "python_random_shuffle_dp_v1"
    python_version: str
    papers: list[PaperMeta]
    ordered_groups: list[PaperGroup]
    shuffled_groups: list[PaperGroup]
    selected_shuffled_positions: list[int]
    development_ids: list[str] = Field(min_length=10, max_length=10)
    final_test_ids: list[str] = Field(min_length=40, max_length=40)
    represented_domains: list[str]
    preprint_assignments: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_assignment(self) -> Self:
        development = set(self.development_ids)
        final_test = set(self.final_test_ids)
        if len(development) != 10 or len(final_test) != 40 or development & final_test:
            raise ValueError("split requires disjoint 10/40 paper IDs")
        all_groups = [paper for group in self.ordered_groups for paper in group.paper_ids]
        if set(all_groups) != development | final_test or len(all_groups) != 50:
            raise ValueError("split groups must cover 50 distinct papers")
        if [paper.paper_id for paper in self.papers] != sorted(development | final_test):
            raise ValueError("ordered input papers must match split groups")
        for group in self.ordered_groups:
            if not (set(group.paper_ids) <= development or set(group.paper_ids) <= final_test):
                raise ValueError("related group crosses split")
        return self


def make_split(
    papers: list[PaperMeta],
    related_groups: list[PaperGroup],
    *,
    seed: int,
    corpus_version: str = "corpus_v1",
    corpus_content_sha256: str | None = None,
    preprint_ids: tuple[str, ...] = (),
) -> PaperSplit:
    """Maximize covered primary domains among group-constrained ten-paper subsets.

    For each (paper count, domain mask), retain the lexicographically earliest tuple
    of positions in the seeded shuffled group order. Equal-coverage terminal states
    use that same tuple comparison. No dictionary iteration order affects selection.
    """
    by_id = {paper.paper_id: paper for paper in papers}
    if len(papers) != 50 or len(by_id) != 50:
        raise ValueError("split requires 50 unique papers")
    claimed: set[str] = set()
    groups: list[PaperGroup] = []
    for related in related_groups:
        ids = sorted(related.paper_ids)
        if len(ids) < 2 or any(i not in by_id or i in claimed for i in ids):
            raise ValueError("invalid or overlapping related group")
        claimed.update(ids)
        groups.append(PaperGroup(paper_ids=ids, basis=related.basis))
    groups.extend(
        PaperGroup(paper_ids=[i], basis="singleton: no confirmed related version")
        for i in sorted(by_id)
        if i not in claimed
    )
    ordered = sorted(groups, key=lambda group: group.paper_ids)
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    domains = sorted({paper.primary_domain for paper in papers})
    bits = {domain: 1 << index for index, domain in enumerate(domains)}
    states: dict[tuple[int, int], tuple[int, ...]] = {(0, 0): ()}
    for position, group in enumerate(shuffled):
        mask = 0
        for paper_id in group.paper_ids:
            mask |= bits[by_id[paper_id].primary_domain]
        next_states = dict(states)
        for (count, domain_mask), positions in states.items():
            new_count = count + len(group.paper_ids)
            if new_count > 10:
                continue
            key = (new_count, domain_mask | mask)
            candidate = (*positions, position)
            if key not in next_states or candidate < next_states[key]:
                next_states[key] = candidate
        states = next_states
    feasible = [
        (mask.bit_count(), positions, mask)
        for (count, mask), positions in states.items()
        if count == 10
    ]
    if not feasible:
        raise ValueError("no group-constrained ten-paper assignment exists")
    _, positions, mask = min(feasible, key=lambda item: (-item[0], item[1]))
    development = sorted(paper for position in positions for paper in shuffled[position].paper_ids)
    final_test = sorted(set(by_id) - set(development))
    if any(i not in by_id for i in preprint_ids):
        raise ValueError("unknown preprint paper ID")
    return PaperSplit(
        corpus_version=corpus_version,
        corpus_content_sha256=corpus_content_sha256,
        seed=seed,
        python_version=sys.version.split()[0],
        papers=[by_id[i] for i in sorted(by_id)],
        ordered_groups=ordered,
        shuffled_groups=shuffled,
        selected_shuffled_positions=list(positions),
        development_ids=development,
        final_test_ids=final_test,
        represented_domains=[domain for domain in domains if mask & bits[domain]],
        preprint_assignments={
            i: "development" if i in development else "final_test" for i in sorted(preprint_ids)
        },
    )
