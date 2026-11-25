"""Chunking invariants on seeded random documents, checked against an independent oracle.

The oracle is plain arithmetic on the fixture: a paragraph either fits the construction limit
with its section context or it does not. Fitting paragraphs must each be delivered exactly once,
in source order, and never over the limit; oversized ones must reach no chunk and stay
accounted for. Standard-library random with fixed seeds, no extra dependency.
"""

import random

import pytest
from test_chunking import build

LIMIT = 192
PER_WORD = 20  # the counter charges every word the same, so the fixture arithmetic is exact
CONTEXT_WORDS = 3  # "[section]", the title word and "[/section]" are repeated in every chunk


def count(text):
    return PER_WORD * len(text.split())


def random_document(seed):
    rng = random.Random(seed)
    paragraphs = []
    body = []
    for n in range(rng.randint(3, 12)):
        # 6 words plus the context cost 180 and fit; 7 words cost 200 and exceed the limit
        words = rng.choice([1, 2, 3, 6, 7, 12])
        text = " ".join(f"w{seed}x{n}x{k}" for k in range(words))
        paragraphs.append((text, words <= 6))
        body.append(f"<p>{text}</p>")
    xml = "<article><body><sec><title>Methods</title>" + "".join(body) + "</sec></body></article>"
    return xml, paragraphs


@pytest.mark.parametrize("seed", range(25))
def test_fitting_paragraphs_are_delivered_once_in_order_and_oversized_ones_are_accounted(seed):
    xml, paragraphs = random_document(seed)
    _doc, _sel, _cfg_ref, _cfg, result = build(xml, budget=LIMIT, counter=count)
    texts = [c.rendered_text for c in result.chunks]

    assert all(c.construction_tokens == count(c.rendered_text) <= LIMIT for c in result.chunks)
    assert all(c.rendered_text.count("[section]") <= 1 for c in result.chunks)

    positions = []
    for text, fits in paragraphs:
        holders = [i for i, chunk_text in enumerate(texts) if text in chunk_text]
        assert len(holders) == (1 if fits else 0), text
        if fits:
            positions.append((holders[0], texts[holders[0]].index(text)))
    assert positions == sorted(positions)  # source order survives packing

    statuses = [a.status for a in result.chunk_set.accounting]
    assert statuses.count("oversized") == sum(not fits for _, fits in paragraphs)
    # the title is delivered as its own unit and again as context, so only paragraphs are counted
    assert statuses.count("included") == 1 + sum(fits for _, fits in paragraphs)


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_rebuilding_gives_the_same_chunk_identities(seed):
    xml, _ = random_document(seed)
    first = build(xml, budget=LIMIT, counter=count)[4]
    second = build(xml, budget=LIMIT, counter=count)[4]
    assert [c.rendered_text for c in first.chunks] == [c.rendered_text for c in second.chunks]
    assert first.chunk_set.model_dump() == second.chunk_set.model_dump()
