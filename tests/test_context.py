"""Literal packing expectations, independent of publication and tokenizer assets."""

import pytest
from test_chunking import build

from biorag_repro.retrieval import context


def chunks(texts):
    return [
        build(f"<article><body><p>{text}</p></body></article>", counter=lambda s: 1)[4].chunks[0]
        for text in texts
    ]


def test_empty():
    assert context.pack([], len) == ([], "", 0, "exhausted")


def test_wrappers_and_whole_context_recount():
    calls = []

    def count(text):
        calls.append(text)
        return len(text)

    decisions, text, tokens, stop = context.pack(chunks(["alpha", "beta"]), count)
    assert text == "[evidence]\nalpha\n[/evidence]\n\n[evidence]\nbeta\n[/evidence]"
    assert calls == ["[evidence]\nalpha\n[/evidence]", text]
    assert tokens == len(text)
    assert [d.decision for d in decisions] == ["included", "included"]
    assert stop == "exhausted"


@pytest.mark.parametrize("size,accepted", [(2048, True), (2049, False)])
def test_boundary(size, accepted):
    ds, text, tokens, stop = context.pack(chunks(["x"]), lambda s: size)
    assert [d.decision for d in ds] == ["included" if accepted else "nonfitting"]
    assert tokens == (2048 if accepted else 0)
    assert bool(text) == accepted
    assert stop == ("exhausted" if accepted else "first_nonfitting")


def test_first_nonfit_never_skips():
    ds, text, tokens, stop = context.pack(
        chunks(["a", "large", "z"]), lambda s: 2049 if "large" in s else 1
    )
    assert [d.decision for d in ds] == ["included", "nonfitting", "not_processed"]
    assert text == "[evidence]\na\n[/evidence]"
    assert tokens == 1 and stop == "first_nonfitting"


@pytest.mark.parametrize("n", [1, 7, 20, 21, 25])
def test_original_top20(n):
    ds, text, _tokens, stop = context.pack(chunks([str(i) for i in range(n)]), lambda s: 1)
    assert len(ds) == min(n, 20)
    assert [d.rank for d in ds] == list(range(1, min(n, 20) + 1))
    assert stop == ("top20" if n >= 20 else "exhausted")
    assert "[evidence]\n20\n" not in text


def test_same_text_same_source_only():
    a, b = chunks(["same", "same"])
    different = b.model_copy(
        update={
            "contributors": [b.contributors[0].model_copy(update={"occurrence_id": "distinct"})]
        }
    )
    ds, text, _tokens, _stop = context.pack([a, b, different, b], lambda s: 1)
    assert [d.decision for d in ds] == ["included", "duplicate", "included", "duplicate"]
    assert [d.duplicate_of_rank for d in ds] == [None, 1, None, 1]
    assert text == "[evidence]\nsame\n[/evidence]\n\n[evidence]\nsame\n[/evidence]"


def test_nonadditive_counter_uses_proposed_complete_text():
    a, b = chunks(["a", "b"])

    def counter(text):
        # A tokenizer merge across concatenation changes the full count.
        return 2048 if "[/evidence]\n\n[evidence]" in text else 1200

    ds, _text, tokens, stop = context.pack([a, b], counter)
    assert [d.decision for d in ds] == ["included", "included"]
    assert tokens == 2048 and stop == "exhausted"


def test_duplicate_before_nonfit_and_twentieth_rank():
    a, b, c = chunks(["a", "large", "small"])
    ds, _text, _tokens, stop = context.pack([a, a, b, c], lambda s: 2049 if "large" in s else 1)
    assert [d.decision for d in ds] == ["included", "duplicate", "nonfitting", "not_processed"]
    assert ds[1].duplicate_of_rank == 1
    base = chunks([str(i) for i in range(19)])
    ds, _, _, stop = context.pack(base + [base[0]], lambda s: 1)
    assert len(ds) == 20 and stop == "top20"
    assert ds[-1].decision == "duplicate" and ds[-1].duplicate_of_rank == 1


@pytest.mark.parametrize(
    "values",
    [
        ["37 in condition A", "37 in condition B"],
        ["increased", "not increased"],
        ["α −2.5", "α -2.5"],
    ],
)
def test_distinct_scientific_conditions_and_qualifiers(values):
    ds, _, _, _ = context.pack(chunks(values), lambda s: 1)
    assert [d.decision for d in ds] == ["included", "included"]


def test_containing_node_does_not_create_another_source_occurrence():
    a = chunks(["same"])[0]
    b = a.model_copy(
        update={
            "contributors": [
                c.model_copy(update={"node_id": "another-containing-node"}) for c in a.contributors
            ]
        }
    )
    ds, _, _, _ = context.pack([a, b], lambda s: 1)
    assert [d.decision for d in ds] == ["included", "duplicate"]
    assert ds[1].duplicate_of_rank == 1


def test_duplicates_do_not_refill_beyond_twenty():
    a, b = chunks(["first", "rank21"])
    ds, text, n, stop = context.pack([a] * 20 + [b], lambda s: 1)
    assert [d.decision for d in ds] == ["included"] + ["duplicate"] * 19
    assert text == "[evidence]\nfirst\n[/evidence]"
    assert n == 1 and stop == "top20"


@pytest.mark.parametrize("coordinate", ["paper_id", "representation", "canonical"])
def test_identical_text_with_different_scientific_identity_is_retained(coordinate):
    a = chunks(["same"])[0]
    if coordinate == "paper_id":
        value = "P002"
    elif coordinate == "representation":
        value = "PDF_DOCLING"
    else:
        value = a.input.canonical.model_copy(update={"artifact_id": "different-canonical"})
    b = a.model_copy(update={"input": a.input.model_copy(update={coordinate: value})})
    ds, _, _, _ = context.pack([a, b], lambda s: 1)
    assert [d.decision for d in ds] == ["included", "included"]


def test_wrapper_policy_is_part_of_resolved_scientific_configuration():
    from test_retrieval_models import settings

    from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

    original = settings()
    packed = original.model_copy(update={"context_rendering": context.POLICY})
    assert sha256_bytes(canonical_json_bytes(original.model_dump())) != sha256_bytes(
        canonical_json_bytes(packed.model_dump())
    )
