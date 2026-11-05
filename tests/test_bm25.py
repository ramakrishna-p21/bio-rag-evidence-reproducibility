"""Literal lexical expectations and independent BM25 arithmetic."""

import math

import pytest

from biorag_repro.retrieval import bm25


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A study of RNA.", ["A", "study", "of", "RNA", "."]),
        (
            "BRCA1 c.68_69delAG −1.2e−3 +2 <0.05 ≤5 ≥7",
            [
                "BRCA1",
                "c",
                ".",
                "68_69delAG",
                "−",
                "1",
                ".",
                "2e",
                "−",
                "3",
                "+",
                "2",
                "<",
                "0",
                ".",
                "05",
                "≤",
                "5",
                "≥",
                "7",
            ],
        ),
        ("α β₂ μg café e\u0301 中文", ["α", "β₂", "μg", "café", "e", "\u0301", "中文"]),
        ("a-b/γ: (x)!=y", ["a", "-", "b", "/", "γ", ":", "(", "x", ")", "!", "=", "y"]),
        ("RNA rna RNA", ["RNA", "rna", "RNA"]),
        ("", []),
        (" \n\t", []),
    ],
)
def test_literal_python_unicode_tokens(text, expected):
    assert bm25.tokenize(text) == expected


def test_independent_scores_and_statistics():
    # N=3, lengths 3,2,1; avgdl=2. df(a)=2; df(b)=df(c)=df(d)=1.
    # Raw IDFs: -log(5/3) for a and log(5/3) for b, c and d.
    model = bm25.scorer([["a", "a", "b"], ["a", "c"], ["d"]])
    l = math.log(5 / 3)
    # Four terms give average idf=l/2 and floor=l/8.
    assert model.doc_len == [3, 2, 1]
    assert model.avgdl == 2
    assert model.doc_freqs == [{"a": 2, "b": 1}, {"a": 1, "c": 1}, {"d": 1}]
    assert model.average_idf == pytest.approx(l / 2)
    assert model.idf == pytest.approx({"a": l / 8, "b": l, "c": l, "d": l})
    # First a contribution: (l/8)*5/(2+1.5*(.25+.75*3/2)).
    first = (l / 8) * 5 / 4.0625
    assert list(model.get_scores(["a"])) == pytest.approx([first, l / 8, 0])
    assert list(model.get_scores(["a", "a", "absent"])) == pytest.approx([2 * first, l / 4, 0])
    assert list(model.get_scores(["absent"])) == [0, 0, 0]
    assert (model.k1, model.b, model.epsilon) == (1.5, 0.75, 0.25)


def test_negative_average_is_preserved_exactly():
    # Pinned package floor itself remains negative for a one-term corpus.
    model = bm25.scorer([["x"], ["x", "x"]])
    assert model.idf["x"] == pytest.approx(-math.log(5) / 4)
    assert all(v < 0 for v in model.get_scores(["x"]))


@pytest.mark.parametrize("corpus", [[], [[]], [["a"], []]])
def test_unusable_corpus_is_explicit(corpus):
    with pytest.raises(ValueError, match="empty"):
        bm25.scorer(corpus)
