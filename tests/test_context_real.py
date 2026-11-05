"""Real pinned Qwen evidence-token counting; assets are explicitly selected."""

import os
from pathlib import Path

import pytest

from biorag_repro.retrieval import context


def test_real_qwen_boundaries_and_nonadditivity():
    root = os.environ.get("BIORAG_QWEN_ROOT")
    if root is None:
        pytest.skip("explicit local Qwen tokenizer assets not selected")
    counter = context.QwenCounter(Path(root), context.tokenizer_snapshot())
    assert counter("") == 0
    assert counter("hello") == 1
    assert counter("hellohello") != counter("hello") * 2 or counter("ab") != counter("a") + counter(
        "b"
    )
    assert counter("α = −2.5 × 10⁻³; P < 0.05.") == 21
    # Literal repeated space-prefixed words each occupy one token in this snapshot.
    assert counter(" x" * 2048) == 2048
    assert counter(" x" * 2049) == 2049
    bge_root = os.environ.get("BIORAG_BGE_ROOT")
    if bge_root is not None:
        from tokenizers import Tokenizer

        bge = Tokenizer.from_file(str(Path(bge_root) / "tokenizer.json"))
        assert len(bge.encode("α = −2.5 × 10⁻³; P < 0.05.", add_special_tokens=False).ids) != 21
    t = counter.tokenizer
    for text in ["", "hello", "α = −2.5 × 10⁻³; P < 0.05."]:
        assert counter(text) == len(t.backend_tokenizer.encode(text, add_special_tokens=False).ids)
        assert (
            t(text, add_special_tokens=True)["input_ids"]
            == t(text, add_special_tokens=False)["input_ids"]
        )
    invalid = context.tokenizer_snapshot().model_copy(update={"revision": "a" * 40})
    with pytest.raises(ValueError):
        context.QwenCounter(Path(root), invalid)


def test_real_wrapped_exact_fit_and_overflow():
    from test_context import chunks

    root = os.environ.get("BIORAG_QWEN_ROOT")
    if root is None:
        pytest.skip("explicit local Qwen tokenizer assets not selected")
    counter = context.QwenCounter(Path(root), context.tokenizer_snapshot())
    ds, text, n, stop = context.pack(chunks([" x" * 2040]), counter)
    assert n == 2048 and stop == "exhausted" and ds[0].decision == "included"
    assert (
        len(counter.tokenizer.backend_tokenizer.encode(text, add_special_tokens=False).ids) == 2048
    )
    ds, text, n, stop = context.pack(chunks([" x" * 2041, "small"]), counter)
    assert n == 0 and text == "" and stop == "first_nonfitting"
    assert [d.decision for d in ds] == ["nonfitting", "not_processed"]
