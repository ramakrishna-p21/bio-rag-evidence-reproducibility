"""Independent vector arithmetic and complete-input boundary expectations."""

from types import SimpleNamespace

import numpy as np
import pytest

from biorag_repro.retrieval import dense


def test_cls_normalization_uses_first_token():
    import torch

    hidden = torch.tensor([[[3.0, 4.0], [12.0, 0.0]], [[0.0, 2.0], [4.0, 4.0]]])
    result = dense.cls_normalize(hidden, 2)
    np.testing.assert_allclose(result, [[0.6, 0.8], [0.0, 1.0]], atol=1e-7)
    assert not np.allclose(result[0], np.array([15.0, 4.0]) / np.sqrt(241.0))


@pytest.mark.parametrize("values", [[[0.0, 0.0]], [[float("nan"), 1.0]], [[1.0, float("inf")]]])
def test_invalid_pooling_fails(values):
    import torch

    with pytest.raises(ValueError):
        dense.cls_normalize(torch.tensor(values)[:, None, :], 2)


def test_exact_dot_products_ties_zero_negative():
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [1.0, 0.0]], dtype="float32")
    q = np.array([1.0, 0.0], dtype="float32")
    # Four independent dot products: 1, 0, -1, 1. Equal scores sort by ID.
    assert dense.rank_vectors(vectors, q, ["z", "c", "b", "a"], ["P001"] * 4, None) == [
        (3, 1.0),
        (0, 1.0),
        (1, 0.0),
        (2, -1.0),
    ]
    assert dense.rank_vectors(
        vectors, q, ["z", "c", "b", "a"], ["P001", "P002", "P002", "P001"], "P002"
    ) == [(1, 0.0), (2, -1.0)]


@pytest.mark.parametrize(
    "value,rows,dim",
    [
        ([[0.0, 0.0]], 1, 2),
        ([[1.0, 1.0]], 1, 2),
        ([[float("nan"), 0.0]], 1, 2),
        ([[float("inf"), 0.0]], 1, 2),
        ([[1.0, 0.0]], 2, 2),
        ([[1.0, 0.0]], 1, 3),
    ],
)
def test_vector_validation(value, rows, dim):
    with pytest.raises(ValueError):
        dense.validate_vectors(np.array(value, dtype="float32"), rows, dim)


class Tokenizer:
    pad_token_id = 0
    model_max_length = 8
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        assert kwargs["truncation"] is False
        assert kwargs["add_special_tokens"] is True
        rows = [[101] + [10 + i for i, _ in enumerate(t.split())] + [102] for t in texts]
        masks = [[1] * len(row) for row in rows]
        if kwargs["padding"]:
            width = max(map(len, rows))
            for row, mask in zip(rows, masks):
                n = width - len(row)
                row.extend([0] * n)
                mask.extend([0] * n)
        out = {
            "input_ids": rows,
            "attention_mask": masks,
            "token_type_ids": [[0] * len(r) for r in rows],
        }
        if kwargs.get("return_tensors"):
            import torch

            return {k: torch.tensor(v) for k, v in out.items()}
        return out


class Model:
    training = False

    def __init__(self):
        self.calls = []

    def __call__(self, **batch):
        import torch

        self.calls.append({k: v.tolist() for k, v in batch.items()})
        assert not torch.is_grad_enabled()
        hidden = torch.zeros((*batch["input_ids"].shape, 2))
        hidden[:, 0, 0] = 3
        hidden[:, 0, 1] = 4
        hidden[:, 1:, 0] = 9
        return SimpleNamespace(last_hidden_state=hidden)


def test_complete_input_boundary_and_padding():
    tokenizer = Tokenizer()
    texts = ["one two three four five sentinel", "one"]
    rows = dense.prepare_inputs(tokenizer, texts, ["long", "short"], "passage", 8)
    assert rows[0]["input_ids"] == [101, 10, 11, 12, 13, 14, 15, 102]
    model = Model()
    vectors, evidence = dense.encode_inputs(
        model, tokenizer, texts, ["long", "short"], "passage", 8, 2, 2, "cpu"
    )
    assert [e.token_count for e in evidence] == [8, 3]
    assert model.calls[0]["input_ids"][0] == rows[0]["input_ids"]
    assert model.calls[0]["attention_mask"][1] == [1, 1, 1, 0, 0, 0, 0, 0]
    separate, same = dense.encode_inputs(
        Model(), tokenizer, texts, ["long", "short"], "passage", 8, 2, 1, "cpu"
    )
    np.testing.assert_array_equal(vectors, separate)
    assert evidence == same
    with pytest.raises(ValueError, match="too-long.*9.*8"):
        dense.prepare_inputs(tokenizer, [texts[0] + " over"], ["too-long"], "passage", 8)


def test_query_instruction_and_special_tokens():
    tok = Tokenizer()
    query = dense.prepare_inputs(tok, ["alpha"], ["q"], "query", 32)[0]
    passage = dense.prepare_inputs(tok, ["alpha"], ["p"], "passage", 32)[0]
    assert dense.QUERY_INSTRUCTION == "Represent this sentence for searching relevant passages: "
    assert len(query["input_ids"]) == 10  # seven instruction words, one query word, two specials
    assert passage["input_ids"] == [101, 10, 102]


def test_silent_batch_truncation_rejected():
    class Truncating(Tokenizer):
        def __call__(self, texts, **kwargs):
            out = super().__call__(texts, **kwargs)
            if kwargs.get("return_tensors"):
                out = {k: v[:, :-1] for k, v in out.items()}
            return out

    with pytest.raises(ValueError, match="complete input"):
        dense.encode_inputs(
            Model(), Truncating(), ["one sentinel"], ["p"], "passage", 8, 2, 1, "cpu"
        )


def test_shape_dtype_count_and_top_twenty():
    import torch

    with pytest.raises(ValueError, match="dimension"):
        dense.cls_normalize(torch.ones((1, 3, 4)), 2)
    with pytest.raises(ValueError, match="float32"):
        dense.validate_vectors(np.array([[1.0, 0.0]], dtype="float64"), 1, 2)
    vectors = np.tile(np.array([[1.0, 0.0]], dtype="float32"), (24, 1))
    ids = [f"c{i:02}" for i in reversed(range(24))]
    ranked = dense.rank_vectors(vectors, vectors[0], ids, ["P001"] * 24, None)
    assert [ids[i] for i, _ in ranked] == [f"c{i:02}" for i in range(20)]


@pytest.mark.parametrize(
    "installed, accepted",
    [
        ("2.15.0+cpu", True),
        ("2.15.0+cu132", True),  # the approved generation lock's build
        ("2.15.0+cu128", False),
        ("2.15.0", False),
        ("2.15.1+cpu", False),
        ("2.14.0+cu132", False),
    ],
)
def test_only_the_cpu_and_approved_lock_torch_builds_are_accepted(installed, accepted):
    from biorag_repro.retrieval.dense import check_torch_build

    if accepted:
        assert check_torch_build(installed) == installed
    else:
        with pytest.raises(ValueError, match="version mismatch"):
            check_torch_build(installed)


def test_dense_execution_records_only_cpu_inference():
    from typing import Literal

    from biorag_repro.retrieval.models import DenseExecution

    assert DenseExecution.model_fields["device"].annotation == Literal["cpu"]
