"""Opt-in pinned BGE check; local acceptance requires this actual inference test."""

import os
from pathlib import Path

import numpy as np
import pytest
from test_bm25_publication import inputs  # noqa: F401

from biorag_repro.retrieval import dense


@pytest.fixture(scope="module")
def real_encoder():
    root = os.environ.get("BIORAG_BGE_ROOT")
    if root is None:
        pytest.skip(
            "pinned local BGE assets not selected; required for dense scientific acceptance"
        )
    return dense.BgeEncoder(Path(root), batch_size=2, threads=4)


def test_pinned_model_complete_inputs_and_independent_cls(real_encoder):
    import torch

    encoder = real_encoder
    instruction = "Represent this sentence for searching relevant passages: "
    # 510 meaningful wordpieces plus CLS/SEP exactly fill the verified 512 positions.
    passage = " ".join(["alpha"] * 509 + ["sentinel"])
    texts = ["RNA sequencing", passage]
    references = ["safe-short", "safe-boundary-sentinel"]
    for role in ("query", "passage"):
        selected = (
            texts if role == "passage" else ["RNA sequencing", "fixed safe synthetic question"]
        )
        full = [instruction + t for t in selected] if role == "query" else selected
        expected = encoder.tokenizer(
            full, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt"
        )
        actual_ids = []
        hook = encoder.model.get_input_embeddings().register_forward_pre_hook(
            lambda module, args, actual_ids=actual_ids: actual_ids.append(args[0].detach().clone())
        )
        try:
            observed, evidence = encoder.encode(selected, references, role)
        finally:
            hook.remove()
        assert len(actual_ids) == 1
        torch.testing.assert_close(actual_ids[0], expected["input_ids"], rtol=0, atol=0)
        assert [e.token_count for e in evidence] == expected["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            final = encoder.model(**expected).last_hidden_state
            direct = torch.nn.functional.normalize(final[:, 0, :], p=2, dim=1).cpu().numpy()
            mean = torch.nn.functional.normalize(final.mean(dim=1), p=2, dim=1).cpu().numpy()
        np.testing.assert_allclose(observed, direct, rtol=0, atol=2e-7)
        assert not np.allclose(observed, mean, atol=1e-4)
        assert observed.shape == (2, 768)
        repeated, same = encoder.encode(selected, references, role)
        np.testing.assert_array_equal(observed, repeated)
        assert evidence == same
        # Padding/batch shape may change floating point order; do not claim bitwise equality.
        alone, _ = encoder.encode([selected[0]], [references[0]], role)
        np.testing.assert_allclose(alone[0], observed[0], rtol=0, atol=2e-6)
    rows = dense.prepare_inputs(encoder.tokenizer, [passage], ["boundary"], "passage", 512)
    assert len(rows[0]["input_ids"]) == 512
    assert rows[0]["input_ids"][-2] == encoder.tokenizer.convert_tokens_to_ids("sentinel")
    with pytest.raises(ValueError, match="over.*513.*512"):
        encoder.encode([passage + " alpha"], ["over"], "passage")
    query_words = " ".join(["alpha"] * 502)
    # The documented instruction consists of eight wordpieces (colon included).
    assert (
        len(
            dense.prepare_inputs(encoder.tokenizer, [query_words], ["q-boundary"], "query", 512)[0][
                "input_ids"
            ]
        )
        == 512
    )
    with pytest.raises(ValueError, match="q-over.*513.*512"):
        encoder.encode([query_words + " alpha"], ["q-over"], "query")


def test_real_omitted_query_instruction_counterfeit(real_encoder, tmp_path, request):
    # The shared fixture supplies a fully verified two-paper synthetic source chain.
    sources, cfg = request.getfixturevalue("inputs")
    encoder = real_encoder
    index, vectors = dense.build_index(sources, cfg, tmp_path, encoder)
    from test_bm25_publication import query

    from biorag_repro.retrieval.models import RankedHit
    from biorag_repro.retrieval.storage import record_ref

    selected_query = query(text="alpha common")
    valid = dense.retrieve(index, vectors, sources, selected_query, cfg, tmp_path, encoder)
    dense.validate_retrieval(valid, index, vectors, sources, selected_query, cfg, tmp_path, encoder)
    # Keep plausible declared instruction and input count, but omit the instruction
    # in actual model execution and renew the scores, ordering and record identity.
    wrong, _ = encoder.encode([selected_query.question], [selected_query.query_id], "passage")
    assert not np.allclose(wrong[0], valid.dense_query.vector, atol=1e-5)
    counterfeit = valid.model_copy(deep=True)
    counterfeit.dense_query.vector = wrong[0].tolist()
    _, chunks = dense._selected(sources, cfg, tmp_path, encoder)
    scores = vectors @ wrong[0]
    eligible = sorted(
        [i for i, p in enumerate(index.chunk_paper_ids) if p == "P001"],
        key=lambda i: (-float(scores[i]), index.ordered_chunk_ids[i]),
    )
    counterfeit.hits = [
        RankedHit(rank=n, chunk=record_ref(chunks[i]), paper_id="P001", score=float(scores[i]))
        for n, i in enumerate(eligible, 1)
    ]
    assert record_ref(counterfeit) != record_ref(valid)
    with pytest.raises(ValueError, match="query"):
        dense.validate_retrieval(
            counterfeit, index, vectors, sources, selected_query, cfg, tmp_path, encoder
        )
