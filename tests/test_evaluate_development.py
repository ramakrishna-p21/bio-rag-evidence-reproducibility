"""The official postprocessor cannot turn a synthetic batch into scientific evidence."""

import json
from argparse import Namespace

import pytest

from scripts.evaluate_development import evaluate
from scripts.run_generation import Refused


def test_synthetic_batch_refused_before_evaluation_writes(tmp_path):
    batch = tmp_path / "runs" / "batch"
    batch.mkdir(parents=True)
    (batch / "batch.json").write_text(json.dumps({"identity": {"phase": "development"}}))
    args = Namespace(
        out=batch, analysis_root=tmp_path, published_context_root=tmp_path, review_root=tmp_path
    )
    before = set(tmp_path.rglob("*"))
    with pytest.raises(Refused, match="official"):
        evaluate(args)
    assert set(tmp_path.rglob("*")) == before
