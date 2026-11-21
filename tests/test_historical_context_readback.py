import pytest

from biorag_repro.evaluation.historical_context_readback import _read, check_delivery, read_case


def test_exact_historical_context_and_spans():
    context = "[evidence]\nanswer\n[/evidence]"
    assert check_delivery(context, context, [{"start": 11, "end": 17, "text": "answer"}])
    with pytest.raises(ValueError, match="supplied context changed"):
        check_delivery(context, context + "x", [])
    with pytest.raises(ValueError, match="evidence span"):
        check_delivery(context, context, [{"start": 0, "end": 6, "text": "answer"}])


def test_changed_retained_hash_is_rejected(tmp_path):
    path = tmp_path / "context.json"
    path.write_text('{"exact_context":"original"}')
    with pytest.raises(ValueError, match="artifact hash differs"):
        _read(path, "0" * 64)


@pytest.mark.parametrize("phase,number", [("final_test", 1), ("historical_development", 284)])
def test_historical_route_rejects_other_population(phase, number):
    with pytest.raises(ValueError, match="authorized development case"):
        read_case(
            packet={"number": number},
            draft={},
            accounting=None,
            known_failures=None,
            observations={},
            data_root=None,
            corpus_root=None,
            canonical_root=None,
            foreign_canonical_root=None,
            selected_canonical_roots={},
            frozen_sources={},
            source_selection={},
            item=None,
            phase=phase,
        )
