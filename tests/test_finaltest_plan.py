"""Synthetic checks of the frozen final-test execution coordinates."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from scripts.run_generation import Planned, Refused, load_inputs, validate_final_test_plan


def synthetic_plan(papers=40):
    items = [f"synthetic_{n:03d}" for n in range(305)]
    package = {"header": {"item_ids": items, "arms": list(FROZEN_MATRIX)}}
    plan = []
    for index, item in enumerate(items):
        for arm, (conditions, repeats) in FROZEN_MATRIX.items():
            for condition in conditions:
                for repeat in repeats:
                    record = {
                        "item_id": item,
                        "paper_id": f"synthetic_paper_{index % papers:02d}",
                        "matrix_arm": arm,
                        "condition": condition,
                        "split": "final_test",
                    }
                    plan.append(
                        Planned(
                            f"synthetic_key_{len(plan)}",
                            record,
                            repeat,
                            SimpleNamespace(configuration_sha256="a" * 64),
                        )
                    )
    return package, plan


def test_complete_synthetic_final_test_plan():
    package, plan = synthetic_plan()
    validate_final_test_plan(package, plan)
    assert len(plan) == 4575


def test_plan_with_one_zero_item_split_paper_is_accepted():
    package, plan = synthetic_plan(papers=39)
    validate_final_test_plan(package, plan)
    assert len({p.record["paper_id"] for p in plan}) == 39


def test_protected_inputs_are_not_read_without_execution_authorization(monkeypatch):
    def forbidden_read(_self):
        raise AssertionError("protected input was read")

    monkeypatch.setattr(Path, "read_bytes", forbidden_read)
    with pytest.raises(Refused, match="execution authorization"):
        load_inputs(SimpleNamespace(phase="final_test", contexts=Path("protected.json")))


@pytest.mark.parametrize("damage", ["missing", "duplicate", "wrong_split", "wrong_paper"])
def test_invalid_final_test_plan_refused(damage):
    package, plan = synthetic_plan()
    if damage == "missing":
        plan.pop()
    elif damage == "duplicate":
        plan[-1] = plan[0]
    elif damage == "wrong_split":
        plan[0].record["split"] = "development"
    else:
        plan[0].record["paper_id"] = "foreign_paper"
    with pytest.raises(Refused):
        validate_final_test_plan(package, plan)
