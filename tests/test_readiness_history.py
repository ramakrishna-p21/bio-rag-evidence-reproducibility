import ast
import pathlib

import pytest

from biorag_repro.readiness import history
from biorag_repro.readiness.history import HistoryEntry, HistoryInventory, classify_history
from biorag_repro.readiness.models import HISTORY_FAIL_CODES, HISTORY_PASS_CODES

E1 = HistoryEntry("B1", "observation", True)


def inv(entries=(), complete=True, authorized="B1", approved=frozenset(), contradictions=False):
    return HistoryInventory(complete, tuple(entries), authorized, approved, contradictions)


@pytest.mark.parametrize(
    ("inventory", "expected"),
    [
        (inv(), "no_prior_final_test_execution"),
        (inv([E1, HistoryEntry("B1", "attempt", True)]), "same_batch_valid_history"),
        (
            inv([HistoryEntry("B0", "observation", True)], approved=frozenset({"B0"})),
            "other_approved_batch_valid_history",
        ),
        (
            inv([E1, HistoryEntry("B0", "observation", True)], approved=frozenset({"B0"})),
            "other_approved_batch_valid_history",
        ),
        (inv([HistoryEntry("B9", "observation", True)]), "unauthorized_final_test_artifact"),
        (inv([E1], contradictions=True), "conflicting_final_test_artifact"),
        (inv([HistoryEntry("B1", "attempt", False)]), "unverified_final_test_artifact"),
        (inv(complete=False), "history_inventory_incomplete"),
        (inv([E1], complete=False), "history_inventory_incomplete"),
        (inv(authorized=None), "no_prior_final_test_execution"),
    ],
)
def test_classes(inventory, expected):
    assert classify_history(inventory) == expected


def test_every_returned_code_is_a_known_history_code():
    known = set(HISTORY_PASS_CODES) | set(HISTORY_FAIL_CODES) | {history.INCOMPLETE}
    assert known >= {classify_history(inv())}


def test_nothing_in_the_real_path_can_reach_the_classifier():
    package = pathlib.Path(history.__file__).parent
    for name in ("evidence.py", "evaluate.py", "publish.py", "cli.py", "gather.py"):
        tree = ast.parse((package / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom | ast.Import):
                text = ast.unparse(node)
                assert "history" not in text.replace("final_test_history", ""), (name, text)
            if isinstance(node, ast.Name | ast.Attribute):
                assert "classify_history" not in ast.unparse(node)
