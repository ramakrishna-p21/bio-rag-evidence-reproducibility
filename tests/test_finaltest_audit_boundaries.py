"""The final-test record contracts add no execution path and no authority."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

import biorag_repro
from biorag_repro.generation.attempts import RunRecord

SRC = Path(biorag_repro.__file__).parent
FINALTEST = SRC / "finaltest"


def tree(path: Path) -> ast.AST:
    return ast.parse(path.read_text(encoding="utf-8"))


def mentions(path: Path, name: str) -> bool:
    return any(
        (isinstance(n, ast.Name) and n.id == name)
        or (isinstance(n, ast.Attribute) and n.attr == name)
        or (isinstance(n, ast.alias) and n.name == name)
        for n in ast.walk(tree(path))
    )


def test_existing_final_test_refusals_stand():
    with pytest.raises(ValidationError):
        RunRecord.model_validate({"phase": "final_test"})


def test_no_run_store_subclass_exists():
    found = {}
    for path in SRC.rglob("*.py"):
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.ClassDef) and any(
                getattr(b, "id", "") == "RunStore" for b in node.bases
            ):
                found[node.name] = path.name
    assert found == {}
