"""The final-test record contracts add no execution path and no authority."""

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

import biorag_repro
from biorag_repro.finaltest import layout
from biorag_repro.generation.attempts import RunRecord, RunStore

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


def test_run_record_type_is_known_only_to_its_schema_and_the_verifier():
    users = {p.name for p in SRC.rglob("*.py") if mentions(p, "FinalTestRunRecord")}
    assert users == {
        "runrecord.py",
        "inventory.py",
        "completion.py",
    }


def test_no_run_store_constructor_takes_the_final_test_record():
    for method in (RunStore.create, RunStore.open, RunStore.__init__):
        names = method.__annotations__
        assert "FinalTestRunRecord" not in " ".join(str(v) for v in names.values())
    assert RunRecord.model_fields["phase"].annotation.__args__ == ("development", "qualification")


def test_input_contracts_import_no_execution_or_publication_module():
    imported = {
        n.module
        for name in ("inputs.py", "runrecord.py")
        for n in ast.walk(tree(FINALTEST / name))
        if isinstance(n, ast.ImportFrom) and n.module
    }
    assert not any(
        m.endswith(("registry", "guard", "driver", "runner", "roster", "storage")) for m in imported
    )


def test_registry_events_are_not_activated():
    assert layout.OPERATIONAL_EVENTS == (
        "root_initialized",
        "batch_reserved",
        "batch_materialized",
        "batch_closed_incomplete",
    )
    assert set(layout.REQUIRED_ROLES) == {
        "batch_reserved",
        "batch_materialized",
        "batch_closed_incomplete",
    }
    # The rehearsal start, resume and completion that published these events were removed in
    # Phase 2.5; only the record and verifier modules still mention them
    assert layout.REHEARSAL_EVENTS == (
        "batch_inputs_bound",
        "batch_authorized",
        "batch_resumed",
        "batch_completed",
    )
    for text in ("batch_inputs_bound", "batch_authorized"):
        users = {p.name for p in SRC.rglob("*.py") if text in p.read_text(encoding="utf-8")}
        assert {"layout.py"} <= users <= {"layout.py", "inventory.py", "registry.py"}, (text, users)
    users = {p.name for p in SRC.rglob("*.py") if "batch_resumed" in p.read_text(encoding="utf-8")}
    assert {"layout.py"} <= users <= {"layout.py", "inventory.py", "registry.py"}, users


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
