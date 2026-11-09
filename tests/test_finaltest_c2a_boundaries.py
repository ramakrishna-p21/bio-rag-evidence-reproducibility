"""P10-C2A adds contracts and a read-only audit: no execution path, no authority."""

import ast
import types
from pathlib import Path

import pytest
from pydantic import ValidationError

import biorag_repro
from biorag_repro.finaltest import layout
from biorag_repro.generation import driver
from biorag_repro.generation.attempts import RunRecord, RunStore
from biorag_repro.generation.roster import RosterMismatch, _check_scope

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


def test_run_record_type_is_known_only_to_its_schema_and_the_audit():
    users = {p.name for p in SRC.rglob("*.py") if mentions(p, "FinalTestRunRecord")}
    assert users == {
        "runrecord.py",
        "audit.py",
        "inventory.py",
        "start.py",
        "finaltest_store.py",
        "resume.py",
    }


def test_no_run_store_constructor_takes_the_final_test_record():
    for method in (RunStore.create, RunStore.open, RunStore.__init__):
        names = method.__annotations__
        assert "FinalTestRunRecord" not in " ".join(str(v) for v in names.values())
    assert RunRecord.model_fields["phase"].annotation.__args__ == ("development", "qualification")


def test_audit_module_has_no_write_operations_and_no_publisher_imports():
    source = tree(FINALTEST / "audit.py")
    written = {
        "write_bytes",
        "write_text",
        "mkdir",
        "unlink",
        "rename",
        "replace",
        "chmod",
        "touch",
    }
    called = {
        n.func.attr
        for n in ast.walk(source)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not called & written
    assert not called & {"publish_bytes", "publish_manifest", "publish_input_record"}
    imported = {n.module for n in ast.walk(source) if isinstance(n, ast.ImportFrom) and n.module}
    assert not any(m.endswith(("registry", "guard", "driver", "runner")) for m in imported)
    opens = [
        n
        for n in ast.walk(source)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"
    ]
    assert not opens


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
    # P10-C2B publishes two rehearsal events and P10-C3 a third; only the rehearsal-gated
    # modules mention them
    assert layout.REHEARSAL_EVENTS == ("batch_inputs_bound", "batch_authorized", "batch_resumed")
    for text in ("batch_inputs_bound", "batch_authorized"):
        users = {p.name for p in SRC.rglob("*.py") if text in p.read_text(encoding="utf-8")}
        assert (
            {"layout.py", "start.py"}
            <= users
            <= {"layout.py", "inventory.py", "registry.py", "start.py", "resume.py"}
        ), (text, users)
    users = {p.name for p in SRC.rglob("*.py") if "batch_resumed" in p.read_text(encoding="utf-8")}
    assert (
        {"layout.py", "resume.py"}
        <= users
        <= {
            "layout.py",
            "inventory.py",
            "registry.py",
            "resume.py",
        }
    ), users


def test_existing_final_test_refusals_stand(monkeypatch, tmp_path):
    fake_scope = types.SimpleNamespace(phase="final_test")
    prepared = types.SimpleNamespace(
        scope=fake_scope, config=types.SimpleNamespace(settings=types.SimpleNamespace())
    )
    monkeypatch.setattr(driver, "check_placement", lambda *_: None)
    monkeypatch.setattr(driver, "prepare", lambda *_: prepared)
    plan = types.SimpleNamespace(run_root=tmp_path / "run", anchor_dir=tmp_path / "anchor")
    with pytest.raises(driver.DriverError, match="final_test scope cannot be executed here"):
        driver.execute(plan, [], mode="mock", start=lambda _: pytest.fail("started"))
    assert not (tmp_path / "run").exists()
    with pytest.raises(RosterMismatch, match="neither development nor qualification"):
        _check_scope(types.SimpleNamespace(phase="final_test"), {}, ("b", "c"))
    with pytest.raises(ValidationError):
        RunRecord.model_validate({"phase": "final_test"})


def test_the_only_run_store_subclasses_are_the_read_only_view_and_the_guarded_store():
    found = {}
    for path in SRC.rglob("*.py"):
        for node in ast.walk(tree(path)):
            if isinstance(node, ast.ClassDef) and any(
                getattr(b, "id", "") == "RunStore" for b in node.bases
            ):
                found[node.name] = path.name
    assert found == {"ReadOnlyRunView": "readonly.py", "FinalTestStore": "finaltest_store.py"}
