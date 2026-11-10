"""What the resume code may and may not touch, checked on the source."""

import ast
from pathlib import Path

import pytest

import biorag_repro
from biorag_repro.generation.finaltest_store import (
    FinalTestStore,
    StoreOperationRefused,
    reopen_final_test_store,
)

SRC = Path(biorag_repro.__file__).parent
RESUME = SRC / "finaltest" / "resume.py"


def calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }


def test_the_resume_module_writes_only_through_the_registry_publishers_and_the_store():
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
    assert not calls(RESUME) & written
    assert not calls(RESUME) & {"publish_bytes", "publish_manifest", "open"}


def test_no_final_test_module_opens_a_run_store_or_vouches_for_a_stopped_worker():
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "previous_worker_stopped=True" in text:
            assert path.name in ("finaltest_store.py", "attempts.py", "driver.py"), path
        if "finaltest" in path.parts:
            tree = ast.parse(text)
            opened = [
                n
                for n in ast.walk(tree)
                if isinstance(n, ast.Attribute)
                and (
                    (n.attr == "open" and getattr(n.value, "id", "") == "RunStore")
                    or n.attr == "recover_interrupted"
                )
            ]
            assert not opened, path


def test_reopening_is_used_only_by_the_resume_module():
    users = {
        p.name
        for p in SRC.rglob("*.py")
        if "reopen_final_test_store" in p.read_text(encoding="utf-8")
    }
    assert users == {"finaltest_store.py", "resume.py"}


def test_a_stored_run_still_cannot_be_opened_or_recovered_generically(tmp_path):
    for name in ("create", "open", "recover_interrupted"):
        with pytest.raises(StoreOperationRefused):
            getattr(FinalTestStore, name)(tmp_path)


def test_recovery_needs_the_stopped_worker_evidence_not_a_flag(tmp_path):
    store = FinalTestStore.__new__(FinalTestStore)
    with pytest.raises(StoreOperationRefused, match="previous worker stopped"):
        store.recover_after_session(True)  # type: ignore[arg-type]


def test_reopening_a_final_test_root_run_is_refused(tmp_path):
    record = type("R", (), {"root_kind": "final_test"})()
    with pytest.raises(StoreOperationRefused, match="rehearsal"):
        reopen_final_test_store(tmp_path, record, None, None, None)  # type: ignore[arg-type]


def test_reopening_checks_the_run_files_against_the_record(tmp_path):
    from finaltest_resume_support import interrupted

    from biorag_repro.finaltest.layout import serialize
    from biorag_repro.finaltest.runrecord import FinalTestRunRecord
    from biorag_repro.generation.config import resolve_generator_config

    w, started = interrupted(tmp_path)
    run_dir = w.root / "runs" / started.run_id
    record = FinalTestRunRecord.model_validate_json((run_dir / "run.json").read_bytes())
    config = resolve_generator_config(w.request.config_toml)
    assert serialize(record) == (run_dir / "run.json").read_bytes()
    reopen_final_test_store(w.root, record, config, lambda *_a: None, None)  # the control
    for name in ("config.toml", "config.canonical.json", "run.json"):
        path = run_dir / name
        original = path.read_bytes()
        path.chmod(0o644)
        path.write_bytes(original + b" ")
        with pytest.raises(StoreOperationRefused, match=name):
            reopen_final_test_store(w.root, record, config, lambda *_a: None, None)
        path.write_bytes(original)
