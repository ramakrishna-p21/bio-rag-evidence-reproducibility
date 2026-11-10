"""Boundaries of the start path: the read-only view, the guarded store, production refusal, development.

These tests classify every writable path of `RunStore`, so a method added later fails here
until someone decides which side of the line it is on.
"""

import ast
import hashlib
import inspect
import textwrap
import typing
from pathlib import Path

import pytest
from finaltest_audit_support import CONFIGS, build_run
from pydantic import ValidationError
from test_generation_attempts import config_for

import biorag_repro
from biorag_repro.finaltest.runrecord import (
    FinalTestRunRecord,
    run_record_projection,
    run_record_sha256,
)
from biorag_repro.generation import attempts, finaltest_store, readonly
from biorag_repro.generation.attempts import RunPhase, RunRecord, RunStore
from biorag_repro.generation.finaltest_store import FinalTestStore

SRC = Path(biorag_repro.__file__).parent

# Methods that only read. Anything not listed here or in WRITERS fails the classification test.
READERS = {
    "run_id", "run_dir", "retry_policy", "observation_keys", "_load", "_check_event",
    "_check_content", "_check_completion", "_receipt_path", "_check_stored_receipt",
    "_check_receipt", "_verification_sha256", "_check_verification_citation",
    "_check_environment_file", "_check_environment", "_check_coordinates", "key_for", "_state",
    "_seal_path", "_seal_manifest", "_check_seal", "observation", "verify", "pending",
    "receipt_of", "_running", "_raw_ref",
}  # fmt: skip
# Methods that write or can lead to writing: refused by the read-only view, gated or refused
# by the guarded store.
WRITERS = {
    "create", "open", "plan", "begin_attempt", "complete", "fail", "recover_interrupted",
    "publish_input", "_append", "_seal",
}  # fmt: skip
GATED = {"plan", "publish_input", "begin_attempt", "complete", "fail", "_append", "_seal"}
REFUSED = {"create", "open", "recover_interrupted"}


def members():
    return {
        name
        for name, value in inspect.getmembers(RunStore)
        if not name.startswith("__")
        and (callable(value) or isinstance(value, property))
        and name not in {"model_fields"}
    }


def test_every_runstore_method_is_classified():
    assert members() == READERS | WRITERS, members() ^ (READERS | WRITERS)


def test_the_read_only_view_refuses_every_writer_and_inherits_only_readers():
    for name in WRITERS:
        assert getattr(readonly.ReadOnlyRunView, name) is readonly._refuse, name
    for name in READERS:
        assert name not in vars(readonly.ReadOnlyRunView), name


def test_the_guarded_store_gates_or_refuses_every_writer():
    for name in WRITERS:
        assert name in vars(FinalTestStore), name
    for name in REFUSED:
        assert vars(FinalTestStore)[name] is finaltest_store._refused
    for name in GATED:
        assert "_gate(" in inspect.getsource(vars(FinalTestStore)[name]), name
    assert set(vars(FinalTestStore)) & READERS == set()


def test_reader_methods_contain_no_write_operations():
    written = {
        "publish_bytes", "publish_manifest", "write_bytes", "write_text", "mkdir", "unlink",
        "rename", "replace", "chmod", "touch", "_append", "_seal", "open",
    }  # fmt: skip
    for name in READERS:
        member = inspect.getattr_static(RunStore, name)
        function = member.fget if isinstance(member, property) else member
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        called = {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        assert not called & written, (name, called & written)


def test_the_projection_covers_every_run_record_field_and_is_validated(tmp_path):
    built = build_run(tmp_path)
    projection = run_record_projection(built.record)
    assert set(RunRecord.model_fields) - {"schema_version"} <= set(FinalTestRunRecord.model_fields)
    for name in RunRecord.model_fields:
        if name not in ("phase", "schema_version"):
            assert getattr(projection, name) == getattr(built.record, name), name
    assert projection.phase == "final_test"
    assert typing.get_args(RunPhase) == ("development", "qualification")
    # the projection cannot be stored as a RunRecord
    with pytest.raises(ValidationError):
        RunRecord.model_validate(projection.model_dump())
    # every other field is validated: an inconsistent record does not project
    broken = built.record.model_copy(update={"backend_kind": "official"})
    with pytest.raises(ValidationError):
        run_record_projection(broken)
    source = (SRC / "finaltest/audit.py").read_text()
    assert "model_construct" not in source


def test_a_rehearsal_record_can_never_be_official(tmp_path):
    built = build_run(tmp_path)
    values = built.record.model_dump()
    with pytest.raises(ValidationError, match="never holds an official run"):
        FinalTestRunRecord.model_validate(
            values | {"backend_kind": "official", "purpose": "development_candidate"}
        )
    assert built.record.official is False
    official = FinalTestRunRecord.model_validate(
        values
        | {
            "root_kind": "final_test",
            "backend_kind": "official",
            "purpose": "development_candidate",
        }
    )
    assert official.official and official.purpose == "development_candidate"
    assert run_record_sha256(official) != run_record_sha256(built.record)


def test_the_purpose_decision_leaves_configuration_identities_alone():
    from biorag_repro.generation.config import GeneratorConfiguration

    assert typing.get_args(GeneratorConfiguration.model_fields["purpose"].annotation) == (
        "synthetic_test",
        "development_candidate",
    )
    # the same development-candidate configuration is a development and a final-test candidate
    assert config_for("official").settings.purpose == "development_candidate"


def test_development_run_files_are_byte_identical(tmp_path, monkeypatch):
    """Digests taken from the code before P10-C2B; `RunStore` and `RunRecord` did not change."""
    monkeypatch.setattr(attempts, "new_run_id", lambda: "0123456789abcdef0123456789abcdef")
    store = RunStore.create(
        tmp_path,
        config=config_for("mock"),
        phase="development",
        benchmark_version="bench_v1",
        backend_kind="mock",
        now=lambda: "2026-11-05T12:00:00Z",
    )
    digests = {
        name: hashlib.sha256((store.run_dir / name).read_bytes()).hexdigest()
        for name in ("run.json", "config.canonical.json", "config.toml")
    }
    assert digests == {
        "run.json": "74adaad4f30cb385758fcf487688245826dafbeb11d685c9dbb3b0fed1c37a2f",
        "config.canonical.json": "4a92bec69d909fb9d5a0a17ff86308ac9cad0f141c82c402c1cf7f8a50a51945",
        "config.toml": "5d547df1d342286f4a568a0e5d220c357aa260658f4873d282a168840c6df7a2",
    }


def test_run_roster_and_run_store_are_unchanged_by_the_final_test_store():
    assert "before_attempt" not in inspect.getsource(
        __import__("biorag_repro.generation.runner", fromlist=["x"]).run_roster
    )
    assert "final_test" not in inspect.getsource(RunStore)
    assert FinalTestStore.__mro__[1] is RunStore


# ---- production refusal and reader isolation -----------------------------------------------------


def trees():
    for path in SRC.rglob("*.py"):
        yield path, ast.parse(path.read_text(encoding="utf-8"))


def test_the_orchestrator_and_the_guard_are_referenced_only_where_expected():
    def users(name):
        found = set()
        for path, tree in trees():
            for node in ast.walk(tree):
                if (
                    (isinstance(node, ast.Name) and node.id == name)
                    or (isinstance(node, ast.Attribute) and node.attr == name)
                    or (isinstance(node, ast.alias) and node.name == name)
                ):
                    found.add(path.name)
        return found

    # defined in start.py, called by nothing in the package: only tests reach the orchestrator
    assert users("_start_with") == set()
    assert users("open_session") == set()
    assert users("authorize_final_test") == {"start.py"}  # guard.py only defines it
    assert users("create_final_test_store") == {"start.py"}


def test_start_batch_calls_the_guard_then_refuses_and_nothing_else():
    source = (SRC / "finaltest/start.py").read_text(encoding="utf-8")
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "start_batch"
    )
    calls = [
        n.func.id
        for n in ast.walk(function)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert calls[0] == "_production_assessed"
    assert set(calls) <= {"_production_assessed", "ProductionStartRefused"}
    assert [n for n in function.body if isinstance(n, ast.Raise)]
    from biorag_repro.finaltest import start

    assert start.PRODUCTION_START is False


def test_nothing_in_the_package_implements_the_protected_input_reader():
    implementations = []
    for path, tree in trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
                if {"items", "context"} <= methods and node.name != "ProtectedInputs":
                    implementations.append((path.name, node.name))
    assert implementations == []


def test_the_new_modules_do_not_read_the_real_benchmark_items():
    for name in ("start.py", "protected.py", "execution.py", "session.py"):
        source = (SRC / "finaltest" / name).read_text(encoding="utf-8")
        for forbidden in (
            "read_development_view",
            "items.jsonl",
            "release.items",
            "read_regular(data_root",
        ):
            assert forbidden not in source, (name, forbidden)


def test_configurations_used_by_the_fixtures_are_rehearsal_configurations():
    for config in CONFIGS.values():
        assert config.settings.purpose == "synthetic_test"
