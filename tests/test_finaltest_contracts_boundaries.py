"""Contracts only: no execution path, no authority, no change to the real project."""

import ast
import typing
from pathlib import Path

import pytest
from readiness_support import REAL_POLICY
from test_finaltest_g13 import NO_AUTHORITY, Setup, world  # noqa: F401  (fixture)

import biorag_repro
from biorag_repro.generation.attempts import RunPhase
from biorag_repro.readiness.guard import authorize_final_test

SRC = Path(biorag_repro.__file__).parent


@pytest.fixture
def setup(world, tmp_path):  # noqa: F811
    return Setup(world, tmp_path / "storage")


def names_defined(tree: ast.AST) -> set[str]:
    return {
        n.name
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }


def test_no_execution_entrypoint_exists():
    # No final-test start or resume entry point exists in the package
    forbidden = {"create_final_test", "open_final_test", "start_batch", "resume_batch"}
    for path in SRC.rglob("*.py"):
        assert not names_defined(ast.parse(path.read_text(encoding="utf-8"))) & forbidden, path


def test_the_guard_still_has_no_caller_in_the_package():
    for path in SRC.rglob("*.py"):
        if path.name == "guard.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        used = any(
            (isinstance(n, ast.Name) and n.id == "authorize_final_test")
            or (isinstance(n, ast.Attribute) and n.attr == "authorize_final_test")
            or (isinstance(n, ast.alias) and n.name == "authorize_final_test")
            for n in ast.walk(tree)
        )
        assert not used, path


def test_run_phase_and_the_real_policy_are_untouched():
    assert typing.get_args(RunPhase) == ("development", "qualification")
    assert REAL_POLICY.slots.restart_evidence_acceptance is None
    assert REAL_POLICY.policy_version == "3"


def test_the_real_guard_still_refuses(tmp_path):
    from biorag_repro.readiness.guard import AuthorizationRefused

    with pytest.raises(AuthorizationRefused):
        authorize_final_test(tmp_path, "0" * 64, repo=tmp_path)
