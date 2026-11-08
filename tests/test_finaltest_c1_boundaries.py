"""P10-C1 adds contracts only: no execution path, no authority, no change to the real project."""

import ast
import hashlib
import typing
from pathlib import Path

import pytest
from test_finaltest_g13 import NO_AUTHORITY, Setup, world  # noqa: F401  (fixture)

import biorag_repro
from biorag_repro.finaltest import layout
from biorag_repro.finaltest.head import HeadAttestation, attestation_sha256
from biorag_repro.generation.attempts import RunPhase
from biorag_repro.readiness import evidence
from biorag_repro.readiness.guard import authorize_final_test
from biorag_repro.readiness.policy import REAL_POLICY

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
    assert all(value is None for value in REAL_POLICY.slots.model_dump().values())


def test_the_real_guard_still_refuses(tmp_path):
    from biorag_repro.readiness.guard import AuthorizationRefused

    with pytest.raises(AuthorizationRefused):
        authorize_final_test(tmp_path, "0" * 64, repo=tmp_path)


def test_a_stored_head_attestation_changes_nothing_about_the_real_g13(setup):
    auth, _ = setup.approve()
    setup.initialize(auth)
    root_id = layout.root_id_for(auth)
    claim = HeadAttestation(
        root_id=root_id,
        sequence=1,
        event_file_sha256=hashlib.sha256(
            (Path(auth.authorized_root) / "registry/000001.json").read_bytes()
        ).hexdigest(),
        provenance="test_fixture",
    )
    folder = setup.world.data / "readiness/approvals/head_attestation"
    folder.mkdir(parents=True)
    (folder / f"{attestation_sha256(claim)}.json").write_bytes(layout.serialize(claim))
    for policy in (REAL_POLICY, None):
        got = setup.g13(policy)
        assert (got.state, got.reason) == ("unavailable", NO_AUTHORITY)
    assert evidence.history_integrity().state == "unavailable"


def test_the_continuity_and_head_modules_import_no_execution_code():
    banned = {"generation", "retrieval", "evaluation", "ingestion", "chunking"}
    for name in ("head.py", "continuity.py"):
        tree = ast.parse((SRC / "finaltest" / name).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[1] if "." in node.module else node.module)
            elif isinstance(node, ast.Import):
                imported.update(a.name.split(".")[1] for a in node.names if "." in a.name)
        assert not imported & banned, (name, imported & banned)
