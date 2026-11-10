"""P10-C4: what the completion code may and may not do, checked on the source and the routes."""

import ast
import inspect
from pathlib import Path

import pytest
from finaltest_completion_support import finished
from finaltest_resume_support import renew
from test_finaltest_resume import _mk

import biorag_repro
from biorag_repro.finaltest import finish
from biorag_repro.finaltest.finish import (
    PRODUCTION_COMPLETION,
    ProductionCompletionRefused,
    complete_batch,
)
from biorag_repro.readiness.guard import AuthorizationRefused

SRC = Path(biorag_repro.__file__).parent
FINISH = SRC / "finaltest" / "finish.py"
COMPLETION = SRC / "finaltest" / "completion.py"


def calls(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }


WRITERS = {"write_bytes", "write_text", "mkdir", "unlink", "rename", "replace", "chmod", "touch"}


def test_the_completion_code_writes_only_through_the_registry_publishers():
    assert not calls(FINISH) & WRITERS
    assert not calls(FINISH) & {"publish_bytes", "publish_manifest", "open", "rmtree", "copy"}
    assert not calls(COMPLETION) & (WRITERS | {"publish_bytes", "open", "read_bytes"})


def test_completion_never_touches_a_run_store_or_vouches_for_a_worker():
    called = calls(FINISH)
    store_writers = {
        "begin_attempt",
        "plan",
        "publish_input",
        "complete",
        "fail",
        "recover_after_session",
        "recover_interrupted",
        "_seal",
        "_append",
        "reopen_final_test_store",
        "prove_previous_worker_stopped",
    }
    assert not called & store_writers
    text = FINISH.read_text(encoding="utf-8")
    assert "RunStore" not in text and "previous_worker_stopped" not in text


def test_only_the_registry_publishers_can_publish_a_completion_event():
    users = {
        p.name
        for p in SRC.rglob("*.py")
        if "batch_completed" in p.read_text(encoding="utf-8") and p.name != "layout.py"
    }
    assert {"finish.py", "inventory.py"} <= users <= {"finish.py", "inventory.py", "resume.py"}


def test_complete_batch_has_no_policy_evaluator_flag_or_authorization_parameter():
    assert list(inspect.signature(complete_batch).parameters) == [
        "data_root",
        "report_sha256",
        "repo",
        "batch_id",
    ]
    assert PRODUCTION_COMPLETION is False


def test_the_public_completion_ends_in_a_production_refusal_even_if_the_guard_accepts(
    tmp_path, monkeypatch
):
    w, started = finished(tmp_path / "rehearsal")
    renew(w)
    seen = []
    monkeypatch.setattr(
        finish, "_production_assessed", lambda *args: seen.append(args) or w.assess()
    )
    before = w.tree()
    with pytest.raises(ProductionCompletionRefused) as caught:
        complete_batch(tmp_path, w.fixture.sha, repo=tmp_path, batch_id=started.batch_id)
    assert caught.value.code == "production_disabled" and len(seen) == 1  # the guard ran first
    monkeypatch.setattr(finish, "PRODUCTION_COMPLETION", True)
    with pytest.raises(ProductionCompletionRefused) as caught:
        complete_batch(tmp_path, w.fixture.sha, repo=tmp_path, batch_id=started.batch_id)
    assert caught.value.code == "no_production_request"
    assert w.tree() == before


def test_the_public_completion_refuses_a_fixture_ready_report_with_the_real_reason(
    tmp_path, monkeypatch
):
    import test_readiness_guard as g
    from readiness_world import build_world

    from biorag_repro.readiness import publish

    real = build_world(_mk(tmp_path / "real"), monkeypatch)
    sha = publish.publish_report(g.build(g.support.fixture_policy()), real.data).report_sha256
    before = {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()}
    with pytest.raises(AuthorizationRefused) as caught:
        complete_batch(real.data, sha, repo=real.repo, batch_id="a" * 64)
    assert caught.value.reason == "report_unverifiable"
    assert {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()} == before


def test_a_final_test_root_is_refused_by_the_fixture_route(tmp_path):
    from finaltest_completion_support import assess_it
    from finaltest_start_support import make_world

    from biorag_repro.finaltest.finish import CompletionRefused, complete_in_new_session
    from biorag_repro.finaltest.session import open_session

    w = make_world(tmp_path)
    forged = w.auth.model_copy(update={"root_kind": "final_test"})
    request = w.with_request(authorization=forged)
    with open_session(w.root) as session, pytest.raises(CompletionRefused) as caught:
        complete_in_new_session(request, w.assess, session, load_report=lambda _s: None)  # type: ignore[arg-type,return-value]
    assert caught.value.code == "root_kind"
    assert assess_it  # imported for symmetry with the other fixtures
