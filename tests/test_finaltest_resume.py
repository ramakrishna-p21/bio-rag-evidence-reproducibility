"""Authorized resume of an interrupted rehearsal batch, on synthetic data.

The positive path uses disposable rehearsal roots, fixture readiness evidence and the mock
backend. Negative tests name the exact refusal and compare the storage before and after.
"""

import pytest
from finaltest_audit_support import CONFIGS
from finaltest_resume_support import (
    assess,
    interrupted,
    renew,
    resume,
    root_problems,
    run_to_the_end,
)
from finaltest_start_support import mock_environment

from biorag_repro.finaltest import layout
from biorag_repro.finaltest.batches import citation_sha256
from biorag_repro.finaltest.session import open_session


def test_a_crashed_batch_is_resumable_with_exactly_the_unfinished_work(tmp_path):
    w, started = interrupted(tmp_path)
    found = assess(w, started.batch_id)
    assert found.classification == "resumable" and found.code == "work_remains"
    assert not found.grants_authorization
    # 10 observations: 3 complete, 1 left running by the dead worker, 6 only planned
    assert len(found.pending) == 7 and len(found.running) == 1


def test_resume_completes_the_batch_without_touching_finished_observations(tmp_path):
    w, started = interrupted(tmp_path)
    run_dir = w.root / "runs" / started.run_id
    before = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    complete_before = {p for p in before if p.name == "final.build.json"}
    assert len(complete_before) == 3
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        assert (resumed.batch_id, resumed.run_id) == (started.batch_id, started.run_id)
        summary = run_to_the_end(resumed)
    after = {p: p.read_bytes() for p in run_dir.rglob("*") if p.is_file()}
    for path, data in before.items():
        assert after[path] == data  # append-only: nothing that existed was rewritten
    assert (summary.complete, summary.attempts) == (10, 7)  # only the unfinished 7 were attempted
    assert summary.terminal_failures == summary.retries_exhausted == 0
    final = assess(w, started.batch_id)
    assert final.classification == "already_complete"
    assert root_problems(w) == []


def test_the_resume_citation_chains_to_the_first_and_cites_the_historical_head(tmp_path):
    w, started = interrupted(tmp_path)
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
    found = assess(w, started.batch_id)
    first, second = found.citations
    assert (first.kind, second.kind) == ("initial", "resume")
    assert second.previous_citation_sha256 == citation_sha256(first)
    assert second.initial_report_sha256 == first.initial_report_sha256
    assert second.readiness_report_sha256 != first.readiness_report_sha256
    assert resumed.citation_sha256 == citation_sha256(second)
    # the cited head is the event just before the resume event, not the latest head
    head = resumed.head[0]
    events = sorted((w.root / layout.REGISTRY_DIR).glob("0*.json"))
    assert second.registry_head_sequence == head - 1 == len(events) - 1
    assert second.head_authority == "not_established"


def _mk(path):
    path.mkdir()
    return path


# ---- helpers for the refusal tests ----------------------------------------------------------

import readiness_support as support
from finaltest_start_support import SyntheticInputs

from biorag_repro.finaltest.resume import ResumeRefused
from biorag_repro.readiness.guard import AuthorizationRefused


def refused(w, code, *, new_report=True, **changes):
    """Resume must refuse with `code` and leave the whole storage byte-identical."""
    if new_report:
        renew(w, **changes.pop("report", {}))
    before = w.tree()
    with open_session(w.root) as session:
        with pytest.raises(ResumeRefused) as caught:
            resume(w, session, **changes)
        assert caught.value.code == code, str(caught.value)
        message = str(caught.value)
    # the session may have written its own worker record; the evidence must be unchanged
    after = w.tree()
    assert {k: v for k, v in after.items() if not k.endswith("session.holder")} == {
        k: v for k, v in before.items() if not k.endswith("session.holder")
    }
    return message


@pytest.fixture
def crashed(tmp_path):
    return interrupted(tmp_path)


# ---- fresh authorization ----------------------------------------------------------------------


@pytest.mark.parametrize("where", ["stored", "fresh"])
def test_not_ready_reports_are_refused_by_the_guard_and_nothing_is_written(crashed, where):
    from test_readiness_guard import build

    w, _ = crashed
    blocked = {"G12_shared_gpu_authorization": "pending"}
    if where == "stored":
        renew(w, blocked=blocked)
    else:  # the report on disk is READY, the present evaluation is not
        renew(w)
        w.fixture.live = build(w.fixture.policy, blocked=blocked)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(AuthorizationRefused) as caught:
        resume(w, session)
    assert caught.value.reason == ("stored_not_ready" if where == "stored" else "live_not_ready")
    assert w.tree() == before


def test_historical_ready_report_is_not_current_authorization(crashed):
    w, _ = crashed
    # the first report is a valid, READY historical record, and the guard can assess it again
    first = w.fixture.sha
    renew(w)
    w.fixture.sha, w.fixture.live = first, w.fixture.live
    message = refused(w, "report_reused", new_report=False)
    assert "one session" in message


def test_the_same_report_cannot_authorize_two_resumes(crashed):
    w, _ = crashed
    renew(w)
    with open_session(w.root) as session:
        resumed = resume(w, session)
        assert run_to_the_end(resumed).attempts > 0
    refused(w, "already_complete", new_report=False)


# ---- continuity -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("parts", "component"),
    [
        (
            {"candidate": support.candidate(scientific_source_identity_sha256="f" * 64)},
            "scientific_source_identity",
        ),
        ({"evaluator": support.evaluator(version="another-evaluator")}, "evaluator"),
        ({"host": support.host(hostname="another-host")}, "host"),
        ({"host": support.host(gpu_names=["Another GPU"])}, "host"),
    ],
)
def test_drift_between_the_two_authorizations_is_not_continuous(crashed, parts, component):
    w, _ = crashed
    cfg = {w.arm: CONFIGS[w.arm].configuration_sha256}
    if "candidate" not in parts:
        pass
    elif parts["candidate"].configuration_sha256_by_arm != cfg:
        parts["candidate"] = parts["candidate"].model_copy(
            update={"configuration_sha256_by_arm": cfg}
        )
    message = refused(w, "not_continuous", report=parts)
    assert component in message


def test_a_changed_policy_is_not_continuous(crashed):
    w, _ = crashed
    first_policy = w.fixture.policy
    w.fixture.policy = support.fixture_policy("another-policy")
    renew(w)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, initial_policy=first_policy)
    assert caught.value.code == "not_continuous" and "policy" in str(caught.value)
    assert w.tree() == before


def test_changed_host_or_gpu_at_run_time_blocks_resume(crashed):
    w, _ = crashed
    base = mock_environment()
    other_host = base.model_copy(update={"hostname": "elsewhere"})
    other_gpu = base.model_copy(update={"cuda_runtime": "99.9"})
    for changed in (other_host, other_gpu):
        refused(w, "host_changed", environment=lambda changed=changed: changed)


def test_missing_environment_record_blocks_resume(crashed):
    w, started = crashed
    environments = w.root / "runs" / started.run_id / "environments"
    for path in environments.iterdir():
        path.unlink()
    renew(w)
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("contradictory", "observation_corrupt")
    refused(w, "contradictory", new_report=False)


# ---- the inputs stay the ones the batch was bound to ------------------------------------------


def test_changed_question_or_context_identity_is_binding_drift(crashed):
    w, _ = crashed

    def other_question(kind, value):
        if kind == "items":
            first = value[0]
            return [
                type(first)(first.item_id, first.paper_id, "other?", first.item_sha256),
                *value[1:],
            ]
        return value[2] if kind == "context" else value

    def other_context(kind, value):
        if kind == "context":
            _item, condition, found = value
            if condition == "PDF_DOCLING":
                from dataclasses import replace

                found = replace(found, exact_context=found.exact_context + " changed")
            return found
        return value

    for change in (other_question, other_context):
        refused(w, "binding_drift", inputs=SyntheticInputs(w.arm, change=change))


def test_another_arm_configuration_is_refused_before_anything_is_read(crashed):
    w, _ = crashed
    from finaltest_start_support import FakeRenderer

    renew(w)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, renderer=FakeRenderer(CONFIGS["closed_book"]))
    assert caught.value.code == "configuration"
    assert w.tree() == before


def test_an_arm_that_was_never_started_has_no_batch_to_resume(crashed):
    w, _ = crashed
    from finaltest_audit_support import settings_for
    from finaltest_start_support import FakeRenderer
    from generation_support import to_toml

    renew(w)
    request = {
        "matrix_arm": "closed_book",
        "config_toml": to_toml(settings_for("closed_book")).encode(),
        "renderer": FakeRenderer(CONFIGS["closed_book"]),
    }
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, **request)
    assert caught.value.code == "contradictory" and "unknown_batch" in str(caught.value)


def test_a_final_test_root_is_refused_by_the_fixture_route(crashed, tmp_path):
    from finaltest_support import authorization

    w, _ = crashed
    final = authorization(tmp_path / "other", "final_test")
    before, calls = w.tree(), w.guard_calls
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, authorization=final)
    assert caught.value.code == "root_kind"
    assert w.tree() == before and w.guard_calls == calls


# ---- production stays closed ------------------------------------------------------------------


def test_the_public_resume_ends_in_a_production_refusal_even_if_the_guard_accepts(
    crashed, monkeypatch, tmp_path
):
    from biorag_repro.finaltest import resume as module
    from biorag_repro.finaltest.resume import ProductionResumeRefused, resume_batch

    w, started = crashed
    renew(w)
    monkeypatch.setattr(module, "_production_assessed", lambda *_: w.assess())
    before = w.tree()
    reads = w.request.inputs.calls  # the crashed batch already read its inputs
    with pytest.raises(ProductionResumeRefused) as caught:
        resume_batch(tmp_path, w.fixture.sha, repo=tmp_path, batch_id=started.batch_id)
    assert caught.value.code == "production_disabled"
    monkeypatch.setattr(module, "PRODUCTION_RESUME", True)
    with pytest.raises(ProductionResumeRefused) as caught:
        resume_batch(tmp_path, w.fixture.sha, repo=tmp_path, batch_id=started.batch_id)
    assert caught.value.code == "no_production_request"
    assert w.tree() == before
    assert w.request.inputs.calls == reads


def test_the_public_resume_refuses_a_fixture_ready_report_with_the_real_reason(
    crashed, tmp_path, monkeypatch
):
    import test_readiness_guard as g
    from readiness_world import build_world

    from biorag_repro.finaltest.resume import resume_batch
    from biorag_repro.readiness import publish

    real = build_world(_mk(tmp_path / "real"), monkeypatch)
    sha = publish.publish_report(g.build(g.support.fixture_policy()), real.data).report_sha256
    before = {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()}
    with pytest.raises(AuthorizationRefused) as caught:
        resume_batch(real.data, sha, repo=real.repo, batch_id="a" * 64)
    assert caught.value.reason == "report_unverifiable"
    assert {str(p): p.read_bytes() for p in real.data.rglob("*") if p.is_file()} == before


def test_resume_batch_has_no_policy_evaluator_flag_or_authorization_parameter():
    import inspect

    from biorag_repro.finaltest.resume import resume_batch

    assert list(inspect.signature(resume_batch).parameters) == [
        "data_root",
        "report_sha256",
        "repo",
        "batch_id",
    ]


# ---- identity of the campaign, and the host history -------------------------------------------


def test_a_plan_for_another_campaign_or_benchmark_is_refused_before_anything_is_read(crashed):
    from finaltest_audit_support import PLAN

    w, _ = crashed
    renew(w)
    other = PLAN.model_copy(update={"split_sha256": "d" * 64})
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, plan=other)
    assert caught.value.code == "benchmark_identity"
    assert w.tree() == before


def test_recorded_attempts_on_two_different_hosts_are_a_conflict(tmp_path):
    from types import SimpleNamespace

    from biorag_repro.core.storage import serialize_project_json
    from biorag_repro.finaltest.resume import _hardware_of
    from biorag_repro.generation.environment import EnvironmentRecord, environment_sha256

    base = mock_environment()
    other = base.model_copy(update={"hostname": "elsewhere"})
    folder = tmp_path / "runs" / "r" / "environments"
    folder.mkdir(parents=True)
    events = []
    for env in (base, other):
        digest = environment_sha256(env)
        (folder / f"{digest}.json").write_bytes(serialize_project_json(env, EnvironmentRecord))
        events.append(SimpleNamespace(state="running", environment_sha256=digest))
    one = _hardware_of(tmp_path, "r", [SimpleNamespace(events=events[:1])])
    two = _hardware_of(tmp_path, "r", [SimpleNamespace(events=events)])
    assert len(one) == 1 and len(two) == 2  # two identities: assess_resume refuses as host_conflict
    # an environment file whose bytes do not match its recorded hash is refused by name
    bad = SimpleNamespace(state="running", environment_sha256="c" * 64)
    (folder / ("c" * 64 + ".json")).write_bytes(b"{}")
    assert _hardware_of(tmp_path, "r", [SimpleNamespace(events=[bad])]) == "environment_altered"
    gone = SimpleNamespace(state="running", environment_sha256="d" * 64)
    assert _hardware_of(tmp_path, "r", [SimpleNamespace(events=[gone])]) == "environment_missing"


def test_two_recorded_hosts_make_the_batch_contradictory(crashed, monkeypatch):
    from biorag_repro.finaltest import resume as module

    w, started = crashed
    one, two = ({"hostname": "a"}, {"hostname": "b"})
    monkeypatch.setattr(module, "_hardware_of", lambda *_a: (one, two))
    found = assess(w, started.batch_id)
    assert (found.classification, found.code) == ("contradictory", "host_conflict")


def test_a_report_that_is_not_the_one_the_batch_started_on_is_refused(crashed):
    w, _ = crashed
    renew(w)
    elsewhere = renew(
        w,
        "5" * 32,
        candidate=support.candidate(
            configuration_sha256_by_arm={w.arm: CONFIGS[w.arm].configuration_sha256},
            scientific_source_identity_sha256="e" * 64,
        ),
    )
    from finaltest_resume_support import loader

    other_report = loader(w)(elsewhere)
    renew(w, "6" * 32)
    before = w.tree()
    with open_session(w.root) as session, pytest.raises(ResumeRefused) as caught:
        resume(w, session, load_report=lambda _sha: other_report)
    assert caught.value.code == "initial_report_mismatch"
    assert w.tree() == before
