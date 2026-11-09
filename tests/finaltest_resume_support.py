"""Builders for resume tests: an interrupted rehearsal batch and a renewed fixture authorization."""

import pytest
import readiness_support as support
from finaltest_audit_support import CONFIGS, PLAN
from finaltest_start_support import MockBackend, World, make_world, mock_environment
from test_readiness_guard import build

from biorag_repro.finaltest.execution import StartedBatch, run_batch
from biorag_repro.finaltest.inventory import HEAD_NOT_ESTABLISHED_TEXT, verify_root
from biorag_repro.finaltest.resume import _resume_with, assess_resume
from biorag_repro.finaltest.session import Session, open_session
from biorag_repro.finaltest.start import _start_with
from biorag_repro.readiness import publish

__all__ = [
    "DiesAfter",
    "Stop",
    "assess",
    "interrupted",
    "plan_only",
    "renew",
    "resume",
    "root_problems",
    "run_to_the_end",
]


class Stop(BaseException):
    """Stands for a worker that dies in the middle of an attempt."""


class DiesAfter(MockBackend):
    def __init__(self, calls: int):
        super().__init__()
        self.limit = calls

    def generate(self, rendered, *, seed, max_new_tokens):
        if len(self.calls) == self.limit:
            raise Stop
        return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)


def interrupted(base, arm: str = "dense_robustness", after: int = 3) -> tuple[World, StartedBatch]:
    """A started batch whose worker died during the attempt after `after` completions."""
    w = make_world(base, arm=arm)
    renew(w, "0" * 32)  # the first authorization names the arm's configuration as approved
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        with pytest.raises(Stop):
            run_batch(started, DiesAfter(after), capture_environment=mock_environment)
    return w, started


def renew(w: World, nonce: str = "1" * 32, **parts) -> str:
    """Make the fixture guard assess a new READY report (different bytes, same science)."""
    candidate = support.candidate(
        configuration_sha256_by_arm={w.arm: CONFIGS[w.arm].configuration_sha256}
    )
    parts = {"candidate": candidate} | parts
    evaluation = build(w.fixture.policy, nonce=nonce, **parts)
    w.fixture.sha = publish.publish_report(evaluation, w.fixture.data).report_sha256
    w.fixture.live = evaluation
    return w.fixture.sha


def loader(w: World, policy=None):
    """Reads a stored report back under the policy it was made under (default: the present one)."""
    return lambda sha: publish.read_report(w.fixture.data, sha, policy or w.fixture.policy).report


def resume(
    w: World,
    session: Session,
    *,
    environment=mock_environment,
    initial_policy=None,
    load_report=None,
    **changes,
):
    request = w.with_request(**changes) if changes else w.request
    return _resume_with(
        request,
        w.assess,
        session,
        load_report=load_report or loader(w, initial_policy),
        capture_environment=environment,
        now=w.clock,
    )


def assess(w: World, batch_id: str):
    return assess_resume(
        w.root,
        w.auth,
        w.anchors,
        w.witnesses,
        plan=PLAN,
        item_papers=dict(w.request.population.item_papers),
        batch_id=batch_id,
    )


def root_problems(w: World) -> list[str]:
    verdict = verify_root(w.root, w.auth, w.anchors, w.witnesses)
    return [p for p in verdict.problems if p != HEAD_NOT_ESTABLISHED_TEXT]


def run_to_the_end(started: StartedBatch):
    return run_batch(started, MockBackend(), capture_environment=mock_environment)


assert CONFIGS  # the arms' configurations are built on import


def plan_only(base, arm: str = "dense_robustness", *, planned: int | None = None):
    """A started batch whose worker died after planning, before any attempt began.

    With `planned`, it died while planning: only that many observations exist.
    """
    from finaltest_start_support import make_world

    w = make_world(base, arm=arm)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        patch = pytest.MonkeyPatch()
        if planned is not None:
            real = type(started.store).plan
            seen = {"n": 0}

            def plan(self, coordinates):
                seen["n"] += 1
                if seen["n"] > planned:
                    raise Stop
                return real(self, coordinates)

            patch.setattr(type(started.store), "plan", plan)

        def no_environment():
            raise Stop

        try:
            with pytest.raises(Stop):
                run_batch(
                    started,
                    MockBackend(),
                    capture_environment=mock_environment if planned is not None else no_environment,
                )
        finally:
            patch.undo()
    return w, started
