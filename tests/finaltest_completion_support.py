"""Builders for completion tests: batches in each state a completion can meet."""

from finaltest_audit_support import ITEMS, PLAN
from finaltest_resume_support import DiesAfter, Stop, loader, renew, resume
from finaltest_start_support import MockBackend, World, make_world, mock_environment

from biorag_repro.finaltest.campaign import expected_keys
from biorag_repro.finaltest.execution import StartedBatch, run_batch
from biorag_repro.finaltest.finish import (
    CompletionResult,
    assess_completion,
    complete_in_new_session,
    complete_in_session,
    verify_completion,
)
from biorag_repro.finaltest.session import open_session
from biorag_repro.finaltest.start import _start_with

__all__ = [
    "ARM",
    "FailsOn",
    "assess_it",
    "complete_fresh",
    "expected_for",
    "finished",
    "in_session",
    "stopped",
    "verify_it",
]

ARM = "dense_robustness"  # ten synthetic observations: JATS and PDF_DOCLING


class FailsOn(MockBackend):
    """Raises an unclassified error (a terminal failure) on the call numbers given."""

    def __init__(self, *calls: int):
        super().__init__()
        self.fail_on = set(calls)

    def generate(self, rendered, *, seed, max_new_tokens):
        if len(self.calls) in self.fail_on:
            self.calls.append(rendered)
            raise RuntimeError("synthetic terminal failure")
        return super().generate(rendered, seed=seed, max_new_tokens=max_new_tokens)


def expected_for(arm: str = ARM) -> list[str]:
    return expected_keys(PLAN, ITEMS, arm)


def finished(base, arm: str = ARM, backend=None) -> tuple[World, StartedBatch]:
    """A batch run to the end in its own session, which is then closed. Nothing is completed."""
    w = make_world(base, arm=arm)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        run_batch(started, backend or MockBackend(), capture_environment=mock_environment)
    return w, started


def stopped(base, arm: str = ARM, after: int = 3) -> tuple[World, StartedBatch]:
    """A batch whose worker died during the attempt after `after` completions."""
    w = make_world(base, arm=arm)
    renew(w, "0" * 32)
    with open_session(w.root) as session:
        started = _start_with(w.request, w.assess, session, now=w.clock)
        try:
            run_batch(started, DiesAfter(after), capture_environment=mock_environment)
        except Stop:
            pass
    return w, started


def in_session(w: World, started: StartedBatch, session) -> CompletionResult:
    return complete_in_session(w.request, session, started, now=w.clock)


def complete_fresh(w: World, nonce: str = "1" * 32, *, new_report: bool = True, **parts):
    """A new session with a new fixture-authorized report completes the batch."""
    if new_report:
        renew(w, nonce, **parts)
    with open_session(w.root) as session:
        return complete_in_new_session(
            w.request, w.assess, session, load_report=loader(w), now=w.clock
        )


def assess_it(w: World, batch_id: str):
    return assess_completion(
        w.root,
        w.auth,
        w.anchors,
        w.witnesses,
        plan=PLAN,
        item_papers=dict(w.request.population.item_papers),
        batch_id=batch_id,
    )


def verify_it(w: World, batch_id: str):
    return verify_completion(
        w.root,
        w.auth,
        w.anchors,
        w.witnesses,
        plan=PLAN,
        item_papers=dict(w.request.population.item_papers),
        batch_id=batch_id,
    )


assert resume  # re-exported for tests that resume before completing
