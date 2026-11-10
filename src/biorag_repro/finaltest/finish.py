"""Completing one rehearsal batch: eligibility, authorization, publication and read-back.

Two entry points, as in `start.py` and `resume.py`.

`complete_batch(data_root, report_sha256, repo=..., batch_id=...)` is the production route. It
calls the readiness guard and then refuses (`PRODUCTION_COMPLETION` is False).

`complete_in_session` and `complete_in_new_session` are reached by tests with a fixture assessor
and accept rehearsal roots only. A completion is a scientifically consequential publication, so
a historical citation, a valid registry or the session lock alone never permit it:

    same session     the session that started or resumed the batch still holds its lock and
                     the batch is exactly as that session left it (the session gate over every
                     bound file and the registry head); the authorization is the batch's latest
                     citation, which the completion cites
    new session      a fresh guard result on a report not cited before, compared with the first
                     report of the batch and found continuous; the continuity assessment is
                     published and the completion cites it

`assess_completion` and `verify_completion` are read-only. Eligibility is decided from the
observation states (`completion.check_decided`) and the verified registry, not from a resume
classification. Nothing here repairs, adopts or regenerates: an observation that is not decided
is work for an authorized resume, and an unfinished or conflicting publication stays so.

Order of one completion, inside the session lock and then the registry lock; nothing is written
before step 4:

    1. verify the whole root and assess the batch: it must be completable
    2. authorization: same-session checks, or guard + continuity comparison
    3. build the completion record in memory (it cites the registry head before its own event;
       the registry lock is held, so the head cannot move before the event)
    4. `batch_completed` event (+ witness), continuity file (new session only), completion file
    5. read the batch back: it must verify as completed, and the stored record must equal the
       record rebuilt from the run

A crash between the publications leaves a root the verifier reports as incomplete. An intact,
fully published completion whose read-back was interrupted verifies read-only; it is never
published again.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, NoReturn

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import layout, registry
from biorag_repro.finaltest.batches import batch_id_for, citation_sha256
from biorag_repro.finaltest.campaign import CampaignPlan, expected_keys, plan_sha256
from biorag_repro.finaltest.completion import (
    BatchCompletion,
    NotCompletable,
    Outcome,
    PublicationAuthorization,
    build_completion,
    check_decided,
    completion_sha256,
)
from biorag_repro.finaltest.continuity import (
    BatchScope,
    compare_assessments,
    continuity_assessment_sha256,
    scientific_identity_sha256,
)
from biorag_repro.finaltest.execution import StartedBatch
from biorag_repro.finaltest.inputs import binding_sha256
from biorag_repro.finaltest.layout import EvidenceCitation, StorageAuthorization
from biorag_repro.finaltest.resume import ResumeAssessment, assess_resume
from biorag_repro.finaltest.runrecord import run_record_sha256
from biorag_repro.finaltest.session import Session
from biorag_repro.finaltest.start import (
    Assessed,
    StartRefused,
    StartRequest,
    _check_request,
    _checked,
    _production_assessed,
)
from biorag_repro.generation.environment import utc_now
from biorag_repro.readiness.models import ReadinessReport

PRODUCTION_COMPLETION: Final = False

Status = Literal[
    "completable",
    "completed",
    "work_remains",
    "closed_incomplete",
    "structurally_incomplete",
    "contradictory",
]


class CompletionRefused(Exception):
    """The completion was refused. `code` says at which check; nothing before step 4 is written."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class ProductionCompletionRefused(CompletionRefused):
    """The production route is closed in this version."""


@dataclass(frozen=True)
class Completability:
    """Read-only answer to "may this batch be completed?". It grants nothing."""

    status: Status
    code: str
    detail: str
    assessment: ResumeAssessment
    outcome: Outcome | None = None  # set when completable or completed

    @property
    def grants_authorization(self) -> Literal[False]:
        return False


@dataclass(frozen=True)
class CompletionResult:
    completion: BatchCompletion
    sha256: str


def complete_batch(data_root: Path, report_sha256: str, *, repo: Path, batch_id: str) -> NoReturn:
    """Production completion. Always refuses in this version; see the module text."""
    _production_assessed(data_root, report_sha256, repo)  # AuthorizationRefused when not READY
    if not PRODUCTION_COMPLETION:
        raise ProductionCompletionRefused(
            "production_disabled", "final-test completion is not enabled"
        )
    raise ProductionCompletionRefused("no_production_request", "no production request exists")


# ---- read-only ----------------------------------------------------------------------------


def _outcome(assessment: ResumeAssessment) -> Outcome:
    complete = all(s.status == "complete" for s in assessment.states)
    return "complete_all_observed" if complete else "complete_with_recorded_missing"


def assess_completion(
    root: Path,
    storage: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    *,
    plan: CampaignPlan,
    item_papers: Mapping[str, str],
    batch_id: str,
) -> Completability:
    """Classify `batch_id` for completion. Reads only; takes no lock, so the answer can go stale."""
    found = assess_resume(
        root, storage, anchors, witnesses, plan=plan, item_papers=item_papers, batch_id=batch_id
    )

    def answer(status: Status, outcome: Outcome | None = None) -> Completability:
        return Completability(status, found.code, found.detail, found, outcome)

    if found.classification == "completed":
        return answer("completed", _outcome(found))
    if found.classification == "resumable":
        return answer("work_remains")
    if found.classification == "structurally_incomplete":
        return answer("structurally_incomplete")
    if found.classification == "contradictory":
        return answer("contradictory")
    if found.code == "batch_closed_incomplete":
        return answer("closed_incomplete")
    # complete, or only terminal outcomes left: the observations themselves decide
    assert found.manifest is not None
    expected = expected_keys(plan, sorted(item_papers), found.manifest.matrix_arm)
    try:
        check_decided(found.states, expected)
    except NotCompletable as error:
        return Completability("work_remains", error.code, str(error), found)
    return answer("completable", _outcome(found))


def verify_completion(
    root: Path,
    storage: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    *,
    plan: CampaignPlan,
    item_papers: Mapping[str, str],
    batch_id: str,
) -> BatchCompletion:
    """Read a published completion back and check it against the run. Writes nothing.

    The stored record must equal, field for field, the record rebuilt from the verified run, so
    a completion that no longer describes the observations is refused. Raises `CompletionRefused`.
    """
    result = assess_completion(
        root, storage, anchors, witnesses, plan=plan, item_papers=item_papers, batch_id=batch_id
    )
    if result.status != "completed":
        raise CompletionRefused(result.status, f"{result.code} {result.detail}".strip())
    found = result.assessment
    path = root / layout.BATCHES_DIR / batch_id / layout.COMPLETION_NAME
    stored = BatchCompletion.model_validate_json(path.read_bytes(), strict=True)
    assert found.manifest is not None and found.record is not None
    assert found.authorization_event_sha256 is not None
    expected = expected_keys(plan, sorted(item_papers), found.manifest.matrix_arm)
    try:
        rebuilt = build_completion(
            manifest=found.manifest,
            run=found.record,
            citations=found.citations,
            authorizing_event_sha256=found.authorization_event_sha256,
            head=(stored.registry_head_sequence, stored.registry_head_event_sha256),
            authorization=stored.authorization,
            manifest_sha256=stored.manifest_sha256,
            states=found.states,
            expected=expected,
            created_utc=stored.created_utc,
        )
    except NotCompletable as error:
        raise CompletionRefused("completion_not_supported", str(error)) from error
    if layout.serialize(rebuilt) != path.read_bytes():
        raise CompletionRefused(
            "completion_differs_from_run", "the record is not what the run shows"
        )
    return stored


# ---- orchestration ------------------------------------------------------------------------


def complete_in_session(
    request: StartRequest,
    session: Session,
    started: StartedBatch,
    *,
    now: Callable[[], str] = utc_now,
) -> CompletionResult:
    """Complete the batch under the authorization of the session that ran it, still held."""
    return _complete(request, session, started=started, now=now)


def complete_in_new_session(
    request: StartRequest,
    assess: Callable[[], Assessed],
    session: Session,
    *,
    load_report: Callable[[str], ReadinessReport],
    now: Callable[[], str] = utc_now,
) -> CompletionResult:
    """Complete the batch in a session that did not run it, on a fresh report.

    `load_report` reads the first report of the batch back for the continuity comparison only.
    """
    return _complete(request, session, assess=assess, load_report=load_report, now=now)


def _complete(
    request: StartRequest,
    session: Session,
    *,
    started: StartedBatch | None = None,
    assess: Callable[[], Assessed] | None = None,
    load_report: Callable[[str], ReadinessReport] | None = None,
    now: Callable[[], str],
) -> CompletionResult:
    storage, root, arm, plan = request.authorization, request.root, request.matrix_arm, request.plan
    try:
        _check_request(request, session, ("rehearsal",))
    except StartRefused as error:
        raise CompletionRefused(error.code, str(error)) from error
    with session.registry() as held:
        batch = batch_id_for(layout.root_id_for(storage), plan_sha256(plan), arm)
        item_papers = dict(request.population.item_papers)
        # step 1
        verdict = assess_completion(
            root, storage, request.anchors, request.witnesses,
            plan=plan, item_papers=item_papers, batch_id=batch,
        )  # fmt: skip
        if verdict.status != "completable":
            raise CompletionRefused(verdict.status, f"{verdict.code} {verdict.detail}".strip())
        found = verdict.assessment
        assert found.manifest is not None and found.record is not None
        assert found.binding is not None and found.run_id is not None
        assert found.authorization_event_sha256 is not None
        head_before = registry.registry_head(root)
        # step 2
        continuity_bytes: bytes | None = None
        if started is not None:
            authorization = _same_session(session, started, found, head_before, batch)
        else:
            assert assess is not None and load_report is not None
            authorization, continuity_bytes = _fresh_session(
                found, assess, load_report, head_before, batch, root
            )
        # step 3
        record = build_completion(
            manifest=found.manifest,
            run=found.record,
            citations=found.citations,
            authorizing_event_sha256=found.authorization_event_sha256,
            head=head_before,
            authorization=authorization,
            manifest_sha256=sha256_bytes(layout.serialize(found.manifest)),
            states=found.states,
            expected=expected_keys(plan, sorted(item_papers), arm),
            created_utc=now(),
        )
        record_bytes, record_digest = layout.serialize(record), completion_sha256(record)
        # step 4
        args = (held, root, storage)
        registry.append_event_locked(
            *args,
            request.witnesses,
            "batch_completed",
            [
                EvidenceCitation(role="batch_id", sha256=batch),
                EvidenceCitation(
                    role="previous_authorization_event", sha256=found.authorization_event_sha256
                ),
                EvidenceCitation(role="authorization_basis", sha256=authorization.basis_sha256),
                EvidenceCitation(role="completion_record", sha256=record_digest),
                EvidenceCitation(role="readiness_report", sha256=authorization.report_sha256),
                EvidenceCitation(role="run_record", sha256=run_record_sha256(found.record)),
            ],
            now=now,
        )
        base = f"{layout.BATCHES_DIR}/{batch}"
        if continuity_bytes is not None:
            registry.publish_batch_file(
                *args, f"{base}/{layout.COMPLETION_CONTINUITY_NAME}", continuity_bytes
            )
        registry.publish_batch_file(*args, f"{base}/{layout.COMPLETION_NAME}", record_bytes)
    # step 5: read-only, so it needs no lock
    stored = verify_completion(
        root, storage, request.anchors, request.witnesses,
        plan=plan, item_papers=item_papers, batch_id=batch,
    )  # fmt: skip
    return CompletionResult(stored, record_digest)


def _same_session(
    session: Session,
    started: StartedBatch,
    found: ResumeAssessment,
    head: tuple[int, str],
    batch: str,
) -> PublicationAuthorization:
    last = found.citations[-1]
    if started.session is not session or started.batch_id != batch:
        raise CompletionRefused("not_the_running_session", "the batch was not run by this session")
    try:
        session.assert_held()
        started.store.check_unchanged()  # every bound file and the head, as the session left them
    except Exception as error:  # SessionLost or GateRefused: the authorization no longer holds
        raise CompletionRefused("session_unchanged_check", str(error)[:200]) from error
    if (
        started.run_id != found.run_id
        or started.report_sha256 != last.readiness_report_sha256
        or started.binding_sha256 != last.inputs_set_sha256
        or started.binding_sha256 != binding_sha256(found.binding)  # type: ignore[arg-type]
        or started.run_record_sha256 != run_record_sha256(found.record)  # type: ignore[arg-type]
        or head[1] != found.authorization_event_sha256
    ):
        raise CompletionRefused(
            "authorization_not_current", "the session's authorization is not the batch's latest"
        )
    return PublicationAuthorization(
        mode="same_session",
        report_sha256=last.readiness_report_sha256,
        report_binding_sha256=last.report_binding_sha256,
        policy_sha256=last.policy_sha256,
        evaluator_code_sha256=last.evaluator_code_sha256,
        scientific_identity_sha256=last.scientific_identity_sha256,
        basis_sha256=citation_sha256(last),
    )


def _fresh_session(
    found: ResumeAssessment,
    assess: Callable[[], Assessed],
    load_report: Callable[[str], ReadinessReport],
    head_before: tuple[int, str],
    batch: str,
    root: Path,
) -> tuple[PublicationAuthorization, bytes]:
    assert found.manifest is not None and found.binding is not None
    first = found.citations[0]
    try:
        authorization, report = _checked(assess())
    except StartRefused as error:
        raise CompletionRefused(error.code, str(error)) from error
    if registry.registry_head(root) != head_before:
        raise CompletionRefused("head_changed", "the registry moved while the guard ran")
    if any(c.readiness_report_sha256 == authorization.report_sha256 for c in found.citations):
        raise CompletionRefused("report_reused", "a report that already authorized this batch")
    try:
        initial = load_report(first.initial_report_sha256)
    except Exception as error:  # any failure to read it back means no continuity evidence
        raise CompletionRefused("initial_report_unavailable", str(error)[:200]) from error
    if initial.binding_sha256 != first.report_binding_sha256:
        raise CompletionRefused("initial_report_mismatch", "not the report the batch started on")
    manifest = found.manifest
    scope = BatchScope(
        campaign_plan_sha256=manifest.campaign_plan_sha256,
        batch_id=batch,
        matrix_arm=manifest.matrix_arm,
        resolved_config_sha256=manifest.resolved_config_sha256,
        ownership_sha256=manifest.ownership,
        key_set_sha256=manifest.key_set_sha256,
        inputs_set_sha256=binding_sha256(found.binding),
    )
    continuity = compare_assessments(initial, report, scope, scope)
    if continuity.verdict != "continuous":
        blocking = [f.component for f in continuity.findings if f.verdict == "blocking"]
        raise CompletionRefused("not_continuous", ", ".join(blocking))
    identity = scientific_identity_sha256(report, scope)  # equal to the first one: continuous
    return (
        PublicationAuthorization(
            mode="fresh_session",
            report_sha256=authorization.report_sha256,
            report_binding_sha256=authorization.report_binding_sha256,
            policy_sha256=authorization.policy_sha256,
            evaluator_code_sha256=authorization.evaluator_code_sha256,
            scientific_identity_sha256=identity,
            basis_sha256=continuity_assessment_sha256(continuity),
        ),
        layout.serialize(continuity),
    )


__all__ = [
    "CompletionRefused",
    "assess_completion",
    "complete_batch",
    "complete_in_new_session",
    "complete_in_session",
    "verify_completion",
]
