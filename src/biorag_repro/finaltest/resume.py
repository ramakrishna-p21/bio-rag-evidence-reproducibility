"""Authorized resume of one interrupted rehearsal batch, under the no-supersession rule.

A batch has one run, one set of observation keys and one scientific input binding, all fixed
when it started. A resume continues exactly that batch. It never creates a replacement, rebinds
an input, reassigns an observation, regenerates a completed observation or resets an attempt
counter; an interrupted attempt keeps counting toward the retry limit.

Two entry points, as in `start.py`.

`resume_batch(data_root, report_sha256, repo=..., batch_id=...)` is the production route. It
calls the readiness guard and then refuses (`PRODUCTION_RESUME` is False). After genesis the
real G13 gate is unavailable, so the guard refuses first; no real batch can exist to resume.

`_resume_with` is the orchestration, reached by tests with a fixture assessor. It accepts
rehearsal roots only.

`assess_resume` is read-only and grants nothing. It classifies a batch before anything is
written, and never calls `RunStore.open`, which would publish a missing seal:

    resumable                work remains and every record verifies
    completed                batch_completed is published and verifies (read-only classification)
    already_complete         every observation is complete and sealed
    terminally_closed        closed incomplete, or only terminal outcomes are left
    structurally_incomplete  a publication is unfinished (never repaired or adopted)
    contradictory            records disagree with each other or with the expected identity

Order of one resume, inside the session lock and then the registry lock; nothing is written
before step 7:

    1. verify the whole root and assess the batch: it must be `resumable`
    2. if an attempt is left running, the previous worker must be shown to have stopped
    3. run the guard on a fresh report, then check the registry head did not move
    4. read the first report back (continuity input only, never authority) and render the inputs
       again: the binding must equal the stored binding
    5. compare the live host and GPU with the run's recorded environments
    6. compare both reports (`compare_assessments`) and build the continuity assessment and the
       resume citation in memory, then check the whole citation chain
    7. `batch_resumed` event, continuity file, citation file
    8. verify the root and the batch again; only then reopen the store and recover attempts

Hash dependencies are acyclic: reports -> continuity assessment -> resume citation (cites the
assessment, its predecessor and the registry head before the event) -> event (cites both, the
previous authorizing event and the unchanged run record). The citation records the head
*before* its event; the verifier requires that to be exactly the event before it. That is the
historical head of an assessment, not the latest head: the latest head is whatever the verified
registry now holds, and no authority for it exists beyond genesis (G13).

A crash at any publication leaves a batch the verifier reports as incomplete or conflicting,
and it is not resumed again; nothing is overwritten or adopted.
"""

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal, NoReturn

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest import layout, registry
from biorag_repro.finaltest.audit import AuditContext, audit_run
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    BatchManifest,
    CitationChainError,
    batch_id_for,
    citation_sha256,
    verify_citation_chain,
)
from biorag_repro.finaltest.campaign import CampaignPlan, plan_sha256
from biorag_repro.finaltest.continuity import (
    BatchScope,
    ContinuityAssessment,
    compare_assessments,
    continuity_assessment_sha256,
    scientific_identity_sha256,
)
from biorag_repro.finaltest.execution import SessionGate, StartedBatch
from biorag_repro.finaltest.head import HeadAttestation, assess_head
from biorag_repro.finaltest.inputs import (
    BatchInputBinding,
    ConditionRoster,
    binding_sha256,
    build_binding,
)
from biorag_repro.finaltest.inventory import (
    HEAD_NOT_ESTABLISHED_TEXT,
    AuthorityNotEstablished,
    verify_root,
)
from biorag_repro.finaltest.layout import EvidenceCitation, StorageAuthorization
from biorag_repro.finaltest.protected import ProtectedInputError, build_rosters, facts_by_key
from biorag_repro.finaltest.runrecord import FinalTestRunRecord, run_record_sha256
from biorag_repro.finaltest.session import Session, WorkerNotStopped
from biorag_repro.finaltest.start import (
    Assessed,
    StartRefused,
    StartRequest,
    _check_request,
    _checked,
    _production_assessed,
)
from biorag_repro.finaltest.store import hardware_identity, reopen_final_test_store
from biorag_repro.generation.attempts import ObservationState
from biorag_repro.generation.environment import EnvironmentRecord, utc_now
from biorag_repro.readiness.models import ReadinessReport

PRODUCTION_RESUME: Final = False

Classification = Literal[
    "resumable",
    "completed",
    "already_complete",
    "terminally_closed",
    "structurally_incomplete",
    "contradictory",
]

# observations that still need work, by the audit code that reports them
_PENDING: Final = (
    "observation_absent",
    "never_attempted",
    "interrupted_attempt",
    "retryable_attempt",
)
_TERMINAL: Final = ("recorded_terminal_failure", "recorded_retries_exhausted")
# registry head classifications that leave the latest authorizing event inside the registry
_HEAD_OK: Final = ("claim_matches_head", "registry_ahead_of_claim", "genesis_anchored")


class ResumeRefused(Exception):
    """The resume was refused. `code` says at which check; nothing before step 7 is written."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code


class ProductionResumeRefused(ResumeRefused):
    """The production route is closed in this version."""


@dataclass(frozen=True)
class ResumeAssessment:
    classification: Classification
    code: str
    batch_id: str
    detail: str = ""
    run_id: str | None = None
    pending: tuple[str, ...] = ()
    running: tuple[str, ...] = ()
    authorization_event_sha256: str | None = None
    manifest: BatchManifest | None = None
    binding: BatchInputBinding | None = None
    rosters: Mapping[str, ConditionRoster] = field(default_factory=dict)
    citations: tuple[AuthorizationCitation, ...] = ()
    assessments: Mapping[str, ContinuityAssessment] = field(default_factory=dict)
    record: FinalTestRunRecord | None = None
    hardware: tuple[dict[str, Any], ...] = ()  # distinct host/GPU identities of past attempts
    states: tuple[ObservationState, ...] = ()  # every observation the verified run reports

    @property
    def grants_authorization(self) -> Literal[False]:
        return False


def resume_batch(data_root: Path, report_sha256: str, *, repo: Path, batch_id: str) -> NoReturn:
    """Production resume. Always refuses in this version; see the module text."""
    _production_assessed(data_root, report_sha256, repo)  # AuthorizationRefused when not READY
    if not PRODUCTION_RESUME:
        raise ProductionResumeRefused("production_disabled", "final-test resume is not enabled")
    raise ProductionResumeRefused("no_production_request", "no production request builder exists")


# ---- read-only assessment -----------------------------------------------------------------


def _load(path: Path, model: Any) -> Any:
    """The record if the file holds exactly its canonical bytes; otherwise ValueError."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{path.name} is not a regular file")
    data = path.read_bytes()
    record = model.model_validate_json(data, strict=True)
    if layout.serialize(record) != data:
        raise ValueError(f"{path.name} is not in canonical form")
    return record


def _hardware_of(root: Path, run_id: str, states: Any) -> tuple[dict[str, Any], ...] | str:
    """Distinct host/GPU identities of the environments the run's attempts recorded.

    Returns a code string if a recorded environment cannot be read back.
    """
    found: dict[str, dict[str, Any]] = {}
    for state in states:
        for event in state.events:
            if event.state != "running":
                continue
            path = root / "runs" / run_id / "environments" / f"{event.environment_sha256}.json"
            try:
                data = path.read_bytes()
                if sha256_bytes(data) != event.environment_sha256:
                    return "environment_altered"
                identity = hardware_identity(EnvironmentRecord.model_validate_json(data))
            except (OSError, ValueError):
                return "environment_missing"
            found[json.dumps(identity, sort_keys=True)] = identity
    return tuple(found.values())


def assess_resume(
    root: Path,
    storage: StorageAuthorization,
    anchors: Path,
    witnesses: Path,
    *,
    plan: CampaignPlan,
    item_papers: Mapping[str, str],
    batch_id: str,
) -> ResumeAssessment:
    """Classify `batch_id` for resume. Reads only; takes no lock, so the answer can go stale."""

    def verdict_of(
        classification: Classification, code: str, detail: str = "", **known: Any
    ) -> ResumeAssessment:
        return ResumeAssessment(classification, code, batch_id, detail, **known)

    try:
        verdict = verify_root(root, storage, anchors, witnesses)
    except AuthorityNotEstablished as error:
        return verdict_of("structurally_incomplete", "root_unavailable", str(error))
    info = verdict.batches.get(batch_id)
    if info is None:
        return verdict_of("contradictory", "unknown_batch")
    if info.state == "closed_incomplete":
        return verdict_of("terminally_closed", "batch_closed_incomplete")
    left = [p for p in verdict.problems if p != HEAD_NOT_ESTABLISHED_TEXT]
    if left:
        only_incomplete = all(p.partition(":")[0] == "incomplete" for p in left)
        return verdict_of(
            "structurally_incomplete" if only_incomplete else "contradictory",
            "root_unverified",
            "; ".join(left[:2]),
        )
    if info.state not in ("authorized", "completed") or info.run_id is None:
        return verdict_of("structurally_incomplete", f"batch_{info.state}")

    base = root / layout.BATCHES_DIR / batch_id
    try:
        manifest = _load(base / layout.MANIFEST_NAME, BatchManifest)
        binding = _load(base / layout.INPUTS_DIR / layout.BINDING_NAME, BatchInputBinding)
        rosters = {}
        for ref in binding.rosters:
            path = base / layout.INPUTS_DIR / layout.ROSTERS_DIR / f"{ref.roster_sha256}.json"
            rosters[ref.condition] = _load(path, ConditionRoster)
        citations = [
            _load(base / layout.CITATIONS_DIR / layout.citation_name(n), AuthorizationCitation)
            for n in range(1, info.resumes + 2)
        ]
        assessments = {}
        for n in range(2, info.resumes + 2):
            path = base / layout.CITATIONS_DIR / layout.continuity_name(n)
            assessment = _load(path, ContinuityAssessment)
            assessments[continuity_assessment_sha256(assessment)] = assessment
        record = _load(root / layout.RUNS_DIR / info.run_id / "run.json", FinalTestRunRecord)
        verify_citation_chain(citations, assessments)
    except CitationChainError as error:
        return verdict_of("contradictory", "citation_chain_invalid", str(error)[:200])
    except (OSError, ValueError) as error:
        return verdict_of("contradictory", "record_unreadable", str(error)[:200])
    known: dict[str, Any] = {
        "run_id": info.run_id,
        "authorization_event_sha256": info.authorization_event_sha256,
        "manifest": manifest,
        "binding": binding,
        "rosters": rosters,
        "citations": tuple(citations),
        "assessments": assessments,
        "record": record,
    }

    # the latest authorizing event must lie in the verified registry (rollback where supported)
    assert info.authorization_event_sequence is not None and info.authorization_event_sha256
    claim = HeadAttestation(
        root_id=verdict.root_id,
        sequence=info.authorization_event_sequence,
        event_file_sha256=info.authorization_event_sha256,
        campaign_plan_sha256=info.campaign_plan_sha256,
        batch_id=batch_id,
        provenance="unverified_claim",
    )
    head = assess_head(root, storage, anchors, witnesses, claim)
    if head.classification not in _HEAD_OK:
        return verdict_of("contradictory", f"registry_head_{head.classification}", **known)

    audit = audit_run(
        root,
        info.run_id,
        AuditContext(
            root_id=verdict.root_id,
            root_kind=storage.root_kind,
            plan=plan,
            item_papers=item_papers,
            manifest=manifest,
            binding=binding,
            rosters=rosters,
            citations=tuple(citations),
            assessments=assessments,
        ),
    )
    structure = [f.code for f in audit.of_kind("structure")]
    claims = [f.code for f in audit.of_kind("claim")]
    incomplete = [f for f in audit.findings if f.kind == "incomplete"]
    if structure and set(structure) <= {"seal_missing", "run_missing"}:
        return verdict_of("structurally_incomplete", structure[0], **known)
    if structure or claims:
        return verdict_of("contradictory", (structure + claims)[0], **known)
    assert audit.observed is not None
    hardware = _hardware_of(root, info.run_id, audit.observed.states)
    if isinstance(hardware, str):
        return verdict_of("contradictory", hardware, **known)
    if len(hardware) > 1:
        return verdict_of("contradictory", "host_conflict", "attempts ran on different hosts")
    known["hardware"] = hardware
    known["states"] = tuple(audit.observed.states)
    pending = tuple(sorted(f.subject for f in incomplete if f.code in _PENDING))
    if info.state == "completed":
        # a completed batch must still show only decided observations: nothing may run after it
        if pending:
            return verdict_of("contradictory", "open_observation_after_completion", **known)
        return verdict_of("completed", "batch_completed", **known)
    running = tuple(sorted(f.subject for f in incomplete if f.code == "interrupted_attempt"))
    known["pending"], known["running"] = pending, running
    if pending:
        return verdict_of("resumable", "work_remains", **known)
    if any(f.code in _TERMINAL for f in incomplete):
        return verdict_of("terminally_closed", "no_eligible_observation", **known)
    return verdict_of("already_complete", "all_observations_complete", **known)


# ---- orchestration ------------------------------------------------------------------------


def _published_files(root: Path, batch: str, run_id: str, head: int) -> dict[str, str]:
    """Root-relative path -> SHA-256 of everything a session gate watches."""
    base = f"{layout.BATCHES_DIR}/{batch}"
    paths = [f"{layout.REGISTRY_DIR}/{n:06d}.json" for n in range(1, head + 1)]
    paths += [f"{base}/{layout.MANIFEST_NAME}", f"{base}/{layout.INPUTS_DIR}/{layout.BINDING_NAME}"]
    paths += [f"{layout.RUNS_DIR}/{run_id}/run.json"]
    for folder in (
        f"{base}/{layout.INPUTS_DIR}/{layout.ROSTERS_DIR}",
        f"{base}/{layout.CITATIONS_DIR}",
    ):
        paths += [f"{folder}/{p.name}" for p in sorted((root / folder).iterdir())]
    return {path: sha256_bytes((root / path).read_bytes()) for path in paths}


def _resume_with(
    request: StartRequest,
    assess: Callable[[], Assessed],
    session: Session,
    *,
    load_report: Callable[[str], ReadinessReport],
    capture_environment: Callable[[], EnvironmentRecord],
    allowed_root_kinds: tuple[str, ...] = ("rehearsal",),
    now: Callable[[], str] = utc_now,
) -> StartedBatch:
    """Resume the batch of `request.matrix_arm`. Everything about the batch is read from disk.

    `load_report` reads a stored report back for the continuity comparison only.
    `capture_environment` describes the host and GPU this session runs on.
    """
    storage = request.authorization
    root, arm, plan = request.root, request.matrix_arm, request.plan
    try:
        resolved, retry = _check_request(request, session, allowed_root_kinds)
    except StartRefused as error:
        raise ResumeRefused(error.code, str(error)) from error

    with session.registry() as held:
        root_id = layout.root_id_for(storage)
        batch = batch_id_for(root_id, plan_sha256(plan), arm)
        # step 1
        item_papers = dict(request.population.item_papers)
        found = assess_resume(
            root, storage, request.anchors, request.witnesses,
            plan=plan, item_papers=item_papers, batch_id=batch,
        )  # fmt: skip
        if found.classification != "resumable":
            raise ResumeRefused(found.classification, f"{found.code} {found.detail}".strip())
        assert found.record is not None and found.manifest is not None
        assert found.binding is not None and found.run_id is not None
        assert found.authorization_event_sha256 is not None
        record, manifest, first = found.record, found.manifest, found.citations[0]
        # step 2: the lock shows no cooperating process holds the session; an attempt left
        # running needs more than that before it is closed on the previous worker's behalf
        stopped = None
        if found.running:
            try:
                stopped = session.prove_previous_worker_stopped()
            except WorkerNotStopped as error:
                raise ResumeRefused("worker_not_stopped", str(error)) from error
        head_before = registry.registry_head(root)
        # step 3
        try:
            authorization, report = _checked(assess())
        except StartRefused as error:
            raise ResumeRefused(error.code, str(error)) from error
        if registry.registry_head(root) != head_before:
            raise ResumeRefused("head_changed", "the registry moved while the guard ran")
        # step 4: the first report is read back as a record, never as permission
        try:
            initial = load_report(first.initial_report_sha256)
        except Exception as error:  # any failure to read it back means no continuity evidence
            raise ResumeRefused("initial_report_unavailable", str(error)[:200]) from error
        if any(c.readiness_report_sha256 == authorization.report_sha256 for c in found.citations):
            raise ResumeRefused("report_reused", "a report authorizes one session only")
        try:
            prepared = build_rosters(
                inputs=request.inputs,
                population=request.population,
                matrix_arm=arm,
                batch_id=batch,
                renderer=request.renderer,
                seeds=request.seeds,
            )
            binding = build_binding(
                manifest=manifest,
                plan=plan,
                resolved=resolved,
                retry=retry,
                seeds=request.seeds,
                rosters=prepared.rosters,
            )
        except ValueError as error:
            code = "inputs" if isinstance(error, ProtectedInputError) else "inputs_invalid"
            raise ResumeRefused(code, str(error)[:200]) from error
        # the stored binding was verified against plan, manifest, configuration and seeds when
        # it was bound; the same bytes rebuilt from the inputs carry all of that over
        if binding != found.binding:
            raise ResumeRefused(
                "binding_drift", "the inputs are not the ones the batch is bound to"
            )
        binding_sha = binding_sha256(binding)
        # step 5
        hardware = hardware_identity(capture_environment())
        if found.hardware and found.hardware[0] != hardware:
            raise ResumeRefused("host_changed", "host or GPU differs from the recorded attempts")
        # step 6
        scope = BatchScope(
            campaign_plan_sha256=manifest.campaign_plan_sha256,
            batch_id=batch,
            matrix_arm=arm,
            resolved_config_sha256=manifest.resolved_config_sha256,
            ownership_sha256=manifest.ownership,
            key_set_sha256=manifest.key_set_sha256,
            inputs_set_sha256=binding_sha,
        )
        if initial.binding_sha256 != first.report_binding_sha256:
            raise ResumeRefused("initial_report_mismatch", "not the report the batch started on")
        continuity = compare_assessments(initial, report, scope, scope)
        if continuity.verdict != "continuous":
            blocking = [f.component for f in continuity.findings if f.verdict == "blocking"]
            raise ResumeRefused("not_continuous", ", ".join(blocking))
        assessment_bytes = layout.serialize(continuity)
        assessment_digest = continuity_assessment_sha256(continuity)
        source, head_sha = (
            report.candidate.scientific_source_identity_sha256,
            report.candidate.head_sha,
        )
        assert source is not None and head_sha is not None
        sequence, head_file = head_before
        citation = AuthorizationCitation(
            kind="resume",
            batch_id=batch,
            initial_report_sha256=first.initial_report_sha256,
            readiness_report_sha256=authorization.report_sha256,
            policy_sha256=authorization.policy_sha256,
            evaluator_code_sha256=authorization.evaluator_code_sha256,
            report_binding_sha256=authorization.report_binding_sha256,
            head_sha=head_sha,
            scientific_source_identity_sha256=source,
            scientific_identity_sha256=scientific_identity_sha256(report, scope),
            registry_head_sequence=sequence,
            registry_head_event_sha256=head_file,
            head_authority="not_established",  # a resume is always beyond genesis
            inputs_set_sha256=binding_sha,
            previous_citation_sha256=citation_sha256(found.citations[-1]),
            continuity_assessment_sha256=assessment_digest,
            created_utc=now(),
        )
        try:
            verify_citation_chain(
                [*found.citations, citation], {**found.assessments, assessment_digest: continuity}
            )
        except CitationChainError as error:
            raise ResumeRefused("citation_chain_invalid", str(error)[:200]) from error
        citation_bytes, citation_digest = layout.serialize(citation), citation_sha256(citation)
        ordinal = len(found.citations) + 1

        # step 7: the first writes
        args = (held, root, storage)
        registry.append_event_locked(
            *args,
            request.witnesses,
            "batch_resumed",
            [
                EvidenceCitation(role="batch_id", sha256=batch),
                EvidenceCitation(
                    role="previous_authorization_event", sha256=found.authorization_event_sha256
                ),
                EvidenceCitation(role="continuity_assessment", sha256=assessment_digest),
                EvidenceCitation(role="authorization_citation", sha256=citation_digest),
                EvidenceCitation(role="readiness_report", sha256=authorization.report_sha256),
                EvidenceCitation(role="run_record", sha256=run_record_sha256(record)),
            ],
            now=now,
        )
        base = f"{layout.BATCHES_DIR}/{batch}/{layout.CITATIONS_DIR}"
        registry.publish_batch_file(
            *args, f"{base}/{layout.continuity_name(ordinal)}", assessment_bytes
        )
        registry.publish_batch_file(
            *args, f"{base}/{layout.citation_name(ordinal)}", citation_bytes
        )

        # step 8: nothing runs on records that were only written; they must verify as a chain
        after = assess_resume(
            root, storage, request.anchors, request.witnesses,
            plan=plan, item_papers=item_papers, batch_id=batch,
        )  # fmt: skip
        if after.classification != "resumable":
            raise ResumeRefused("publication_unverified", f"{after.classification} {after.code}")
        head = registry.registry_head(root)
        session.head = head
        gate = SessionGate(
            session,
            _published_files(root, batch, found.run_id, head[0]),
            head,
            facts_by_key(prepared),
        )
        store = reopen_final_test_store(
            root, record, resolved, gate, found.hardware[0] if found.hardware else hardware, now=now
        )
        if stopped is not None:
            store.recover_after_session(stopped)
    return StartedBatch(
        session=session,
        store=store,
        batch_id=batch,
        run_id=found.run_id,
        binding_sha256=binding_sha,
        citation_sha256=citation_digest,
        run_record_sha256=run_record_sha256(record),
        report_sha256=authorization.report_sha256,
        head=head,
        rows=facts_by_key(prepared),
        entries=prepared.entries,
        seeds=request.seeds,
        root=root,
    )
