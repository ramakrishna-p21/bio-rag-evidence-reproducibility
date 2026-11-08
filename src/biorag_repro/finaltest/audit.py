"""Read-only audit of a stored final-test run against its expected scientific inputs.

The audit never executes, repairs, recovers or publishes anything. It opens files without
following symbolic links, checks that stored records are the canonical bytes of their models,
and delegates the hash chain, state transitions, completions, receipts and seal comparison to
`RunStore._load` through a view whose every mutating entry point raises. Nothing in `RunStore`
is changed: `RunStore.open` seals missing final events and is not used here. A missing seal is
reported, never created.

Three questions are kept apart, and the answer to the last one is fixed:

* structure: do the stored records verify, and do they match the batch, plan and rosters;
* claims: are the run's references (inputs binding, authorization citation) consistent with
  the supplied citations. A consistent citation is a historical claim, not permission;
* authorization: `not_established` always. No authority for the registry head beyond genesis
  exists (G13), so no verdict can say a run was authorized, and event timestamps are not an
  independent clock, so ordering is `unverified`.
"""

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.finaltest.batches import (
    AuthorizationCitation,
    BatchManifest,
    CitationChainError,
    citation_sha256,
    verify_citation_chain,
)
from biorag_repro.finaltest.campaign import CampaignPlan, expected_keys, plan_sha256
from biorag_repro.finaltest.completeness import ObservedBatch
from biorag_repro.finaltest.continuity import ContinuityAssessment
from biorag_repro.finaltest.inputs import (
    BatchInputBinding,
    ConditionRoster,
    Finding,
    RosterIdentityRow,
    binding_sha256,
    retry_policy_sha256,
)
from biorag_repro.finaltest.layout import serialize
from biorag_repro.finaltest.runrecord import (
    FinalTestRunRecord,
    run_record_projection,
    run_record_sha256,
)
from biorag_repro.generation.attempts import (
    AttemptEvent,
    CorruptRecord,
    DuplicateCompletion,
    LifecycleError,
    NotPlanned,
    ObservationState,
    check_no_duplicate_completions,
    retry_policy_of,
)
from biorag_repro.generation.config import resolve_generator_config
from biorag_repro.generation.readonly import ReadOnlyRunView, read_only_view

_FINAL = ("complete", "terminal_failure", "retries_exhausted")
# observations that are not verified completions, by the status the log reports
_OPEN_CODE = {
    "planned": "never_attempted",
    "running": "interrupted_attempt",
    "retryable": "retryable_attempt",
    "terminal_failure": "recorded_terminal_failure",
    "retries_exhausted": "recorded_retries_exhausted",
}


@dataclass(frozen=True)
class AuditContext:
    """What the run is expected to be. Records are supplied by the caller; none is loaded here."""

    root_id: str
    root_kind: str
    plan: CampaignPlan
    item_papers: Mapping[str, str]
    manifest: BatchManifest | None
    binding: BatchInputBinding | None
    rosters: Mapping[str, ConditionRoster]
    citations: Sequence[AuthorizationCitation] = ()
    assessments: Mapping[str, ContinuityAssessment] = field(default_factory=dict)


@dataclass(frozen=True)
class RunAudit:
    run_id: str
    findings: tuple[Finding, ...]
    observed: ObservedBatch | None
    authorization: Literal["not_established"] = "not_established"
    ordering: Literal["unverified"] = "unverified"

    def of_kind(self, kind: str) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.kind == kind)

    @property
    def structure_valid(self) -> bool:
        return not self.of_kind("structure")

    @property
    def claims_consistent(self) -> bool:
        return not self.of_kind("claim")

    @property
    def complete(self) -> bool:
        return not self.of_kind("incomplete")


def _read(path: Path) -> bytes:
    """Read a regular file without following a final symbolic link."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        return handle.read()


def _scan(run_dir: Path) -> list[Finding]:
    """Entries that are not plain files and directories, or leftover temporary files."""
    out: list[Finding] = []
    pending = [run_dir]
    while pending:
        with os.scandir(pending.pop()) as entries:
            for entry in sorted(entries, key=lambda e: e.name):
                name = os.path.relpath(entry.path, run_dir)
                if entry.is_symlink():
                    out.append(Finding("structure", "symlink_in_run", name))
                elif entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif not entry.is_file(follow_symlinks=False):
                    out.append(Finding("structure", "special_file_in_run", name))
                elif entry.name.startswith("."):
                    out.append(Finding("structure", "temporary_file_in_run", name))
    return out


def _canonical(path: Path, model: Any) -> tuple[Any, bytes] | None:
    """Parse strictly and require the stored bytes to equal the canonical serialization."""
    data = _read(path)
    value = model.model_validate(json.loads(data.decode("utf-8")), strict=True)
    return (value, data) if serialize(value) == data else None


def _state_findings(view: ReadOnlyRunView, key: str, add: Any) -> ObservationState | None:
    try:
        state = view.observation(key)
    except DuplicateCompletion:
        add("structure", "duplicate_completion", key)
    except NotPlanned:
        add("structure", "observation_not_planned", key)
    except CorruptRecord as error:
        add("structure", "observation_corrupt", key, str(error)[:200])
    except (LifecycleError, OSError, ValueError) as error:
        add("structure", "observation_unreadable", key, str(error)[:200])
    else:
        return state
    return None


def open_view(
    data_root: Path, record: FinalTestRunRecord, run_sha: str, config: Any
) -> ReadOnlyRunView:
    """The read-only store view of a final-test run.

    The projection validates every field as a development record and only then replaces the
    phase (`runrecord.run_record_projection`); it is never serialized or stored.
    """
    return read_only_view(data_root, run_record_projection(record), run_sha, config)


def audit_run(data_root: Path, run_id: str, context: AuditContext) -> RunAudit:
    """Audit `runs/<run_id>` of `data_root`. Writes nothing."""
    out: list[Finding] = []

    def add(kind: Any, code: str, subject: str, detail: str = "") -> None:
        out.append(Finding(kind, code, subject, detail))

    def done(observed: ObservedBatch | None = None) -> RunAudit:
        return RunAudit(run_id, tuple(out), observed)

    runs, run_dir = data_root / "runs", data_root / "runs" / run_id
    for directory in (runs, run_dir):
        if directory.is_symlink() or not directory.is_dir():
            add("structure", "run_missing", run_id)
            return done()
    out.extend(_scan(run_dir))
    if any(f.code in ("symlink_in_run", "special_file_in_run") for f in out):
        return done()

    try:
        loaded = _canonical(run_dir / "run.json", FinalTestRunRecord)
    except (OSError, ValueError) as error:
        add("structure", "run_record_invalid", run_id, str(error)[:200])
        return done()
    if loaded is None:
        add("structure", "run_record_not_canonical", run_id)
        return done()
    record: FinalTestRunRecord = loaded[0]
    if record.run_id != run_id:
        add("structure", "run_record_invalid", run_id, "run id differs from the directory")
        return done()
    run_sha = run_record_sha256(record)

    manifest, binding = context.manifest, context.binding
    if manifest is None:
        add("incomplete", "manifest_missing", record.batch_id)
    elif (
        record.batch_id,
        record.root_id,
        record.campaign_plan_sha256,
        record.matrix_arm,
        record.resolved_config_sha256,
        record.benchmark_version,
        record.corpus_content_sha256,
        record.benchmark_release_sha256,
    ) != (
        manifest.batch_id,
        manifest.root_id,
        manifest.campaign_plan_sha256,
        manifest.matrix_arm,
        manifest.resolved_config_sha256,
        manifest.benchmark_version,
        manifest.corpus_content_sha256,
        manifest.benchmark_release_sha256,
    ):
        add("structure", "foreign_batch", record.batch_id)
    if (record.root_id, record.root_kind) != (context.root_id, context.root_kind):
        add("structure", "foreign_root", record.root_id)
    if record.campaign_plan_sha256 != plan_sha256(context.plan):
        add("structure", "foreign_plan", record.campaign_plan_sha256)

    try:
        config_canonical = _read(run_dir / "config.canonical.json")
        config_raw = _read(run_dir / "config.toml")
        config = resolve_generator_config(config_raw)
    except (OSError, ValueError) as error:
        add("structure", "config_unreadable", run_id, str(error)[:200])
        return done()
    if (
        sha256_bytes(config_canonical) != record.resolved_config_sha256
        or sha256_bytes(config_raw) != record.raw_config_sha256
        or config.configuration_sha256 != record.resolved_config_sha256
        or (config.settings.purpose, config.settings.matrix_arm)
        != (record.purpose, record.matrix_arm)
        or record.retry != retry_policy_of(config.settings)
        or record.retry_policy_sha256 != retry_policy_sha256(record.retry)
    ):
        add("structure", "config_mismatch", run_id)
        return done()

    view = open_view(data_root, record, run_sha, config)

    rows: dict[str, RosterIdentityRow] = {
        row.observation_key: row for roster in context.rosters.values() for row in roster.rows
    }
    expected = set(expected_keys(context.plan, sorted(context.item_papers), record.matrix_arm))
    states: list[ObservationState] = []
    for key in view.observation_keys():
        events_dir = run_dir / "observations" / key / "events"
        try:
            for path in sorted(events_dir.glob("0*.json")):
                if _canonical(path, AttemptEvent) is None:
                    add("structure", "event_not_canonical", key, path.name)
        except (OSError, ValueError) as error:
            add("structure", "observation_unreadable", key, str(error)[:200])
            continue
        state = _state_findings(view, key, add)
        if state is None:
            continue
        states.append(state)
        if key not in expected:
            add("structure", "foreign_observation_key", key)
        coordinates = state.coordinates
        if coordinates.repeat_index == 0:
            add("structure", "repeat_zero", key)
        if state.status in _FINAL and not os.path.lexists(view._seal_path(key)):
            add("structure", "seal_missing", key)
        if code := _OPEN_CODE.get(state.status):
            add("incomplete", code, key)
        row = rows.get(key)
        if rows and row is None:
            add("structure", "unbound_observation", key)
        elif row is not None:
            planned = (
                coordinates.item_id,
                coordinates.paper_id,
                coordinates.matrix_arm,
                coordinates.representation_or_condition,
                coordinates.repeat_index,
                coordinates.seed,
            )
            if planned != (
                row.item_id,
                row.paper_id,
                row.matrix_arm,
                row.condition,
                row.repeat_index,
                row.seed,
            ):
                add("structure", "roster_coordinates_mismatch", key)
            started = next((e for e in state.events if e.state == "running"), None)
            if started is not None and started.rendered_input_sha256 != row.rendered_input_sha256:
                add("structure", "rendered_input_mismatch", key)
    for key in sorted(expected - {s.key for s in states} - set(view.observation_keys())):
        add("incomplete", "observation_absent", key)
    if any(e.state == "running" for s in states for e in s.events):
        add("note", "ordering_unverified", run_id, "event timestamps are not an independent clock")

    try:
        check_no_duplicate_completions(data_root)
    except DuplicateCompletion as error:
        add("structure", "duplicate_completion_across_runs", run_id, str(error)[:200])
    except (OSError, ValueError):
        add("structure", "completion_scan_unreadable", run_id)

    # claims: what the run says about its inputs and its authorization
    if not context.rosters:
        add("incomplete", "rosters_missing", record.batch_id)
    if binding is None:
        add("claim", "binding_missing", record.batch_id)
    elif binding_sha256(binding) != record.inputs_set_sha256:
        add("claim", "inputs_binding_mismatch", record.batch_id)
    citations = list(context.citations)
    cited = next(
        (c for c in citations if citation_sha256(c) == record.authorization_citation_sha256), None
    )
    if not citations:
        add("claim", "citation_missing", record.batch_id)
    else:
        try:
            verify_citation_chain(citations, context.assessments)
        except CitationChainError as error:
            add("claim", "citation_chain_invalid", record.batch_id, str(error)[:200])
        if cited is None:
            add("claim", "citation_reference_unknown", record.authorization_citation_sha256)
        else:
            if cited.batch_id != record.batch_id:
                add("claim", "citation_other_batch", cited.batch_id)
            if cited.inputs_set_sha256 != record.inputs_set_sha256:
                add("claim", "citation_inputs_mismatch", record.batch_id)
    add("note", "authority_not_established", record.batch_id, "no independent head authority")

    observed = ObservedBatch(
        phase="final_test",
        benchmark_version=record.benchmark_version,
        matrix_arm=record.matrix_arm,
        resolved_config_sha256=record.resolved_config_sha256,
        states=tuple(states),
    )
    return done(observed)
