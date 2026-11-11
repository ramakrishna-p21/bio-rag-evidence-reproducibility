"""Completed-run manifest: one record that closes a run over its authorized roster.

A run is complete when every observation of the authorized roster has reached a final
status (a completed answer, a terminal failure, or exhausted retries) and nothing else is
stored in the run. Completion publishes, once and without replacing anything, a roster
record, a completion record and a build manifest that cites the run record, the
configuration, the roster and every final event.

Read-back needs the manifest hash that publication returned. With that hash retained
somewhere the run directory's writer cannot reach, a deleted or replaced final event,
seal, raw output, roster or completion record is found. Without it, whoever can rewrite
the run directory can also write a new, self-consistent manifest: this is an integrity
check, not tamper-proof storage.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from pydantic import BaseModel, StringConstraints

from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.records import Record
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
    verify_manifest_outputs,
)
from biorag_repro.generation.attempts import (
    Coordinates,
    CorruptRecord,
    LifecycleError,
    ObservationState,
    RunStore,
    check_no_duplicate_completions,
)
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import BackendKind
from biorag_repro.generation.roster import RosterRecord, roster_sha256
from biorag_repro.generation.runner import RosterMismatch

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
FinalStatus = Literal["complete", "terminal_failure", "retries_exhausted"]

ROSTER = "roster.json"
COMPLETION = "completion.json"
MANIFEST = "completion.build.json"
_FINAL: tuple[str, ...] = ("complete", "terminal_failure", "retries_exhausted")


class RunIncomplete(LifecycleError):
    """Not every authorized observation has reached a final status."""


class FinalObservation(Record):
    observation_key: str
    item_id: str
    condition: str
    repeat_index: int
    status: FinalStatus
    attempts: int
    final_event_sequence: int
    final_event_sha256: Sha256
    raw_output: ArtifactRef | None


class RunCompletion(Record):
    schema_version: Literal["1"] = "1"
    run_id: str
    run_record_sha256: Sha256
    resolved_config_sha256: Sha256
    raw_config_sha256: Sha256
    backend_kind: BackendKind
    purpose: str
    matrix_arm: str
    roster_sha256: Sha256
    environments: list[Sha256]  # every environment record an attempt was started under
    observations: list[FinalObservation]
    files: dict[str, Sha256]  # every other file of the run directory, by relative path


@dataclass(frozen=True)
class CompletedRun:
    """A run whose stored records agree with a retained completion anchor."""

    run_id: str
    manifest_sha256: str
    completion: RunCompletion
    roster: RosterRecord
    states: list[ObservationState]
    store: RunStore


def _coordinates(row_item: str, paper: str, condition: str, repeat: int, arm: str) -> Coordinates:
    return Coordinates(
        item_id=row_item,
        paper_id=paper,
        matrix_arm=arm,
        representation_or_condition=condition,
        repeat_index=repeat,
    )


def _roster_bytes(record: RosterRecord) -> bytes:
    return serialize_project_json(record, RosterRecord)


def publish_roster(store: RunStore, record: RosterRecord) -> str:
    """Store the authorized roster with the run. The same roster may be stored again."""
    content = _roster_bytes(record)
    digest = sha256_bytes(content)
    path = store.run_dir / ROSTER
    if path.exists():
        if path.read_bytes() != content:
            raise CorruptRecord("a different roster is already stored for this run")
        return digest
    publish_bytes(path, content, digest)
    return digest


def _inventory(store: RunStore) -> dict[str, str]:
    files = {}
    for path in sorted(store.run_dir.rglob("*")):
        if path.name.startswith("."):
            raise RunIncomplete(f"temporary file {path.name} is present; a writer may be active")
        relative = path.relative_to(store.run_dir).as_posix()
        if path.is_file() and relative not in (ROSTER, COMPLETION, MANIFEST):
            files[relative] = sha256_bytes(path.read_bytes())
    return files


def _describe(store: RunStore, roster: RosterRecord) -> RunCompletion:
    """What the run directory must contain for the roster to be fully decided."""
    record = store.record
    if (roster.resolved_config_sha256, roster.matrix_arm) != (
        record.resolved_config_sha256,
        record.matrix_arm,
    ) or (roster.phase, roster.benchmark_version) != (record.phase, record.benchmark_version):
        raise RosterMismatch("the roster belongs to another configuration, arm or benchmark")

    rows = {
        store.key_for(
            _coordinates(r.item_id, r.paper_id, r.condition, r.repeat_index, record.matrix_arm)
        ): r
        for r in roster.rows
    }
    states = {s.key: s for s in store.verify()}
    if stray := set(states) - set(rows):
        raise RosterMismatch(f"{len(stray)} stored observations are not in the roster")
    if unplanned := set(rows) - set(states):
        raise RunIncomplete(f"{len(unplanned)} authorized observations were never planned")

    for key, row in rows.items():
        state = states[key]
        if (
            row.input_record is not None
            and not (store.run_dir / row.input_record.logical_ref).is_file()
        ):
            raise RosterMismatch(f"the stored input cited for {key} is not in the run")
        c = state.coordinates
        if (c.item_id, c.paper_id, c.representation_or_condition, c.repeat_index) != (
            row.item_id,
            row.paper_id,
            row.condition,
            row.repeat_index,
        ):
            raise RosterMismatch(f"stored coordinates of {key} differ from the roster")
        for event in state.events:
            if (
                event.state == "running"
                and event.rendered_input_sha256 != row.rendered_input_sha256
            ):
                raise RosterMismatch(f"the rendered input of {key} is not the rostered input")

    open_counts: dict[str, int] = {}
    for state in states.values():
        if state.status not in _FINAL:
            open_counts[state.status] = open_counts.get(state.status, 0) + 1
    if open_counts:
        detail = ", ".join(f"{name}={n}" for name, n in sorted(open_counts.items()))
        raise RunIncomplete(f"{sum(open_counts.values())} of {len(states)} undecided: {detail}")

    observations = []
    environments: set[str] = set()
    for key in sorted(states):
        state, row = states[key], rows[key]
        if not store._seal_path(key).exists():
            raise CorruptRecord(f"seal of {key} is missing")
        final = state.events[-1]
        environments |= {e.environment_sha256 for e in state.events if e.environment_sha256}
        observations.append(
            FinalObservation(
                observation_key=key,
                item_id=row.item_id,
                condition=row.condition,
                repeat_index=row.repeat_index,
                status=state.status,  # type: ignore[arg-type]  # narrowed by the final-status check
                attempts=state.attempts,
                final_event_sequence=final.sequence,
                final_event_sha256=state.last_event_sha256,
                raw_output=final.raw_output,
            )
        )
    files = _inventory(store)
    files[ROSTER] = sha256_bytes(_roster_bytes(roster))
    return RunCompletion(
        run_id=store.run_id,
        run_record_sha256=store.run_sha256,
        resolved_config_sha256=record.resolved_config_sha256,
        raw_config_sha256=record.raw_config_sha256,
        backend_kind=record.backend_kind,
        purpose=record.purpose,
        matrix_arm=record.matrix_arm,
        roster_sha256=roster_sha256(roster),
        environments=sorted(environments),
        observations=observations,
        files=dict(sorted(files.items())),
    )


def _manifest(completion: RunCompletion) -> BuildManifest:
    base = f"runs/{completion.run_id}"
    inputs = [
        BuildInput(
            role="run_record",
            artifact=ArtifactRef(
                artifact_id=f"run:{completion.run_id}",
                logical_ref=f"{base}/run.json",
                sha256=completion.run_record_sha256,
            ),
        ),
        BuildInput(
            role="configuration",
            artifact=ArtifactRef(
                artifact_id=f"config:{completion.resolved_config_sha256}",
                logical_ref=f"{base}/config.canonical.json",
                sha256=completion.resolved_config_sha256,
            ),
        ),
        BuildInput(
            role="roster",
            artifact=ArtifactRef(
                artifact_id=f"roster:{completion.roster_sha256}",
                logical_ref=f"{base}/{ROSTER}",
                sha256=completion.roster_sha256,
            ),
        ),
    ]
    for obs in completion.observations:
        name = f"{obs.final_event_sequence:06d}.json"
        inputs.append(
            BuildInput(
                role="final_event",
                artifact=ArtifactRef(
                    artifact_id=f"{obs.observation_key}:{obs.final_event_sequence:06d}",
                    logical_ref=f"{base}/observations/{obs.observation_key}/events/{name}",
                    sha256=obs.final_event_sha256,
                ),
            )
        )
    content_sha = sha256_bytes(serialize_project_json(completion, RunCompletion))
    lineage = [ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in inputs]
    settings = completion.resolved_config_sha256
    return BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id="generation_run_completion_v1",
        transformation_settings_sha256=settings,
        outputs=[
            DerivedOutput(
                artifact=ArtifactRef(
                    artifact_id=derived_artifact_id(
                        artifact_kind="generation_run_completion",
                        inputs=lineage,
                        transformation_id="generation_run_completion_v1",
                        transformation_settings_sha256=settings,
                        output_sha256=content_sha,
                        output_format="json",
                        serialization_version="1",
                    ),
                    logical_ref=f"{base}/{COMPLETION}",
                    sha256=content_sha,
                ),
                artifact_kind="generation_run_completion",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )


def complete_run(store: RunStore, roster: RosterRecord) -> str:
    """Publish the completion of a fully decided run and return its manifest hash.

    The returned hash is the anchor: keep it where the run directory's writer cannot
    change it. Calling this again for an unchanged run returns the same hash.
    """
    completion = _describe(store, roster)
    check_no_duplicate_completions(store.root)
    manifest = _manifest(completion)
    publish_roster(store, roster)
    content = serialize_project_json(completion, RunCompletion)
    digest = sha256_bytes(content)
    path = store.run_dir / COMPLETION
    if path.exists():
        if path.read_bytes() != content:
            raise CorruptRecord("a different completion record is already stored for this run")
    else:
        publish_bytes(path, content, digest)
    manifest_path = store.run_dir / MANIFEST
    try:
        return publish_manifest(
            manifest, manifest_path, {manifest.outputs[0].artifact.logical_ref: path}
        )
    except FileExistsError:
        stored = sha256_bytes(manifest_path.read_bytes())
        if read_manifest(manifest_path, stored) != manifest:
            raise CorruptRecord("a different completion manifest is already stored") from None
        return stored


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read(path: Path, digest: str, model: type[ModelT], what: str) -> ModelT:
    try:
        return read_project_json(path, digest, model)
    except (OSError, ValueError) as error:
        raise CorruptRecord(f"{what} is missing, altered or unreadable: {error}") from error


def verify_completed_run(
    root: Path,
    run_id: str,
    manifest_sha256: str,
    *,
    config: ResolvedConfiguration[GeneratorConfiguration],
    backend_kind: BackendKind,
) -> CompletedRun:
    """Re-derive the completion from the stored records and require it to equal the anchor's."""
    run_dir = root / "runs" / run_id
    manifest = _read(run_dir / MANIFEST, manifest_sha256, BuildManifest, "completion manifest")
    if len(manifest.outputs) != 1:
        raise CorruptRecord("completion manifest must name exactly one output")
    output = manifest.outputs[0].artifact
    completion = _read(run_dir / COMPLETION, output.sha256, RunCompletion, "completion record")
    try:
        verify_manifest_outputs(manifest, {output.logical_ref: run_dir / COMPLETION})
    except (OSError, ValueError) as error:
        raise CorruptRecord(f"completion output does not match its manifest: {error}") from error
    if manifest != _manifest(completion) or completion.run_id != run_id:
        raise CorruptRecord("completion manifest does not describe the completion record")
    roster = _read(run_dir / ROSTER, completion.roster_sha256, RosterRecord, "roster")

    # Seals are checked before the store is opened, because opening republishes a missing one.
    for obs in completion.observations:
        if not (run_dir / "observations" / obs.observation_key / "final.build.json").is_file():
            raise CorruptRecord(f"seal of {obs.observation_key} is missing")
    store = RunStore.open(root, run_id, config=config, backend_kind=backend_kind)
    try:
        rebuilt = _describe(store, roster)
    except RunIncomplete as error:
        raise CorruptRecord(f"a run recorded as complete is no longer complete: {error}") from error
    if rebuilt != completion:
        changed = sorted(
            name
            for name in set(rebuilt.files) | set(completion.files)
            if rebuilt.files.get(name) != completion.files.get(name)
        )
        reason = f"file inventory differs at {changed[:3]}" if changed else "records differ"
        raise CorruptRecord(f"stored run does not match its completion: {reason}")
    return CompletedRun(run_id, manifest_sha256, completion, roster, store.verify(), store)
