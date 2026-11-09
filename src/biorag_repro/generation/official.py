"""The only path from a generation run to official development evidence.

A result qualifies only if it comes from a completed run whose records agree with a
retained completion anchor and whose configuration names the approved tokenizer, and only
if every attempt's environment shows the approved checkpoint on one explicit CUDA device in
the configured precision. Every roster row must cite a stored exact input that re-tokenizes
to its recorded ids, and every completed observation must carry an execution receipt for
that input. The roster must cite the verified development-only benchmark view. A bare
`GenerationResult`, a backend label or a mock wrapper proves nothing and is refused by
`check_official_lineage`.

What this does not prove: which bytes the process executed. The environment holds a hash
inventory of the model directory taken before and after loading, and the backend's receipt
reports what it passed to the model; both come from the executing process. Together with
the anchor they make substitution, truncation and relabelling detectable, but they are not
a hardware attestation. `APPROVED_CHECKPOINTS` is empty until Rama approves a manifest.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.evaluation.analysis_models import GeneratorInput
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.evaluation.stage_c import generation_digest
from biorag_repro.evaluation.stage_c_models import GenerationResult
from biorag_repro.generation import config as generator_config
from biorag_repro.generation import hf_backend
from biorag_repro.generation.attempts import (
    Coordinates,
    LifecycleError,
    NonOfficialBackend,
    ObservationState,
    RunRecord,
)
from biorag_repro.generation.completion import CompletedRun, verify_completed_run
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.environment import EnvironmentRecord
from biorag_repro.generation.environment_gate import EnvironmentEvidence, verify_run_environment
from biorag_repro.generation.hf_backend import HF_BACKEND_NAME
from biorag_repro.generation.inputs import (
    CorruptInput,
    check_tokenization,
    input_record_ref,
    read_input_record,
)
from biorag_repro.generation.receipt import ExecutionFacts, decode_output
from biorag_repro.generation.roster import RosterRow
from biorag_repro.generation.runner import decided_results
from biorag_repro.retrieval import context as phase5

# Implementations accepted as the official generator. A name is necessary, not sufficient:
# the checkpoint, device, inputs and receipts below must all verify as well.
OFFICIAL_BACKENDS: frozenset[str] = frozenset({HF_BACKEND_NAME})


class NotVerifiedOfficial(LifecycleError):
    """A result is not part of a verified official run."""


class NoExecutionReceipt(LifecycleError):
    """No receipt shows that the generator was invoked for this attempt."""


@dataclass(frozen=True)
class VerifiedOfficialRun:
    run_id: str
    manifest_sha256: str
    results: list[GenerationResult]
    digests: frozenset[str]
    completed: CompletedRun
    environment: EnvironmentEvidence


def _check_environment(
    environment: EnvironmentRecord, settings: GeneratorConfiguration, approved_manifest: str
) -> None:
    backend = environment.backend
    if backend.kind != "official" or backend.name not in OFFICIAL_BACKENDS:
        raise NonOfficialBackend(f"backend {backend.name!r} is not a recognised official backend")
    execution = environment.execution
    if execution is None:
        raise NonOfficialBackend("the environment records no loaded model")
    if execution.snapshot.manifest_sha256 != approved_manifest:
        raise NonOfficialBackend("model file manifest differs from the approved manifest")
    if execution.dtype != settings.model.dtype:
        raise NonOfficialBackend(
            f"model ran in dtype {execution.dtype}, not {settings.model.dtype}"
        )
    if not hf_backend.is_official_device(execution.device):
        raise NonOfficialBackend(f"{execution.device!r} is not one explicit CUDA device")
    if environment.gpu_status != "ok":
        raise NonOfficialBackend("no GPU was read from the machine")
    if execution.tokenizer_file_sha256 != settings.tokenizer.file_sha256:
        raise NonOfficialBackend("tokenizer files differ from the configuration")
    if execution.max_position_embeddings < settings.budgets.context_window_tokens:
        raise NonOfficialBackend("configured context window exceeds the model's positions")


def row_key(completed: CompletedRun, row: RosterRow) -> str:
    return completed.store.key_for(
        Coordinates(
            item_id=row.item_id,
            paper_id=row.paper_id,
            matrix_arm=completed.completion.matrix_arm,
            representation_or_condition=row.condition,
            repeat_index=row.repeat_index,
        )
    )


def _check_output(
    completed: CompletedRun,
    state: ObservationState,
    facts: ExecutionFacts,
    tokenizer: Any,
    name: object,
) -> None:
    """The stored raw text must be what the reported output token ids decode to."""
    if tokenizer.eos_token_id not in facts.generation.eos_token_ids:
        raise NonOfficialBackend(f"{name}: the reported stop tokens omit the chat end token")
    raw_ref = state.events[-1].raw_output
    assert raw_ref is not None
    raw = (completed.store.root / raw_ref.logical_ref).read_text(encoding="utf-8")
    if decode_output(tokenizer, facts.output_token_ids, facts.stop_reason) != raw:
        raise NonOfficialBackend(
            f"{name}: the output token ids do not decode to the stored raw output"
        )


def _check_inputs(completed: CompletedRun, tokenizer: Any) -> None:
    states = {s.key: s for s in completed.states}
    run_dir = completed.store.run_dir
    for row in completed.roster.rows:
        name = (row.item_id, row.condition, row.repeat_index)
        ref = row.input_record
        if ref is None:
            raise NonOfficialBackend(f"{name}: the roster cites no stored input")
        key = row_key(completed, row)
        try:
            record = read_input_record(run_dir, ref.sha256)
        except CorruptInput as error:
            raise NonOfficialBackend(f"{name}: stored input is unusable: {error}") from error
        if input_record_ref(record) != ref:
            raise NonOfficialBackend(f"{name}: stored input differs from the one the roster cites")
        if record.observation_key != key:
            raise NonOfficialBackend(f"{name}: stored input belongs to another observation")
        rendered = record.rendered
        if (rendered.rendered_sha256, rendered.context_sha256, rendered.input_tokens) != (
            row.rendered_input_sha256,
            row.context_sha256,
            row.input_tokens,
        ) or record.context != row.context:
            raise NonOfficialBackend(f"{name}: stored input differs from the roster row")
        try:
            check_tokenization(record, tokenizer)
        except ValueError as error:
            raise NonOfficialBackend(f"{name}: {error}") from error
        state = states[key]
        if state.status == "complete":
            receipt = completed.store.receipt_of(state)
            if receipt is None:
                raise NonOfficialBackend(f"{name}: completed without an execution receipt")
            if receipt.input_record_sha256 != ref.sha256:
                raise NonOfficialBackend(f"{name}: the receipt names another stored input")
            _check_output(completed, state, receipt.facts, tokenizer, name)


def verify_official_run(
    root: Path,
    run_id: str,
    manifest_sha256: str,
    *,
    config: ResolvedConfiguration[GeneratorConfiguration],
    view_directory: Path,
    view_record_sha256: str,
    tokenizer_root: Path,
) -> VerifiedOfficialRun:
    run_path = root / "runs" / run_id / "run.json"
    try:
        record = RunRecord.model_validate_json(run_path.read_bytes())
    except (OSError, ValueError) as error:
        raise NonOfficialBackend(f"run record of {run_id} is unreadable") from error
    if not record.official:
        raise NonOfficialBackend(
            f"a {record.backend_kind} run with a {record.purpose} configuration is not official"
        )
    # The peeked record is not trusted by itself: its hash is bound by the completion anchor.
    completed = verify_completed_run(
        root, run_id, manifest_sha256, config=config, backend_kind=record.backend_kind
    )

    settings = config.settings
    if (
        settings.tokenizer.revision != phase5.REVISION
        or settings.tokenizer.file_sha256 != phase5.ASSETS
    ):
        raise NonOfficialBackend("the configuration does not name the approved tokenizer")
    approved = generator_config.APPROVED_CHECKPOINTS.get(
        (settings.model.repository, settings.model.revision)
    )
    if approved is None:
        raise NonOfficialBackend(
            f"checkpoint {settings.model.repository}@{settings.model.revision} is not approved"
        )
    environments = [
        read_project_json(
            root / "runs" / run_id / "environments" / f"{digest}.json", digest, EnvironmentRecord
        )
        for digest in completed.completion.environments
    ]
    for environment in environments:
        _check_environment(environment, settings, approved)
    environment_evidence = verify_run_environment(completed, environments)

    view, items = read_development_view(view_directory, view_record_sha256)
    roster = completed.roster
    if (roster.development_items, roster.benchmark_version, roster.corpus_version) != (
        view.output_items,
        view.benchmark_version,
        view.corpus_version,
    ):
        raise NonOfficialBackend("the roster does not cite this development view")
    by_id = {item.item_id: item for item in items}
    for row in roster.rows:
        item = by_id.get(row.item_id)
        if item is None:
            raise NonOfficialBackend("a rostered item is not in the development view")
        if (row.paper_id, row.item_sha256) != (item.paper_id, item_digest(item)):
            raise NonOfficialBackend("a rostered item differs from the view record")

    tokenizer = phase5.QwenCounter(tokenizer_root, phase5.tokenizer_snapshot()).tokenizer
    _check_inputs(completed, tokenizer)

    results = decided_results(completed.store)
    return VerifiedOfficialRun(
        run_id,
        manifest_sha256,
        results,
        frozenset(generation_digest(r) for r in results),
        completed,
        environment_evidence,
    )


def check_official_lineage(
    results: Iterable[GenerationResult], verified: VerifiedOfficialRun
) -> None:
    """Every result must be byte-identical to one produced by the verified run."""
    for result in results:
        if generation_digest(result) not in verified.digests:
            raise NotVerifiedOfficial(
                f"result for {result.observation_key} is not part of verified run {verified.run_id}"
            )


def _link(
    completed: CompletedRun,
    result: GenerationResult,
    provenance: Literal["recorded", "simulated"],
) -> GeneratorInput:
    if result.status != "completed":
        raise ValueError("a generator input link needs an executed (completed) result")
    if result.run_id != completed.run_id:
        raise NotVerifiedOfficial("the result belongs to another run")
    state: ObservationState | None = next(
        (s for s in completed.states if s.key == result.observation_key), None
    )
    if (
        state is None
        or state.status != "complete"
        or state.events[-1].attempt_id != result.attempt_id
    ):
        raise NotVerifiedOfficial("the result is not the completed attempt of its observation")
    if completed.store.receipt_of(state) is None:
        raise NoExecutionReceipt(f"{result.observation_key} has no execution receipt")
    row = next(r for r in completed.roster.rows if row_key(completed, r) == state.key)
    if row.context is None or row.context_sha256 is None or row.input_record is None:
        raise ValueError("a closed-book observation or a row without a stored input has no link")
    return GeneratorInput(
        observation_key=result.observation_key,
        attempt_id=result.attempt_id,
        context=row.context,
        exact_context_sha256=row.context_sha256,
        input=ArtifactRef(
            artifact_id=row.input_record.artifact_id,
            logical_ref=f"runs/{completed.run_id}/{row.input_record.logical_ref}",
            sha256=row.input_record.sha256,
        ),
        provenance=provenance,
    )


def recorded_generator_input(
    verified: VerifiedOfficialRun, result: GenerationResult
) -> GeneratorInput:
    """Only a verified official run can say a generator received the recorded input."""
    check_official_lineage([result], verified)
    return _link(verified.completed, result, "recorded")


def simulated_generator_input(completed: CompletedRun, result: GenerationResult) -> GeneratorInput:
    """A mock or smoke run yields a link marked simulated, never a recorded delivery."""
    if completed.store.record.official:
        raise NonOfficialBackend("an official run's input is recorded through verify_official_run")
    return _link(completed, result, "simulated")
