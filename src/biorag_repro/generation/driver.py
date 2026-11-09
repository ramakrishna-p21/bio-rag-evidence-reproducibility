"""Execution driver: dry run, mock or smoke execution, anchoring and verification.

The driver chains the pieces that already exist: the authorized roster built from verified
Phase 5 contexts, the run store, a configured backend, the completion manifest, an anchor
kept outside the run, and read-back verification. It is generic: nothing in it knows which
items or papers are run.

Three things are kept apart. A dry run validates everything that can be checked without a
model and writes nothing. Mock and smoke execution produce engineering artifacts that are
labelled as such and can never be verified as official. Official execution stays disabled
until a checkpoint manifest has been approved and the configuration is a development
candidate; the backend is not even created before that gate passes.

The anchor is the hash that lets a later reader detect a rewritten run. It is stored
outside the run root, read-only, and read back after writing. That makes tampering visible
only to readers who trust the anchor's own location: keep it, or its printed hash, where
whoever can write the run cannot.
"""

import argparse
import fcntl
import importlib
import json
import os
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import StringConstraints

from biorag_repro.benchmark.models import Record
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.storage import publish_bytes, serialize_project_json
from biorag_repro.evaluation.analysis_models import PlannedRoster
from biorag_repro.generation import config as generator_config
from biorag_repro.generation.attempts import LifecycleError, RunStore
from biorag_repro.generation.completion import CompletedRun, complete_run, verify_completed_run
from biorag_repro.generation.config import GeneratorConfiguration, resolve_generator_config
from biorag_repro.generation.environment import (
    BackendIdentity,
    BackendKind,
    EnvironmentRecord,
    capture_environment,
    system_probes,
)
from biorag_repro.generation.environment_gate import (
    EnvironmentEvidence,
    check_approved,
    publish_environment_evidence,
)
from biorag_repro.generation.official import VerifiedOfficialRun, verify_official_run
from biorag_repro.generation.prompt import InputRenderer, PromptArtifact, RenderedInput
from biorag_repro.generation.roster import BuiltRoster, ContextSource, build_roster
from biorag_repro.generation.runner import (
    GeneratorBackend,
    NonOfficialResults,
    RunSummary,
    nonofficial_results,
    run_roster,
)
from biorag_repro.generation.seeds import GenerationSeedRegistry, check_seeds_for_configuration
from biorag_repro.generation.sources import SourceError, load_context_sources

Mode = Literal["mock", "smoke", "official"]
MODES: tuple[str, ...] = get_args(Mode)
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
MOCK_TEXT = json.dumps({"answer": "mock answer", "abstained": False})


class DriverError(Exception):
    """The request cannot be carried out as given; nothing was run."""


class OfficialDisabled(DriverError):
    """Official execution is not permitted for this configuration yet."""


class ConcurrentRun(DriverError):
    """Another driver process holds the run root."""


@dataclass(frozen=True)
class DriverPlan:
    """Where everything lives. Paths are configurable; no directory layout is assumed."""

    run_root: Path
    anchor_dir: Path
    config: Path
    prompt: Path
    seeds: Path | None
    tokenizer_root: Path
    view_directory: Path
    view_record_sha256: str
    scope: Path
    repository: Path  # checkout whose git state each environment record captures
    context_sources: Path | None = None  # serialized Phase 5 context sources (generation.sources)
    context_sources_sha256: str | None = None  # their pinned hash; required with the file


@dataclass(frozen=True)
class Prepared:
    config: ResolvedConfiguration[GeneratorConfiguration]
    seeds: GenerationSeedRegistry | None
    renderer: InputRenderer
    scope: PlannedRoster
    built: BuiltRoster


@dataclass(frozen=True)
class DryRunReport:
    mode: str
    observations: int
    items: int
    conditions: list[str]
    repeats: list[int]
    min_input_tokens: int
    max_input_tokens: int
    input_token_limit: int
    official_blockers: list[str]


@dataclass(frozen=True)
class ExecutionOutcome:
    run_id: str
    mode: str
    summary: RunSummary
    manifest_sha256: str
    anchor_file: Path
    anchor_file_sha256: str


class Anchor(Record):
    """What must be retained outside the run to recognise it later."""

    schema_version: Literal["1"] = "1"
    run_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]
    backend_kind: BackendKind
    resolved_config_sha256: Sha256
    roster_sha256: Sha256
    manifest_sha256: Sha256


Engine = tuple[GeneratorBackend, Callable[[], EnvironmentRecord]]


class MockBackend:
    """Fixed-text backend for engineering runs. Its output is not an observation."""

    identity = BackendIdentity(kind="mock", name=f"{__name__}.MockBackend", version=None)

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate(self, rendered: RenderedInput, *, seed: int | None, max_new_tokens: int) -> str:
        self.calls.append(rendered.rendered_sha256)
        return MOCK_TEXT


# ---- plan and inputs ---------------------------------------------------------------------


_PLAN_KEYS = {f for f in DriverPlan.__dataclass_fields__}
_OPTIONAL_KEYS = {"seeds", "context_sources", "context_sources_sha256"}
_LITERAL_KEYS = {"view_record_sha256", "context_sources_sha256"}


def load_plan(path: Path) -> DriverPlan:
    """Read a JSON plan; relative paths are relative to the plan file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DriverError(f"plan {path} is unreadable: {error}") from error
    if not isinstance(data, dict):
        raise DriverError("a plan is one JSON object")
    if unknown := sorted(set(data) - _PLAN_KEYS):
        raise DriverError(f"unknown plan keys: {unknown}")
    if missing := sorted(_PLAN_KEYS - set(data) - _OPTIONAL_KEYS):
        raise DriverError(f"plan keys missing: {missing}")
    base = path.resolve().parent
    values: dict[str, object] = {}
    for key in _PLAN_KEYS:
        value = data.get(key)
        if value is None:
            values[key] = None
        elif key in _LITERAL_KEYS:
            values[key] = value
        else:
            values[key] = base / str(value)
    if (values["context_sources"] is None) != (values["context_sources_sha256"] is None):
        raise DriverError("context_sources and context_sources_sha256 are given together")
    return DriverPlan(**values)  # type: ignore[arg-type]


def check_placement(run_root: Path, anchor_dir: Path) -> None:
    """The anchor must not live inside the run root, nor the run root inside the anchor's."""
    runs, anchors = run_root.resolve(), anchor_dir.resolve()
    if runs == anchors or runs.is_relative_to(anchors) or anchors.is_relative_to(runs):
        raise DriverError("the anchor directory must lie outside the run root")


def prepare(plan: DriverPlan, sources: Sequence[ContextSource]) -> Prepared:
    """Validate configuration, prompt, seeds, scope and contexts; write nothing."""
    try:
        config = resolve_generator_config(plan.config.read_bytes())
        prompt = PromptArtifact.model_validate_json(plan.prompt.read_bytes())
        scope = PlannedRoster.model_validate_json(plan.scope.read_bytes())
    except (OSError, ValueError) as error:
        raise DriverError(f"configuration, prompt or scope is not usable: {error}") from error
    settings = config.settings
    seeds = None
    if plan.seeds is not None:
        try:
            seeds = GenerationSeedRegistry.model_validate_json(plan.seeds.read_bytes())
            check_seeds_for_configuration(settings, seeds)
        except (OSError, ValueError) as error:
            raise DriverError(f"seed registry is not usable: {error}") from error
    if settings.decoding.do_sample and seeds is None:
        raise DriverError("sampling decoding needs a seed registry (plan key seeds)")
    try:
        renderer = InputRenderer(plan.tokenizer_root, config, prompt)
        built = build_roster(
            view_directory=plan.view_directory,
            view_record_sha256=plan.view_record_sha256,
            scope=scope,
            renderer=renderer,
            sources=sources,
        )
    except (OSError, ValueError) as error:
        raise DriverError(f"the authorized roster cannot be built: {error}") from error
    return Prepared(config, seeds, renderer, scope, built)


def official_blockers(settings: GeneratorConfiguration) -> list[str]:
    blockers = []
    if settings.purpose != "development_candidate":
        blockers.append(f"configuration purpose is {settings.purpose}, not development_candidate")
    if (settings.model.repository, settings.model.revision) not in (
        generator_config.APPROVED_CHECKPOINTS
    ):
        blockers.append(
            f"checkpoint {settings.model.repository}@{settings.model.revision} has no "
            "approved weight manifest"
        )
    if not generator_config.APPROVED_ENVIRONMENT_LOCKS:
        blockers.append("no environment lock has been approved")
    return blockers


def _check_mode(mode: str) -> None:
    if mode not in MODES:
        raise DriverError(f"unknown mode {mode!r}; expected one of {MODES}")


def dry_run(plan: DriverPlan, sources: Sequence[ContextSource], *, mode: str) -> DryRunReport:
    """Everything that can be checked without model weights or generation."""
    _check_mode(mode)
    check_placement(plan.run_root, plan.anchor_dir)
    if plan.run_root.exists() and not plan.run_root.is_dir():
        raise DriverError("the run root exists and is not a directory")
    prepared = prepare(plan, sources)
    rows = prepared.built.record.rows
    settings = prepared.config.settings
    tokens = [row.input_tokens for row in rows]
    return DryRunReport(
        mode=mode,
        observations=len(rows),
        items=len({row.item_id for row in rows}),
        conditions=sorted({row.condition for row in rows}),
        repeats=sorted({row.repeat_index for row in rows}),
        min_input_tokens=min(tokens),
        max_input_tokens=max(tokens),
        input_token_limit=settings.budgets.max_input_tokens,
        official_blockers=official_blockers(settings),
    )


# ---- single writer and anchor ------------------------------------------------------------


@contextmanager
def single_writer(run_root: Path) -> Iterator[None]:
    """Hold an exclusive advisory lock on the run root; refuse if another process holds it.

    flock is reliable on a local file system. Some network file systems do not honour it,
    so on shared storage keep one driver per run root by other means as well.
    """
    descriptor = os.open(run_root / "driver.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ConcurrentRun(f"another driver holds {run_root}") from None
        yield
    finally:
        os.close(descriptor)


def _anchor_path(anchor_dir: Path, run_id: str) -> Path:
    return anchor_dir / f"{run_id}.anchor.json"


def write_anchor(anchor_dir: Path, run_root: Path, anchor: Anchor) -> str:
    """Store the anchor read-only outside the run root and return its file hash."""
    check_placement(run_root, anchor_dir)
    anchor_dir.mkdir(parents=True, exist_ok=True)
    content = serialize_project_json(anchor, Anchor)
    digest = sha256_bytes(content)
    path = _anchor_path(anchor_dir, anchor.run_id)
    try:
        publish_bytes(path, content, digest)
    except FileExistsError:
        if path.read_bytes() != content:
            raise DriverError(
                f"a different anchor is already stored for run {anchor.run_id}"
            ) from None
    path.chmod(0o444)
    return digest


def read_anchor(anchor_dir: Path, run_id: str, file_sha256: str | None = None) -> Anchor:
    path = _anchor_path(anchor_dir, run_id)
    if not path.is_file():
        raise DriverError(f"no anchor for run {run_id} in {anchor_dir}")
    content = path.read_bytes()
    if file_sha256 is not None and sha256_bytes(content) != file_sha256:
        raise DriverError(f"anchor file of run {run_id} differs from the retained hash")
    try:
        anchor = Anchor.model_validate_json(content)
    except ValueError as error:
        raise DriverError(f"anchor file of run {run_id} is unreadable: {error}") from error
    if anchor.run_id != run_id:
        raise DriverError("anchor file names another run")
    return anchor


# ---- execution and verification ---------------------------------------------------------


def execute(
    plan: DriverPlan,
    sources: Sequence[ContextSource],
    *,
    mode: str,
    start: Callable[[Prepared], Engine],
    resume: str | None = None,
    recover_interrupted: bool = False,
    environment_evidence: EnvironmentEvidence | None = None,
) -> ExecutionOutcome:
    """Run the authorized roster to completion, anchor it, and read the anchor back.

    `start` creates the backend and its environment capture; it is called only after
    every check that can refuse the run, so refused runs never load a model.
    An official run also needs `environment_evidence`: an approved lock and its verification
    against a live inventory, checked here and stored with the run before any attempt.
    `recover_interrupted` asserts that no worker other than this driver is still writing
    the run: the lock excludes other drivers, not other tools.
    """
    _check_mode(mode)
    check_placement(plan.run_root, plan.anchor_dir)
    prepared = prepare(plan, sources)
    settings = prepared.config.settings
    run_phase = prepared.scope.phase
    if run_phase == "final_test":
        raise DriverError("a final_test scope cannot be executed here")
    if mode == "official" and (blockers := official_blockers(settings)):
        raise OfficialDisabled("official execution is disabled: " + "; ".join(blockers))
    if mode == "official":
        if environment_evidence is None:
            raise OfficialDisabled("official execution is disabled: no environment evidence")
        check_approved(environment_evidence)

    plan.run_root.mkdir(parents=True, exist_ok=True)
    with single_writer(plan.run_root):
        backend, capture = start(prepared)
        if backend.identity.kind != mode:
            raise DriverError(
                f"mode is {mode} but the backend is {backend.identity.kind}; "
                "the mode must match what the backend verified about itself"
            )
        built = prepared.built
        if resume is None:
            store = RunStore.create(
                plan.run_root,
                config=prepared.config,
                phase=run_phase,
                benchmark_version=built.record.benchmark_version,
                backend_kind=mode,
            )
        else:
            store = RunStore.open(
                plan.run_root,
                resume,
                config=prepared.config,
                backend_kind=mode,
            )
        if mode == "official" and environment_evidence is not None:
            publish_environment_evidence(store.run_dir, environment_evidence)
        if recover_interrupted:
            store.recover_interrupted(previous_worker_stopped=True)
        summary = run_roster(
            store, built.entries, backend, capture_environment=capture, seeds=prepared.seeds
        )
        manifest = complete_run(store, built.record)
        anchor = Anchor(
            run_id=store.run_id,
            backend_kind=mode,
            resolved_config_sha256=prepared.config.configuration_sha256,
            roster_sha256=built.sha256,
            manifest_sha256=manifest,
        )
        file_sha = write_anchor(plan.anchor_dir, plan.run_root, anchor)
        retained = read_anchor(plan.anchor_dir, store.run_id, file_sha)
        verify_completed_run(
            plan.run_root,
            store.run_id,
            retained.manifest_sha256,
            config=prepared.config,
            backend_kind=mode,
        )
    return ExecutionOutcome(
        store.run_id, mode, summary, manifest, _anchor_path(plan.anchor_dir, store.run_id), file_sha
    )


def verify(
    plan: DriverPlan, run_id: str, *, mode: str, anchor_file_sha256: str | None = None
) -> NonOfficialResults | VerifiedOfficialRun:
    """Verify a completed run against its retained anchor and return the labelled handoff.

    Mock and smoke runs return `NonOfficialResults`; only an official run that passes
    every official check returns a `VerifiedOfficialRun`.
    """
    _check_mode(mode)
    check_placement(plan.run_root, plan.anchor_dir)
    anchor = read_anchor(plan.anchor_dir, run_id, anchor_file_sha256)
    if anchor.backend_kind != mode:
        raise DriverError(f"run {run_id} is a {anchor.backend_kind} run, not {mode}")
    try:
        config = resolve_generator_config(plan.config.read_bytes())
    except (OSError, ValueError) as error:
        raise DriverError(f"generator configuration is not usable: {error}") from error
    if mode == "official":
        return verify_official_run(
            plan.run_root,
            run_id,
            anchor.manifest_sha256,
            config=config,
            view_directory=plan.view_directory,
            view_record_sha256=plan.view_record_sha256,
            tokenizer_root=plan.tokenizer_root,
        )
    completed: CompletedRun = verify_completed_run(
        plan.run_root,
        run_id,
        anchor.manifest_sha256,
        config=config,
        backend_kind=mode,
    )
    return nonofficial_results(completed.store)


# ---- command line -------------------------------------------------------------------------


def resolve_sources(plan: DriverPlan, provider: str | None) -> Sequence[ContextSource]:
    """The plan's pinned source file, or an explicit provider function; never both."""
    if provider is not None and plan.context_sources is not None:
        raise DriverError("give either --sources or plan context_sources, not both")
    if provider is not None:
        return _load_sources(provider)
    if plan.context_sources is None or plan.context_sources_sha256 is None:
        raise DriverError("context sources are needed: plan context_sources or --sources")
    try:
        return load_context_sources(
            plan.context_sources,
            plan.context_sources_sha256,
            plan.view_directory,
            plan.view_record_sha256,
        )
    except SourceError as error:
        raise DriverError(str(error)) from error


def _load_sources(spec: str) -> Sequence[ContextSource]:
    """`package.module:function`, called with no arguments, returns the context sources.

    For sources a spec file cannot describe (dense retrieval needs an encoder object).
    """
    module, _, function = spec.partition(":")
    if not module or not function:
        raise DriverError("--sources must look like package.module:function")
    try:
        provider = getattr(importlib.import_module(module), function)
    except (ImportError, AttributeError) as error:
        raise DriverError(f"cannot load context sources from {spec}: {error}") from error
    return list(provider())


def _hf_engine(plan: DriverPlan, model_dir: Path, device: str) -> Callable[[Prepared], Engine]:
    def start(prepared: Prepared) -> Engine:
        from biorag_repro.generation.hf_backend import HuggingFaceBackend

        backend = HuggingFaceBackend(
            prepared.config,
            model_dir=model_dir,
            tokenizer_root=plan.tokenizer_root,
            device=device,
        )
        probes = system_probes(plan.repository)
        return backend, lambda: backend.environment(probes)

    return start


def _mock_engine(plan: DriverPlan) -> Callable[[Prepared], Engine]:
    def start(prepared: Prepared) -> Engine:
        settings = prepared.config.settings
        backend = MockBackend()
        probes = system_probes(plan.repository)

        def environment() -> EnvironmentRecord:
            return capture_environment(
                backend=backend.identity,
                model=(settings.model.repository, settings.model.revision),
                tokenizer=(settings.tokenizer.repository, settings.tokenizer.revision),
                packages=(),
                probes=probes,
            )

        return backend, environment

    return start


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("operation", choices=["dry-run", "run", "verify"])
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument(
        "--sources", help="package.module:function returning sources (else the plan's pinned file)"
    )
    parser.add_argument("--resume", help="run id to continue")
    parser.add_argument("--recover-interrupted", action="store_true")
    parser.add_argument("--run-id", help="run to verify")
    parser.add_argument("--anchor-file-sha256", help="retained hash of the anchor file")
    parser.add_argument("--model-dir", type=Path, help="local checkpoint (smoke and official)")
    parser.add_argument("--device", help="torch device (smoke and official)")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.operation == "verify":
            if not args.run_id:
                raise DriverError("verify needs --run-id")
            result = verify(
                plan, args.run_id, mode=args.mode, anchor_file_sha256=args.anchor_file_sha256
            )
            kind = "official" if isinstance(result, VerifiedOfficialRun) else args.mode
            print(f"verified {kind} run {args.run_id}: {len(result.results)} decided results")
            return 0
        sources = resolve_sources(plan, args.sources)
        if args.operation == "dry-run":
            report = dry_run(plan, sources, mode=args.mode)
            print(json.dumps(report.__dict__, indent=1))
            return 0
        if args.mode == "mock":
            start = _mock_engine(plan)
        elif args.model_dir and args.device:
            start = _hf_engine(plan, args.model_dir, args.device)
        else:
            raise DriverError("smoke and official modes need --model-dir and --device")
        outcome = execute(
            plan,
            sources,
            mode=args.mode,
            start=start,
            resume=args.resume,
            recover_interrupted=args.recover_interrupted,
        )
    except (DriverError, LifecycleError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    print(f"{outcome.mode} run {outcome.run_id}: {outcome.summary}")
    print(f"manifest sha256 {outcome.manifest_sha256}")
    print(f"anchor file {outcome.anchor_file} sha256 {outcome.anchor_file_sha256}")
    print("retain the anchor file hash where the run's writer cannot change it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
