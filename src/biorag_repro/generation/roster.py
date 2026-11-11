"""Build the authorized observation roster for one generator configuration.

Membership comes from the published development-only benchmark view and the declared
scope, never from the caller's list of contexts. Every retrieval context is read back
through the context reader, so its text, token count and lineage are recomputed from the
independent selection; the generator input is rendered from that exact text and nothing
is truncated. A closed-book roster carries no context at all.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self

from pydantic import StringConstraints, model_validator

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation.analysis_models import PlannedRoster
from biorag_repro.evaluation.analysis_table import ARM_RETRIEVAL
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.generation.inputs import build_input_record, input_record_ref
from biorag_repro.generation.prompt import InputRenderer, RenderedInput
from biorag_repro.generation.runner import RosterEntry, RosterMismatch
from biorag_repro.retrieval import context as phase5
from biorag_repro.retrieval.models import ConfigurationRef, ContextRecord

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]


@dataclass(frozen=True)
class ContextSource:
    """A published context and the independent selection it must be recomputed from."""

    context: ArtifactRef
    manifest: ArtifactRef
    selection: phase5.Selection
    configuration: ConfigurationRef
    configuration_root: Path
    root: Path


class RosterRow(Record):
    item_id: Nonblank
    item_sha256: Sha256
    paper_id: Nonblank
    condition: Nonblank
    repeat_index: int
    rendered_input_sha256: Sha256
    input_tokens: int
    context: ArtifactRef | None
    context_sha256: Sha256 | None
    input_record: ArtifactRef | None = None  # run-relative reference to the stored exact input


class RosterRecord(Record):
    """The exact set of authorized observations and the inputs they were built from."""

    schema_version: str = "1"
    phase: str
    benchmark_version: Nonblank
    corpus_version: Nonblank
    development_items: ArtifactRef
    scope_sha256: Sha256
    resolved_config_sha256: Sha256
    matrix_arm: Nonblank
    rows: list[RosterRow]

    @model_validator(mode="after")
    def sorted_unique(self) -> Self:
        keys = [(r.item_id, r.condition, r.repeat_index) for r in self.rows]
        if not keys or keys != sorted(set(keys)):
            raise ValueError("roster rows must be non-empty, unique and sorted")
        return self


def roster_sha256(record: RosterRecord) -> str:
    return sha256_bytes(serialize_project_json(record, RosterRecord))


@dataclass(frozen=True)
class BuiltRoster:
    record: RosterRecord
    entries: list[RosterEntry]
    sha256: str


def _check_scope(
    scope: PlannedRoster, view_items: dict[str, BenchmarkItem], view_versions: tuple[str, str]
) -> None:
    # Qualification draws on development items only; the view check below enforces that.
    if scope.phase not in ("development", "qualification"):
        raise RosterMismatch("the scope phase is neither development nor qualification")
    if (scope.benchmark_version, scope.corpus_version) != view_versions:
        raise RosterMismatch("the scope names another benchmark or corpus version than the view")
    declared = {i.item_id: i for i in scope.items}
    if outside := set(declared) - set(view_items):
        raise RosterMismatch(f"{len(outside)} declared items are not in the development view")
    if undeclared := set(view_items) - set(declared):
        raise RosterMismatch(f"the scope does not declare {len(undeclared)} development items")
    for item_id, row in declared.items():
        if row.state != "included":
            continue
        item = view_items[item_id]
        if (row.paper_id, row.item_sha256) != (item.paper_id, item_digest(item)):
            raise RosterMismatch(f"included item {item_id} differs from the view record")


def _read_source(source: ContextSource, renderer: InputRenderer) -> ContextRecord:
    return phase5.read(
        source.context,
        source.manifest,
        source.selection,
        source.configuration,
        source.configuration_root,
        renderer.tokenizer_root,
        source.root,
    )


@dataclass(frozen=True)
class ItemFacts:
    """What rendering needs to know about one item. The question never leaves this step."""

    item_id: str
    paper_id: str
    question: str
    item_sha256: str


@dataclass(frozen=True)
class ObservationContext:
    """An assembled context as the renderer receives it; None stands for closed book."""

    exact_context: str
    generator_tokens: int
    reference: ArtifactRef | None


def render_observations(
    *,
    phase: str,
    benchmark_version: str,
    items: Mapping[str, ItemFacts],
    conditions: Sequence[str],
    repeats: Sequence[int],
    renderer: InputRenderer,
    contexts: Mapping[tuple[str, str], ObservationContext | None],
) -> tuple[list[RosterEntry], list[RosterRow]]:
    """Rendered input, stored input record and roster row of every (item, condition, repeat).

    Shared by the development roster and the final-test pipeline so that rendering, the
    observation key and the stored input are one implementation. Membership, the question
    source and the context source are the caller's responsibility; the result is sorted.
    """
    settings = renderer.config.settings
    entries: list[RosterEntry] = []
    rows: list[RosterRow] = []
    wanted = {(item_id, condition) for item_id in items for condition in conditions}
    for item_id, condition in sorted(wanted):
        item = items[item_id]
        assembled = contexts[(item_id, condition)]
        try:
            rendered: RenderedInput = renderer.render(
                item.question,
                None if assembled is None else assembled.exact_context,
                None if assembled is None else assembled.generator_tokens,
            )
        except ValueError as error:
            raise RosterMismatch(f"{item_id}/{condition}: {error}") from error
        if assembled is not None and rendered.context != assembled.exact_context:
            raise RosterMismatch(f"{item_id}/{condition}: rendered context differs from assembled")
        ref = None if assembled is None else assembled.reference
        for repeat in repeats:
            obs_key = observation_key(
                phase=phase,
                benchmark_version=benchmark_version,
                item_id=item_id,
                matrix_arm=settings.matrix_arm,
                representation_or_condition=condition,
                repeat_index=repeat,
                resolved_config_sha256=renderer.config.configuration_sha256,
            )
            try:
                stored = build_input_record(
                    renderer, rendered, observation_key=obs_key, context=ref
                )
            except ValueError as error:
                raise RosterMismatch(f"{item_id}/{condition}: {error}") from error
            entries.append(
                RosterEntry(
                    item_id=item_id,
                    paper_id=item.paper_id,
                    representation_or_condition=condition,
                    repeat_index=repeat,
                    rendered=rendered,
                    input_record=stored,
                )
            )
            rows.append(
                RosterRow(
                    item_id=item_id,
                    item_sha256=item.item_sha256,
                    paper_id=item.paper_id,
                    condition=condition,
                    repeat_index=repeat,
                    rendered_input_sha256=rendered.rendered_sha256,
                    input_tokens=rendered.input_tokens,
                    context=ref,
                    context_sha256=rendered.context_sha256,
                    input_record=input_record_ref(stored),
                )
            )
    rows.sort(key=lambda r: (r.item_id, r.condition, r.repeat_index))
    entries.sort(key=lambda e: (e.item_id, e.representation_or_condition, e.repeat_index))
    return entries, rows


def build_roster(
    *,
    view_directory: Path,
    view_record_sha256: str,
    scope: PlannedRoster,
    renderer: InputRenderer,
    sources: Sequence[ContextSource],
) -> BuiltRoster:
    """Rendered inputs for every authorized (item, condition, repeat), in a fixed order."""
    view, view_items = read_development_view(view_directory, view_record_sha256)
    by_id = {item.item_id: item for item in view_items}
    _check_scope(scope, by_id, (view.benchmark_version, view.corpus_version))

    settings = renderer.config.settings
    arm = next((a for a in scope.arms if a.matrix_arm == settings.matrix_arm), None)
    if arm is None:
        raise RosterMismatch(f"the scope declares no {settings.matrix_arm} arm")
    included = sorted(i.item_id for i in scope.items if i.state == "included")
    closed_book = settings.matrix_arm == "closed_book"
    wanted = {(item_id, condition) for item_id in included for condition in arm.conditions}

    contexts: dict[tuple[str, str], ContextRecord | None] = {}
    sources_by_key: dict[tuple[str, str], ContextSource] = {}
    for source in sources:
        recomputed = _read_source(source, renderer)
        query = source.selection.query
        key = (query.query_id, recomputed.representation)
        if key not in wanted:
            raise RosterMismatch(f"context source {key} is not authorized by the scope")
        if key in contexts:
            raise RosterMismatch(f"duplicate context source for {key}")
        if recomputed.configuration.resolved_sha256 != settings.retrieval_config_sha256:
            raise RosterMismatch(
                f"{key}: context is not from the generator's retrieval configuration"
            )
        item = by_id[key[0]]
        expected_scope, expected_retriever = ARM_RETRIEVAL[settings.matrix_arm]
        if query.question != item.question:
            raise RosterMismatch(f"{key}: the retrieval question is not the item's question")
        if (query.scope, source.selection.method) != (expected_scope, expected_retriever):
            raise RosterMismatch(f"{key}: retrieval scope or retriever differs from the arm")
        if expected_scope == "designated_paper" and query.designated_paper_id != item.paper_id:
            raise RosterMismatch(f"{key}: the designated paper is not the item's paper")
        contexts[key] = recomputed
        sources_by_key[key] = source
    if not closed_book and (missing := sorted(wanted - set(contexts))):
        raise RosterMismatch(
            f"no context source for {len(missing)} authorized pairs: {missing[:3]}"
        )

    observation_contexts: dict[tuple[str, str], ObservationContext | None] = {}
    for pair in wanted:
        chosen = None if closed_book else contexts[pair]
        observation_contexts[pair] = (
            None
            if chosen is None
            else ObservationContext(
                exact_context=chosen.exact_context,
                generator_tokens=chosen.generator_tokens,
                reference=sources_by_key[pair].context,
            )
        )
    entries, rows = render_observations(
        phase=scope.phase,
        benchmark_version=view.benchmark_version,
        items={
            item_id: ItemFacts(
                item_id=item_id,
                paper_id=by_id[item_id].paper_id,
                question=by_id[item_id].question,
                item_sha256=item_digest(by_id[item_id]),
            )
            for item_id in included
        },
        conditions=arm.conditions,
        repeats=arm.repeats,
        renderer=renderer,
        contexts=observation_contexts,
    )
    record = RosterRecord(
        phase=scope.phase,
        benchmark_version=view.benchmark_version,
        corpus_version=view.corpus_version,
        development_items=view.output_items,
        scope_sha256=sha256_bytes(canonical_json_bytes(scope.model_dump(mode="json"))),
        resolved_config_sha256=renderer.config.configuration_sha256,
        matrix_arm=settings.matrix_arm,
        rows=rows,
    )
    return BuiltRoster(record, entries, roster_sha256(record))
