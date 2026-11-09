"""The only way final-test question text and retrieval contexts enter the start procedure.

`ProtectedInputs` is a narrow interface. In this version nothing in `src/` implements it for
real final-test content: the only implementations are synthetic and live in the tests, and
the public start path stops at the readiness guard before it could call one. A future reader
of the real benchmark items and the Phase 5 contexts has to be written, reviewed and enabled
separately.

`build_rosters` renders every observation with the same function the development roster uses
(`generation.roster.render_observations`) and keeps only digests: the returned
`ConditionRoster` records carry no question, gold answer or context. The question text is used
once, by the renderer; error messages here name identifiers and counts only.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.finaltest.campaign import PHASE, Population
from biorag_repro.finaltest.inputs import (
    CLOSED_BOOK,
    ConditionRoster,
    RosterIdentityRow,
    ordered_digest,
)
from biorag_repro.generation.prompt import InputRenderer
from biorag_repro.generation.roster import ItemFacts, ObservationContext, render_observations
from biorag_repro.generation.runner import RosterEntry
from biorag_repro.generation.seeds import GenerationSeedRegistry, repeat_seed


class ProtectedInputError(ValueError):
    """The protected inputs do not match the verified population. Messages carry ids only."""


@dataclass(frozen=True)
class ProtectedItem:
    item_id: str
    paper_id: str
    question: str  # protected: reaches only the renderer
    item_sha256: str


@dataclass(frozen=True)
class ProtectedContext:
    """The assembled context of one (item, representation), as recomputed by its reader."""

    exact_context: str  # protected
    generator_tokens: int
    evidence_ids: tuple[str, ...]  # in rank order
    source_identity_sha256: str  # identity of the raw source of the representation
    reference: ArtifactRef | None


class ProtectedInputs(Protocol):
    def items(self) -> Sequence[ProtectedItem]: ...

    def context(self, item_id: str, condition: str) -> ProtectedContext | None: ...


@dataclass(frozen=True)
class PreparedRosters:
    rosters: tuple[ConditionRoster, ...]  # in matrix condition order
    entries: tuple[RosterEntry, ...]  # exact generator inputs, kept in memory for the run


def build_rosters(
    *,
    inputs: ProtectedInputs,
    population: Population,
    matrix_arm: str,
    batch_id: str,
    renderer: InputRenderer,
    seeds: GenerationSeedRegistry | None,
) -> PreparedRosters:
    settings = renderer.config.settings
    if matrix_arm != settings.matrix_arm or matrix_arm not in FROZEN_MATRIX:
        raise ProtectedInputError("the configuration is not for this matrix arm")
    conditions, repeats = FROZEN_MATRIX[matrix_arm]
    items = {item.item_id: item for item in inputs.items()}
    if sorted(items) != sorted(population.final_test_item_ids):
        raise ProtectedInputError("the protected items are not the final-test population")
    for item in items.values():
        if population.item_papers[item.item_id] != item.paper_id:
            raise ProtectedInputError(f"item {item.item_id} names another paper")

    contexts: dict[tuple[str, str], ObservationContext | None] = {}
    found: dict[tuple[str, str], ProtectedContext] = {}
    for item_id in items:
        for condition in conditions:
            if matrix_arm == CLOSED_BOOK:
                if inputs.context(item_id, condition) is not None:
                    raise ProtectedInputError(f"closed-book item {item_id} carries a context")
                contexts[(item_id, condition)] = None
                continue
            context = inputs.context(item_id, condition)
            if context is None:
                raise ProtectedInputError(f"item {item_id} has no context for {condition}")
            found[(item_id, condition)] = context
            contexts[(item_id, condition)] = ObservationContext(
                exact_context=context.exact_context,
                generator_tokens=context.generator_tokens,
                reference=context.reference,
            )
    entries, _rows = render_observations(
        phase=PHASE,
        benchmark_version=population.benchmark_version,
        items={
            item_id: ItemFacts(item_id, item.paper_id, item.question, item.item_sha256)
            for item_id, item in items.items()
        },
        conditions=conditions,
        repeats=repeats,
        renderer=renderer,
        contexts=contexts,
    )
    by_condition: dict[str, list[RosterIdentityRow]] = {c: [] for c in conditions}
    for entry in entries:
        condition, item_id = entry.representation_or_condition, entry.item_id
        record = entry.input_record
        assert record is not None  # render_observations always stores the exact input
        context = found.get((item_id, condition))
        sampling = settings.decoding.do_sample
        by_condition[condition].append(
            RosterIdentityRow(
                item_id=item_id,
                paper_id=entry.paper_id,
                item_sha256=items[item_id].item_sha256,
                matrix_arm=matrix_arm,
                condition=condition,
                repeat_index=entry.repeat_index,
                resolved_config_sha256=renderer.config.configuration_sha256,
                seed=repeat_seed(seeds, entry.repeat_index) if sampling and seeds else None,
                source_identity_sha256=None if context is None else context.source_identity_sha256,
                context_sha256=entry.rendered.context_sha256,
                evidence_sequence_sha256=None
                if context is None
                else ordered_digest(context.evidence_ids),
                rendered_input_sha256=entry.rendered.rendered_sha256,
                observation_key=record.observation_key,
            )
        )
    rosters = tuple(
        ConditionRoster(
            batch_id=batch_id,
            matrix_arm=matrix_arm,
            condition=condition,
            benchmark_version=population.benchmark_version,
            resolved_config_sha256=renderer.config.configuration_sha256,
            rows=sorted(by_condition[condition], key=lambda r: (r.item_id, r.repeat_index)),
        )
        for condition in conditions
    )
    return PreparedRosters(rosters, tuple(entries))


def facts_by_key(prepared: PreparedRosters) -> Mapping[str, RosterIdentityRow]:
    return {row.observation_key: row for roster in prepared.rosters for row in roster.rows}
