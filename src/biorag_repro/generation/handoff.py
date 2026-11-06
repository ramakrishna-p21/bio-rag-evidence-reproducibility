"""Analysis-table observations from a verified official run.

This is the only place where generation results become analysis-table input, and it
accepts nothing but a `VerifiedOfficialRun`: mock and smoke results, bare
`GenerationResult` lists and unverified runs are refused. Stage A and Stage B evidence is
supplied by the caller from the published judgments; it is passed through unchanged, so an
observation without a Stage B judgment (for example the recorded-placement non-deliveries
that stay unassessed) remains unassessed and gets no generator-input link.
"""

from collections.abc import Callable, Sequence

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.evaluation import stage_c
from biorag_repro.evaluation.analysis_table import Evidence, Observation
from biorag_repro.generation.official import (
    NotVerifiedOfficial,
    VerifiedOfficialRun,
    recorded_generator_input,
    row_key,
)

EvidenceFor = Callable[[str, str], Evidence | None]  # (item_id, condition) -> published judgments


def official_observations(
    verified: VerifiedOfficialRun, items: Sequence[BenchmarkItem], evidence_for: EvidenceFor
) -> list[Observation]:
    """One observation per rostered (item, condition, repeat), scored from the verified run."""
    if not isinstance(verified, VerifiedOfficialRun):
        raise NotVerifiedOfficial("only a verified official run can be exported")
    completed = verified.completed
    if completed.store.record.phase != "development":
        raise NotVerifiedOfficial(
            f"a {completed.store.record.phase} run is not a source of scientific observations"
        )
    by_item = {item.item_id: item for item in items}
    results = {r.observation_key: r for r in verified.results}
    retrieval = completed.store.config.settings.retrieval_config_sha256
    observations = []
    for row in completed.roster.rows:
        item = by_item.get(row.item_id)
        if item is None:
            raise NotVerifiedOfficial(f"rostered item {row.item_id} was not supplied")
        result = results.get(row_key(completed, row))
        if result is None:
            raise NotVerifiedOfficial(f"no decided result for rostered item {row.item_id}")
        evidence = evidence_for(row.item_id, row.condition)
        executed = result.status == "completed"
        has_context_judgment = evidence is not None and evidence.stage_b is not None
        observations.append(
            Observation(
                item=item,
                phase="development",
                matrix_arm=completed.completion.matrix_arm,  # type: ignore[arg-type]
                condition=row.condition,  # type: ignore[arg-type]
                repeat_index=row.repeat_index,
                resolved_config_sha256=completed.completion.resolved_config_sha256,
                retrieval_config_sha256=retrieval,
                result=result,
                score=stage_c.score(item, result),  # a missing generation scores as missing
                evidence=evidence,
                generator_input=(
                    recorded_generator_input(verified, result)
                    if executed and has_context_judgment and row.context is not None
                    else None
                ),
            )
        )
    return observations
