"""Stage C: is the generated answer scientifically correct for the designated item?

Scoring reads only the benchmark item and the generation result. It takes no
Stage A or Stage B input, so correctness cannot depend on evidence preservation
or delivery. Operational failures are missing, never incorrect; scientific
ambiguity is non-evaluable until a reviewer records a decision.
"""

from typing import Any

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import observation_key
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.evaluation.answer_scoring import Verdict, score_answer
from biorag_repro.evaluation.stage_a import item_digest
from biorag_repro.evaluation.stage_c_models import (
    ADJUDICABLE,
    GenerationResult,
    StageCScore,
    Status,
    utc_stamp,
)
from biorag_repro.evaluation.stage_records import stage_record_ref


def check_result(item: BenchmarkItem, result: GenerationResult) -> None:
    """Refuse a result that is not the designated item's observation."""
    if (result.benchmark_version, result.item_id, result.paper_id) != (
        item.benchmark_version,
        item.item_id,
        item.paper_id,
    ):
        raise ValueError("generation result is not for the designated benchmark item and paper")
    expected = observation_key(
        phase=result.phase,
        benchmark_version=result.benchmark_version,
        item_id=result.item_id,
        matrix_arm=result.matrix_arm,
        representation_or_condition=result.representation_or_condition,
        repeat_index=result.repeat_index,
        resolved_config_sha256=result.resolved_config_sha256,
    )
    if result.observation_key != expected:
        raise ValueError("observation key does not match the result's scientific coordinates")


def generation_digest(result: GenerationResult) -> str:
    return sha256_bytes(serialize_project_json(result, GenerationResult))


def _identity(item: BenchmarkItem, result: GenerationResult, rules: str) -> dict[str, Any]:
    return {
        "scoring_rules": rules,
        "benchmark_version": item.benchmark_version,
        "corpus_version": item.corpus_version,
        "item_id": item.item_id,
        "item_sha256": item_digest(item),
        "paper_id": item.paper_id,
        "answer_class": item.answer_class,
        "observation_key": result.observation_key,
        "repeat_index": result.repeat_index,
        "attempt_id": result.attempt_id,
        "generation_sha256": generation_digest(result),
        "output": result.output,
    }


def score(item: BenchmarkItem, result: GenerationResult, rules: str = "v1") -> StageCScore:
    """The automatic score. Gold is checked first; a missing generation is not an error.

    `rules` selects the scoring-rule version and defaults to "v1"; v2 is never implied.
    """
    check_result(item, result)
    if item.candidate_status != "accepted":
        v = Verdict("non_evaluable", "gold_unusable", "item is not an accepted benchmark item")
    elif result.status == "operational_failure":
        v = Verdict("missing", "operational_failure", result.failure or "")
    elif result.parse_status == "unparseable":
        v = Verdict("incorrect", "unparseable", "completed output could not be parsed")
    elif result.abstained:
        v = Verdict("incorrect", "abstained", "completed abstention on an answerable item")
    else:
        assert result.answer is not None
        v = score_answer(item.gold, result.answer, rules)
    return StageCScore(
        **_identity(item, result, rules),
        status=v.status,
        basis="deterministic",
        reason=v.reason,
        detail=v.detail,
        interpreted=list(v.interpreted),
    )


def adjudicate(
    automatic: StageCScore, *, status: Status, reviewer: str, at_utc: str, rationale: str
) -> StageCScore:
    """A reviewer's recorded decision on an unresolved scoring ambiguity."""
    if automatic.basis != "deterministic" or automatic.status != "non_evaluable":
        raise ValueError("a reviewer resolves only automatic non-evaluable scores")
    if automatic.reason not in ADJUDICABLE:
        raise ValueError(f"{automatic.reason} is not a reviewable scoring ambiguity")
    if status not in {"correct", "incorrect"}:
        raise ValueError("a reviewer decides correct or incorrect only")
    utc_stamp(at_utc)
    identity = {k: getattr(automatic, k) for k in _identity_fields()}
    return StageCScore(
        **identity,
        status=status,
        basis="reviewer",
        reason=None if status == "correct" else "adjudicated_incorrect",
        detail=f"adjudicated; automatic reason {automatic.reason}: {automatic.detail}",
        interpreted=automatic.interpreted,
        automatic=score_ref(automatic),
        rationale=rationale,
        reviewer=reviewer,
        at_utc=at_utc,
    )


def check_consistent(scores: list[StageCScore]) -> None:
    """Refuse different reviewer decisions on the same categorical answer to the same item.

    One reviewed proposition gets one decision across arms, conditions and repeats.
    The key is the scorer's normalized answer text, so this adds no equivalence of its own.
    """
    decisions: dict[tuple[str, ...], str] = {}
    for s in scores:
        if s.basis != "reviewer" or s.answer_class != "categorical_conclusion":
            continue
        key = (s.item_id, s.item_sha256, s.scoring_rules, *s.interpreted)
        if decisions.setdefault(key, s.status) != s.status:
            raise ValueError(f"different reviewer decisions for one answer to {s.item_id}")


def _identity_fields() -> list[str]:
    return [
        "scoring_rules",
        "benchmark_version",
        "corpus_version",
        "item_id",
        "item_sha256",
        "paper_id",
        "answer_class",
        "observation_key",
        "repeat_index",
        "attempt_id",
        "generation_sha256",
        "output",
    ]


def validate_score(value: StageCScore, item: BenchmarkItem, result: GenerationResult) -> None:
    """Scientific read-back: the record must be what scoring these inputs produces."""
    automatic = score(item, result, value.scoring_rules)
    if value.basis == "deterministic":
        expected = automatic
    else:
        if value.status not in {"correct", "incorrect"}:
            raise ValueError("a reviewer decides correct or incorrect only")
        expected = adjudicate(
            automatic,
            status=value.status,
            reviewer=value.reviewer or "",
            at_utc=value.at_utc or "",
            rationale=value.rationale or "",
        )
    if value != expected:
        raise ValueError("score differs from the evaluator output for these inputs")


def dependencies(value: StageCScore) -> list[tuple[str, ArtifactRef]]:
    items = [
        (
            "benchmark_item",
            ArtifactRef(
                artifact_id=value.item_id,
                logical_ref=f"benchmark/{value.benchmark_version}/items/{value.item_id}",
                sha256=value.item_sha256,
            ),
        ),
        (
            "generation_result",
            ArtifactRef(
                artifact_id=f"{value.observation_key}:{value.attempt_id}",
                logical_ref=f"generation/{value.observation_key}/{value.attempt_id}",
                sha256=value.generation_sha256,
            ),
        ),
    ]
    if value.output is not None:
        items.append(("generator_output", value.output))
    if value.automatic is not None:
        items.append(("automatic_score", value.automatic))
    return items


def transformation_id(value: StageCScore) -> str:
    return f"stage_c_score_{value.scoring_rules}"


def score_ref(value: StageCScore) -> ArtifactRef:
    """Identify a score by its bytes and the exact item, generation and output it came from."""
    return stage_record_ref(
        value, "stage_c_score", "stage_c", dependencies(value), transformation_id(value)
    )
