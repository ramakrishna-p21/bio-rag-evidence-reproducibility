"""Final-test evaluation: authorization, masked review queues and reviewer-record ingestion.

Nothing here decides a scientific label. Stage A/B/C labels still come from the existing
validators (`stage_a.validate_judgment`, `stage_b.validate_judgment`, `stage_c.adjudicate`).
This module adds what the final test needs around them:

- the researcher's evaluation authorization, which binds the evaluation protocol, the custody
  exception and the exact batch, and without which nothing protected is read;
- Stage B review records turned into v2 judgments, and the narrow rule for applying one review
  decision to another context (same item, Stage A judgment, exact context and included chunks);
- the masked Stage C adjudication queue and the application of its decisions.
"""

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from biorag_repro.benchmark.models import BenchmarkItem
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.records import Record
from biorag_repro.evaluation import stage_b, stage_c
from biorag_repro.evaluation.answer_scoring import normalized_answer
from biorag_repro.evaluation.stage_a import item_digest, stage_a_ref
from biorag_repro.evaluation.stage_b_models import (
    Citation,
    ComponentRef,
    StageBCandidate,
    StageBIdentity,
    StageBJudgment,
)
from biorag_repro.evaluation.stage_c_models import ADJUDICABLE, StageCScore, utc_stamp

RESEARCHER = "Rama Krishna Pudota"
PROTOCOL_FILE = "docs/frozen/EVALUATION_PROTOCOL_V1.md"
CUSTODY_FILE = "docs/frozen/CUSTODY_EXCEPTION_0f1251d9_V1.md"
AUTHORIZATION_FILE = "approvals/final_test_evaluation.json"
SCOPE_AMENDMENT_FILE = "docs/frozen/EVALUATION_PROTOCOL_V1_AMENDMENT_CONFIRMATORY_SCOPE_V1.md"
PLANNED_OBSERVATIONS = 4575


class EvaluationRefused(ValueError):
    """Final-test evaluation cannot proceed with the supplied records."""


# ---- authorization ---------------------------------------------------------------------------


class EvaluationAuthorization(Record):
    """The researcher's approval to evaluate one preserved final-test batch.

    It is written by the researcher, never by this code. It binds the exact protocol and
    custody-exception texts and the batch identities the custody exception describes.
    """

    schema_version: Literal["final_test_evaluation_v1"]
    decision: Literal["approved"]
    approved_by: Literal["Rama Krishna Pudota"]
    approved_utc: str
    protocol_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    custody_exception_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    campaign_id: str
    run_id: str
    drive_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    head_request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    new_head_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    observations: Literal[4575]
    evaluation_source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    # Absent in the original record: the complete matrix. A confirmatory-only release must be
    # named by the researcher's own record; the code never widens or narrows it.
    scope: Literal["complete_matrix", "confirmatory_only"] = "complete_matrix"
    scope_amendment_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    reliability_design: str = Field(min_length=1)
    ai_assistance: str = Field(min_length=1)


def read_authorization(
    path: Path, expected_sha256: str, repo: Path
) -> tuple[EvaluationAuthorization, ArtifactRef]:
    """The approved authorization, checked against its trusted hash and the local texts."""
    if not path.is_file():
        raise EvaluationRefused("no evaluation authorization has been recorded")
    data = path.read_bytes()
    if sha256_bytes(data) != expected_sha256:
        raise EvaluationRefused("evaluation authorization differs from its trusted SHA-256")
    try:
        value = EvaluationAuthorization.model_validate_json(data)
        utc_stamp(value.approved_utc)
    except ValueError as error:
        raise EvaluationRefused(f"evaluation authorization is not an approval: {error}") from None
    if (value.scope == "confirmatory_only") != (value.scope_amendment_sha256 is not None):
        raise EvaluationRefused("a confirmatory-only scope needs its approved amendment hash")
    texts = [(PROTOCOL_FILE, value.protocol_sha256), (CUSTODY_FILE, value.custody_exception_sha256)]
    if value.scope_amendment_sha256:
        texts.append((SCOPE_AMENDMENT_FILE, value.scope_amendment_sha256))
    for name, digest in texts:
        text = repo / name
        body = text.read_bytes() if text.is_file() else b""
        if sha256_bytes(body) != digest:
            raise EvaluationRefused(f"{name} differs from the approved text")
        if b"PENDING RESEARCHER APPROVAL" in body:
            raise EvaluationRefused(f"{name} is still marked pending approval")
    ref = ArtifactRef(
        artifact_id="evaluation_authorization_" + expected_sha256,
        logical_ref=AUTHORIZATION_FILE,
        sha256=expected_sha256,
    )
    return value, ref


def check_source(repo: Path, revision: str) -> None:
    """Evaluation runs only from the exact, clean code revision the researcher approved."""
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    if git("rev-parse", "HEAD") != revision:
        raise EvaluationRefused("evaluation code is not the approved evaluation source revision")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise EvaluationRefused("evaluation code has uncommitted tracked changes")


# ---- Stage B ---------------------------------------------------------------------------------

REVIEW_FIELDS = (
    "status",
    "reason",
    "rationale",
    "reviewer",
    "at_utc",
    "citations",
    "missing_components",
    "unresolved_reason",
    "flags",
    "conflict_citations",
)


def reuse_key(inputs: stage_b.Inputs) -> tuple[str, ...]:
    """Contexts with one key are the same review question: same item, Stage A judgment,
    exact supplied text and ordered included chunks (so every cited span stays valid)."""
    return (
        item_digest(inputs.item),
        stage_a_ref(inputs.stage_a).sha256,
        sha256_bytes(inputs.context.exact_context.encode("utf-8")),
        *inputs.context.included_chunk_ids,
    )


def _judgment(
    fields: dict[str, Any], candidate: StageBCandidate, inputs: stage_b.Inputs
) -> StageBJudgment:
    identity = {k: getattr(candidate, k) for k in StageBIdentity.model_fields}
    judged = StageBJudgment.model_validate(
        {
            **identity,
            **fields,
            "basis": "reviewer",
            "rules": "v2",
            "candidate": stage_b.stage_b_ref(candidate).model_dump(),
            "delivered": [],
        }
    )
    stage_b.validate_judgment(judged, inputs)
    return judged


def apply_review(
    review: StageBJudgment, source: stage_b.Inputs, inputs: stage_b.Inputs
) -> StageBJudgment:
    """The same review decision recorded for an identical context of another arm."""
    if review.rules != "v2" or review.basis != "reviewer":
        raise EvaluationRefused("only a v2 whole-context review can be applied")
    if reuse_key(source) != reuse_key(inputs):
        raise EvaluationRefused("contexts differ; the review cannot be reused")
    fields = review.model_dump(mode="json", include=set(REVIEW_FIELDS))
    return _judgment(fields, stage_b.evaluate(inputs), inputs)


def stage_b_view(inputs: stage_b.Inputs) -> tuple[dict[str, Any], dict[str, str]]:
    """Answer-free, representation-masked review view and its chunk-handle mapping."""
    candidate = stage_b.evaluate(inputs)
    if not stage_b.needs_review_v2(candidate, inputs):
        raise EvaluationRefused("this context has a deterministic v2 outcome")
    included = {cid: inputs.chunks[cid] for cid in inputs.context.included_chunk_ids}
    handles = {f"C{n}": cid for n, cid in enumerate(included, 1)}
    view = {
        "question": inputs.item.question,
        "evidence_sets": [
            [{"role": c.role, "text": c.text} for c in s.components]
            for s in inputs.item.evidence_sets
        ],
        "chunks": [
            {
                "chunk": handle,
                "designated_paper": included[cid].input.paper_id == inputs.item.paper_id,
                "text": included[cid].rendered_text,
            }
            for handle, cid in handles.items()
        ],
    }
    return view, handles


def judgment_from_record(
    record: dict[str, Any], handles: dict[str, str], inputs: stage_b.Inputs, reviewer: str
) -> StageBJudgment:
    """A completed reviewer record (quotes by chunk handle) as a validated v2 judgment."""

    def span(entry: dict[str, Any]) -> dict[str, Any]:
        if entry["chunk"] not in handles:
            raise EvaluationRefused(f"unknown chunk handle {entry['chunk']}")
        cited = stage_b.cite(
            inputs, handles[entry["chunk"]], entry["quote"], entry.get("occurrence")
        )
        return cited.model_dump()

    fields = {
        "status": record["status"],
        "reason": record.get("reason"),
        "rationale": record["rationale"],
        "reviewer": reviewer,
        "at_utc": record["at_utc"],
        "citations": [
            Citation(
                **span(c), set_index=c["set_index"], component_index=c["component_index"]
            ).model_dump()
            for c in record.get("citations", [])
        ],
        "missing_components": [
            ComponentRef(**m).model_dump() for m in record.get("missing_components", [])
        ],
        "unresolved_reason": record.get("unresolved_reason"),
        "flags": record.get("flags", []),
        "conflict_citations": [span(c) for c in record.get("conflict_citations", [])],
    }
    return _judgment(fields, stage_b.evaluate(inputs), inputs)


def review_order(keys: Iterable[str], seed: int) -> list[str]:
    """Seeded review order, independent of arm, representation and outcome."""
    order = sorted(set(keys))
    random.Random(seed).shuffle(order)
    return order


# ---- Stage C ---------------------------------------------------------------------------------


def decision_key(score: StageCScore, answer: Any) -> tuple[str, ...]:
    """One reviewer decision per item, rules, automatic reason and normalized answer.

    `interpreted` is empty for several adjudicable reasons, so the normalized answer is part
    of the key itself: different answers never share a decision."""
    return (
        score.item_id,
        score.item_sha256,
        score.scoring_rules,
        score.reason or "",
        *score.interpreted,
        json.dumps(normalized_answer(answer, score.answer_class), ensure_ascii=False),
    )


def needs_adjudication(score: StageCScore) -> bool:
    return (
        score.basis == "deterministic"
        and score.status == "non_evaluable"
        and score.reason in ADJUDICABLE
    )


def stage_c_queue(
    scores: list[StageCScore],
    items: dict[str, BenchmarkItem],
    seed: int,
    answers: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    """Masked entries (one per decision key) and the private handle -> keys mapping.

    An entry shows the frozen question, gold, answer class, source references, automatic
    reason, the scorer's reading and the distinct answer texts (`answers` maps an observation
    key to the parsed answer); never the arm, condition, repeat, Stage A/B or observation."""
    grouped: dict[tuple[str, ...], list[StageCScore]] = {}
    for s in scores:
        if needs_adjudication(s):
            grouped.setdefault(decision_key(s, answers[s.observation_key]), []).append(s)
    keyed = {json.dumps(list(k)): k for k in grouped}
    entries, mapping = [], {}
    for n, text in enumerate(review_order(keyed, seed), 1):
        key = keyed[text]
        first = grouped[key][0]
        item = items[first.item_id]
        handle = f"M{n:04d}"
        mapping[handle] = sorted(s.observation_key for s in grouped[key])
        entries.append(
            {
                "handle": handle,
                "question": item.question,
                "answer_class": item.answer_class,
                "gold": item.gold.model_dump(mode="json"),
                "source_references": {
                    "jats": [r.model_dump(mode="json") for r in item.jats_refs],
                    "pdf": [r.model_dump(mode="json") for r in item.pdf_refs],
                },
                "automatic_reason": first.reason,
                "automatic_detail": first.detail,
                "interpreted": list(first.interpreted),
                "answers": sorted({json.dumps(answers[s.observation_key]) for s in grouped[key]}),
            }
        )
    return entries, mapping


def apply_decisions(
    scores: list[StageCScore],
    mapping: dict[str, list[str]],
    decisions: dict[str, dict[str, Any]],
    reviewer: str,
) -> list[StageCScore]:
    """Scores with every recorded decision applied; undecided entries stay non-evaluable."""
    unknown = set(decisions) - set(mapping)
    if unknown:
        raise EvaluationRefused(f"decisions for unknown handles {sorted(unknown)[:5]}")
    by_observation = {}
    for handle, decision in decisions.items():
        if decision.get("status") == "non_evaluable":
            if not (decision.get("rationale") or "").strip():
                raise EvaluationRefused(f"{handle}: an unclear case keeps its recorded reason")
            continue
        for key in mapping[handle]:
            by_observation[key] = decision
    out = []
    for s in scores:
        decision = by_observation.get(s.observation_key)
        if decision is None or not needs_adjudication(s):
            out.append(s)
            continue
        out.append(
            stage_c.adjudicate(
                s,
                status=decision["status"],
                reviewer=reviewer,
                at_utc=decision["at_utc"],
                rationale=decision["rationale"],
            )
        )
    stage_c.check_consistent(out)
    return out
