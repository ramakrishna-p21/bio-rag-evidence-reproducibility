"""Completeness of a final-test campaign, judged from observation identities.

Non-executable. The input is what a verified run store reports for each batch: its
`ObservationState` list (`generation.attempts`) and the identity of the run, together with the
registry verdict for the structural batches. This module reuses the store's own attempt
statuses and adds no state machine of its own; it also never asks for an observation to be
regenerated. Counting files proves nothing: a key is accounted for only if the state reported
for it carries exactly the coordinates and run identity that produce that key.

Trust limits: the result is only as good as the verified store behind it and the registry head,
which the batch layout cannot independently establish; `limits` says so whenever it applies.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from biorag_repro.core.identity import observation_key
from biorag_repro.finaltest.batches import batch_id_for
from biorag_repro.finaltest.campaign import CampaignPlan, expected_keys, plan_sha256
from biorag_repro.finaltest.inventory import HEAD_NOT_ESTABLISHED_TEXT, RootVerdict
from biorag_repro.generation.attempts import ObservationState

VERIFIED: Final = "verified_complete"
MISSING_RECORDED: Final = "missing_recorded"  # failed or retries exhausted, attempts recorded
NEVER_ATTEMPTED: Final = "never_attempted"
INTERRUPTED: Final = "interrupted_recoverable"  # eligible only for a future authorized resume

_INCOMPLETE_KINDS: Final = ("batch_closed_incomplete", "registry_incomplete")
_CLASS = {
    "complete": VERIFIED,
    "terminal_failure": MISSING_RECORDED,
    "retries_exhausted": MISSING_RECORDED,
    "planned": NEVER_ATTEMPTED,
    "retryable": INTERRUPTED,
    "running": INTERRUPTED,
}


@dataclass(frozen=True)
class ObservedBatch:
    """The identity of a verified run and the states it reports."""

    phase: str
    benchmark_version: str
    matrix_arm: str
    resolved_config_sha256: str
    states: Sequence[ObservationState]


@dataclass(frozen=True)
class Anomaly:
    kind: str
    subject: str
    detail: str


@dataclass(frozen=True)
class CampaignCompleteness:
    verdict: str  # complete_all_observed, complete_with_recorded_missing, incomplete, invalid
    per_arm: Mapping[str, Mapping[str, int]]
    outcomes: Mapping[str, str]  # expected key -> class
    anomalies: tuple[Anomaly, ...]
    limits: tuple[str, ...] = field(default_factory=tuple)


def check_campaign(
    plan: CampaignPlan,
    item_ids: Sequence[str],
    root_id: str,
    registry: RootVerdict,
    observed: Mapping[str, Sequence[ObservedBatch]],
) -> CampaignCompleteness:
    """`observed` maps a matrix arm to the verified runs found for it (normally one)."""
    anomalies: list[Anomaly] = []
    limits: list[str] = []
    open_problems = [p for p in registry.problems if p != HEAD_NOT_ESTABLISHED_TEXT]
    if open_problems:
        # unfinished publications leave the campaign incomplete; anything else is untrustworthy
        unfinished = all(p.startswith("incomplete:") for p in open_problems)
        kind = "registry_incomplete" if unfinished else "registry_not_verified"
        anomalies.append(Anomaly(kind, registry.root_id, registry.code))
    if registry.head_authority != "genesis_anchored":
        limits.append("the latest registry head is not independently established")
    plan_sha = plan_sha256(plan)

    outcomes: dict[str, str] = {}
    per_arm: dict[str, Counter[str]] = {}
    structural_incomplete = False
    for batch in plan.batches:
        arm = batch.matrix_arm
        expected = set(expected_keys(plan, item_ids, arm))
        record = registry.batches.get(batch_id_for(root_id, plan_sha, arm))
        if record is None or record.state not in ("materialized", "completed"):
            structural_incomplete = True
            if record is not None and record.state == "closed_incomplete":
                anomalies.append(
                    Anomaly("batch_closed_incomplete", arm, "closed before completion")
                )
        seen: dict[str, str] = {}
        for run in observed.get(arm, ()):
            _check_run(plan, batch.resolved_config_sha256, arm, run, anomalies)
            for state in run.states:
                _account(plan, item_ids, arm, run, state, expected, seen, anomalies)
        for key in expected:
            outcomes[key] = seen.get(key, NEVER_ATTEMPTED)
        per_arm[arm] = Counter(outcomes[k] for k in expected)
    for arm in set(observed) - {b.matrix_arm for b in plan.batches}:
        anomalies.append(Anomaly("wrong_arm", arm, "observations for an arm outside the plan"))

    return CampaignCompleteness(
        verdict=_verdict(outcomes, anomalies, structural_incomplete),
        per_arm={arm: dict(counts) for arm, counts in per_arm.items()},
        outcomes=outcomes,
        anomalies=tuple(anomalies),
        limits=tuple(limits),
    )


def _check_run(
    plan: CampaignPlan, config: str, arm: str, run: ObservedBatch, anomalies: list[Anomaly]
) -> None:
    if (run.phase, run.benchmark_version) != ("final_test", plan.benchmark_version):
        anomalies.append(Anomaly("wrong_campaign", arm, f"{run.phase}/{run.benchmark_version}"))
    if run.matrix_arm != arm:
        anomalies.append(Anomaly("wrong_arm", arm, f"run is for {run.matrix_arm}"))
    if run.resolved_config_sha256 != config:
        anomalies.append(Anomaly("wrong_configuration", arm, run.resolved_config_sha256[:12]))


def _account(
    plan: CampaignPlan,
    item_ids: Sequence[str],
    arm: str,
    run: ObservedBatch,
    state: ObservationState,
    expected: set[str],
    seen: dict[str, str],
    anomalies: list[Anomaly],
) -> None:
    coordinates = state.coordinates
    recomputed = observation_key(
        phase=run.phase,
        benchmark_version=run.benchmark_version,
        item_id=coordinates.item_id,
        matrix_arm=coordinates.matrix_arm,
        representation_or_condition=coordinates.representation_or_condition,
        repeat_index=coordinates.repeat_index,
        resolved_config_sha256=run.resolved_config_sha256,
    )
    if recomputed != state.key:
        anomalies.append(Anomaly("identity_mismatch", state.key, "key differs from coordinates"))
        return
    if state.key not in expected:
        anomalies.append(
            Anomaly(_why_foreign(plan, item_ids, arm, state), state.key, "not expected")
        )
        return
    if state.key in seen:
        kind = (
            "duplicate_completion"
            if VERIFIED in (seen[state.key], _CLASS[state.status])
            else ("duplicate_state")
        )
        anomalies.append(Anomaly(kind, state.key, "reported more than once"))
        return
    seen[state.key] = _CLASS[state.status]


def _why_foreign(
    plan: CampaignPlan, item_ids: Sequence[str], arm: str, state: ObservationState
) -> str:
    coordinates = state.coordinates
    batch = next(b for b in plan.batches if b.matrix_arm == arm)
    cells = {c.condition: c.repeats for c in batch.cells}
    if coordinates.representation_or_condition not in cells:
        return "wrong_condition"
    if coordinates.repeat_index not in cells[coordinates.representation_or_condition]:
        return "wrong_repeat"
    if coordinates.item_id not in set(item_ids):
        return "foreign_key"
    return "wrong_configuration"


def _verdict(
    outcomes: Mapping[str, str], anomalies: Sequence[Anomaly], structural_incomplete: bool
) -> str:
    if any(a.kind not in _INCOMPLETE_KINDS for a in anomalies):
        return "invalid"
    classes = set(outcomes.values())
    if (
        structural_incomplete
        or any(a.kind in _INCOMPLETE_KINDS for a in anomalies)
        or classes - {VERIFIED, MISSING_RECORDED}
    ):
        return "incomplete"
    return "complete_all_observed" if classes == {VERIFIED} else "complete_with_recorded_missing"
