"""Prospective simple random sampling of proposed accepted benchmark items."""

import random
import sys
from typing import Self

from pydantic import Field, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record


class AuditDraw(Record):
    schema_version: str = "1"
    seed_name: str = "benchmark_audit"
    seed: int
    algorithm: str = "python_random_sample_v1"
    python_version: str
    eligible_count: int
    eligible_ids_sha256: str
    selected_ids: list[str] = Field(default_factory=list)


class AuditedItem(Record):
    item_id: str
    decision_event_id: str
    defect_found: bool
    finding: str
    repair_event_id: str | None = None


class ExpandedReview(Record):
    trigger_item_id: str
    stratum: str
    item_ids: list[str]
    decision_event_ids: list[str]


class AuditReport(Record):
    schema_version: str = "1"
    draw: AuditDraw
    proposed_item_ids: list[str]
    audited_items: list[AuditedItem]
    expanded_reviews: list[ExpandedReview] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_draw(self) -> Self:
        ids = sorted(self.proposed_item_ids)
        if len(ids) != len(set(ids)) or self.draw.eligible_count != len(ids):
            raise ValueError("audit proposed population must contain unique IDs")
        if self.draw.eligible_ids_sha256 != sha256_bytes(canonical_json_bytes(ids)):
            raise ValueError("audit population hash differs from proposed IDs")
        if self.draw != sample_audit(ids, seed=self.draw.seed):
            raise ValueError("audit draw does not replay from the recorded seed and population")
        audited = [item.item_id for item in self.audited_items]
        if set(audited) != set(self.draw.selected_ids) or len(audited) != len(set(audited)):
            raise ValueError("every selected audit item needs one finding")
        if any(item.defect_found and not item.repair_event_id for item in self.audited_items):
            raise ValueError("audited defect needs repair or exclusion provenance")
        return self


def sample_audit(accepted_ids: list[str], *, seed: int) -> AuditDraw:
    """Draw uniformly without replacement from all stable sorted accepted IDs."""
    ids = sorted(accepted_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("audit population contains duplicate item IDs")
    return AuditDraw(
        seed=seed,
        python_version=sys.version.split()[0],
        eligible_count=len(ids),
        eligible_ids_sha256=sha256_bytes(canonical_json_bytes(ids)),
        selected_ids=random.Random(seed).sample(ids, min(100, len(ids))),
    )
