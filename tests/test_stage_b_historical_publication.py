"""Versioned historical Stage B accounting keeps delivery and provenance separate."""

import pytest

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.evaluation.stage_b_historical_publication import (
    APPROVAL_SHA256,
    CASES_SHA256,
    CONTEXT_AMENDMENT_SHA256,
    PACKAGE_SHA256,
    PROVENANCE_AMENDMENT_SHA256,
    HistoricalStageBAccounting,
)


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, logical_ref=f"test/{name}.json", sha256="a" * 64)


def accounting(case_number: int) -> dict:
    failed = case_number == 14
    return {
        "case_number": case_number,
        "judgment": ref("judgment").model_dump(),
        "judgment_manifest": ref("manifest").model_dump(),
        "observation_keys": ["obs_1"],
        "item_id": "q_1",
        "paper_id": "P001",
        "representation": "JATS",
        "source_sha256": "b" * 64,
        "canonical_sha256": "c" * 64,
        "context_sha256": "d" * 64,
        "source_authenticity": "PASS",
        "exact_context_delivery": "PASS",
        "evidence_sufficiency": "PASS",
        "selected_context_provenance": "FAILED" if failed else "PASS",
        "complete_set_provenance": "FAILED",
        "complete_set_failure_count": 1,
        "selected_failures": [
            {
                "chunk_id": "chunk_1",
                "occurrence_id": "occurrence_1",
                "declared_node_id": "node_1",
                "owner_node_id": "node_2",
                "canonical_sha256": "c" * 64,
                "source_artifact_id": "source_1",
                "source_sha256": "b" * 64,
            }
        ]
        if failed
        else [],
        "researcher_disposition": "APPROVED" if failed else "NOT_NEEDED",
        "approval_sha256": APPROVAL_SHA256,
        "package_sha256": PACKAGE_SHA256,
        "historical_readback_sha256": CASES_SHA256,
        "provenance_amendment_sha256": PROVENANCE_AMENDMENT_SHA256,
        "context_amendment_sha256": CONTEXT_AMENDMENT_SHA256,
    }


def test_yes_with_failed_selected_provenance_is_explicit():
    value = HistoricalStageBAccounting.model_validate(accounting(14))
    assert value.evidence_sufficiency == "PASS"
    assert value.selected_context_provenance == "FAILED"
    with pytest.raises(ValueError, match="selected-context accounting"):
        HistoricalStageBAccounting.model_validate(
            accounting(14) | {"selected_context_provenance": "PASS"}
        )


def test_complete_set_failure_can_coexist_with_valid_selected_context():
    value = HistoricalStageBAccounting.model_validate(accounting(1))
    assert value.selected_context_provenance == "PASS"
    assert value.complete_set_provenance == "FAILED"
    with pytest.raises(ValueError, match="complete-set accounting"):
        HistoricalStageBAccounting.model_validate(
            accounting(1) | {"complete_set_provenance": "PASS"}
        )
