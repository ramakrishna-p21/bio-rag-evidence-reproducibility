"""The development accounting reader must retain every approved decision identity."""

import copy
from collections import Counter, defaultdict

import pytest

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.evaluation import stage_a_accounting_v2 as accounting


def _store(root, name, value):
    raw = canonical_json_bytes(value)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {"artifact_id": name.replace("/", "_"), "logical_ref": name, "sha256": sha256_bytes(raw)}


@pytest.fixture
def inventory(tmp_path, monkeypatch):
    root = tmp_path
    labels = [label for label, count in accounting.EXPECTED_LABELS.items() for _ in range(count)]
    decisions, coordinates, entries = [], {}, []
    approvals = set()
    for index, label in enumerate(labels, 1):
        handle = f"R{index:03d}"
        item_id = f"item_{index:03d}"
        paper_id = f"P{(index - 1) // 2:03d}"
        representation = "JATS" if index % 2 else "PDF_DOCLING"
        item_sha = sha256_bytes(item_id.encode())
        canonical = {"artifact_id": f"canonical_{index}"}
        rationale = f"Approved rationale {index}"
        evidence = [f"source_{index}"]
        decisions.append(
            {
                "handle": handle,
                "researcher_concurred_label": label,
                "researcher_concurred_rationale": rationale,
                "supporting_evidence_handles": evidence,
            }
        )
        coordinates[handle] = {
            "item_sha256": item_sha,
            "item": {"item_id": item_id, "paper_id": paper_id, "split": "development"},
            "representation": representation,
            "canonical": canonical,
        }
        entry = {
            "handle": handle,
            "item_id": item_id,
            "item_sha256": item_sha,
            "paper_id": paper_id,
            "representation": representation,
            "canonical": canonical,
            "classification": label,
            "rationale": rationale,
            "original_evidence_handles": evidence,
            "verification": "provenance_incomplete",
            "incomplete_reason": accounting.INCOMPLETE,
            "judgment": None,
            "resolution": None,
        }
        if index <= 106:
            resolution = None
            if handle not in {"R050", "R116", "R152"}:
                resolution = _store(root, f"sources/resolution_{handle}.json", {"handle": handle})
                approvals.add(resolution["sha256"])
            judgment = {
                "handle": handle,
                "item_sha256": item_sha,
                "label": label,
                "rationale": rationale,
                "canonical": canonical,
                "component_resolution": resolution,
            }
            entry.update(
                verification="verified",
                incomplete_reason=None,
                judgment=_store(root, f"sources/judgment_{handle}.json", judgment),
                resolution=resolution,
            )
        entries.append(entry)
    # The three original records are among the 106 verified entries.
    for handle in ("R116", "R152"):
        entry = entries[int(handle[1:]) - 1]
        entry.update(
            verification="verified",
            incomplete_reason=None,
            judgment=_store(
                root,
                f"sources/judgment_{handle}.json",
                {
                    "handle": handle,
                    "item_sha256": entry["item_sha256"],
                    "label": entry["classification"],
                    "rationale": entry["rationale"],
                    "canonical": entry["canonical"],
                    "component_resolution": None,
                },
            ),
            resolution=None,
        )
    # Replace two focused entries so the fixture still has 106 verified and 50 incomplete.
    for index in (104, 105):
        entry = entries[index]
        entry.update(
            verification="provenance_incomplete",
            incomplete_reason=accounting.INCOMPLETE,
            judgment=None,
            resolution=None,
        )
    by_paper = defaultdict(Counter)
    by_representation = defaultdict(Counter)
    for entry in entries:
        by_paper[entry["paper_id"]][entry["representation"] + "/" + entry["verification"]] += 1
        by_representation[entry["representation"]][entry["verification"]] += 1
    decision_ref = _store(root, "sources/decisions.json", {"records": decisions})
    coordinate_ref = _store(root, "sources/coordinates.json", coordinates)
    session_ref = _store(root, "sources/session.json", {"seed": 2026})
    authorization_ref = _store(
        root, "sources/authorization.json", {"approved_by": "Rama Krishna Pudota"}
    )
    monkeypatch.setattr(
        accounting.retrospective, "APPROVED_DECISIONS_SHA256", decision_ref["sha256"]
    )
    monkeypatch.setattr(accounting.retrospective, "SESSION_SHA256", session_ref["sha256"])
    monkeypatch.setattr(
        accounting.retrospective, "AUTHORIZATION_SHA256", authorization_ref["sha256"]
    )
    monkeypatch.setattr(
        accounting.retrospective, "APPROVED_COMPONENT_RESOLUTIONS", frozenset(approvals)
    )
    data = {
        "schema_version": "2",
        "protocol": accounting.PROTOCOL,
        "scope": "development_only",
        "status": "complete_accounting_not_complete_evidence_verification",
        "approved_decisions": decision_ref,
        "coordinate_map": coordinate_ref,
        "session_manifest": session_ref,
        "authorization": authorization_ref,
        "entries": entries,
        "by_paper": {k: dict(v) for k, v in by_paper.items()},
        "by_representation": {k: dict(v) for k, v in by_representation.items()},
    }
    path = root / "evaluation/stage_a/accounting_v2/inventory.json"

    def read(value):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json_bytes(value)
        path.write_bytes(raw)
        return accounting.read_inventory(path, sha256_bytes(raw))

    return data, read


def test_complete_inventory_readback(inventory):
    data, read = inventory
    assert read(data) == data
    assert Counter(e["verification"] for e in data["entries"]) == {
        "verified": 106,
        "provenance_incomplete": 50,
    }


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d["entries"][0].update(classification="absent"),
        lambda d: d["entries"][0].update(representation="PDF_DOCLING"),
        lambda d: d["entries"][106].update(verification="verified"),
        lambda d: d["entries"][0].update(judgment=None),
        lambda d: d["entries"][0].update(resolution=None),
        lambda d: d["authorization"].update(sha256="0" * 64),
        lambda d: d["session_manifest"].update(sha256="0" * 64),
        lambda d: d["by_representation"]["JATS"].update(verified=0),
        lambda d: d["entries"].pop(),
    ],
)
def test_accounting_tampering_is_rejected(inventory, change):
    data, read = inventory
    damaged = copy.deepcopy(data)
    change(damaged)
    with pytest.raises((ValueError, KeyError, TypeError)):
        read(damaged)
