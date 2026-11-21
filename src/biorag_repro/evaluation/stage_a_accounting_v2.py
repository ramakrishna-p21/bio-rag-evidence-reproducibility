"""Development-only accounting for the approved retrospective Stage A package."""

from collections import Counter, defaultdict
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes
from biorag_repro.evaluation import stage_a_retrospective as retrospective
from biorag_repro.evaluation.stage_a import item_digest, stage_a_ref
from biorag_repro.evaluation.stage_a_storage import publish_candidate, read_candidate
from biorag_repro.readiness.files import loads_strict

PROTOCOL = "researcher_approved_retrospective_accounting_v2"
INCOMPLETE = "retrospective_provenance_incomplete"
EXPECTED_LABELS = {
    "preserved": 150,
    "altered_but_sufficient": 3,
    "scientifically_corrupted": 1,
    "unresolved": 2,
}


def _verified_bytes(path: Path, expected: str) -> bytes:
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected:
        raise ValueError(f"content hash differs: {path}")
    return raw


def build_inventory(
    *,
    decisions: Path,
    coordinates: Path,
    session: Path,
    authorization: Path,
    verified_sources: dict[str, dict[str, str]],
    original_drafts: dict[str, Path],
    root: Path,
    amendment_sha256: str,
    approval_sha256s: list[str],
) -> dict:
    """Publish a dry-run inventory; never writes to the official destination."""
    from scripts.stage_a_review import load_analysis
    from scripts.transcribe_stage_a import source_ref

    raw_sources = {
        "approved_decisions": decisions.read_bytes(),
        "coordinate_map": coordinates.read_bytes(),
        "session_manifest": session.read_bytes(),
        "authorization": authorization.read_bytes(),
    }
    refs = {name: source_ref(raw) for name, raw in raw_sources.items()}
    if (
        refs["approved_decisions"].sha256,
        refs["session_manifest"].sha256,
        refs["authorization"].sha256,
    ) != (
        retrospective.APPROVED_DECISIONS_SHA256,
        retrospective.SESSION_SHA256,
        retrospective.AUTHORIZATION_SHA256,
    ):
        raise ValueError("retrospective source package differs from approved exact identities")
    source_rows = loads_strict(raw_sources["approved_decisions"])["records"]
    coordinate_rows = loads_strict(raw_sources["coordinate_map"])
    session_data = loads_strict(raw_sources["session_manifest"])
    handles = [f"R{i:03d}" for i in range(1, 157)]
    if [r["handle"] for r in source_rows] != handles or list(coordinate_rows) != handles:
        raise ValueError("accounting inventory requires the complete original 156 handles")
    if Counter(r["researcher_concurred_label"] for r in source_rows) != EXPECTED_LABELS:
        raise ValueError("frozen scientific classification counts differ")
    if (
        set(verified_sources) & set(original_drafts)
        or len(verified_sources) != 103
        or len(original_drafts) != 3
    ):
        raise ValueError("verified source roster differs from approved 103 plus three")
    if set(original_drafts) != {"R050", "R116", "R152"}:
        raise ValueError("original valid record identities differ")
    if {
        v["sha256"] for v in verified_sources.values()
    } - retrospective.APPROVED_COMPONENT_RESOLUTIONS:
        raise ValueError("a focused resolution is not registered")
    for ref, raw in zip(refs.values(), raw_sources.values(), strict=True):
        target = root / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_bytes(target, raw, ref.sha256, reuse_identical=True)
    entries = []
    for row in source_rows:
        handle = row["handle"]
        coordinate = coordinate_rows[handle]
        analysis = load_analysis(handle, session_data["seed"], coordinate)
        item, document, canonical = (
            analysis.source.item,
            analysis.source.document,
            analysis.source.canonical,
        )
        if coordinate["item_sha256"] != item_digest(item):
            raise ValueError("item identity differs")
        entry = {
            "handle": handle,
            "item_id": item.item_id,
            "item_sha256": coordinate["item_sha256"],
            "paper_id": item.paper_id,
            "representation": document.representation,
            "canonical": canonical.model_dump(mode="json"),
            "classification": row["researcher_concurred_label"],
            "rationale": row["researcher_concurred_rationale"],
            "original_evidence_handles": row["supporting_evidence_handles"],
            "verification": "provenance_incomplete",
            "incomplete_reason": INCOMPLETE,
            "judgment": None,
            "resolution": None,
        }
        if handle in verified_sources or handle in original_drafts:
            source = verified_sources.get(handle)
            path = Path(source["judgment"]) if source else original_drafts[handle]
            value = retrospective.RetrospectiveJudgment.model_validate(
                loads_strict(path.read_bytes())
            )
            if source:
                resolution_raw = _verified_bytes(Path(source["resolution"]), source["sha256"])
                resolution_ref = value.component_resolution
                if resolution_ref is None or resolution_ref.sha256 != source["sha256"]:
                    raise ValueError(f"{handle}: resolution reference differs")
                target = root / resolution_ref.logical_ref
                target.parent.mkdir(parents=True, exist_ok=True)
                publish_bytes(target, resolution_raw, resolution_ref.sha256, reuse_identical=True)
            else:
                resolution_ref = None
                if value.component_resolution is not None:
                    raise ValueError(f"{handle}: original record unexpectedly has a resolution")
            retrospective.validate_retrospective(value, item, document, canonical, root)
            candidate_ref = stage_a_ref(analysis.candidate)
            if (root / candidate_ref.logical_ref).exists():
                read_candidate(candidate_ref, root, item, document, canonical)
            else:
                publish_candidate(analysis.candidate, item, document, canonical, root)
            judgment_ref, _ = retrospective.publish_judgment(value, item, document, canonical, root)
            retrospective.read_judgment(judgment_ref, root, item, document, canonical)
            entry.update(
                verification="verified",
                incomplete_reason=None,
                judgment=judgment_ref.model_dump(mode="json"),
                resolution=resolution_ref.model_dump(mode="json") if resolution_ref else None,
            )
        entries.append(entry)
    by_paper = defaultdict(lambda: Counter())
    by_representation = defaultdict(lambda: Counter())
    for entry in entries:
        by_paper[entry["paper_id"]][entry["representation"] + "/" + entry["verification"]] += 1
        by_representation[entry["representation"]][entry["verification"]] += 1
    inventory = {
        "schema_version": "2",
        "protocol": PROTOCOL,
        "scope": "development_only",
        "status": "complete_accounting_not_complete_evidence_verification",
        "amendment_sha256": amendment_sha256,
        "approval_sha256s": sorted(approval_sha256s),
        "approved_decisions": refs["approved_decisions"].model_dump(mode="json"),
        "coordinate_map": refs["coordinate_map"].model_dump(mode="json"),
        "session_manifest": refs["session_manifest"].model_dump(mode="json"),
        "authorization": refs["authorization"].model_dump(mode="json"),
        "entries": entries,
        "by_paper": {k: dict(sorted(v.items())) for k, v in sorted(by_paper.items())},
        "by_representation": {
            k: dict(sorted(v.items())) for k, v in sorted(by_representation.items())
        },
    }
    raw = canonical_json_bytes(inventory)
    target = root / "evaluation/stage_a/accounting_v2/inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(target, raw, sha256_bytes(raw), reuse_identical=True)
    return inventory


def read_inventory(path: Path, expected_sha256: str) -> dict:
    """Read the version 2 accounting envelope without treating it as a Stage A judgment."""
    data = loads_strict(_verified_bytes(path, expected_sha256))
    entries = data.get("entries")
    if (
        data.get("protocol") != PROTOCOL
        or data.get("scope") != "development_only"
        or not isinstance(entries, list)
    ):
        raise ValueError("not a development retrospective accounting inventory")
    if [e.get("handle") for e in entries] != [f"R{i:03d}" for i in range(1, 157)]:
        raise ValueError("inventory does not account for exactly 156 original handles")
    if Counter(e.get("classification") for e in entries) != EXPECTED_LABELS:
        raise ValueError("inventory changes approved classifications")
    if (
        data.get("schema_version") != "2"
        or data.get("status") != "complete_accounting_not_complete_evidence_verification"
    ):
        raise ValueError("inventory has an unapproved accounting status")
    root = path.parents[3]
    decision_ref = ArtifactRef(**data["approved_decisions"])
    coordinate_ref = ArtifactRef(**data["coordinate_map"])
    session_ref = ArtifactRef(**data["session_manifest"])
    authorization_ref = ArtifactRef(**data["authorization"])
    if decision_ref.sha256 != retrospective.APPROVED_DECISIONS_SHA256:
        raise ValueError("inventory names another approved decision package")
    if (
        session_ref.sha256 != retrospective.SESSION_SHA256
        or authorization_ref.sha256 != retrospective.AUTHORIZATION_SHA256
    ):
        raise ValueError("inventory names another retrospective source or authorization")
    _verified_bytes(root / session_ref.logical_ref, session_ref.sha256)
    _verified_bytes(root / authorization_ref.logical_ref, authorization_ref.sha256)
    decisions = loads_strict(_verified_bytes(root / decision_ref.logical_ref, decision_ref.sha256))[
        "records"
    ]
    coordinates = loads_strict(
        _verified_bytes(root / coordinate_ref.logical_ref, coordinate_ref.sha256)
    )
    by_paper = defaultdict(lambda: Counter())
    by_representation = defaultdict(lambda: Counter())
    for e, decision in zip(entries, decisions, strict=True):
        if (
            e["handle"] != decision["handle"]
            or e["classification"] != decision["researcher_concurred_label"]
            or e["rationale"] != decision["researcher_concurred_rationale"]
            or e["original_evidence_handles"] != decision["supporting_evidence_handles"]
            or e["item_sha256"] != coordinates[e["handle"]]["item_sha256"]
            or e["item_id"] != coordinates[e["handle"]]["item"]["item_id"]
            or e["paper_id"] != coordinates[e["handle"]]["item"]["paper_id"]
            or coordinates[e["handle"]]["item"]["split"] != "development"
            or e["representation"] != coordinates[e["handle"]]["representation"]
            or e["canonical"] != coordinates[e["handle"]]["canonical"]
        ):
            raise ValueError(
                "inventory changes an approved scientific decision or coordinate identity"
            )
        if e.get("verification") == "verified":
            if not e.get("judgment") or e.get("incomplete_reason") is not None:
                raise ValueError("verified entry lacks its judgment")
            ref = ArtifactRef(**e["judgment"])
            judgment = loads_strict(_verified_bytes(root / ref.logical_ref, ref.sha256))
            if (
                judgment["handle"] != e["handle"]
                or judgment["item_sha256"] != e["item_sha256"]
                or judgment["label"] != e["classification"]
                or judgment["rationale"] != e["rationale"]
                or judgment["canonical"] != e["canonical"]
                or judgment.get("component_resolution") != e["resolution"]
            ):
                raise ValueError("verified judgment differs from its accounting entry")
            if e["resolution"] is not None:
                resolution_ref = ArtifactRef(**e["resolution"])
                if resolution_ref.sha256 not in retrospective.APPROVED_COMPONENT_RESOLUTIONS:
                    raise ValueError("verified entry names an unregistered resolution")
                _verified_bytes(root / resolution_ref.logical_ref, resolution_ref.sha256)
            elif e["handle"] not in {"R050", "R116", "R152"}:
                raise ValueError("focused verified entry has no approved resolution")
        elif e.get("verification") == "provenance_incomplete":
            if (
                e.get("judgment") is not None
                or e.get("resolution") is not None
                or e.get("incomplete_reason") != INCOMPLETE
            ):
                raise ValueError("incomplete entry masquerades as verified evidence")
        else:
            raise ValueError("unknown evidence-verification status")
        by_paper[e["paper_id"]][e["representation"] + "/" + e["verification"]] += 1
        by_representation[e["representation"]][e["verification"]] += 1
    if data["by_paper"] != {
        k: dict(sorted(v.items())) for k, v in sorted(by_paper.items())
    } or data["by_representation"] != {
        k: dict(sorted(v.items())) for k, v in sorted(by_representation.items())
    }:
        raise ValueError("inventory completeness counts differ from its entries")
    return data
