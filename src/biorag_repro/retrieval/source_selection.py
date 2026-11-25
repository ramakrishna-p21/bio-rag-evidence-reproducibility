"""Bounded source selection for retrieval, without publication approval claims."""

from pathlib import Path

from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.authorization import ScientificAuthority, validate_scientific_pair
from biorag_repro.documents.ingestion import Phase4GateIndex, Phase4Validation
from biorag_repro.retrieval.validation import CanonicalAuthority


def select_pair(
    *,
    corpus_root: Path,
    artifact_root: Path,
    release_path: Path,
    release_sha256: str,
    paper_id: str,
    jats_validation: ArtifactRef,
    pdf_validation: ArtifactRef,
    verification_revision: str,
) -> CanonicalAuthority:
    """Revalidate caller-selected lineage; this confers no human acceptance or authentication."""
    jats = read_project_json(
        artifact_root / jats_validation.logical_ref, jats_validation.sha256, Phase4Validation
    )
    pdf = read_project_json(
        artifact_root / pdf_validation.logical_ref, pdf_validation.sha256, Phase4Validation
    )
    if pdf.attempt_artifact is None or pdf.environment_sha256 is None:
        raise ValueError("PDF selection requires qualified attempt and environment")
    if pdf.asset_inventory_sha256 is None:
        raise ValueError("PDF selection requires qualified assets")
    scientific = ScientificAuthority(
        release_path=release_path,
        release_sha256=release_sha256,
        native_attempt=pdf.attempt_artifact,
        jats_configuration_sha256=jats.configuration_sha256,
        pdf_configuration_sha256=pdf.configuration_sha256,
        environment_sha256=pdf.environment_sha256,
        asset_inventory_sha256=pdf.asset_inventory_sha256,
    )
    # This is an in-memory evidence index, never a published complete gate.
    evidence = Phase4GateIndex(
        corpus_version="corpus_v1",
        paper_id=paper_id,
        corpus_release_sha256=release_sha256,
        implementation_git_sha=verification_revision,
        jats_validation=jats_validation,
        pdf_validation=pdf_validation,
    )
    validate_scientific_pair(
        evidence, artifact_root, corpus_root, scientific, verification_revision
    )
    return CanonicalAuthority(
        corpus_root, artifact_root, scientific, evidence, verification_revision
    )
