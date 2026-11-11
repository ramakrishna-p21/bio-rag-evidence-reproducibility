"""Fresh deterministic reproduction, retained as pending human review."""

import json
import platform
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, read_project_json
from biorag_repro.documents.authorization import (
    ReproductionCandidate,
    ScientificAuthority,
    observed_clean_revision,
    publish_candidate,
    validate_scientific_pair,
)
from biorag_repro.documents.ingestion import (
    Phase4GateIndex,
    Phase4Validation,
    build_arm,
    frozen_source,
    load_frozen_paper,
    publish_validation,
    validate_build,
)
from biorag_repro.documents.pdf_ingest import ConversionAttempt, canonicalize_saved_native


def reproduce_pair(
    reference: Phase4GateIndex,
    reference_root: Path,
    output: Path,
    corpus: Path,
    authority: ScientificAuthority,
    repository: Path,
    *,
    load_native: Any = None,
) -> ArtifactRef:
    """Reproduce into a fresh directory; never rerun the Docling converter.

    The reference validations identify expected content, not a historical producer.
    This local observation does not authenticate itself or authorize a final gate.
    """
    revision = observed_clean_revision(repository)
    started = datetime.now(UTC).isoformat()
    expected = reference.model_copy(update={"implementation_git_sha": revision})
    validate_scientific_pair(expected, reference_root, corpus, authority, revision)
    if output.exists():
        raise FileExistsError("reproduction requires a fresh output root")
    _, paper = load_frozen_paper(
        corpus, authority.release_path, authority.release_sha256, "corpus_v1", reference.paper_id
    )
    reports = [
        read_project_json(reference_root / ref.logical_ref, ref.sha256, Phase4Validation)
        for ref in (reference.jats_validation, reference.pdf_validation)
    ]
    original = read_project_json(
        reference_root / authority.native_attempt.logical_ref,
        authority.native_attempt.sha256,
        ConversionAttempt,
    )
    # Copy only retained dependencies needed by saved-native reuse, not old canonical outputs.
    dependencies = [
        authority.native_attempt,
        original.native_artifact,
        original.native_build_manifest,
    ]
    output.mkdir(parents=True)
    for ref in dependencies:
        assert ref is not None
        verify_file_sha256(reference_root / ref.logical_ref, ref.sha256)
        target = output / ref.logical_ref
        target.parent.mkdir(parents=True, exist_ok=True)
        publish_bytes(target, (reference_root / ref.logical_ref).read_bytes(), ref.sha256)
    inventory = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": dict(sorted((d.metadata["Name"], d.version) for d in metadata.distributions())),
    }
    content = (json.dumps(inventory, sort_keys=True, indent=2) + "\n").encode()
    digest = sha256_bytes(content)
    runtime = ArtifactRef(
        artifact_id="runtime_" + digest,
        logical_ref=f"corpus/derived/runtime/{digest}.json",
        sha256=digest,
    )
    path = output / runtime.logical_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(path, content, digest)
    jats = build_arm(
        paper, paper.paper_id, "JATS", corpus, output, authority.jats_configuration_sha256
    )
    source_path, source = frozen_source(paper, "PDF_DOCLING", corpus)
    pdf = canonicalize_saved_native(
        source_path,
        source,
        paper.paper_id,
        original,
        authority.pdf_configuration_sha256,
        output,
        load_native=load_native,
        original_data_root=reference_root,
    )
    refs = []
    for built, old in zip((jats, pdf), reports, strict=True):
        if (
            built.canonical_artifact != old.canonical_artifact
            or built.build_manifest != old.build_manifest
        ):
            raise ValueError(
                "deterministic reproduction mismatch; preserve outputs for investigation"
            )
        checked = validate_build(built, output, corpus_root=corpus, corpus_version="corpus_v1")
        refs.append(publish_validation(checked, output))
    if observed_clean_revision(repository) != revision:
        raise ValueError("implementation changed during reproduction")
    candidate = ReproductionCandidate(
        reproduction_git_sha=revision,
        started_utc=started,
        ended_utc=datetime.now(UTC).isoformat(),
        runtime_inventory=runtime,
        corpus_version="corpus_v1",
        paper_id=paper.paper_id,
        corpus_release_sha256=authority.release_sha256,
        jats_validation=refs[0],
        pdf_validation=refs[1],
    )
    return publish_candidate(candidate, output, corpus, authority, revision)
