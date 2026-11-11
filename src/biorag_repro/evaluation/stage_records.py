"""Identity, publication and read-back shared by the Stage A, B and C records.

Each stage keeps its own models, dependencies and scientific validation; what is common is how a
record is identified by its bytes and lineage, published with one build manifest, and read back.
"""

from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_project_json,
    serialize_project_json,
)

RecordT = TypeVar("RecordT", bound=BaseModel)
Dependencies = list[tuple[str, ArtifactRef]]


def stage_record_ref(
    value: BaseModel,
    kind: str,
    stage: str,
    dependencies: Dependencies,
    transformation_id: str,
) -> ArtifactRef:
    """Identify a record by its exact bytes and the lineage it was derived from."""
    digest = sha256_bytes(serialize_project_json(value, type(value)))
    artifact_id = derived_artifact_id(
        artifact_kind=kind,
        inputs=[ArtifactInput(role, ref.artifact_id, ref.sha256) for role, ref in dependencies],
        transformation_id=transformation_id,
        transformation_settings_sha256=None,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=artifact_id,
        logical_ref=f"evaluation/{stage}/{kind}/{artifact_id}.json",
        sha256=digest,
    )


def publish_record(
    value: BaseModel,
    ref: ArtifactRef,
    kind: str,
    stage: str,
    dependencies: Dependencies,
    transformation_id: str,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    """Write the record and its build manifest; returns (record, build manifest) references."""
    model = type(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, model),
        ref.sha256,
        validate_temp=lambda path: read_project_json(path, ref.sha256, model),
    )
    manifest = BuildManifest(
        schema_version="1",
        inputs=[BuildInput(role=r, artifact=a) for r, a in dependencies],
        transformation_id=transformation_id,
        outputs=[
            DerivedOutput(
                artifact=ref,
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )
    logical_ref = f"evaluation/{stage}/build/{ref.artifact_id}.build.json"
    (root / logical_ref).parent.mkdir(parents=True, exist_ok=True)
    digest = publish_manifest(manifest, root / logical_ref, {ref.logical_ref: target})
    return ref, ArtifactRef(artifact_id="build_" + digest, logical_ref=logical_ref, sha256=digest)


def read_record(
    ref: ArtifactRef,
    root: Path,
    model: type[RecordT],
    ref_of: Callable[[RecordT], ArtifactRef],
    label: str,
) -> RecordT:
    """Read the stored bytes and require them to carry exactly the identity `ref` names."""
    value = read_project_json(root / ref.logical_ref, ref.sha256, model)
    if ref_of(value) != ref:
        raise ValueError(f"stored {label} does not match its reference identity")
    return value
