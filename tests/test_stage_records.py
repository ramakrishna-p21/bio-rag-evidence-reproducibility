"""The identity, publication and read-back shared by the Stage A, B and C records.

A tiny synthetic record stands in for a stage record. The stage tests exercise the same code
through the real records; this file pins the parts they do not name: the layout of the
published paths, what the build manifest says, and that a stored record is accepted only under
the identity it was published with.
"""

import pytest
from pydantic import BaseModel

from biorag_repro.core.hashing import HashMismatchError, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_manifest
from biorag_repro.evaluation.stage_records import publish_record, read_record, stage_record_ref


class Tiny(BaseModel):
    note: str


LINEAGE = [
    ("context", ArtifactRef(artifact_id="ctx_1", logical_ref="a/ctx.json", sha256="1" * 64)),
    ("item", ArtifactRef(artifact_id="item_1", logical_ref="a/item.json", sha256="2" * 64)),
]
TRANSFORMATION = "stage_x_tiny_v1"


def published(root, note="alpha"):
    value = Tiny(note=note)
    ref = stage_record_ref(value, "stage_x_tiny", "stage_x", LINEAGE, TRANSFORMATION)
    return (
        value,
        ref,
        publish_record(value, ref, "stage_x_tiny", "stage_x", LINEAGE, TRANSFORMATION, root),
    )


def test_a_record_is_identified_by_its_bytes_and_lineage_under_its_stage_directory():
    value = Tiny(note="alpha")
    ref = stage_record_ref(value, "stage_x_tiny", "stage_x", LINEAGE, TRANSFORMATION)
    assert ref.logical_ref == f"evaluation/stage_x/stage_x_tiny/{ref.artifact_id}.json"
    assert ref.sha256 == sha256_bytes(b'{\n  "note": "alpha"\n}\n')
    other_lineage = stage_record_ref(value, "stage_x_tiny", "stage_x", LINEAGE[:1], TRANSFORMATION)
    other_bytes = stage_record_ref(
        Tiny(note="beta"), "stage_x_tiny", "stage_x", LINEAGE, TRANSFORMATION
    )
    other_transformation = stage_record_ref(value, "stage_x_tiny", "stage_x", LINEAGE, "other_v1")
    assert len({ref.artifact_id, other_lineage.artifact_id, other_bytes.artifact_id}) == 3
    assert other_transformation.artifact_id != ref.artifact_id


def test_publication_writes_the_record_and_one_manifest_that_describes_it(tmp_path):
    _, ref, (record, build) = published(tmp_path)
    assert record == ref
    assert (tmp_path / ref.logical_ref).read_bytes() == b'{\n  "note": "alpha"\n}\n'
    assert build.logical_ref == f"evaluation/stage_x/build/{ref.artifact_id}.build.json"
    manifest = read_manifest(tmp_path / build.logical_ref, build.sha256)
    assert manifest.transformation_id == TRANSFORMATION
    assert manifest.status == "complete"
    assert [(i.role, i.artifact) for i in manifest.inputs] == LINEAGE
    (output,) = manifest.outputs
    assert (output.artifact, output.artifact_kind, output.output_format) == (
        ref,
        "stage_x_tiny",
        "json",
    )


def test_a_stored_record_is_read_back_only_under_the_identity_it_was_published_with(tmp_path):
    value, ref, _ = published(tmp_path)

    def ref_of(record):
        return stage_record_ref(record, "stage_x_tiny", "stage_x", LINEAGE, TRANSFORMATION)

    assert read_record(ref, tmp_path, Tiny, ref_of, "tiny record") == value

    def ref_with_other_lineage(record):
        return stage_record_ref(record, "stage_x_tiny", "stage_x", LINEAGE[:1], TRANSFORMATION)

    with pytest.raises(ValueError, match="stored tiny record does not match its reference"):
        read_record(ref, tmp_path, Tiny, ref_with_other_lineage, "tiny record")


def test_altered_stored_bytes_are_refused_before_any_identity_is_computed(tmp_path):
    _, ref, _ = published(tmp_path)
    target = tmp_path / ref.logical_ref
    target.chmod(0o644)
    target.write_bytes(b'{"note":"omega"}\n')
    with pytest.raises(HashMismatchError):
        read_record(ref, tmp_path, Tiny, lambda record: ref, "tiny record")
