"""What a build manifest must agree on before a derived document is trusted."""

import pytest
from pydantic import BaseModel

from biorag_repro.core.hashing import HashMismatchError
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_manifest, read_manifest
from biorag_repro.documents.lineage import check_manifest
from biorag_repro.evaluation.stage_records import publish_record, stage_record_ref


class Tiny(BaseModel):
    note: str


INPUTS = [
    ("source", ArtifactRef(artifact_id="src_1", logical_ref="raw/a.xml", sha256="1" * 64)),
    ("native", ArtifactRef(artifact_id="nat_1", logical_ref="raw/a.json", sha256="2" * 64)),
]


@pytest.fixture
def built(tmp_path):
    value = Tiny(note="canonical")
    ref = stage_record_ref(value, "tiny", "stage_x", INPUTS, "tiny_v1")
    _, build = publish_record(value, ref, "tiny", "stage_x", INPUTS, "tiny_v1", tmp_path)
    return tmp_path, ref, build, [artifact for _, artifact in INPUTS]


def test_a_manifest_that_describes_the_output_and_inputs_is_accepted(built):
    root, output, build, inputs = built
    manifest = check_manifest(build, root, output, inputs)
    assert manifest.outputs[0].artifact == output


def test_a_build_reference_must_carry_the_manifest_hash_in_its_identity(built):
    root, output, build, inputs = built
    forged = build.model_copy(update={"artifact_id": "build_" + "0" * 64})
    with pytest.raises(ValueError, match="build manifest artifact identity mismatch"):
        check_manifest(forged, root, output, inputs)


def test_the_output_must_be_the_expected_one(built):
    root, output, build, inputs = built
    other = output.model_copy(update={"artifact_id": "other_output"})
    with pytest.raises(ValueError, match="build output lineage mismatch"):
        check_manifest(build, root, other, inputs)


@pytest.mark.parametrize(
    "inputs",
    [lambda i: i[:1], lambda i: i[::-1], lambda i: [*i, i[0]]],
    ids=["fewer", "order", "more"],
)
def test_the_inputs_must_be_exactly_the_expected_ones_in_order(built, inputs):
    root, output, build, expected = built
    with pytest.raises(ValueError, match="build input lineage mismatch"):
        check_manifest(build, root, output, inputs(expected))


def test_a_manifest_that_does_not_say_complete_is_refused(built):
    root, output, build, inputs = built
    manifest = read_manifest(root / build.logical_ref, build.sha256)
    failed = manifest.model_copy(update={"status": "failed", "failure_reason": "interrupted"})
    path = root / "failed.build.json"
    digest = publish_manifest(failed, path, {output.logical_ref: root / output.logical_ref})
    ref = ArtifactRef(artifact_id="build_" + digest, logical_ref="failed.build.json", sha256=digest)
    with pytest.raises(ValueError, match="complete build manifest is required"):
        check_manifest(ref, root, output, inputs)


def test_altered_output_bytes_are_refused(built):
    root, output, build, inputs = built
    target = root / output.logical_ref
    target.chmod(0o644)
    target.write_bytes(b'{\n  "note": "changed"\n}\n')
    with pytest.raises(HashMismatchError):
        check_manifest(build, root, output, inputs)
