import json
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError
from test_manifest import manifest_data

from biorag_repro.core.hashing import HashMismatchError, sha256_bytes, sha256_file
from biorag_repro.core.manifest import BuildManifest
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    serialize_manifest,
    verify_manifest_outputs,
)


def valid_manifest():
    return BuildManifest.model_validate(manifest_data())


def test_stored_json_is_deterministic_utf8_and_has_one_final_newline():
    manifest = valid_manifest()
    encoded = serialize_manifest(manifest)
    assert encoded == serialize_manifest(manifest)
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert b'"schema_version": "1"' in encoded
    assert b"\n  " in encoded
    assert json.loads(encoded)["status"] == "complete"


def test_unicode_round_trip_and_exact_stored_byte_hash(tmp_path):
    data = manifest_data("failed")
    data["failure_reason"] = "Greek letter β was absent"
    manifest = BuildManifest.model_validate(data)
    encoded = serialize_manifest(manifest)
    assert "β".encode() in encoded
    target = tmp_path / "build.json"
    target.write_bytes(encoded)
    assert read_manifest(target, sha256_bytes(encoded)) == manifest
    assert sha256_file(target) == sha256_bytes(encoded)


@pytest.mark.parametrize(
    "content", [b"{", b'{"schema_version":"1"}', b'{"x": NaN}', b'{"x": Infinity}']
)
def test_read_rejects_malformed_schema_invalid_or_nonfinite_json(tmp_path, content):
    target = tmp_path / "invalid.json"
    target.write_bytes(content)
    with pytest.raises((ValueError, ValidationError)):
        read_manifest(target, sha256_bytes(content))


def test_serialize_revalidates_mutated_model_and_rejects_nonfinite():
    manifest = valid_manifest()
    manifest.inputs[0].artifact.sha256 = "bad"
    with pytest.raises((ValidationError, PydanticSerializationError)):
        serialize_manifest(manifest)
    manifest = valid_manifest()
    manifest.failure_reason = float("nan")  # type: ignore[assignment]
    with pytest.raises(PydanticSerializationError):
        serialize_manifest(manifest)


def test_publication_is_no_clobber_and_cleans_temporary_file(tmp_path):
    target = tmp_path / "artifact.bin"
    digest = publish_bytes(target, b"complete bytes", sha256_bytes(b"complete bytes"))
    assert digest == sha256_file(target)
    assert target.read_bytes() == b"complete bytes"
    assert list(tmp_path.iterdir()) == [target]
    with pytest.raises(FileExistsError):
        publish_bytes(target, b"other", sha256_bytes(b"other"))
    assert target.read_bytes() == b"complete bytes"
    assert list(tmp_path.iterdir()) == [target]


def test_integrity_mismatch_never_publishes_target(tmp_path):
    target = tmp_path / "artifact.bin"
    with pytest.raises(HashMismatchError):
        publish_bytes(target, b"wrong", sha256_bytes(b"expected"))
    assert list(tmp_path.iterdir()) == []


def test_temporary_file_is_in_target_directory_and_cleaned_on_validation_failure(tmp_path):
    target = tmp_path / "artifact.bin"
    seen = []

    def reject(temp: Path):
        seen.append(temp.parent)
        assert temp.read_bytes() == b"abc"
        raise ValueError("invalid stored content")

    with pytest.raises(ValueError, match="invalid stored content"):
        publish_bytes(target, b"abc", sha256_bytes(b"abc"), validate_temp=reject)
    assert seen == [tmp_path]
    assert list(tmp_path.iterdir()) == []


def test_hard_link_failure_is_explicit_and_cleans_temp(tmp_path):
    target = tmp_path / "artifact.bin"
    with (
        patch("biorag_repro.core.storage.os.link", side_effect=OSError("hard links unavailable")),
        pytest.raises(OSError, match="hard links unavailable"),
    ):
        publish_bytes(target, b"abc", sha256_bytes(b"abc"))
    assert list(tmp_path.iterdir()) == []


def test_complete_manifest_requires_verified_published_output(tmp_path):
    manifest = valid_manifest()
    target = tmp_path / "build.json"
    output = tmp_path / "output.json"
    logical_ref = manifest.outputs[0].artifact.logical_ref
    with pytest.raises(ValueError):
        publish_manifest(manifest, target, {})
    assert not target.exists()
    output.write_bytes(b"wrong")
    with pytest.raises(HashMismatchError):
        publish_manifest(manifest, target, {logical_ref: output})
    assert not target.exists()
    output.write_bytes(b"")
    with pytest.raises(HashMismatchError):
        verify_manifest_outputs(manifest, {logical_ref: output})
    assert not target.exists()


def test_complete_manifest_publishes_after_output_integrity_check(tmp_path):
    data = manifest_data()
    content = b"canonical document"
    data["outputs"][0]["artifact"]["sha256"] = sha256_bytes(content)
    from biorag_repro.core.identity import ArtifactInput, derived_artifact_id

    output = data["outputs"][0]
    source = data["inputs"][0]["artifact"]
    output["artifact"]["artifact_id"] = derived_artifact_id(
        artifact_kind=output["artifact_kind"],
        inputs=[ArtifactInput("source", source["artifact_id"], source["sha256"])],
        transformation_id=data["transformation_id"],
        transformation_settings_sha256=data["transformation_settings_sha256"],
        output_sha256=output["artifact"]["sha256"],
        output_format=output["output_format"],
        serialization_version=output["serialization_version"],
        identity_version=output["identity_version"],
    )
    manifest = BuildManifest.model_validate(data)
    output_path = tmp_path / "output.json"
    publish_bytes(output_path, content, sha256_bytes(content))
    target = tmp_path / "build.json"
    digest = publish_manifest(manifest, target, {output["artifact"]["logical_ref"]: output_path})
    assert digest == sha256_file(target)
    assert read_manifest(target, digest) == manifest


def test_failed_manifest_retains_verified_partial_output(tmp_path):
    data = manifest_data("failed")
    manifest = BuildManifest.model_validate(data)
    target = tmp_path / "failed.json"
    with pytest.raises(FileNotFoundError):
        publish_manifest(
            manifest, target, {manifest.outputs[0].artifact.logical_ref: tmp_path / "missing"}
        )
    assert not target.exists()


def test_explicit_identical_reuse_checks_bytes_and_never_rewrites(tmp_path):
    target = tmp_path / "record.json"
    content = b"complete immutable publication\n"
    digest = sha256_bytes(content)
    publish_bytes(target, content, digest)
    before = target.stat()
    checked = []
    assert (
        publish_bytes(
            target,
            content,
            digest,
            reuse_identical=True,
            validate_temp=lambda path: checked.append(path.read_bytes()),
        )
        == digest
    )
    assert checked == [content]
    assert target.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError):
        publish_bytes(target, b"altered", digest, reuse_identical=True)
    target.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        publish_bytes(target, content, digest, reuse_identical=True)


def test_identical_reuse_refuses_symlinks(tmp_path):
    original = tmp_path / "original"
    original.write_bytes(b"same")
    link = tmp_path / "link"
    link.symlink_to(original)
    with pytest.raises(ValueError):
        publish_bytes(link, b"same", sha256_bytes(b"same"), reuse_identical=True)
