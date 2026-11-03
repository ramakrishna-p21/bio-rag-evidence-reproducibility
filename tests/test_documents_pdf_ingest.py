"""Attempt and immutable lineage behavior without model inference."""

import json
from pathlib import Path
from types import SimpleNamespace as Item

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.pdf_ingest import ConversionAttempt, publish_pdf_result


def source(tmp_path):
    content = b"%PDF-1.4 fixture"
    digest = sha256_bytes(content)
    ref = ArtifactRef(
        artifact_id=raw_source_artifact_id(paper_id="P001", representation="PDF", sha256=digest),
        logical_ref="corpus/raw/P001/PDF/source.pdf",
        sha256=digest,
    )
    path = tmp_path / ref.logical_ref
    path.parent.mkdir(parents=True)
    path.write_bytes(content)
    return path, ref


class FakeDocument:
    def save_as_json(self, path, **kwargs):
        Path(path).write_text(
            json.dumps({"schema_name": "DoclingDocument", "version": "1.10.0", "body": "fixture"})
        )


def load_fake(path):
    assert json.loads(Path(path).read_text())["schema_name"] == "DoclingDocument"
    body = Item(self_ref="#/body", parent=None, label="unspecified", prov=[], captions=[])
    return Item(iterate_items=lambda with_groups: iter([(body, 0)]))


def test_success_publishes_saved_native_then_canonical_and_manifest(tmp_path):
    path, ref = source(tmp_path)
    result = Item(status=Item(value="success"), errors=[], pages=[], document=FakeDocument())
    attempt = publish_pdf_result(
        result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
    )
    assert attempt.status == "success"
    assert attempt.build_status == "complete"
    assert attempt.native_artifact and attempt.canonical_artifact and attempt.build_manifest
    native = tmp_path / attempt.native_artifact.logical_ref
    canonical = tmp_path / attempt.canonical_artifact.logical_ref
    assert native.is_file() and canonical.is_file()
    assert (
        json.loads(canonical.read_text())["native_artifact"]["artifact_id"]
        == attempt.native_artifact.artifact_id
    )
    assert (tmp_path / attempt.build_manifest.logical_ref).is_file()
    with pytest.raises(FileExistsError):
        publish_pdf_result(
            result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
        )


@pytest.mark.parametrize("status", ["failure", "partial_success"])
def test_non_success_retains_attempt_without_complete_manifest(tmp_path, status):
    path, ref = source(tmp_path)
    result = Item(
        status=Item(value=status), errors=[Item(error_message="invalid")], pages=[], document=None
    )
    attempt = publish_pdf_result(
        result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
    )
    assert attempt.status == status
    assert attempt.build_status == ("failed" if status == "failure" else "incomplete")
    assert attempt.native_artifact is None and attempt.canonical_artifact is None
    assert attempt.build_manifest is None
    assert (
        ConversionAttempt.model_validate_json((tmp_path / attempt.attempt_logical_ref).read_text())
        == attempt
    )


def test_partial_with_native_preserves_native_without_canonical(tmp_path):
    path, ref = source(tmp_path)
    result = Item(
        status=Item(value="partial_success"), errors=[], pages=[], document=FakeDocument()
    )
    attempt = publish_pdf_result(
        result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
    )
    assert attempt.native_artifact and attempt.native_build_manifest
    assert attempt.canonical_artifact is None and attempt.build_manifest is None


def test_native_reload_failure_records_attempt_without_complete_manifest(tmp_path):
    path, ref = source(tmp_path)
    result = Item(status=Item(value="success"), errors=[], pages=[], document=FakeDocument())

    def broken_loader(_path):
        raise ValueError("invalid native structure")

    attempt = publish_pdf_result(
        result,
        path,
        ref,
        "P001",
        "c" * 64,
        "e" * 64,
        "f" * 64,
        tmp_path,
        load_native=broken_loader,
    )
    assert attempt.status == "success"
    assert attempt.build_status == "failed"
    assert "invalid native structure" in attempt.errors[0]
    assert (tmp_path / attempt.attempt_logical_ref).is_file()
    assert not list(tmp_path.rglob("*.build.json"))
