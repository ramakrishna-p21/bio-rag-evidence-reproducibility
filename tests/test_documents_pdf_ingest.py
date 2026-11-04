"""Attempt and immutable lineage behavior without model inference."""

import json
from pathlib import Path
from types import SimpleNamespace as Item

import pytest

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import raw_source_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.pdf_ingest import (
    ConversionAttempt,
    canonicalize_saved_native,
    publish_pdf_result,
)


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
            json.dumps(
                {
                    "schema_name": "DoclingDocument",
                    "version": "1.10.0",
                    "body": {"self_ref": "#/body", "label": "unspecified", "children": []},
                    "texts": [],
                    "groups": [],
                    "pictures": [],
                    "tables": [],
                    "form_items": [],
                    "key_value_items": [],
                }
            )
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


def test_saved_native_reuse_records_original_attempt_without_new_conversion(tmp_path, monkeypatch):
    from biorag_repro.documents import pdf_ingest

    path, ref = source(tmp_path)
    result = Item(status=Item(value="success"), errors=[], pages=[], document=FakeDocument())
    monkeypatch.setattr(pdf_ingest, "TRANSFORMATION_ID", "docling_pdf_adapter_v1")
    original = publish_pdf_result(
        result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
    )
    monkeypatch.setattr(pdf_ingest, "TRANSFORMATION_ID", "docling_pdf_adapter_v3")
    renewed = canonicalize_saved_native(
        path, ref, "P001", original, "c" * 64, tmp_path, load_native=load_fake
    )
    assert renewed.status == "success" and renewed.build_status == "complete"
    assert renewed.native_artifact == original.native_artifact
    assert renewed.original_attempt_artifact.sha256 == sha256_bytes(
        (tmp_path / original.attempt_logical_ref).read_bytes()
    )
    assert renewed.canonical_artifact != original.canonical_artifact
    assert renewed.build_manifest != original.build_manifest
    from biorag_repro.documents.phase4 import validate_build

    checked = validate_build(renewed, tmp_path)
    assert checked.canonical_build_artifact is not None
    assert checked.attempt_artifact == renewed.original_attempt_artifact
    with pytest.raises(FileExistsError):
        canonicalize_saved_native(
            path, ref, "P001", original, "c" * 64, tmp_path, load_native=load_fake
        )


def test_omitted_native_object_prevents_complete_manifest(tmp_path):
    path, ref = source(tmp_path)

    class IncompleteNative:
        def save_as_json(self, target, **kwargs):
            Path(target).write_text(
                json.dumps(
                    {
                        "schema_name": "DoclingDocument",
                        "version": "1.10.0",
                        "body": {
                            "self_ref": "#/body",
                            "label": "unspecified",
                            "children": [{"$ref": "#/texts/0"}],
                        },
                        "texts": [
                            {
                                "self_ref": "#/texts/0",
                                "label": "text",
                                "text": "lost",
                                "orig": "lost",
                                "parent": {"$ref": "#/body"},
                                "children": [],
                                "prov": [],
                            }
                        ],
                        "groups": [],
                        "pictures": [],
                        "tables": [],
                        "form_items": [],
                        "key_value_items": [],
                    }
                )
            )

    result = Item(status=Item(value="success"), errors=[], pages=[], document=IncompleteNative())
    attempt = publish_pdf_result(
        result, path, ref, "P001", "c" * 64, "e" * 64, "f" * 64, tmp_path, load_native=load_fake
    )
    assert attempt.build_status == "failed"
    assert attempt.build_manifest is None
    assert "native object" in attempt.errors[0] or "native child" in attempt.errors[0]


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
