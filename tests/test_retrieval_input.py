"""Scientific input selection is independent of the submitted record."""

from dataclasses import replace

import pytest
from test_publication_authorization import attack
from test_publication_authorization import authority as authority  # noqa: PLC0414
from test_publication_authorization import pair as pair  # noqa: PLC0414

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import BuildManifest
from biorag_repro.core.storage import read_manifest, read_project_json, serialize_project_json
from biorag_repro.retrieval.models import CanonicalInput
from biorag_repro.retrieval.storage import input_ref, publish_input
from biorag_repro.retrieval.validation import CanonicalAuthority, load_canonical_input


def selected(pair, representation="JATS"):
    report = pair.jr if representation == "JATS" else pair.pr
    return CanonicalInput(
        corpus_version="corpus_v1",
        corpus_release_sha256=pair.index.corpus_release_sha256,
        paper_id="P001",
        representation=representation,
        canonical=report.canonical_artifact,
        validation=pair.index.jats_validation
        if representation == "JATS"
        else pair.index.pdf_validation,
        build_manifest=report.build_manifest,
    )


def trusted(pair, authority):
    return CanonicalAuthority(
        corpus_root=pair.corpus,
        artifact_root=pair.output,
        scientific=authority,
        evidence=pair.index,
        reproduction_revision="f" * 40,
    )


@pytest.mark.parametrize("representation", ["JATS", "PDF_DOCLING"])
def test_positive_scientific_boundary_and_immutable_publication(
    pair, authority, tmp_path, representation
):
    value, context = selected(pair, representation), trusted(pair, authority)
    doc = load_canonical_input(value, context)
    assert (doc.paper_id, doc.representation) == ("P001", representation)
    out = tmp_path / "phase5"
    ref, manifest_ref = publish_input(value, context, out)
    assert ref == input_ref(value)
    assert read_project_json(out / ref.logical_ref, ref.sha256, CanonicalInput) == value
    manifest = read_manifest(out / manifest_ref.logical_ref, manifest_ref.sha256)
    assert manifest.status == "complete"
    assert [i.role for i in manifest.inputs] == [
        "canonical",
        "phase4_validation",
        "canonical_build_manifest",
    ]
    with pytest.raises(FileExistsError):
        publish_input(value, context, out)
    assert read_manifest(out / manifest_ref.logical_ref, manifest_ref.sha256) == manifest


@pytest.mark.parametrize(
    "field,item",
    [
        ("paper_id", "P002"),
        ("representation", "PDF_DOCLING"),
        ("corpus_version", "substitute"),
        ("corpus_release_sha256", "0" * 64),
    ],
)
def test_input_coordinates_cannot_override_independent_selection(pair, authority, field, item):
    value = selected(pair).model_copy(update={field: item})
    with pytest.raises(ValueError, match="selected|selection"):
        load_canonical_input(value, trusted(pair, authority))


@pytest.mark.parametrize(
    "case",
    ["release", "source", "role", "configuration", "validation_id", "warnings", "wrong_revision"],
)
def test_rehashed_phase4_counterfeits_fail_same_input_boundary(pair, authority, case):
    context = trusted(pair, authority)
    load_canonical_input(selected(pair), context)
    counterfeit = attack(pair, case)
    # Submit the renewed evidence as a probe to exercise relationship checks,
    # rather than relying on an unchanged outer hash to reject every attack.
    context = replace(context, evidence=counterfeit)
    with pytest.raises(ValueError):
        load_canonical_input(selected(pair), context)


def test_candidate_cannot_supply_release_authority(pair, authority):
    value = selected(pair)
    with pytest.raises(TypeError, match="independently"):
        load_canonical_input(value, {"scientific": authority, "evidence": pair.index})
    context = trusted(pair, authority)
    bad = replace(authority, release_sha256="0" * 64)
    with pytest.raises(ValueError, match="Phase 2H"):
        load_canonical_input(
            value.model_copy(update={"corpus_release_sha256": "0" * 64}),
            replace(context, scientific=bad),
        )


def test_incomplete_manifest_is_not_accepted_even_with_renewed_hash(pair, authority):
    context = trusted(pair, authority)
    load_canonical_input(selected(pair), context)
    ref = pair.jr.build_manifest
    manifest = read_manifest(pair.output / ref.logical_ref, ref.sha256)
    manifest.status = "failed"
    manifest.failure_reason = "interrupted"
    content = serialize_project_json(manifest, BuildManifest)
    altered = ref.model_copy(
        update={
            "artifact_id": "build_" + sha256_bytes(content),
            "logical_ref": "incomplete.json",
            "sha256": sha256_bytes(content),
        }
    )
    (pair.output / altered.logical_ref).write_bytes(content)
    from biorag_repro.documents.ingestion import publish_validation

    report = pair.jr.model_copy(update={"build_manifest": altered})
    index = pair.index.model_copy(
        update={"jats_validation": publish_validation(report, pair.output)}
    )
    value = selected(pair).model_copy(
        update={"validation": index.jats_validation, "build_manifest": altered}
    )
    with pytest.raises(ValueError, match="manifest"):
        load_canonical_input(value, replace(context, evidence=index))


def test_wrong_canonical_identity_cannot_be_selected(pair, authority):
    value = selected(pair)
    counterfeit = value.canonical.model_copy(update={"artifact_id": "drv_" + "0" * 64})
    with pytest.raises(ValueError, match="canonical|selection"):
        load_canonical_input(
            value.model_copy(update={"canonical": counterfeit}), trusted(pair, authority)
        )


def test_failed_or_interrupted_publication_never_exposes_complete_manifest(
    pair, authority, tmp_path, monkeypatch
):
    from biorag_repro.core import storage

    value, context = selected(pair), trusted(pair, authority)
    original = storage.os.link
    calls = []

    def interrupt(source, target):
        calls.append(target)
        if str(target).endswith(".build.json"):
            raise OSError("interrupted manifest publication")
        return original(source, target)

    monkeypatch.setattr(storage.os, "link", interrupt)
    out = tmp_path / "phase5"
    with pytest.raises(OSError, match="interrupted"):
        publish_input(value, context, out)
    assert len(calls) == 2
    assert not list(out.rglob("*.build.json"))
    assert not list(out.rglob("*.tmp"))
    assert (
        read_project_json(
            out / input_ref(value).logical_ref, input_ref(value).sha256, CanonicalInput
        )
        == value
    )


def test_invalid_input_never_publishes_any_record(pair, authority, tmp_path):
    value = selected(pair).model_copy(update={"paper_id": "P002"})
    out = tmp_path / "phase5"
    with pytest.raises(ValueError):
        publish_input(value, trusted(pair, authority), out)
    assert not out.exists()
