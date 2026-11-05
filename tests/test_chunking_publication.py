"""Publication checks use synthetic inputs and a clearly marked fixture counter."""

import pytest
from test_phase4_publication_authorization import authority as authority  # noqa: PLC0414
from test_phase4_publication_authorization import pair as pair  # noqa: PLC0414
from test_retrieval_input import selected, trusted
from test_retrieval_models import ref, settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import read_manifest, serialize_project_json
from biorag_repro.documents.chunking import (
    chunk_document,
    publish_chunking,
    read_chunking,
)
from biorag_repro.retrieval.models import ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.validation import load_canonical_input


def example(pair, authority, tmp_path):
    config = settings()
    content = serialize_project_json(config, Phase5Configuration)
    (tmp_path / "config.json").write_bytes(content)
    cfgref = ConfigurationRef(
        artifact=ref("config").model_copy(
            update={"logical_ref": "config.json", "sha256": sha256_bytes(content)}
        ),
        resolved_sha256=sha256_bytes(canonical_json_bytes(config.model_dump())),
    )
    context = trusted(pair, authority)
    input = selected(pair)
    doc = load_canonical_input(input, context)
    result = chunk_document(doc, input, cfgref, config, len)
    return context, cfgref, result


def test_complete_readback_no_clobber_and_deterministic_publication(pair, authority, tmp_path):
    context, cfgref, result = example(pair, authority, tmp_path)
    refs = []
    for name in ["one", "two"]:
        out = tmp_path / name
        refs.append(publish_chunking(result, context, cfgref, tmp_path, out, fixture_counter=len))
        assert (
            read_chunking(*refs[-1], context, cfgref, tmp_path, out, fixture_counter=len) == result
        )
        assert read_manifest(out / refs[-1][1].logical_ref, refs[-1][1].sha256).status == "complete"
    assert refs[0] == refs[1]
    with pytest.raises(FileExistsError):
        publish_chunking(result, context, cfgref, tmp_path, tmp_path / "one", fixture_counter=len)


def test_counterfeit_is_checked_before_any_publication(pair, authority, tmp_path):
    context, cfgref, result = example(pair, authority, tmp_path)
    result.chunks[0].rendered_text = "altered scientific value"
    with pytest.raises(ValueError):
        publish_chunking(
            result, context, cfgref, tmp_path, tmp_path / "rejected-chunks", fixture_counter=len
        )
    assert not (tmp_path / "rejected-chunks").exists()


def test_interrupted_final_manifest_is_not_complete(pair, authority, tmp_path, monkeypatch):
    from biorag_repro.core import storage

    context, cfgref, result = example(pair, authority, tmp_path)
    original = storage.os.link

    def interrupt(source, target):
        if str(target).endswith(".chunk-set.build.json"):
            raise OSError("interrupted final publication")
        return original(source, target)

    monkeypatch.setattr(storage.os, "link", interrupt)
    out = tmp_path / "output"
    with pytest.raises(OSError, match="interrupted"):
        publish_chunking(result, context, cfgref, tmp_path, out, fixture_counter=len)
    assert not list(out.rglob("*.chunk-set.build.json"))
    assert not list(out.rglob("*.tmp"))


def test_changed_complete_manifest_is_rejected_after_rehash(pair, authority, tmp_path):
    from biorag_repro.core.manifest import ArtifactRef

    context, cfgref, result = example(pair, authority, tmp_path)
    out = tmp_path / "output"
    meta, manifest_ref = publish_chunking(
        result, context, cfgref, tmp_path, out, fixture_counter=len
    )
    manifest = read_manifest(out / manifest_ref.logical_ref, manifest_ref.sha256)
    manifest.status = "failed"
    manifest.failure_reason = "interrupted"
    content = serialize_project_json(manifest, type(manifest))
    (out / "bad.json").write_bytes(content)
    bad_ref = ArtifactRef(
        artifact_id="build_" + sha256_bytes(content),
        logical_ref="bad.json",
        sha256=sha256_bytes(content),
    )
    with pytest.raises(ValueError, match="manifest"):
        read_chunking(meta, bad_ref, context, cfgref, tmp_path, out, fixture_counter=len)


def test_fixture_counter_cannot_certify_a_development_configuration(pair, authority, tmp_path):
    context, cfgref, result = example(pair, authority, tmp_path)
    config = settings().model_copy(update={"purpose": "development_contract"})
    content = serialize_project_json(config, Phase5Configuration)
    (tmp_path / "config.json").write_bytes(content)
    cfgref.artifact.sha256 = sha256_bytes(content)
    cfgref.resolved_sha256 = sha256_bytes(canonical_json_bytes(config.model_dump()))
    with pytest.raises(ValueError, match="fixture counter"):
        publish_chunking(
            result, context, cfgref, tmp_path, tmp_path / "rejected", fixture_counter=len
        )


def test_construction_counter_requires_exact_verified_assets(tmp_path):
    from biorag_repro.documents.chunking import load_construction_counter

    config = settings()
    with pytest.raises(ValueError, match="pinned"):
        load_construction_counter(config, tmp_path)
    config.construction_tokenizer = config.construction_tokenizer.model_copy(
        update={
            "state": "pinned",
            "revision": "a" * 40,
            "file_hashes": {
                name: "a" * 64
                for name in (
                    "tokenizer.json",
                    "vocab.txt",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                )
            },
        }
    )
    with pytest.raises(FileNotFoundError):
        load_construction_counter(config, tmp_path)
    (tmp_path / "tokenizer.json").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_construction_counter(config, tmp_path)
