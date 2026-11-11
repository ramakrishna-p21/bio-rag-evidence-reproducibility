"""Persisted generator inputs: exact text, token ids and linkage, verified on read-back.

Prompt wording, configuration and questions are synthetic. The pinned Qwen tokenizer is used.
"""

import json

import pytest
from generation_support import needs_tokenizer
from pydantic import ValidationError
from test_generation_roster import make_renderer

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.generation.inputs import (
    CorruptInput,
    GeneratorInputRecord,
    build_input_record,
    check_tokenization,
    input_record_ref,
    publish_input_record,
    read_input_record,
)

pytestmark = needs_tokenizer

SHA = "a" * 64
KEY = "obs_" + "1" * 64
CONTEXT_REF = ArtifactRef(artifact_id="ctx-1", logical_ref="contexts/ctx-1.json", sha256="c" * 64)
CONTEXT = "The assay measured a half-life of 4.2 hours in the treated cells."


@pytest.fixture(scope="module")
def renderer():
    return make_renderer(SHA)


def rendered_with_context(renderer):
    return renderer.render("What was the half-life?", CONTEXT, renderer.counter(CONTEXT))


def record_for(renderer, **changes):
    rendered = rendered_with_context(renderer)
    fields = {"observation_key": KEY, "context": CONTEXT_REF}
    return build_input_record(renderer, rendered, **{**fields, **changes})


def rewrite(record, **changes):
    data = record.model_dump(mode="json")
    data.update(changes)
    return data


def test_record_holds_the_exact_text_context_and_token_ids(renderer):
    rendered = rendered_with_context(renderer)
    record = record_for(renderer)
    ids = renderer.tokenizer(rendered.rendered_text, add_special_tokens=False)["input_ids"]
    assert record.rendered == rendered
    assert record.input_token_ids == ids
    assert record.context == CONTEXT_REF
    assert record.observation_key == KEY
    assert record.rendered.resolved_config_sha256 == renderer.config.configuration_sha256
    assert record.rendered.prompt_sha256 == renderer.config.settings.prompt_sha256
    assert record.tokenizer == renderer.config.settings.tokenizer
    assert sha256_bytes(canonical_json_bytes(ids)) == rendered.input_token_ids_sha256


def test_closed_book_record_has_no_context_reference(renderer):
    closed = make_renderer(None, arm="closed_book")
    rendered = closed.render("What was the half-life?", None, None)
    record = build_input_record(closed, rendered, observation_key=KEY, context=None)
    assert record.context is None
    with pytest.raises(ValueError, match="context"):
        build_input_record(closed, rendered, observation_key=KEY, context=CONTEXT_REF)


def test_a_retrieval_record_needs_its_context_reference(renderer):
    with pytest.raises(ValueError, match="context"):
        record_for(renderer, context=None)


def test_a_rendered_input_with_a_wrong_token_hash_is_refused(renderer):
    rendered = rendered_with_context(renderer).model_copy(
        update={"input_token_ids_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="token"):
        build_input_record(renderer, rendered, observation_key=KEY, context=CONTEXT_REF)


def test_a_rendered_input_under_another_configuration_is_refused(renderer):
    rendered = rendered_with_context(renderer).model_copy(
        update={"resolved_config_sha256": "0" * 64}
    )
    with pytest.raises(ValueError, match="configuration"):
        build_input_record(renderer, rendered, observation_key=KEY, context=CONTEXT_REF)


def test_substituted_token_ids_are_refused_by_the_model(renderer):
    record = record_for(renderer)
    ids = list(record.input_token_ids)
    ids[3] += 1
    with pytest.raises(ValidationError, match="token ids"):
        GeneratorInputRecord.model_validate(rewrite(record, input_token_ids=ids))


def test_dropped_token_ids_are_refused_by_the_model(renderer):
    record = record_for(renderer)
    with pytest.raises(ValidationError, match="token"):
        GeneratorInputRecord.model_validate(
            rewrite(record, input_token_ids=record.input_token_ids[:-1])
        )


def test_self_consistent_but_wrong_token_ids_fail_the_tokenizer_check(renderer):
    """An editor who also recomputes the hashes passes the model, not the tokenizer."""
    record = record_for(renderer)
    ids = list(record.input_token_ids)
    ids[3] += 1
    digest = sha256_bytes(canonical_json_bytes(ids))
    forged_rendered = record.rendered.model_copy(update={"input_token_ids_sha256": digest})
    forged = GeneratorInputRecord.model_validate(
        rewrite(
            record,
            input_token_ids=ids,
            rendered=forged_rendered.model_dump(mode="json"),
        )
    )
    with pytest.raises(ValueError, match="tokenization"):
        check_tokenization(forged, renderer.tokenizer)
    check_tokenization(record, renderer.tokenizer)


def test_context_must_be_the_one_in_the_rendered_text(renderer):
    record = record_for(renderer)
    data = rewrite(record)
    data["rendered"]["context"] = CONTEXT + " extra"
    with pytest.raises(ValidationError):
        GeneratorInputRecord.model_validate(data)


def test_tokenizer_pin_must_be_the_approved_one(renderer):
    record = record_for(renderer)
    data = rewrite(record)
    data["tokenizer"]["revision"] = "0" * 40
    forged = GeneratorInputRecord.model_validate(data)
    with pytest.raises(ValueError, match="tokenizer"):
        check_tokenization(forged, renderer.tokenizer)


def test_reference_is_deterministic_and_names_configuration_prompt_and_context(renderer):
    record = record_for(renderer)
    ref = input_record_ref(record)
    assert ref == input_record_ref(record_for(renderer))
    assert ref.logical_ref == f"inputs/{ref.sha256}.json"
    other_context = ArtifactRef(
        artifact_id="ctx-2", logical_ref="contexts/ctx-2.json", sha256="d" * 64
    )
    assert (
        input_record_ref(record_for(renderer, context=other_context)).artifact_id != ref.artifact_id
    )
    assert (
        input_record_ref(record_for(renderer, observation_key="obs_" + "2" * 64)).sha256
        != ref.sha256
    )


def test_publish_is_immutable_idempotent_and_read_back_equal(tmp_path, renderer):
    record = record_for(renderer)
    ref = publish_input_record(tmp_path, record)
    assert ref == input_record_ref(record)
    assert publish_input_record(tmp_path, record) == ref
    assert read_input_record(tmp_path, ref.sha256) == record
    assert (tmp_path / ref.logical_ref).read_bytes().endswith(b"\n")


def test_publishing_over_different_bytes_is_refused(tmp_path, renderer):
    record = record_for(renderer)
    ref = publish_input_record(tmp_path, record)
    (tmp_path / ref.logical_ref).write_bytes(b"{}")
    with pytest.raises(CorruptInput):
        publish_input_record(tmp_path, record)


@pytest.mark.parametrize("damage", ["flip", "truncate", "delete", "garbage"])
def test_altered_or_missing_files_are_detected_on_read(tmp_path, renderer, damage):
    ref = publish_input_record(tmp_path, record_for(renderer))
    path = tmp_path / ref.logical_ref
    content = path.read_bytes()
    if damage == "flip":
        path.write_bytes(content.replace(b"half-life", b"half-lifx", 1))
    elif damage == "truncate":
        path.write_bytes(content[:-20])
    elif damage == "delete":
        path.unlink()
    else:
        path.write_bytes(b"not json")
    with pytest.raises(CorruptInput):
        read_input_record(tmp_path, ref.sha256)


def test_a_record_stored_under_another_hash_is_not_read(tmp_path, renderer):
    first = publish_input_record(tmp_path, record_for(renderer))
    second_record = record_for(renderer, observation_key="obs_" + "2" * 64)
    (tmp_path / f"inputs/{'e' * 64}.json").write_bytes((tmp_path / first.logical_ref).read_bytes())
    with pytest.raises(CorruptInput):
        read_input_record(tmp_path, "e" * 64)
    assert second_record.observation_key != KEY


def test_stored_json_carries_the_literal_rendered_text(tmp_path, renderer):
    rendered = rendered_with_context(renderer)
    ref = publish_input_record(tmp_path, record_for(renderer))
    stored = json.loads((tmp_path / ref.logical_ref).read_text(encoding="utf-8"))
    assert stored["rendered"]["rendered_text"] == rendered.rendered_text
    assert sha256_bytes(stored["rendered"]["rendered_text"].encode()) == rendered.rendered_sha256
