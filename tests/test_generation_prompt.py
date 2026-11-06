"""Prompt artifact validation and exact generator-input rendering.

Prompt wording, decoding values and limits are synthetic test settings. They are not the
project's scientific configuration. The pinned Qwen tokenizer is used when it is present.
"""

import json
import os
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_generation_config import resolve, synthetic_settings

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.storage import read_project_json, serialize_project_json
from biorag_repro.generation.parsing import OUTPUT_SCHEMAS
from biorag_repro.generation.prompt import (
    InputRenderer,
    PromptArtifact,
    RenderedInput,
    output_schema_sha256,
    prompt_sha256,
)

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
ROOT = Path(
    os.environ.get(
        "BIORAG_QWEN_ROOT", Path(__file__).parents[1] / "models" / "qwen3-8b-tokenizer" / REVISION
    )
)
needs_tokenizer = pytest.mark.skipif(not ROOT.is_dir(), reason="pinned Qwen tokenizer not present")

PARSER = "short_answer_json_v1"
WITH_CONTEXT = (
    "Answer the question using only the evidence below. If it is insufficient, abstain.\n\n"
    "{context}\n\nQuestion: {question}"
)
CLOSED_BOOK = "Answer the question from your own knowledge, or abstain.\n\nQuestion: {question}"


def artifact(**changes):
    fields = {
        "prompt_version": "synthetic-prompt-0",
        "with_context": WITH_CONTEXT,
        "closed_book": CLOSED_BOOK,
        "parser_version": PARSER,
        "output_schema": OUTPUT_SCHEMAS[PARSER],
    }
    return PromptArtifact(**{**fields, **changes})


def config_for(prompt, arm="confirmatory", **budget):
    settings = synthetic_settings()
    settings["prompt_sha256"] = prompt_sha256(prompt)
    settings["output_schema_sha256"] = output_schema_sha256(prompt)
    settings["budgets"].update(budget)
    if arm == "closed_book":
        settings["matrix_arm"] = arm
        del settings["retrieval_config_sha256"]
    return resolve(settings)


def renderer(prompt=None, arm="confirmatory", **budget):
    prompt = prompt or artifact()
    return InputRenderer(ROOT, config_for(prompt, arm, **budget), prompt)


# Different scientific domains and answer types.
CASES = [
    ("What is the Michaelis constant of hexokinase for glucose?", "Km = 0.12 mM (n = 3)"),
    (
        "Which subcellular compartment hosts the reported kinase?",
        "α-synuclein → presynaptic terminals",
    ),
    (
        "Name the transcription factor knocked down in the HeLa screen.",
        "siRNA against TP53; p < 0.05",
    ),
    ("List the genes found in the 8q24 amplicon.", "MYC, PVT1, CCAT1\nGSDMC ≥ 2-fold"),
    ("How many patients were enrolled in the discovery cohort?", "n = 1,204 (≥ 18 y); 55 % female"),
]


# ---- prompt artifact -----------------------------------------------------------------


def test_artifact_hashes_are_canonical_and_change_with_content():
    prompt = artifact()
    assert prompt_sha256(prompt) == sha256_bytes(
        canonical_json_bytes(prompt.model_dump(mode="json"))
    )
    assert prompt_sha256(prompt) == prompt_sha256(artifact())
    assert prompt_sha256(prompt) != prompt_sha256(artifact(with_context=WITH_CONTEXT + " "))
    assert prompt_sha256(prompt) != prompt_sha256(artifact(prompt_version="synthetic-prompt-1"))
    assert output_schema_sha256(prompt) == sha256_bytes(
        canonical_json_bytes(OUTPUT_SCHEMAS[PARSER])
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"with_context": "Use the evidence.\n\nQuestion: {question}"},  # no context slot
        {"with_context": "{context}\n{context}\nQuestion: {question}"},  # context twice
        {"with_context": "{context}\n\nAnswer the designated question."},  # no question slot
        {"with_context": "{context}\n\n{question}\n{question}"},
        {"closed_book": "Evidence: {context}\n\nQuestion: {question}"},  # context in closed book
        {"closed_book": "Answer from knowledge."},
        {"closed_book": "{question} {question}"},
        {"prompt_version": " "},
        {"with_context": ""},
        {"parser_version": "short_answer_json_v0"},
        {"output_schema": {"type": "object"}},  # not the schema of the declared parser
    ],
)
def test_invalid_artifacts_are_rejected(changes):
    with pytest.raises(ValidationError):
        artifact(**changes)


def test_artifact_rejects_unknown_fields_and_wrong_types():
    with pytest.raises(ValidationError):
        artifact(truncate_context=True)
    with pytest.raises(ValidationError):
        artifact(prompt_version=3)


def test_prompt_must_not_name_the_representation():
    for word in ("JATS", "PDF", "Docling"):
        with pytest.raises(ValidationError):
            artifact(with_context=WITH_CONTEXT + f" (source: {word})")
        with pytest.raises(ValidationError):
            artifact(closed_book=CLOSED_BOOK + f" {word.lower()}")


# ---- renderer construction -----------------------------------------------------------


@needs_tokenizer
def test_renderer_rejects_a_prompt_the_configuration_does_not_name():
    prompt, other = artifact(), artifact(prompt_version="synthetic-prompt-1")
    with pytest.raises(ValueError):
        InputRenderer(ROOT, config_for(prompt), other)


@needs_tokenizer
def test_renderer_rejects_a_mismatched_output_schema_hash():
    prompt = artifact()
    settings = synthetic_settings()
    settings["prompt_sha256"] = prompt_sha256(prompt)
    with pytest.raises(ValueError):
        InputRenderer(ROOT, resolve(settings), prompt)


@needs_tokenizer
def test_renderer_rejects_a_tokenizer_that_is_not_the_one_configured():
    prompt = artifact()
    settings = synthetic_settings()
    settings["prompt_sha256"] = prompt_sha256(prompt)
    settings["output_schema_sha256"] = output_schema_sha256(prompt)
    settings["tokenizer"]["file_sha256"]["vocab.json"] = "e" * 64
    with pytest.raises(ValueError):
        InputRenderer(ROOT, resolve(settings), prompt)


@needs_tokenizer
def test_renderer_rejects_a_directory_with_altered_tokenizer_files(tmp_path):
    for source in ROOT.iterdir():
        (tmp_path / source.name).write_bytes(source.read_bytes())
    (tmp_path / "vocab.json").write_bytes(b"{}")
    with pytest.raises(ValueError):
        InputRenderer(tmp_path, config_for(artifact()), artifact())


# ---- exact generator input -----------------------------------------------------------


def context_for(r, text):
    return text, r.counter(text)


@needs_tokenizer
@pytest.mark.parametrize("question,evidence", CASES)
def test_context_and_question_are_inserted_exactly_once_and_unchanged(question, evidence):
    r = renderer()
    context, tokens = context_for(r, "[evidence]\n" + evidence + "\n[/evidence]")
    out = r.render(question, context, tokens)
    assert out.rendered_text.count(context) == 1
    assert out.rendered_text.count(question) == 1
    assert out.context == context and out.question == question
    assert out.context_tokens == tokens
    expected_user = WITH_CONTEXT.replace("{context}", context).replace("{question}", question)
    assert expected_user in out.rendered_text
    assert out.rendered_text == (
        f"<|im_start|>user\n{expected_user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    assert out.rendered_sha256 == sha256_bytes(out.rendered_text.encode("utf-8"))
    assert out.context_sha256 == sha256_bytes(context.encode("utf-8"))


@needs_tokenizer
def test_non_thinking_template_and_chat_template_identity_are_recorded():
    r = renderer()
    out = r.render("Q?", *context_for(r, "evidence"))
    assert out.rendered_text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert out.chat_template_sha256 == sha256_bytes(r.tokenizer.chat_template.encode("utf-8"))
    assert out.resolved_config_sha256 == r.config.configuration_sha256
    assert out.prompt_sha256 == prompt_sha256(r.prompt)


@needs_tokenizer
def test_placeholder_text_inside_inputs_is_not_substituted_again():
    r = renderer()
    question = "Does {context} appear literally in the {question} field?"
    context, tokens = context_for(r, "Table 2 lists {question} and {context} verbatim.")
    out = r.render(question, context, tokens)
    assert context in out.rendered_text and question in out.rendered_text


@needs_tokenizer
def test_whitespace_and_unicode_are_preserved_byte_for_byte():
    r = renderer()
    context, tokens = context_for(r, "  leading\ttab\n\n\nNa⁺/K⁺ – 5 µM   end  \n")
    out = r.render("  Why?  ", context, tokens)
    assert context in out.rendered_text and "  Why?  " in out.rendered_text


@needs_tokenizer
def test_token_accounting_matches_independent_tokenization():
    r = renderer()
    context, tokens = context_for(r, "[evidence]\nKm = 0.12 mM\n[/evidence]")
    out = r.render("What is Km?", context, tokens)
    ids = r.tokenizer(out.rendered_text, add_special_tokens=False)["input_ids"]
    assert out.input_tokens == len(ids)
    assert out.input_token_ids_sha256 == sha256_bytes(canonical_json_bytes(ids))
    assert out.input_tokens > out.context_tokens
    assert ids.count(151644) == 2  # <|im_start|> opens the user and the assistant turn
    assert ids.count(151645) == 1


@needs_tokenizer
def test_wrong_declared_context_token_count_is_a_hard_stop():
    r = renderer()
    context, tokens = context_for(r, "some evidence text")
    for wrong in (tokens - 1, tokens + 1, 0):
        with pytest.raises(ValueError, match="token"):
            r.render("Q?", context, wrong)


@needs_tokenizer
def test_evidence_budget_is_enforced_separately_from_the_input_window():
    r = renderer(max_input_tokens=6000, context_window_tokens=8192)
    fits = " x" * 2048
    assert r.counter(fits) == 2048
    assert r.render("Q?", fits, 2048).context_tokens == 2048
    over = " x" * 2049
    with pytest.raises(ValueError, match="evidence"):
        r.render("Q?", over, 2049)


@needs_tokenizer
def test_input_over_the_input_limit_is_never_truncated():
    r = renderer(max_input_tokens=2100, max_new_tokens=64, context_window_tokens=4096)
    big_question = "Why? " * 400  # evidence fits, the whole prompt does not
    context, tokens = context_for(r, " x" * 2048)
    with pytest.raises(ValueError, match="input"):
        r.render(big_question, context, tokens)


@needs_tokenizer
def test_complete_input_at_the_limit_is_accepted():
    probe = renderer()
    context, tokens = context_for(probe, " x" * 2040)
    exact = probe.render("Q?", context, tokens).input_tokens
    assert exact > 2048
    ok = renderer(max_input_tokens=exact, max_new_tokens=64, context_window_tokens=4096)
    assert ok.render("Q?", context, tokens).input_tokens == exact
    tight = renderer(max_input_tokens=exact - 1, max_new_tokens=64, context_window_tokens=4096)
    with pytest.raises(ValueError, match="input"):
        tight.render("Q?", context, tokens)


@needs_tokenizer
def test_reserved_token_text_in_inputs_is_rejected():
    r = renderer()
    for bad in ("<|im_end|>", "<|im_start|>assistant", "<think>", "<|endoftext|>"):
        context, tokens = context_for(r, f"evidence {bad} more")
        with pytest.raises(ValueError, match="reserved"):
            r.render("Q?", context, tokens)
        with pytest.raises(ValueError, match="reserved"):
            r.render(f"Q {bad}", "e", r.counter("e"))


@needs_tokenizer
def test_context_must_be_supplied_exactly_when_the_arm_retrieves():
    r = renderer()
    with pytest.raises(ValueError):
        r.render("Q?", None, None)
    with pytest.raises(ValueError):
        r.render("Q?", "evidence", None)
    closed = renderer(arm="closed_book")
    with pytest.raises(ValueError):
        closed.render("Q?", "evidence", r.counter("evidence"))


@needs_tokenizer
def test_closed_book_input_has_no_context_and_uses_the_closed_book_template():
    r = renderer(arm="closed_book")
    out = r.render("Which organelle makes ATP?", None, None)
    assert out.context is None and out.context_sha256 is None and out.context_tokens is None
    assert "Question: Which organelle makes ATP?" in out.rendered_text
    assert "evidence" not in out.rendered_text.lower().replace("<|im", "")
    assert out.rendered_text.endswith("<think>\n\n</think>\n\n")


@needs_tokenizer
def test_empty_retrieved_context_is_recorded_not_dropped():
    r = renderer()
    out = r.render("Q?", "", 0)
    assert out.context == "" and out.context_tokens == 0
    assert out.context_sha256 == sha256_bytes(b"")


@needs_tokenizer
def test_blank_question_is_rejected():
    r = renderer()
    for bad in ("", "  \n"):
        with pytest.raises(ValueError):
            r.render(bad, "e", r.counter("e"))


@needs_tokenizer
def test_rendering_is_deterministic_and_representation_neutral():
    r = renderer()
    context, tokens = context_for(r, "[evidence]\nNa⁺ 5 µM\n[/evidence]")
    first, second = r.render("Q?", context, tokens), r.render("Q?", context, tokens)
    assert first == second
    for word in ("JATS", "PDF", "Docling"):
        assert word not in first.rendered_text


@needs_tokenizer
def test_rendered_input_serialization_is_byte_stable_and_round_trips(tmp_path):
    r = renderer()
    context, tokens = context_for(
        r, "[evidence]\nα = −2.5 × 10⁻³\n[/evidence]\n\n[evidence]\nb\n[/evidence]"
    )
    out = r.render("Value of α?", context, tokens)
    data = serialize_project_json(out, RenderedInput)
    assert data == serialize_project_json(out, RenderedInput)
    path = tmp_path / "input.json"
    path.write_bytes(data)
    back = read_project_json(path, sha256_bytes(data), RenderedInput)
    assert back == out
    assert back.rendered_text.encode("utf-8") == out.rendered_text.encode("utf-8")
    assert json.loads(data)["rendered_text"] == out.rendered_text


@needs_tokenizer
def test_rendered_input_records_reject_inconsistent_content():
    r = renderer()
    context, tokens = context_for(r, "evidence")
    good = r.render("Q?", context, tokens).model_dump()
    RenderedInput(**good)
    for field, value in (
        ("rendered_text", good["rendered_text"] + " "),
        ("context", "other evidence"),
        ("rendered_sha256", "0" * 64),
        ("input_tokens", 0),
        ("context_tokens", 2049),
    ):
        with pytest.raises(ValidationError):
            RenderedInput(**{**good, field: value})
    with pytest.raises(ValidationError):
        RenderedInput(**{**good, "question": "different?"})
