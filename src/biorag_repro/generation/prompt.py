"""Prompt artifact and exact generator-input rendering with the pinned Qwen chat template.

The rendered text is the single source of the generator input: its bytes are hashed and
stored, and the token ids a backend must use come from tokenizing that text with no
truncation. Any limit violation stops the render; nothing is ever shortened.
"""

import re
from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import JsonValue, StringConstraints, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.config import ResolvedConfiguration
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, validate_sha256
from biorag_repro.generation.config import GeneratorConfiguration
from biorag_repro.generation.parsing import OUTPUT_SCHEMAS, PARSER_VERSIONS
from biorag_repro.retrieval.context import QwenCounter
from biorag_repro.retrieval.models import Snapshot

Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

_SLOT = re.compile(r"\{(context|question)\}")
_REPRESENTATION_WORDS = re.compile(r"jats|pdf|docling", re.IGNORECASE)
_NON_THINKING_SUFFIX = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _slots(template: str) -> list[str]:
    return _SLOT.findall(template)


def _fill(template: str, values: dict[str, str]) -> str:
    # Single pass: text inserted for one slot is never scanned for slots again.
    return _SLOT.sub(lambda match: values[match.group(1)], template)


class PromptArtifact(Record):
    """Representation-neutral prompt text plus the output schema its parser accepts."""

    schema_version: str = "1"
    prompt_version: Nonblank
    with_context: Nonblank
    closed_book: Nonblank
    parser_version: str
    output_schema: dict[str, JsonValue]

    @field_validator("schema_version")
    @classmethod
    def known_schema(cls, value: str) -> str:
        if value != "1":
            raise ValueError("unsupported prompt artifact schema")
        return value

    @model_validator(mode="after")
    def valid_templates(self) -> Self:
        if sorted(_slots(self.with_context)) != ["context", "question"]:
            raise ValueError("with_context must contain {context} and {question} exactly once")
        if _slots(self.closed_book) != ["question"]:
            raise ValueError("closed_book must contain {question} exactly once and no context")
        if _REPRESENTATION_WORDS.search(self.with_context + self.closed_book):
            raise ValueError("the prompt must not name a document representation")
        if self.parser_version not in PARSER_VERSIONS:
            raise ValueError("unknown parser version")
        if self.output_schema != OUTPUT_SCHEMAS[self.parser_version]:
            raise ValueError("output schema is not the schema of the declared parser")
        return self


def prompt_sha256(prompt: PromptArtifact) -> str:
    return sha256_bytes(canonical_json_bytes(prompt.model_dump(mode="json")))


def output_schema_sha256(prompt: PromptArtifact) -> str:
    return sha256_bytes(canonical_json_bytes(prompt.output_schema))


class RenderedInput(Record):
    """The exact text one generation call receives, with its token accounting."""

    schema_version: str = "1"
    resolved_config_sha256: Sha256
    prompt_sha256: Sha256
    chat_template_sha256: Sha256
    question: str
    context: str | None
    context_sha256: Sha256 | None
    context_tokens: int | None
    rendered_text: str
    rendered_sha256: Sha256
    input_tokens: int
    input_token_ids_sha256: Sha256

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if sha256_bytes(self.rendered_text.encode("utf-8")) != self.rendered_sha256:
            raise ValueError("rendered text does not match its hash")
        if (self.context is None) != (self.context_sha256 is None) or (self.context is None) != (
            self.context_tokens is None
        ):
            raise ValueError("context text, hash and token count are present together")
        if self.context is not None:
            assert self.context_tokens is not None
            if sha256_bytes(self.context.encode("utf-8")) != self.context_sha256:
                raise ValueError("context does not match its hash")
            if not 0 <= self.context_tokens <= 2048:
                raise ValueError("context tokens outside the evidence budget")
            if self.context not in self.rendered_text:
                raise ValueError("context missing from the rendered text")
        if self.question not in self.rendered_text:
            raise ValueError("question missing from the rendered text")
        if self.input_tokens <= 0 or self.input_tokens < (self.context_tokens or 0):
            raise ValueError("input token count is inconsistent")
        validate_sha256(self.input_token_ids_sha256)
        return self


class InputRenderer:
    def __init__(
        self,
        tokenizer_root: Path,
        config: ResolvedConfiguration[GeneratorConfiguration],
        prompt: PromptArtifact,
    ):
        settings = config.settings
        if settings.prompt_sha256 != prompt_sha256(prompt):
            raise ValueError("prompt is not the one the configuration names")
        if (
            settings.output_schema_sha256 != output_schema_sha256(prompt)
            or settings.parser_version != prompt.parser_version
        ):
            raise ValueError("output schema or parser differs from the configuration")
        pin = settings.tokenizer
        snapshot = Snapshot(
            repository=pin.repository,
            state="pinned",
            revision=pin.revision,
            file_hashes=dict(pin.file_sha256),
        )
        self.tokenizer_root = tokenizer_root
        self.counter = QwenCounter(tokenizer_root, snapshot)  # verifies the asset hashes
        self.tokenizer: Any = self.counter.tokenizer
        template = self.tokenizer.chat_template
        if not isinstance(template, str) or "enable_thinking" not in template:
            raise ValueError("tokenizer chat template lacks the enable_thinking switch")
        self.chat_template_sha256 = sha256_bytes(template.encode("utf-8"))
        self.reserved = sorted(self.tokenizer.get_added_vocab())
        self.config = config
        self.prompt = prompt

    def render(
        self, question: str, context: str | None, context_tokens: int | None
    ) -> RenderedInput:
        settings = self.config.settings
        budgets = settings.budgets
        if not question.strip():
            raise ValueError("question must not be blank")
        retrieves = settings.matrix_arm != "closed_book"
        if retrieves != (context is not None) or (context is None) != (context_tokens is None):
            raise ValueError(
                "context and its token count are supplied exactly when the arm retrieves"
            )
        for name, text in (("question", question), ("context", context or "")):
            if any(token in text for token in self.reserved):
                raise ValueError(f"{name} contains reserved tokenizer text")

        if context is not None:
            measured = self.counter(context)
            if measured != context_tokens:
                raise ValueError(
                    f"declared context token count {context_tokens} differs from recount {measured}"
                )
            if measured > budgets.evidence_tokens:
                raise ValueError(
                    f"context exceeds the {budgets.evidence_tokens}-token evidence budget"
                )
            user = _fill(self.prompt.with_context, {"context": context, "question": question})
        else:
            user = _fill(self.prompt.closed_book, {"question": question})

        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The pinned template adds no system turn, so the whole input is fixed by the user
        # turn. Equality proves the context and question were inserted once, unchanged.
        if rendered != f"<|im_start|>user\n{user}<|im_end|>\n{_NON_THINKING_SUFFIX}":
            raise ValueError("chat template output differs from the expected non-thinking input")

        ids = self.tokenizer(rendered, add_special_tokens=False, truncation=False, padding=False)[
            "input_ids"
        ]
        if len(ids) > budgets.max_input_tokens:
            raise ValueError(
                f"input of {len(ids)} tokens exceeds the input limit {budgets.max_input_tokens}"
            )
        return RenderedInput(
            resolved_config_sha256=self.config.configuration_sha256,
            prompt_sha256=settings.prompt_sha256,
            chat_template_sha256=self.chat_template_sha256,
            question=question,
            context=context,
            context_sha256=None if context is None else sha256_bytes(context.encode("utf-8")),
            context_tokens=context_tokens,
            rendered_text=rendered,
            rendered_sha256=sha256_bytes(rendered.encode("utf-8")),
            input_tokens=len(ids),
            input_token_ids_sha256=sha256_bytes(canonical_json_bytes(ids)),
        )
