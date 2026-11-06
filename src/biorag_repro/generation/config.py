"""Resolved generator configuration and its content-addressed identity.

No scientific value has a default. Decoding, limits, prompt and seed references must be
written out, so an omission fails validation instead of silently becoming a setting.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.config import ResolvedConfiguration, resolve_config
from biorag_repro.evaluation.analysis_models import MatrixArm

Nonblank = Annotated[str, StringConstraints(strip_whitespace=False, pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

OFFICIAL_REPOSITORY = "Qwen/Qwen3-8B"
EVIDENCE_BUDGET_TOKENS = 2048  # EXPERIMENT_PLAN section 4; independent of the model window

# Approved official checkpoints: (repository, revision) -> manifest hash of the local file
# inventory (generation.environment.manifest_sha256). Empty on purpose. An entry is added
# only after the downloaded files have been inventoried and the hash approved by Rama.
APPROVED_CHECKPOINTS: dict[tuple[str, str], str] = {}


class ModelPin(Record):
    repository: Nonblank
    revision: GitRevision
    dtype: Literal["bfloat16"]  # published weights; no quantization setting exists


class TokenizerPin(Record):
    repository: Nonblank
    revision: GitRevision
    file_sha256: dict[Nonblank, Sha256] = Field(min_length=1)


class TokenBudgets(Record):
    """The evidence cap is a context-assembly rule; the other limits bound the whole call."""

    evidence_tokens: int
    max_input_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    context_window_tokens: int = Field(gt=0)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.evidence_tokens != EVIDENCE_BUDGET_TOKENS:
            raise ValueError(f"evidence budget is frozen at {EVIDENCE_BUDGET_TOKENS} tokens")
        if self.max_input_tokens <= self.evidence_tokens:
            raise ValueError("input limit must leave room for the prompt beyond the evidence")
        if self.max_input_tokens + self.max_new_tokens > self.context_window_tokens:
            raise ValueError("input limit plus output allowance exceeds the context window")
        return self


class Decoding(Record):
    do_sample: bool
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None

    @model_validator(mode="after")
    def consistent(self) -> Self:
        controls = (self.temperature, self.top_p, self.top_k)
        if not self.do_sample:
            if any(value is not None for value in controls):
                raise ValueError("sampling controls require do_sample = true")
            return self
        if any(value is None for value in controls):
            raise ValueError("sampling requires temperature, top_p and top_k")
        assert self.temperature is not None and self.top_p is not None and self.top_k is not None
        if self.temperature <= 0 or not 0 < self.top_p <= 1 or self.top_k < 0:
            raise ValueError("sampling controls out of range")
        return self


class GeneratorConfiguration(Record):
    schema_version: Literal["1"]
    purpose: Literal["synthetic_test", "development_candidate"]
    experiment_plan_version: Nonblank
    matrix_arm: MatrixArm
    retrieval_config_sha256: Sha256 | None = None
    model: ModelPin
    tokenizer: TokenizerPin
    serving_path: Literal["transformers"]
    thinking_mode: Literal["non_thinking"]
    budgets: TokenBudgets
    decoding: Decoding
    prompt_sha256: Sha256
    output_schema_sha256: Sha256
    parser_version: Literal["short_answer_json_v1"]
    seed_registry_version: Nonblank
    seed_registry_sha256: Sha256
    retry_max_attempts: int = Field(ge=1)
    retryable_failures: list[Nonblank]  # explicit; an empty list means nothing is retried
    scoring_version: Nonblank

    @field_validator("retryable_failures")
    @classmethod
    def sorted_distinct_categories(cls, value: list[str]) -> list[str]:
        """The retry rule is a set; order must not change the configuration identity."""
        if len(set(value)) != len(value):
            raise ValueError("duplicate retryable failure category")
        return sorted(value)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        closed_book = self.matrix_arm == "closed_book"
        if closed_book != (self.retrieval_config_sha256 is None):
            raise ValueError(
                "a retrieval arm names its retrieval configuration; closed book has none"
            )
        if self.purpose == "development_candidate" and not (
            self.model.repository == self.tokenizer.repository == OFFICIAL_REPOSITORY
        ):
            raise ValueError("a development candidate uses the official generator and tokenizer")
        return self


def resolve_generator_config(raw_toml: bytes) -> ResolvedConfiguration[GeneratorConfiguration]:
    return resolve_config(raw_toml, GeneratorConfiguration)


def check_retrieval_reference(config: GeneratorConfiguration, phase5_sha256: str | None) -> None:
    """The configuration must name the Phase 5 retrieval configuration actually used."""
    if config.retrieval_config_sha256 != phase5_sha256:
        raise ValueError("retrieval configuration does not match the one the generator names")
