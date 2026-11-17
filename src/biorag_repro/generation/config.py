"""Resolved generator configuration and its content-addressed identity.

No scientific value has a default. Decoding, limits, prompt and seed references must be
written out, so an omission fails validation instead of silently becoming a setting.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_validator

from biorag_repro.core.config import ResolvedConfiguration, resolve_config
from biorag_repro.core.records import Record
from biorag_repro.evaluation.analysis_models import MatrixArm

Nonblank = Annotated[str, StringConstraints(strip_whitespace=False, pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]

OFFICIAL_REPOSITORY = "Qwen/Qwen3-8B"
APPROVED_CHECKPOINT_GPU = "NVIDIA L4"
EVIDENCE_BUDGET_TOKENS = 2048  # EXPERIMENT_PLAN section 4; independent of the model window

# Qualified settings pins do not confer checkpoint or environment approval.
QUALIFIED_CONFIRMATORY_SHA256 = "11290e58973f3eec13ae2ce2712b9848f19fb57900bc42e7a73bd6f687949927"
QUALIFIED_CLOSED_BOOK_SHA256 = "c247b14982c8bdba5d8110816465e1e2dc07789c8f1d7d8debe0948cfe397888"
QUALIFIED_RETRIEVAL_SHA256 = "e3467647a2aa14b01149b586c9720d6af4a28c8fe66f3d51b3858bd505be7a52"

# Approved official checkpoints: (repository, revision) -> manifest hash of the local file
# inventory (generation.environment.manifest_sha256). Researcher decision and exact scope:
# approvals/researcher_decisions.json, researcher_authority_v1. Environment approval is separate.
APPROVED_CHECKPOINTS: dict[tuple[str, str], str] = {
    (
        OFFICIAL_REPOSITORY,
        "b968826d9c46dd6066d109eabc6255188de91218",
    ): "4a133fa0d16ea3fdd42e987d7aa5135f3f8562ce45c55bf478b2f777f53e6fad",
}

# SHA-256 of each environment lock file Rama has approved for official runs. A lock cannot
# approve itself: its own status field and any hash a caller supplies prove nothing. Entries
# are added only after the lock has been resolved and reviewed on the execution machine.
APPROVED_ENVIRONMENT_LOCKS: frozenset[str] = frozenset(
    {
        "96af1876324098a0fbf6c28d9896b790d10a210ba7943538d41056257f15847a",
    }
)


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
    """The configuration must name the retrieval configuration actually used."""
    if config.retrieval_config_sha256 != phase5_sha256:
        raise ValueError("retrieval configuration does not match the one the generator names")


def require_qualified_settings(settings: GeneratorConfiguration) -> None:
    """Compare controls with Phase 7 while retaining each arm's real configuration identity.

    This is a settings check, never checkpoint/environment approval or execution proof.
    """
    from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

    value = settings.model_dump()
    if settings.matrix_arm == "closed_book":
        expected = QUALIFIED_CLOSED_BOOK_SHA256
    else:
        value.update(matrix_arm="confirmatory", retrieval_config_sha256=QUALIFIED_RETRIEVAL_SHA256)
        expected = QUALIFIED_CONFIRMATORY_SHA256
    if sha256_bytes(canonical_json_bytes(value)) != expected:
        raise ValueError("generator controls differ from the exact qualified settings")
