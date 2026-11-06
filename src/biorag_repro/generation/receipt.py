"""What a backend reports it gave the model and what came back.

Rendering and storing an input proves nothing about execution. A receipt is the backend's
own account of one call: the token ids it passed to the model, the generation settings it
used, and the token ids that came out. The attempt store checks the receipt against the
rest of the record; it is evidence from the executing process, not a hardware attestation.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
OpaqueId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


def token_ids_sha256(ids: list[int]) -> str:
    """Identity of a token sequence; the same rule the renderer uses for its input ids."""
    return sha256_bytes(canonical_json_bytes(ids))


class GenerationSettings(Record):
    """The decoding settings one call consumed, including the library defaults it ran with."""

    max_new_tokens: int = Field(gt=0)
    do_sample: bool
    temperature: float | None
    top_p: float | None
    top_k: int | None
    eos_token_ids: list[int] = Field(min_length=1)
    pad_token_id: int | None
    seed: int | None
    effective: dict[str, JsonValue]  # the complete generation configuration passed to the model


class ExecutionFacts(Record):
    """Reported by the backend for one call."""

    input_token_ids_sha256: Sha256
    input_token_count: int = Field(gt=0)
    generation: GenerationSettings
    output_token_ids: list[int]
    stop_reason: Literal["eos", "max_new_tokens"]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        ids, limit = self.output_token_ids, self.generation.max_new_tokens
        if len(ids) > limit:
            raise ValueError("more output tokens than the generation limit")
        if self.stop_reason == "eos":
            if not ids or ids[-1] not in self.generation.eos_token_ids:
                raise ValueError("an eos stop must end with a stop token")
        elif len(ids) != limit:
            raise ValueError("a length stop must have used the whole output allowance")
        return self


class ExecutionReceipt(Record):
    schema_version: Literal["1"] = "1"
    run_id: OpaqueId
    observation_key: Annotated[str, StringConstraints(pattern=r"^obs_[0-9a-f]{64}$")]
    attempt_id: OpaqueId
    environment_sha256: Sha256
    input_record_sha256: Sha256
    rendered_input_sha256: Sha256
    raw_output_sha256: Sha256
    output_token_ids_sha256: Sha256
    facts: ExecutionFacts

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if token_ids_sha256(self.facts.output_token_ids) != self.output_token_ids_sha256:
            raise ValueError("output token ids do not match their hash")
        return self
