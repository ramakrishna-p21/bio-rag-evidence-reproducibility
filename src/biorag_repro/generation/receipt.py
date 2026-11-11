"""What a backend reports it gave the model and what came back.

Rendering and storing an input proves nothing about execution. A receipt is the backend's
own account of one call: the token ids it passed to the model, the generation settings it
used, and the token ids that came out. The attempt store checks the receipt against the
rest of the record; it is evidence from the executing process, not a hardware attestation.
"""

from typing import Annotated, Any, Literal, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.records import Record

Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
OpaqueId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


def token_ids_sha256(ids: list[int]) -> str:
    """Identity of a token sequence; the same rule the renderer uses for its input ids."""
    return sha256_bytes(canonical_json_bytes(ids))


def decode_output(tokenizer: Any, output_token_ids: list[int], stop_reason: str) -> str:
    """The raw text of an output: its tokens without the stop token, special tokens kept.

    The one decoding rule, used by the backend to produce the text and by verification to
    check that the stored text is what the reported token ids decode to.
    """
    body = output_token_ids[:-1] if stop_reason == "eos" else output_token_ids
    text: str = tokenizer.decode(
        body, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return text


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


class CudaMemoryUnavailable(Record):
    """No CUDA memory reading was taken; the reason is stated and nothing is estimated."""

    status: Literal["unavailable"] = "unavailable"
    reason: Nonblank


class CudaMemoryMeasured(Record):
    """Memory around one call on one CUDA device, in bytes.

    The allocated and reserved peaks are this process's torch counters, reset just before the
    call. `device_used_after_bytes` is a device-wide snapshot taken after the call (total minus
    free): it includes every other process on a shared GPU and is not a peak.
    """

    status: Literal["measured"] = "measured"
    device_index: int = Field(ge=0)
    allocated_before_bytes: int = Field(ge=0)
    peak_allocated_bytes: int = Field(ge=0)
    peak_reserved_bytes: int = Field(ge=0)
    device_total_bytes: int = Field(gt=0)
    device_used_after_bytes: int = Field(ge=0)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if self.peak_allocated_bytes < self.allocated_before_bytes:
            raise ValueError("the allocated peak is below the memory allocated before the call")
        if self.peak_reserved_bytes < self.peak_allocated_bytes:
            raise ValueError("reserved memory cannot be below the allocated peak")
        if self.device_used_after_bytes > self.device_total_bytes:
            raise ValueError("more device memory is used than the device has")
        return self


class ExecutionMeasurements(Record):
    """What the backend measured around one `model.generate` call.

    `generation_ns` is the monotonic interval from just before the call to just after it
    returned, with the CUDA device synchronized before the clock started and again before it
    was read, so queued GPU work is inside the interval and earlier work is not. It excludes
    tokenization, the token check, decoding, validation, receipt writing and environment
    capture. The overall attempt duration is a different quantity, taken from event times.
    """

    clock: Literal["perf_counter_ns"]
    generation_ns: int = Field(ge=0)
    cuda_memory: Annotated[
        CudaMemoryMeasured | CudaMemoryUnavailable, Field(discriminator="status")
    ]


class ExecutionFacts(Record):
    """Reported by the backend for one call."""

    input_token_ids_sha256: Sha256
    input_token_count: int = Field(gt=0)
    generation: GenerationSettings
    output_token_ids: list[int]
    stop_reason: Literal["eos", "max_new_tokens"]
    # Omitted from stored JSON when absent, so receipts written before it keep their bytes.
    measurements: ExecutionMeasurements | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def consistent(self) -> Self:
        ids, limit = self.output_token_ids, self.generation.max_new_tokens
        if len(ids) > limit:
            raise ValueError("more output tokens than the generation limit")
        stops = set(self.generation.eos_token_ids)
        # Generation ends at the first stop token, so one can only ever be the last token.
        if any(token in stops for token in ids[:-1]):
            raise ValueError("a stop token before the end of the output is impossible")
        if self.stop_reason == "eos":
            if not ids or ids[-1] not in stops:
                raise ValueError("an eos stop must end with a stop token")
        else:
            if len(ids) != limit:
                raise ValueError("a length stop must have used the whole output allowance")
            if ids[-1] in stops:
                raise ValueError("a stop token before the end of the output is impossible")
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
