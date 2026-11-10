"""Persisted generator inputs.

A record keeps the exact rendered text, the token ids that text becomes, the pinned
tokenizer, the prompt and configuration identities and the retrieval context it was built
from, for one observation. It is published once under its own hash and verified on every
read. Storing a record says what the generator is meant to receive; whether a call
received it is shown only by an execution receipt.
"""

from pathlib import Path
from typing import Annotated, Any, Self

from pydantic import StringConstraints, model_validator

from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import publish_bytes, read_project_json, serialize_project_json
from biorag_repro.generation.config import TokenizerPin
from biorag_repro.generation.prompt import InputRenderer, RenderedInput
from biorag_repro.generation.receipt import token_ids_sha256
from biorag_repro.retrieval import context as phase5

ObservationKey = Annotated[str, StringConstraints(pattern=r"^obs_[0-9a-f]{64}$")]
_TRANSFORMATION = "generator_input_v1"


class CorruptInput(ValueError):
    """A stored generator input is missing, altered or inconsistent."""


class GeneratorInputRecord(Record):
    schema_version: str = "1"
    observation_key: ObservationKey
    rendered: RenderedInput
    context: ArtifactRef | None  # the context record; None only in the closed-book arm
    tokenizer: TokenizerPin
    input_token_ids: list[int]

    @model_validator(mode="after")
    def consistent(self) -> Self:
        if token_ids_sha256(self.input_token_ids) != self.rendered.input_token_ids_sha256:
            raise ValueError("token ids do not match the recorded token hash")
        if len(self.input_token_ids) != self.rendered.input_tokens:
            raise ValueError("token count does not match the token ids")
        if (self.context is None) != (self.rendered.context is None):
            raise ValueError("a context reference is present exactly when the input has context")
        return self


def build_input_record(
    renderer: InputRenderer,
    rendered: RenderedInput,
    *,
    observation_key: str,
    context: ArtifactRef | None,
) -> GeneratorInputRecord:
    """Tokenize the rendered text again and require it to be the input the renderer counted."""
    settings = renderer.config.settings
    if rendered.resolved_config_sha256 != renderer.config.configuration_sha256:
        raise ValueError("rendered input belongs to another configuration")
    if rendered.prompt_sha256 != settings.prompt_sha256:
        raise ValueError("rendered input used another prompt than the configuration")
    ids = renderer.tokenizer(
        rendered.rendered_text, add_special_tokens=False, truncation=False, padding=False
    )["input_ids"]
    if token_ids_sha256(ids) != rendered.input_token_ids_sha256:
        raise ValueError("rendered text does not tokenize to the recorded token ids")
    return GeneratorInputRecord(
        observation_key=observation_key,
        rendered=rendered,
        context=context,
        tokenizer=settings.tokenizer,
        input_token_ids=ids,
    )


def _bytes(record: GeneratorInputRecord) -> bytes:
    return serialize_project_json(record, GeneratorInputRecord)


def input_record_ref(record: GeneratorInputRecord) -> ArtifactRef:
    """Run-relative reference; the identity names configuration, prompt and context."""
    digest = sha256_bytes(_bytes(record))
    rendered = record.rendered
    lineage = [
        ArtifactInput(
            "configuration",
            f"config:{rendered.resolved_config_sha256}",
            rendered.resolved_config_sha256,
        ),
        ArtifactInput("prompt", f"prompt:{rendered.prompt_sha256}", rendered.prompt_sha256),
    ]
    if record.context is not None:
        lineage.append(ArtifactInput("context", record.context.artifact_id, record.context.sha256))
    artifact_id = derived_artifact_id(
        artifact_kind="generator_input",
        inputs=lineage,
        transformation_id=_TRANSFORMATION,
        transformation_settings_sha256=rendered.resolved_config_sha256,
        output_sha256=digest,
        output_format="json",
        serialization_version="1",
    )
    return ArtifactRef(artifact_id=artifact_id, logical_ref=f"inputs/{digest}.json", sha256=digest)


def publish_input_record(run_dir: Path, record: GeneratorInputRecord) -> ArtifactRef:
    """Store the record without replacing anything. Publishing identical bytes again is a no-op."""
    ref = input_record_ref(record)
    target = run_dir / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    content = _bytes(record)
    try:
        publish_bytes(target, content, ref.sha256)
    except FileExistsError:
        if target.read_bytes() != content:
            raise CorruptInput(f"a different file is stored at {ref.logical_ref}") from None
    return ref


def read_input_record(run_dir: Path, sha256: str) -> GeneratorInputRecord:
    path = run_dir / "inputs" / f"{sha256}.json"
    try:
        return read_project_json(path, sha256, GeneratorInputRecord)
    except (OSError, ValueError) as error:  # includes hash mismatch, JSON and validation errors
        raise CorruptInput(
            f"generator input {sha256} is missing, altered or invalid: {error}"
        ) from error


def check_tokenization(record: GeneratorInputRecord, tokenizer: Any) -> None:
    """The stored ids must be what the approved tokenizer makes of the stored text."""
    approved = phase5.tokenizer_snapshot()
    pin = record.tokenizer
    if (pin.repository, pin.revision, pin.file_sha256) != (
        approved.repository,
        approved.revision,
        approved.file_hashes,
    ):
        raise ValueError("input was not tokenized with the approved tokenizer")
    ids = tokenizer(
        record.rendered.rendered_text, add_special_tokens=False, truncation=False, padding=False
    )["input_ids"]
    if ids != record.input_token_ids:
        raise ValueError("stored token ids differ from the tokenization of the stored text")
