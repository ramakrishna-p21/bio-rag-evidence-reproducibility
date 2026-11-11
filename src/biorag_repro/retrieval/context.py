"""Whole-chunk, first-nonfitting evidence packing with checked source lineage."""

from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from biorag_repro.core.hashing import sha256_bytes, verify_file_sha256
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
    verify_manifest_outputs,
)
from biorag_repro.retrieval import bm25
from biorag_repro.retrieval.chunking import read_chunking, verified_once
from biorag_repro.retrieval.models import (
    Chunk,
    ConfigurationRef,
    ContextDecision,
    ContextRecord,
    Snapshot,
)
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import load_configuration

REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
StopReason = Literal["exhausted", "first_nonfitting", "top20"]
Decision = Literal["included", "duplicate", "nonfitting", "not_processed"]

POLICY = "whole_evidence_wrappers_qwen_no_special_tokens_v1"
ASSETS = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
}


def tokenizer_snapshot() -> Snapshot:
    return Snapshot(
        repository="Qwen/Qwen3-8B", state="pinned", revision=REVISION, file_hashes=ASSETS
    )


class QwenCounter:
    """Verified local fast tokenizer; evidence only, without chat-template tokens."""

    def __init__(self, root: Path, snapshot: Snapshot):
        if snapshot != tokenizer_snapshot():
            raise ValueError("unapproved Qwen tokenizer identity")
        for name, digest in ASSETS.items():
            verify_file_sha256(root / name, digest)
        if {p: version(p) for p in ("transformers", "tokenizers", "huggingface-hub")} != {
            "transformers": "4.57.6",
            "tokenizers": "0.22.2",
            "huggingface-hub": "0.36.2",
        }:
            raise ValueError("unverified Qwen tokenizer runtime")
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            root, local_files_only=True, trust_remote_code=False
        )
        if type(self.tokenizer).__name__ != "Qwen2TokenizerFast" or not self.tokenizer.is_fast:
            raise ValueError("unexpected Qwen tokenizer class/backend")
        if (
            self.tokenizer.bos_token_id is not None
            or self.tokenizer.eos_token_id != 151645
            or self.tokenizer.pad_token_id != 151643
        ):
            raise ValueError("unexpected Qwen special token behavior")
        if self.tokenizer.backend_tokenizer.truncation or self.tokenizer.backend_tokenizer.padding:
            raise ValueError("Qwen backend truncation/padding enabled")
        if self("") != 0 or self("hello") != 1:
            raise ValueError("Qwen tokenizer probe failed")

    def __call__(self, text: str) -> int:
        return len(
            self.tokenizer(text, add_special_tokens=False, truncation=False, padding=False)[
                "input_ids"
            ]
        )


def _same_evidence(chunk: Chunk) -> tuple[Any, ...]:
    # Containing presentations do not create new source occurrences. Include
    # every occurrence (including repeated context) to preserve qualifiers.
    return (
        chunk.rendered_text,
        chunk.input.canonical.artifact_id,
        chunk.input.canonical.sha256,
        chunk.input.paper_id,
        chunk.input.representation,
        tuple(sorted({c.occurrence_id for c in chunk.contributors})),
    )


def pack(
    chunks: list[Chunk], count: Callable[[str], int]
) -> tuple[list[ContextDecision], str, int, StopReason]:
    """Pure packing step; callers must first verify ranking and complete chunk inputs."""
    decisions = []
    retained: dict[tuple[Any, ...], int] = {}
    text, tokens, stopped = "", 0, False
    for rank, chunk in enumerate(chunks[:20], 1):
        cid = record_ref(chunk).artifact_id
        key = _same_evidence(chunk)
        duplicate = None
        decision: Decision
        if stopped:
            decision = "not_processed"
        elif key in retained:
            decision, duplicate = "duplicate", retained[key]
        else:
            wrapped = "[evidence]\n" + chunk.rendered_text + "\n[/evidence]"
            proposed = text + "\n\n" + wrapped if text else wrapped
            measured = count(proposed)
            if measured < 0:
                raise ValueError("negative evidence token count")
            if measured > 2048:
                decision, stopped = "nonfitting", True
            else:
                decision = "included"
                text, tokens = proposed, measured
                retained[key] = rank
        decisions.append(
            ContextDecision(rank=rank, chunk_id=cid, decision=decision, duplicate_of_rank=duplicate)
        )
    reason: StopReason = (
        "first_nonfitting" if stopped else ("top20" if len(chunks) >= 20 else "exhausted")
    )
    return decisions, text, tokens, reason


@dataclass(frozen=True)
class Selection:
    """Independent upstream choices, never taken from a submitted context record."""

    retrieval: ArtifactRef
    retrieval_manifest: ArtifactRef
    index: ArtifactRef
    index_manifest: ArtifactRef
    sources: list[bm25.ChunkSelection]
    query: bm25.Query
    retrieval_configuration: ConfigurationRef
    retrieval_configuration_root: Path
    retrieval_root: Path
    method: str
    encoder: Any = None


def assemble(
    selected: Selection,
    configuration: ConfigurationRef,
    configuration_root: Path,
    tokenizer_root: Path | None,
    *,
    fixture_counter: Callable[[str], int] | None = None,
) -> ContextRecord:
    with verified_once():
        return _assemble(
            selected, configuration, configuration_root, tokenizer_root, fixture_counter
        )


def _assemble(
    selected: Selection,
    configuration: ConfigurationRef,
    configuration_root: Path,
    tokenizer_root: Path | None,
    fixture_counter: Callable[[str], int] | None,
) -> ContextRecord:
    settings = load_configuration(configuration, configuration_root)
    previous = load_configuration(
        selected.retrieval_configuration, selected.retrieval_configuration_root
    )
    if settings.context_rendering != POLICY:
        raise ValueError("context configuration requires the explicit rendering/counting policy")
    if (
        settings.model_copy(
            update={
                "packing_tokenizer": previous.packing_tokenizer,
                "context_rendering": previous.context_rendering,
            }
        )
        != previous
    ):
        raise ValueError("context settings differ beyond approved tokenizer pin")
    if fixture_counter is not None:
        if settings.purpose != "synthetic_fixture":
            raise ValueError("tokenizer doubles require synthetic fixtures")
        count = fixture_counter
    else:
        if tokenizer_root is None:
            raise ValueError("Qwen tokenizer assets required")
        count = QwenCounter(tokenizer_root, settings.packing_tokenizer)
    args = (
        selected.retrieval,
        selected.retrieval_manifest,
        selected.index,
        selected.index_manifest,
        selected.sources,
        selected.query,
        selected.retrieval_configuration,
        selected.retrieval_configuration_root,
    )
    if selected.method == "bm25":
        ranked = bm25.read_retrieval(*args, selected.retrieval_root)
    elif selected.method == "bge":
        from biorag_repro.retrieval import dense

        if selected.encoder is None:
            raise ValueError("BGE scientific reader requires selected encoder")
        ranked = dense.read_retrieval(*args, selected.encoder, selected.retrieval_root)
    else:
        raise ValueError("unknown selected retrieval method")
    chunks: dict[str, Chunk] = {}
    for source in selected.sources:
        result = read_chunking(
            source.chunk_set,
            source.manifest,
            source.authority,
            source.configuration,
            source.configuration_root,
            source.chunk_root,
            tokenizer_root=source.tokenizer_root,
            fixture_counter=source.fixture_counter,
        )
        chunks.update((record_ref(c).artifact_id, c) for c in result.chunks)
    candidates = []
    for hit in ranked.hits:
        chunk = chunks.get(hit.chunk.artifact_id)
        if (
            chunk is None
            or record_ref(chunk) != hit.chunk
            or chunk.input.paper_id != hit.paper_id
            or chunk.input.representation != ranked.representation
        ):
            raise ValueError("ranked candidate source correspondence mismatch")
        candidates.append(chunk)
    decisions, text, tokens, stop = pack(candidates, count)
    return ContextRecord(
        configuration=configuration,
        retrieval=selected.retrieval,
        retrieval_build_manifest=selected.retrieval_manifest,
        packing_policy=POLICY,
        packing_tokenizer=settings.packing_tokenizer,
        representation=ranked.representation,
        decisions=decisions,
        included_chunk_ids=[d.chunk_id for d in decisions if d.decision == "included"],
        exact_context=text,
        generator_tokens=tokens,
        stop_reason=stop,
    )


def validate(
    value: ContextRecord,
    selected: Selection,
    configuration: ConfigurationRef,
    configuration_root: Path,
    tokenizer_root: Path | None,
    *,
    fixture_counter: Callable[[str], int] | None = None,
) -> None:
    expected = assemble(
        selected, configuration, configuration_root, tokenizer_root, fixture_counter=fixture_counter
    )
    if ContextRecord.model_validate(value.model_dump()) != expected:
        raise ValueError(
            "context differs from independently selected scientific inputs and packing"
        )


def build_manifest(value: ContextRecord) -> BuildManifest:
    if value.retrieval_build_manifest is None:
        raise ValueError("context requires complete selected retrieval manifest")
    return BuildManifest(
        schema_version="1",
        inputs=[
            BuildInput(role="retrieval", artifact=value.retrieval),
            BuildInput(role="retrieval_build_manifest", artifact=value.retrieval_build_manifest),
            BuildInput(role="configuration", artifact=value.configuration.artifact),
        ],
        transformation_id="context_record_v1",
        transformation_settings_sha256=value.configuration.resolved_sha256,
        outputs=[
            DerivedOutput(
                artifact=record_ref(value),
                artifact_kind="context",
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )


def manifest_ref(value: ContextRecord) -> ArtifactRef:
    digest = sha256_bytes(serialize_project_json(build_manifest(value), BuildManifest))
    return ArtifactRef(
        artifact_id="build_" + digest,
        logical_ref=record_ref(value).logical_ref + ".build.json",
        sha256=digest,
    )


def publish(
    value: ContextRecord,
    selected: Selection,
    configuration: ConfigurationRef,
    configuration_root: Path,
    tokenizer_root: Path | None,
    root: Path,
    *,
    fixture_counter: Callable[[str], int] | None = None,
) -> tuple[ArtifactRef, ArtifactRef]:
    validate(
        value,
        selected,
        configuration,
        configuration_root,
        tokenizer_root,
        fixture_counter=fixture_counter,
    )
    ref, mr = record_ref(value), manifest_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, ContextRecord),
        ref.sha256,
        validate_temp=lambda p: read_project_json(p, ref.sha256, ContextRecord),
    )
    publish_manifest(build_manifest(value), root / mr.logical_ref, {ref.logical_ref: target})
    return ref, mr


def read(
    ref: ArtifactRef,
    mr: ArtifactRef,
    selected: Selection,
    configuration: ConfigurationRef,
    configuration_root: Path,
    tokenizer_root: Path | None,
    root: Path,
    *,
    fixture_counter: Callable[[str], int] | None = None,
) -> ContextRecord:
    value = read_project_json(root / ref.logical_ref, ref.sha256, ContextRecord)
    manifest = read_manifest(root / mr.logical_ref, mr.sha256)
    if ref != record_ref(value) or mr != manifest_ref(value) or manifest != build_manifest(value):
        raise ValueError("context identity or complete manifest mismatch")
    verify_manifest_outputs(manifest, {ref.logical_ref: root / ref.logical_ref})
    validate(
        value,
        selected,
        configuration,
        configuration_root,
        tokenizer_root,
        fixture_counter=fixture_counter,
    )
    return value
