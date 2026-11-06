"""Serialized description of the Phase 5 contexts a generator roster is built from.

A spec names published artifacts and the roots they are read from; it contains no scientific
content. The question of each query is taken from the development-only benchmark view, and
the roster builder still recomputes every context through the Phase 5 reader, so a wrong
hash or a substituted artifact stops the run instead of being repaired here.

Only BM25 sources can be described. Dense retrieval needs an encoder object that a file
cannot carry, so it is refused rather than approximated.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated, Literal

from pydantic import StringConstraints

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.benchmark.models import Record
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.authorization import ScientificAuthority
from biorag_repro.documents.phase4 import Phase4GateIndex
from biorag_repro.generation.roster import ContextSource
from biorag_repro.retrieval import bm25
from biorag_repro.retrieval import context as phase5
from biorag_repro.retrieval.models import ChunkSet, ConfigurationRef
from biorag_repro.retrieval.validation import CanonicalAuthority

Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Nonblank = Annotated[str, StringConstraints(pattern=r"\S")]
SUPPORTED_RETRIEVERS = ("bm25",)


class SourceError(Exception):
    """The context sources cannot be described or loaded as given."""


class AuthoritySpec(Record):
    """Independently selected canonical authority; each field is checked against the files."""

    corpus_root: Nonblank
    artifact_root: Nonblank
    release_path: Nonblank
    release_sha256: Sha256
    native_attempt: ArtifactRef
    jats_configuration_sha256: Sha256
    pdf_configuration_sha256: Sha256
    environment_sha256: Sha256
    asset_inventory_sha256: Sha256
    evidence: Phase4GateIndex
    reproduction_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]


class ChunkSetSpec(Record):
    chunk_set: ArtifactRef
    manifest: ArtifactRef
    authority: AuthoritySpec  # selected per paper: the evidence index names one paper


class SourceEntry(Record):
    item_id: Nonblank
    representation: Nonblank  # an item has one context per representation
    scope: Literal["designated_paper", "full_corpus"]
    designated_paper_id: Nonblank | None
    chunk_sets: list[ChunkSetSpec]
    index: ArtifactRef
    index_manifest: ArtifactRef
    retrieval: ArtifactRef
    retrieval_manifest: ArtifactRef
    context: ArtifactRef
    context_manifest: ArtifactRef


class ContextSourceSpec(Record):
    schema_version: Literal["1"] = "1"
    retriever: Nonblank
    retrieval_configuration: ConfigurationRef
    context_configuration: ConfigurationRef
    chunk_configuration_root: Nonblank
    chunk_root: Nonblank
    chunk_tokenizer_root: Nonblank
    retrieval_configuration_root: Nonblank
    retrieval_root: Nonblank
    context_root: Nonblank  # holds the context artifacts and the context configuration
    entries: list[SourceEntry]


def _authority_spec(authority: CanonicalAuthority) -> AuthoritySpec:
    sci = authority.scientific
    return AuthoritySpec(
        corpus_root=str(authority.corpus_root),
        artifact_root=str(authority.artifact_root),
        release_path=str(sci.release_path),
        release_sha256=sci.release_sha256,
        native_attempt=sci.native_attempt,
        jats_configuration_sha256=sci.jats_configuration_sha256,
        pdf_configuration_sha256=sci.pdf_configuration_sha256,
        environment_sha256=sci.environment_sha256,
        asset_inventory_sha256=sci.asset_inventory_sha256,
        evidence=authority.evidence,
        reproduction_revision=authority.reproduction_revision,
    )


def _canonical_authority(a: AuthoritySpec, where: Callable[[str], Path]) -> CanonicalAuthority:
    return CanonicalAuthority(
        corpus_root=where(a.corpus_root),
        artifact_root=where(a.artifact_root),
        scientific=ScientificAuthority(
            where(a.release_path),
            a.release_sha256,
            a.native_attempt,
            a.jats_configuration_sha256,
            a.pdf_configuration_sha256,
            a.environment_sha256,
            a.asset_inventory_sha256,
        ),
        evidence=a.evidence,
        reproduction_revision=a.reproduction_revision,
    )


def spec_from_sources(sources: Sequence[ContextSource], *, retriever: str) -> ContextSourceSpec:
    """Describe already-built sources; they must share configurations and roots."""
    if retriever not in SUPPORTED_RETRIEVERS:
        raise SourceError(f"retriever {retriever!r} cannot be described by a source spec")
    if not sources:
        raise SourceError("no context sources to describe")
    first = sources[0]
    chunk = first.selection.sources[0]
    if chunk.tokenizer_root is None:
        raise SourceError("fixture chunk counters cannot be described by a source spec")
    shared = (
        first.selection.retrieval_configuration,
        first.selection.retrieval_configuration_root,
        first.selection.retrieval_root,
        first.configuration,
        first.root,
    )
    chunk_shared = (chunk.configuration_root, chunk.chunk_root, chunk.tokenizer_root)
    entries = []
    for source in sources:
        selection = source.selection
        found = (
            selection.retrieval_configuration,
            selection.retrieval_configuration_root,
            selection.retrieval_root,
            source.configuration,
            source.root,
        )
        if found != shared or any(
            (c.configuration_root, c.chunk_root, c.tokenizer_root) != chunk_shared
            for c in selection.sources
        ):
            raise SourceError("sources differ in configuration or roots")
        if selection.method != retriever:
            raise SourceError("source retriever differs from the described retriever")
        stored = read_project_json(
            chunk.chunk_root / selection.sources[0].chunk_set.logical_ref,
            selection.sources[0].chunk_set.sha256,
            ChunkSet,
        )
        entries.append(
            SourceEntry(
                item_id=selection.query.query_id,
                representation=stored.input.representation,
                scope=selection.query.scope,
                designated_paper_id=selection.query.designated_paper_id,
                chunk_sets=[
                    ChunkSetSpec(
                        chunk_set=c.chunk_set,
                        manifest=c.manifest,
                        authority=_authority_spec(c.authority),
                    )
                    for c in selection.sources
                ],
                index=selection.index,
                index_manifest=selection.index_manifest,
                retrieval=selection.retrieval,
                retrieval_manifest=selection.retrieval_manifest,
                context=source.context,
                context_manifest=source.manifest,
            )
        )
    return ContextSourceSpec(
        retriever=retriever,
        retrieval_configuration=shared[0],
        context_configuration=shared[3],
        chunk_configuration_root=str(chunk_shared[0]),
        chunk_root=str(chunk_shared[1]),
        chunk_tokenizer_root=str(chunk_shared[2]),
        retrieval_configuration_root=str(shared[1]),
        retrieval_root=str(shared[2]),
        context_root=str(shared[4]),
        entries=sorted(entries, key=lambda e: (e.item_id, e.representation)),
    )


def load_context_sources(
    path: Path, expected_sha256: str, view_directory: Path, view_record_sha256: str
) -> list[ContextSource]:
    """Read a pinned spec and return the sources it describes, one per entry.

    Relative paths are relative to the spec file. Beyond the development view and the (hash-checked) chunk-set record that
    names each entry's representation, nothing is read here; the roster builder verifies
    the rest through Phase 5.
    """
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SourceError(f"source spec {path} is unreadable: {error}") from error
    if sha256_bytes(raw) != expected_sha256:
        raise SourceError(f"source spec {path} differs from the pinned hash")
    try:
        spec = ContextSourceSpec.model_validate(json.loads(raw))
    except ValueError as error:
        raise SourceError(f"source spec {path} is not a valid specification: {error}") from error
    if spec.retriever not in SUPPORTED_RETRIEVERS:
        raise SourceError(
            f"retriever {spec.retriever!r} is not supported by source specs "
            f"(supported: {SUPPORTED_RETRIEVERS})"
        )
    try:
        _, items = read_development_view(view_directory, view_record_sha256)
    except (OSError, ValueError) as error:
        raise SourceError(f"development view is not usable: {error}") from error
    questions = {item.item_id: item.question for item in items}

    base = path.resolve().parent

    def where(value: str) -> Path:
        return base / value  # an absolute value replaces the base

    seen: set[tuple[str, str]] = set()
    sources = []
    for entry in spec.entries:
        if (entry.item_id, entry.representation) in seen:
            raise SourceError(f"duplicate source entry for {(entry.item_id, entry.representation)}")
        seen.add((entry.item_id, entry.representation))
        if entry.item_id not in questions:
            raise SourceError(f"{entry.item_id} is not in the development view")
        first_set = entry.chunk_sets[0].chunk_set
        try:
            stored = read_project_json(
                where(spec.chunk_root) / first_set.logical_ref, first_set.sha256, ChunkSet
            )
        except (OSError, ValueError) as error:
            raise SourceError(f"chunk set of {entry.item_id}: {error}") from error
        if stored.input.representation != entry.representation:
            raise SourceError(f"{entry.item_id}: entry representation differs from its chunk set")
        chunks = [
            bm25.ChunkSelection(
                c.chunk_set,
                c.manifest,
                _canonical_authority(c.authority, where),
                spec.retrieval_configuration,
                where(spec.chunk_configuration_root),
                where(spec.chunk_root),
                tokenizer_root=where(spec.chunk_tokenizer_root),
            )
            for c in entry.chunk_sets
        ]
        try:
            query = bm25.Query(
                query_id=entry.item_id,
                question=questions[entry.item_id],
                scope=entry.scope,
                designated_paper_id=entry.designated_paper_id,
            )
        except ValueError as error:
            raise SourceError(f"{entry.item_id}: {error}") from error
        selection = phase5.Selection(
            entry.retrieval,
            entry.retrieval_manifest,
            entry.index,
            entry.index_manifest,
            chunks,
            query,
            spec.retrieval_configuration,
            where(spec.retrieval_configuration_root),
            where(spec.retrieval_root),
            spec.retriever,
        )
        sources.append(
            ContextSource(
                entry.context,
                entry.context_manifest,
                selection,
                spec.context_configuration,
                where(spec.context_root),
                where(spec.context_root),
            )
        )
    return sources
