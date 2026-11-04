"""Exact lexical retrieval over independently selected, scientifically verified chunks."""

import math
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

from pydantic import model_validator

from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
    verify_manifest_outputs,
)
from biorag_repro.documents.chunking import read_chunking
from biorag_repro.documents.models import Record
from biorag_repro.retrieval.models import (
    Bm25Index,
    Bm25Statistics,
    Chunk,
    ConfigurationRef,
    Nonblank,
    PaperId,
    Phase5Configuration,
    RankedHit,
    RankedRetrieval,
)
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import CanonicalAuthority, load_configuration

PREPROCESSING = "unicode_word_punctuation_v1"
PATTERN = r"\w+|[^\w\s]"


def tokenize(text: str) -> list[str]:
    """Python Unicode re, flags=0; preserve case, punctuation and repeated terms."""
    return re.findall(PATTERN, text)


def scorer(corpus: list[list[str]]) -> Any:
    """Use the pinned implementation, including its potentially negative IDF floor."""
    if not corpus or any(not row for row in corpus):
        raise ValueError("empty BM25 corpus or token row")
    if version("rank-bm25") != "0.2.2":
        raise ValueError("BM25 requires rank-bm25==0.2.2")
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    return BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)


@dataclass(frozen=True)
class ChunkSelection:
    """Caller-selected complete chunk evidence; submitted indexes cannot select it."""

    chunk_set: ArtifactRef
    manifest: ArtifactRef
    authority: CanonicalAuthority
    configuration: ConfigurationRef
    configuration_root: Path
    chunk_root: Path
    tokenizer_root: Path | None = None
    fixture_counter: Callable[[str], int] | None = None


class Query(Record):
    """Independently supplied query identity/text/scope, without benchmark gold."""

    query_id: Nonblank
    question: Nonblank
    scope: Literal["designated_paper", "full_corpus"]
    designated_paper_id: PaperId | None

    @model_validator(mode="after")
    def valid_scope(self) -> "Query":
        if (self.scope == "designated_paper") != (self.designated_paper_id is not None):
            raise ValueError("designated-paper query requires exactly its paper ID")
        return self


def statistics(model: Any) -> Bm25Statistics:
    frequencies: Counter[str] = Counter()
    for row in model.doc_freqs:
        frequencies.update(row.keys())
    return Bm25Statistics(
        corpus_size=model.corpus_size,
        document_lengths=model.doc_len,
        term_frequencies=model.doc_freqs,
        document_frequencies=dict(frequencies),
        average_document_length=model.avgdl,
        average_idf=model.average_idf,
        idf=model.idf,
    )


def _selected(sources: list[ChunkSelection]) -> tuple[Bm25Index, list[Chunk]]:
    if not sources:
        raise ValueError("empty selected index membership")
    sources = sorted(sources, key=lambda s: s.chunk_set.artifact_id)
    if len({s.chunk_set.artifact_id for s in sources}) != len(sources):
        raise ValueError("duplicate selected chunk set")
    configuration = sources[0].configuration
    settings = load_configuration(configuration, sources[0].configuration_root)
    if settings.lexical_tokenizer != PREPROCESSING:
        raise ValueError("unapproved lexical preprocessing")
    chunks: list[Chunk] = []
    papers: list[str] = []
    releases: set[str] = set()
    for source in sources:
        if source.configuration != configuration:
            raise ValueError("index configuration differs across chunk sets")
        result = read_chunking(
            source.chunk_set,
            source.manifest,
            source.authority,
            configuration,
            source.configuration_root,
            source.chunk_root,
            tokenizer_root=source.tokenizer_root,
            fixture_counter=source.fixture_counter,
        )
        papers.append(result.chunk_set.input.paper_id)
        releases.add(result.chunk_set.input.corpus_release_sha256)
        if not result.chunks:
            raise ValueError("empty selected chunk set is unusable for BM25")
        chunks.extend(result.chunks)
    if len(set(papers)) != len(papers) or len(releases) != 1:
        raise ValueError("duplicate paper membership or different corpus release")
    if len({c.input.representation for c in chunks}) != 1:
        raise ValueError("index mixes canonical representations")
    chunks.sort(key=lambda c: record_ref(c).artifact_id)
    ids = [record_ref(c).artifact_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate index chunk ID")
    corpus = [tokenize(c.rendered_text) for c in chunks]
    index = Bm25Index(
        configuration=configuration,
        representation=chunks[0].input.representation,
        chunk_tokens=settings.chunk_tokens,
        paper_ids=sorted(set(papers)),
        chunk_paper_ids=[c.input.paper_id for c in chunks],
        chunk_sets=[s.chunk_set for s in sources],
        chunk_set_manifests=[s.manifest for s in sources],
        ordered_chunk_ids=ids,
        tokenized_corpus=corpus,
        statistics=statistics(scorer(corpus)),
    )
    return index, chunks


def build_index(sources: list[ChunkSelection]) -> Bm25Index:
    """Reopen full upstream lineage and derive sorted membership and full statistics."""
    return _selected(sources)[0]


def validate_index(value: Bm25Index, sources: list[ChunkSelection]) -> None:
    expected = build_index(sources)
    if Bm25Index.model_validate(value.model_dump()) != expected:
        raise ValueError("BM25 index differs from independently selected scientific inputs")


def _rank(
    index: Bm25Index,
    chunks: list[Chunk],
    query: Query,
    configuration: ConfigurationRef,
    settings: Phase5Configuration,
    index_settings: Phase5Configuration,
) -> RankedRetrieval:
    query = Query.model_validate(query.model_dump())
    # Scope is part of the execution configuration. Reuse the original index
    # statistics while recording an explicit scope-specific retrieval configuration.
    if (
        settings.scope != query.scope
        or settings.model_copy(update={"scope": index_settings.scope}) != index_settings
    ):
        raise ValueError("retrieval configuration/scope differs from selected index settings")
    if query.designated_paper_id is not None and query.designated_paper_id not in index.paper_ids:
        raise ValueError("designated paper missing from index membership")
    scores = scorer(index.tokenized_corpus).get_scores(tokenize(query.question))
    values = [float(v) for v in scores]
    if any(not math.isfinite(v) for v in values):
        raise ValueError("nonfinite BM25 score")
    # Score the entire index first; filter eligibility without rebuilding statistics.
    eligible = [
        (chunk, score)
        for chunk, score in zip(chunks, values, strict=True)
        if query.scope == "full_corpus" or chunk.input.paper_id == query.designated_paper_id
    ]
    eligible.sort(key=lambda pair: (-pair[1], record_ref(pair[0]).artifact_id))
    return RankedRetrieval(
        configuration=configuration,
        index=record_ref(index),
        index_build_manifest=manifest_ref(index),
        retriever="bm25",
        query_id=query.query_id,
        question=query.question,
        representation=index.representation,
        scope=query.scope,
        designated_paper_id=query.designated_paper_id,
        hits=[
            RankedHit(rank=n, chunk=record_ref(chunk), paper_id=chunk.input.paper_id, score=score)
            for n, (chunk, score) in enumerate(eligible[: settings.context_candidates], 1)
        ],
    )


def retrieve(
    index: Bm25Index,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
) -> RankedRetrieval:
    expected, chunks = _selected(sources)
    if Bm25Index.model_validate(index.model_dump()) != expected:
        raise ValueError("BM25 index differs from independently selected scientific inputs")
    return _rank(
        expected,
        chunks,
        query,
        configuration,
        load_configuration(configuration, configuration_root),
        load_configuration(expected.configuration, sources[0].configuration_root),
    )


def validate_retrieval(
    value: RankedRetrieval,
    index: Bm25Index,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
) -> None:
    expected = retrieve(index, sources, query, configuration, configuration_root)
    if RankedRetrieval.model_validate(value.model_dump()) != expected:
        raise ValueError("ranked retrieval differs from selected query, scope or recomputed scores")


def build_manifest(
    value: Bm25Index | RankedRetrieval,
    index_manifest: ArtifactRef | None = None,
) -> BuildManifest:
    if isinstance(value, Bm25Index):
        kind = "bm25_index"
        inputs = [
            BuildInput(role=role, artifact=ref)
            for pair in zip(value.chunk_sets, value.chunk_set_manifests, strict=True)
            for role, ref in zip(("chunk_set", "chunk_set_build_manifest"), pair, strict=True)
        ]
    else:
        if index_manifest is None:
            raise ValueError("retrieval requires a complete index manifest")
        kind = "ranked_retrieval"
        inputs = [
            BuildInput(role="index", artifact=value.index),
            BuildInput(role="index_build_manifest", artifact=index_manifest),
        ]
    inputs.append(BuildInput(role="configuration", artifact=value.configuration.artifact))
    return BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=f"{kind}_record_v1",
        transformation_settings_sha256=value.configuration.resolved_sha256,
        outputs=[
            DerivedOutput(
                artifact=record_ref(value),
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
        status="complete",
    )


def manifest_ref(
    value: Bm25Index | RankedRetrieval,
    index_manifest: ArtifactRef | None = None,
) -> ArtifactRef:
    content = serialize_project_json(build_manifest(value, index_manifest), BuildManifest)
    digest = sha256_bytes(content)
    return ArtifactRef(
        artifact_id="build_" + digest,
        logical_ref=record_ref(value).logical_ref + ".bm25.build.json",
        sha256=digest,
    )


def _publish(
    value: Bm25Index | RankedRetrieval,
    root: Path,
    index_manifest: ArtifactRef | None = None,
) -> tuple[ArtifactRef, ArtifactRef]:
    ref = record_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, type(value)),
        ref.sha256,
        validate_temp=lambda p: read_project_json(p, ref.sha256, type(value)),
    )
    mr = manifest_ref(value, index_manifest)
    publish_manifest(
        build_manifest(value, index_manifest), root / mr.logical_ref, {ref.logical_ref: target}
    )
    return ref, mr


def publish_index(
    value: Bm25Index, sources: list[ChunkSelection], root: Path
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_index(value, sources)
    return _publish(value, root)


def _check_read(
    value: Bm25Index | RankedRetrieval,
    ref: ArtifactRef,
    mr: ArtifactRef,
    root: Path,
    index_manifest: ArtifactRef | None = None,
) -> None:
    if ref != record_ref(value):
        raise ValueError("BM25 record reference identity mismatch")
    manifest = read_manifest(root / mr.logical_ref, mr.sha256)
    if mr != manifest_ref(value, index_manifest) or manifest != build_manifest(
        value, index_manifest
    ):
        raise ValueError("BM25 complete manifest contract mismatch")
    verify_manifest_outputs(manifest, {ref.logical_ref: root / ref.logical_ref})


def read_index(
    ref: ArtifactRef, mr: ArtifactRef, sources: list[ChunkSelection], root: Path
) -> Bm25Index:
    value = read_project_json(root / ref.logical_ref, ref.sha256, Bm25Index)
    _check_read(value, ref, mr, root)
    validate_index(value, sources)
    return value


def publish_retrieval(
    value: RankedRetrieval,
    index_ref: ArtifactRef,
    index_manifest: ArtifactRef,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    root: Path,
) -> tuple[ArtifactRef, ArtifactRef]:
    index = read_index(index_ref, index_manifest, sources, root)
    validate_retrieval(value, index, sources, query, configuration, configuration_root)
    return _publish(value, root, index_manifest)


def read_retrieval(
    ref: ArtifactRef,
    mr: ArtifactRef,
    index_ref: ArtifactRef,
    index_manifest: ArtifactRef,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    root: Path,
) -> RankedRetrieval:
    value = read_project_json(root / ref.logical_ref, ref.sha256, RankedRetrieval)
    _check_read(value, ref, mr, root, index_manifest)
    index = read_index(index_ref, index_manifest, sources, root)
    validate_retrieval(value, index, sources, query, configuration, configuration_root)
    return value
