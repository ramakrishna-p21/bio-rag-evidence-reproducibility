"""Versioned retrieval and context interfaces; validation does not execute scientific transforms."""

from typing import Annotated, Literal, Self

from pydantic import AfterValidator, Field, model_validator

from biorag_repro.core.hashing import validate_sha256
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.documents.models import ContentState, MappingState, Record, SourceMapping

Sha256 = Annotated[str, AfterValidator(lambda value: _digest(value))]
Nonblank = Annotated[str, Field(min_length=1, pattern=r"\S")]
Representation = Literal["JATS", "PDF_DOCLING"]
OccurrenceField = Literal["segment", "table_cell", "native_orig", "native_marker", "nontextual"]
# content_continuation: a later sentence segment of an occurrence whose first part is the
# content of an earlier segment. It never holds the whole occurrence on its own.
ContributorUsage = Literal[
    "content", "content_continuation", "repeated_context", "containing_reference"
]
AccountingStatus = Literal["included", "oversized", "unsupported", "nontextual"]
PaperId = Annotated[str, Field(pattern=r"^P0(?:0[1-9]|[1-4][0-9]|50)$")]


def _digest(value: str) -> str:
    validate_sha256(value)
    return value


def _unique(values: list[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


class Snapshot(Record):
    """A recorded pin is not a claim of installed assets or qualification."""

    repository: Nonblank
    state: Literal["unresolved", "pinned"]
    revision: str | None = None
    file_hashes: dict[Nonblank, Sha256] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_pin(self) -> Self:
        import re

        if self.state == "unresolved":
            if self.revision is not None or self.file_hashes:
                raise ValueError("unresolved snapshot cannot claim a revision or file hashes")
        elif (
            self.revision is None
            or re.fullmatch(r"[0-9a-f]{40}", self.revision) is None
            or self.revision == "0" * 40
            or not self.file_hashes
        ):
            raise ValueError("pinned snapshot requires exact revision and file hashes")
        return self


class Phase5Configuration(Record):
    """Shared settings exclude the per-condition representation coordinate."""

    schema_version: Literal["1"] = "1"
    purpose: Literal["development_contract", "synthetic_fixture"]
    chunk_tokens: Literal[192, 384]
    construction_tokenizer: Snapshot
    packing_tokenizer: Snapshot
    bge_model: Snapshot
    lexical_tokenizer: Nonblank
    scope: Literal["designated_paper", "full_corpus"]
    # complete_units is the approved policy. The segmenting value is an opt-in development
    # candidate only; it is never a default and is not approved for official runs.
    unit_policy: Literal["complete_units", "complete_units_or_sentence_segments"] = "complete_units"
    overlap_tokens: Literal[0] = 0
    oversized_policy: Literal["diagnose_without_truncation"] = "diagnose_without_truncation"
    statistics: Literal["representation_wide"] = "representation_wide"
    bm25_implementation: Literal["rank-bm25==0.2.2:BM25Okapi"] = "rank-bm25==0.2.2:BM25Okapi"
    k1: float = Field(default=1.5, ge=1.5, le=1.5)
    b: float = Field(default=0.75, ge=0.75, le=0.75)
    epsilon: float = Field(default=0.25, ge=0.25, le=0.25)
    context_rendering: Literal["whole_evidence_wrappers_qwen_no_special_tokens_v1"] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    context_candidates: Literal[20] = 20
    evidence_tokens: Literal[2048] = 2048
    duplicate_policy: Literal["text_and_scientific_provenance"] = "text_and_scientific_provenance"
    stop_policy: Literal["first_nonfitting"] = "first_nonfitting"
    # v3 is an opt-in development candidate only (header-block defaults, row-local coordinates).
    chunking_algorithm: Literal[
        "canonical_whole_units_v1", "canonical_whole_units_v2", "canonical_whole_units_v3"
    ] = "canonical_whole_units_v1"
    construction_counting: Literal["no_special_tokens_no_padding_no_truncation"] = (
        "no_special_tokens_no_padding_no_truncation"
    )
    construction_backend: Literal["tokenizers==0.22.2"] = "tokenizers==0.22.2"

    @model_validator(mode="after")
    def permitted_models(self) -> Self:
        if (
            self.construction_tokenizer.repository != "BAAI/bge-base-en-v1.5"
            or self.bge_model.repository != "BAAI/bge-base-en-v1.5"
            or self.packing_tokenizer.repository != "Qwen/Qwen3-8B"
        ):
            raise ValueError("token roles require the approved BGE and intended Qwen repositories")
        if (
            self.construction_tokenizer.state == self.bge_model.state == "pinned"
            and self.construction_tokenizer.revision != self.bge_model.revision
        ):
            raise ValueError("BGE construction tokenizer and model revisions differ")
        return self

    def require_pinned_assets(self) -> None:
        """Reject unresolved prerequisites without pretending to verify installed files."""
        if any(
            item.state != "pinned"
            for item in (self.construction_tokenizer, self.packing_tokenizer, self.bge_model)
        ):
            raise ValueError("unresolved model/tokenizer acquisition prerequisites")


class ConfigurationRef(Record):
    artifact: ArtifactRef
    resolved_sha256: Sha256


class CanonicalInput(Record):
    schema_version: Literal["1"] = "1"
    corpus_version: Nonblank
    corpus_release_sha256: Sha256
    paper_id: PaperId
    representation: Representation
    canonical: ArtifactRef
    validation: ArtifactRef
    build_manifest: ArtifactRef


class SourceOccurrence(Record):
    """One location/field; containing-node references do not create new evidence."""

    occurrence_id: Nonblank
    canonical: ArtifactRef
    owner_node_id: Nonblank
    field: OccurrenceField
    segment_index: int | None = Field(ge=0)
    mapping: SourceMapping
    text: str | None
    content_state: ContentState
    interpretation_state: MappingState | None
    native_order_state: Literal["linked", "unpositioned"] | None = None
    orig_qualification: (
        Literal["same_as_text", "unverified_formula", "marker_inclusive", "unclassified_variant"]
        | None
    ) = None

    @model_validator(mode="after")
    def valid_content(self) -> Self:
        if (self.field == "segment") != (self.segment_index is not None):
            raise ValueError("only a segment occurrence has a segment index")
        if (self.field == "nontextual") != (self.text is None):
            raise ValueError(
                "nontextual occurrence requires null text; textual fields require text"
            )
        return self


class SourceContributor(Record):
    occurrence_id: Nonblank
    node_id: Nonblank
    usage: ContributorUsage


class Chunk(Record):
    schema_version: Literal["1"] = "1"
    input: CanonicalInput
    configuration: ConfigurationRef
    contributors: list[SourceContributor] = Field(min_length=1)
    rendered_text: Nonblank
    construction_tokens: int = Field(gt=0, le=384)


class OccurrenceAccounting(Record):
    occurrence_id: Nonblank
    status: AccountingStatus
    chunk_ids: list[Nonblank]
    reason: Nonblank | None

    @model_validator(mode="after")
    def valid_disposition(self) -> Self:
        _unique(self.chunk_ids, "accounted chunk")
        if self.status == "included":
            if not self.chunk_ids or self.reason is not None:
                raise ValueError("included occurrence requires chunks and no exclusion reason")
        elif self.chunk_ids or self.reason is None:
            raise ValueError("excluded occurrence requires a reason and no included chunks")
        return self


class ChunkingDiagnostic(Record):
    unit_id: Nonblank
    kind: Nonblank
    code: Literal["oversized", "unsupported", "nontextual", "layout_only"]
    source_node_ids: list[Nonblank] = Field(min_length=1)
    contributors: list[SourceContributor] = Field(min_length=1)
    actual_tokens: int | None = Field(ge=0)
    token_limit: Literal[192, 384]
    rendered_text: str | None
    reason: Nonblank
    uncertainty: list[Nonblank]

    @model_validator(mode="after")
    def valid_oversize(self) -> Self:
        if self.code == "oversized" and (
            self.actual_tokens is None
            or self.actual_tokens <= self.token_limit
            or self.rendered_text is None
        ):
            raise ValueError(
                "oversized diagnostic requires complete text and actual over-limit count"
            )
        return self


class ChunkSet(Record):
    schema_version: Literal["1"] = "1"
    input: CanonicalInput
    configuration: ConfigurationRef
    chunks: ArtifactRef
    chunks_build_manifest: ArtifactRef | None = None
    chunk_ids: list[Nonblank]
    occurrences: list[SourceOccurrence]
    accounting: list[OccurrenceAccounting]
    diagnostics: list[ChunkingDiagnostic] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_accounting(self) -> Self:
        _unique(self.chunk_ids, "chunk ID")
        ids = [item.occurrence_id for item in self.occurrences]
        _unique(ids, "source occurrence")
        _unique([item.occurrence_id for item in self.accounting], "accounting entry")
        if set(ids) != {item.occurrence_id for item in self.accounting}:
            raise ValueError("occurrence accounting is incomplete")
        if any(item.canonical != self.input.canonical for item in self.occurrences):
            raise ValueError("occurrence canonical input differs")
        if any(not set(item.chunk_ids) <= set(self.chunk_ids) for item in self.accounting):
            raise ValueError("accounting references unknown chunk")
        return self


class Bm25Statistics(Record):
    corpus_size: int = Field(gt=0)
    document_lengths: list[Annotated[int, Field(gt=0)]]
    term_frequencies: list[dict[Nonblank, Annotated[int, Field(gt=0)]]]
    document_frequencies: dict[Nonblank, Annotated[int, Field(gt=0)]]
    average_document_length: float = Field(gt=0)
    average_idf: float
    idf: dict[Nonblank, float]


class Bm25Index(Record):
    schema_version: Literal["1"] = "1"
    representation: Representation
    chunk_tokens: Literal[192, 384]
    paper_ids: list[PaperId] = Field(min_length=1)
    chunk_paper_ids: list[PaperId] = Field(min_length=1)
    chunk_set_manifests: list[ArtifactRef] = Field(min_length=1)
    preprocessing: Literal["unicode_word_punctuation_v1"] = "unicode_word_punctuation_v1"
    token_pattern: Literal[r"\w+|[^\w\s]"] = r"\w+|[^\w\s]"
    regex_implementation: Literal["python3.11.re:unicode:flags=0"] = "python3.11.re:unicode:flags=0"
    implementation: Literal["rank-bm25==0.2.2:BM25Okapi"] = "rank-bm25==0.2.2:BM25Okapi"
    k1: float = Field(default=1.5, ge=1.5, le=1.5)
    b: float = Field(default=0.75, ge=0.75, le=0.75)
    epsilon: float = Field(default=0.25, ge=0.25, le=0.25)
    statistics: Bm25Statistics
    configuration: ConfigurationRef
    chunk_sets: list[ArtifactRef] = Field(min_length=1)
    ordered_chunk_ids: list[Nonblank] = Field(min_length=1)
    tokenized_corpus: list[list[Nonblank]]

    @model_validator(mode="after")
    def valid_rows(self) -> Self:
        _unique(self.paper_ids, "index paper")
        if self.paper_ids != sorted(set(self.chunk_paper_ids)):
            raise ValueError("index paper membership differs from rows")
        if len(self.chunk_set_manifests) != len(self.chunk_sets):
            raise ValueError("index chunk-set manifests differ from inputs")
        if len(self.chunk_paper_ids) != len(self.ordered_chunk_ids):
            raise ValueError("index paper rows differ from chunk IDs")
        _unique(self.ordered_chunk_ids, "index chunk")
        _unique([item.artifact_id for item in self.chunk_sets], "index chunk set")
        if len(self.ordered_chunk_ids) != len(self.tokenized_corpus):
            raise ValueError("BM25 token rows and chunk IDs differ")
        if any(not row for row in self.tokenized_corpus):
            raise ValueError("BM25 cannot index empty token rows")
        return self


class DenseExecution(Record):
    model: Snapshot
    tokenizer: Snapshot
    query_instruction: Literal["Represent this sentence for searching relevant passages: "] = (
        "Represent this sentence for searching relevant passages: "
    )
    passage_instruction: Literal[""] = ""
    preprocessing: Literal["bert_uncased_complete_special_tokens_v1"] = (
        "bert_uncased_complete_special_tokens_v1"
    )
    capacity: Literal[512] = 512
    dimension: Literal[768] = 768
    pooling: Literal["final_hidden_state_cls"] = "final_hidden_state_cls"
    normalization: Literal["l2"] = "l2"
    device: Literal["cpu"] = "cpu"
    dtype: Literal["float32"] = "float32"
    attention: Literal["eager"] = "eager"
    evaluation: Literal[True] = True
    inference_only: Literal[True] = True
    deterministic_algorithms: Literal[True] = True
    batch_size: int = Field(gt=0)
    threads: int = Field(gt=0)
    interop_threads: int = Field(gt=0)
    versions: dict[Nonblank, Nonblank] = Field(min_length=1)


class DenseQuery(Record):
    token_count: int = Field(gt=0, le=512)
    model_input_sha256: Sha256
    vector: list[float] = Field(min_length=1)


class DenseIndex(Record):
    schema_version: Literal["2"] = "2"
    configuration: ConfigurationRef
    representation: Representation
    chunk_tokens: Literal[192, 384]
    paper_ids: list[PaperId] = Field(min_length=1)
    chunk_paper_ids: list[PaperId] = Field(min_length=1)
    chunk_set_manifests: list[ArtifactRef] = Field(min_length=1)
    execution: DenseExecution
    chunk_sets: list[ArtifactRef] = Field(min_length=1)
    ordered_chunk_ids: list[Nonblank] = Field(min_length=1)
    vectors: ArtifactRef
    vectors_build_manifest: ArtifactRef
    dimension: int = Field(gt=0)
    vector_count: int = Field(gt=0)
    shape: list[int] = Field(min_length=2, max_length=2)
    input_token_counts: list[Annotated[int, Field(gt=0)]]
    input_id_hashes: list[Sha256]
    dtype: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    similarity: Literal["exact_dot_product"] = "exact_dot_product"

    @model_validator(mode="after")
    def valid_rows(self) -> Self:
        _unique(self.ordered_chunk_ids, "index chunk")
        _unique([item.artifact_id for item in self.chunk_sets], "index chunk set")
        if len(self.ordered_chunk_ids) != len(self.input_token_counts):
            raise ValueError("dense input counts and chunk IDs differ")
        if len(self.input_id_hashes) != len(self.ordered_chunk_ids):
            raise ValueError("dense input hashes and chunk IDs differ")
        if len(self.chunk_paper_ids) != len(self.ordered_chunk_ids):
            raise ValueError("dense paper rows and chunk IDs differ")
        if self.paper_ids != sorted(set(self.chunk_paper_ids)):
            raise ValueError("dense paper membership differs")
        if len(self.chunk_set_manifests) != len(self.chunk_sets):
            raise ValueError("dense chunk-set manifests differ")
        if self.vector_count != len(self.ordered_chunk_ids) or self.shape != [
            self.vector_count,
            self.dimension,
        ]:
            raise ValueError("dense vector count/shape differs from ordered membership")
        return self


class RankedHit(Record):
    rank: int = Field(ge=1, le=20)
    chunk: ArtifactRef
    paper_id: PaperId
    score: float


class RankedRetrieval(Record):
    schema_version: Literal["1"] = "1"
    configuration: ConfigurationRef
    index: ArtifactRef
    index_build_manifest: ArtifactRef | None = None
    retriever: Literal["bm25", "bge"]
    query_id: Nonblank
    question: Nonblank
    representation: Representation
    scope: Literal["designated_paper", "full_corpus"]
    designated_paper_id: PaperId | None
    hits: list[RankedHit] = Field(max_length=20)
    dense_query: DenseQuery | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def valid_ranks(self) -> Self:
        if (self.retriever == "bge") != (self.dense_query is not None):
            raise ValueError("BGE retrieval requires complete query evidence; BM25 has none")
        _unique([item.chunk.artifact_id for item in self.hits], "ranked chunk")
        if [item.rank for item in self.hits] != list(range(1, len(self.hits) + 1)):
            raise ValueError("retrieval ranks must be contiguous in supplied order")
        if (self.scope == "designated_paper") != (self.designated_paper_id is not None):
            raise ValueError("designated-paper scope requires its paper ID")
        if self.scope == "designated_paper" and any(
            item.paper_id != self.designated_paper_id for item in self.hits
        ):
            raise ValueError("ranked hit is outside designated paper")
        keys = [(-item.score, item.chunk.artifact_id) for item in self.hits]
        if keys != sorted(keys):
            raise ValueError("ranking violates score/chunk-ID ordering")
        return self


class ContextDecision(Record):
    rank: int = Field(ge=1, le=20)
    chunk_id: Nonblank
    decision: Literal["included", "duplicate", "nonfitting", "not_processed"]
    duplicate_of_rank: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def valid_duplicate(self) -> Self:
        if self.decision == "duplicate":
            if self.duplicate_of_rank is None or self.duplicate_of_rank >= self.rank:
                raise ValueError("duplicate must reference an earlier rank")
        elif self.duplicate_of_rank is not None:
            raise ValueError("only duplicate decisions have a retained rank")
        return self


class ContextRecord(Record):
    schema_version: Literal["1"] = "1"
    configuration: ConfigurationRef
    retrieval: ArtifactRef
    retrieval_build_manifest: ArtifactRef | None = Field(
        default=None, exclude_if=lambda v: v is None
    )
    packing_policy: Nonblank | None = Field(default=None, exclude_if=lambda v: v is None)
    packing_tokenizer: Snapshot | None = Field(default=None, exclude_if=lambda v: v is None)
    representation: Representation
    decisions: list[ContextDecision] = Field(max_length=20)
    included_chunk_ids: list[Nonblank]
    exact_context: str
    generator_tokens: int = Field(ge=0, le=2048)
    stop_reason: Literal["exhausted", "first_nonfitting", "top20"]

    @model_validator(mode="after")
    def valid_decisions(self) -> Self:
        if [item.rank for item in self.decisions] != list(range(1, len(self.decisions) + 1)):
            raise ValueError("context decision ranks must be contiguous")
        included = [item.chunk_id for item in self.decisions if item.decision == "included"]
        if included != self.included_chunk_ids:
            raise ValueError("included chunks differ from ordered context decisions")
        _unique(included, "included chunk")
        stopped = False
        retained_ranks = {item.rank for item in self.decisions if item.decision == "included"}
        for item in self.decisions:
            if item.decision == "duplicate" and item.duplicate_of_rank not in retained_ranks:
                raise ValueError("duplicate decision must reference a retained inclusion")
            if stopped and item.decision != "not_processed":
                raise ValueError("context cannot process candidates after first nonfitting chunk")
            if item.decision == "nonfitting":
                stopped = True
            elif item.decision == "not_processed" and not stopped:
                raise ValueError("unprocessed candidate precedes stopping point")
        if stopped != (self.stop_reason == "first_nonfitting"):
            raise ValueError("context stopping reason differs from decisions")
        if not included and (self.exact_context or self.generator_tokens):
            raise ValueError("empty inclusion requires empty context and zero evidence tokens")
        return self
