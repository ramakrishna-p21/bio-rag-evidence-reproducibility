"""Complete-input BGE inference and exact retrieval over immutable NumPy vectors."""

import io
import json
import platform
from contextvars import ContextVar
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes
from biorag_repro.core.identity import ArtifactInput, derived_artifact_id
from biorag_repro.core.manifest import ArtifactRef, BuildInput, BuildManifest, DerivedOutput
from biorag_repro.core.storage import (
    publish_bytes,
    publish_manifest,
    read_manifest,
    read_project_json,
    serialize_project_json,
    verify_manifest_outputs,
)
from biorag_repro.documents.models import Record
from biorag_repro.retrieval.bm25 import ChunkSelection, Query, _selection_key
from biorag_repro.retrieval.chunking import read_chunking
from biorag_repro.retrieval.models import (
    Chunk,
    ConfigurationRef,
    DenseIndex,
    DenseQuery,
    RankedHit,
    RankedRetrieval,
    Sha256,
    Snapshot,
)
from biorag_repro.retrieval.models import (
    DenseExecution as Execution,
)
from biorag_repro.retrieval.storage import record_ref
from biorag_repro.retrieval.validation import load_configuration

QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
FloatArray = NDArray[np.float32]


class InputEvidence(Record):
    reference: str
    token_count: int
    model_input_sha256: Sha256


def prepare_inputs(
    tokenizer: Any,
    texts: list[str],
    references: list[str],
    role: Literal["query", "passage"],
    capacity: int,
) -> list[dict[str, list[int]]]:
    if not texts or len(texts) != len(references) or any(not t.strip() for t in texts):
        raise ValueError("missing dense inputs/references")
    if role not in ("query", "passage"):
        raise ValueError("unknown dense input role")
    complete = [QUERY_INSTRUCTION + t if role == "query" else t for t in texts]
    encoded = tokenizer(complete, add_special_tokens=True, truncation=False, padding=False)
    rows = [{k: v[n] for k, v in encoded.items()} for n in range(len(texts))]
    for ref, row in zip(references, rows, strict=True):
        length = len(row["input_ids"])
        if length > capacity:
            raise ValueError(f"dense input {ref}: {length} tokens exceeds capacity {capacity}")
        if row["attention_mask"] != [1] * length:
            raise ValueError(f"complete input mask invalid: {ref}")
    return rows


def validate_vectors(vectors: Any, rows: int, dimension: int) -> None:
    if not isinstance(vectors, np.ndarray) or vectors.dtype != np.dtype("float32"):
        raise ValueError("vectors require float32 NumPy array")
    if vectors.shape != (rows, dimension) or rows < 1 or dimension < 1:
        raise ValueError("vector shape/count/dimension mismatch")
    if not np.isfinite(vectors).all():
        raise ValueError("nonfinite vector")
    # Accumulate norms in float64; float32 unit-vector roundoff is below 1e-6.
    norms = np.linalg.norm(vectors.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("zero or unnormalized vector")


def cls_normalize(hidden: Any, dimension: int) -> FloatArray:
    import torch

    if hidden.ndim != 3 or hidden.shape[2] != dimension or hidden.shape[1] < 1:
        raise ValueError("unexpected final hidden-state dimension")
    cls = hidden[:, 0, :]
    norms = torch.linalg.vector_norm(cls, dim=1)
    if not torch.isfinite(cls).all() or not torch.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("nonfinite or zero CLS embedding")
    result: FloatArray = (cls / norms[:, None]).detach().cpu().numpy()
    validate_vectors(result, hidden.shape[0], dimension)
    return result


def encode_inputs(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    references: list[str],
    role: Literal["query", "passage"],
    capacity: int,
    dimension: int,
    batch_size: int,
    device: str,
) -> tuple[FloatArray, list[InputEvidence]]:
    import torch

    if batch_size < 1 or model.training:
        raise ValueError("dense inference requires positive batch size and evaluation mode")
    rows = prepare_inputs(tokenizer, texts, references, role, capacity)
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch_text = texts[start : start + batch_size]
        complete = [QUERY_INSTRUCTION + t if role == "query" else t for t in batch_text]
        batch = tokenizer(
            complete, add_special_tokens=True, truncation=False, padding=True, return_tensors="pt"
        )
        for n, row in enumerate(rows[start : start + batch_size]):
            length = len(row["input_ids"])
            for key, expected in row.items():
                if batch[key][n, :length].tolist() != expected:
                    raise ValueError("batch differs from accepted complete input IDs/mask")
            if (
                batch["attention_mask"][n, length:].any()
                or (batch["input_ids"][n, length:] != tokenizer.pad_token_id).any()
            ):
                raise ValueError("invalid padding after complete input")
        if batch["input_ids"].shape[1] > capacity:
            raise ValueError("padded batch exceeds model capacity")
        with torch.inference_mode():
            output = model(**{k: v.to(device) for k, v in batch.items()})
            if output.last_hidden_state.shape[:2] != batch["input_ids"].shape:
                raise ValueError("complete sequence did not reach final hidden state")
            vectors.append(cls_normalize(output.last_hidden_state, dimension))
    result = np.concatenate(vectors)
    validate_vectors(result, len(texts), dimension)
    evidence = input_evidence(rows, references)
    return result, evidence


def rank_vectors(
    vectors: FloatArray,
    query: FloatArray,
    ids: list[str],
    papers: list[str],
    designated: str | None,
) -> list[tuple[int, float]]:
    validate_vectors(vectors, len(ids), len(query))
    validate_vectors(query[None, :], 1, vectors.shape[1])
    if len(set(ids)) != len(ids) or len(papers) != len(ids):
        raise ValueError("duplicate/missing chunk/vector membership")
    if designated is not None and designated not in papers:
        raise ValueError("designated paper missing from index")
    scores = vectors @ query
    if not np.isfinite(scores).all():
        raise ValueError("nonfinite similarity score")
    eligible = [
        (n, float(s)) for n, s in enumerate(scores) if designated is None or papers[n] == designated
    ]
    return sorted(eligible, key=lambda pair: (-pair[1], ids[pair[0]]))[:20]


def input_evidence(rows: list[dict[str, list[int]]], references: list[str]) -> list[InputEvidence]:
    return [
        InputEvidence(
            reference=ref,
            token_count=len(row["input_ids"]),
            model_input_sha256=sha256_bytes(canonical_json_bytes(row)),
        )
        for ref, row in zip(references, rows, strict=True)
    ]


ASSETS = {
    "README.md": "fd29e6117a8c2bc83f8eafb73f261d35341702dcd95c9cc0b34f5be75afa8231",
    "config.json": "bc00af31a4a31b74040d73370aa83b62da34c90b75eb77bfa7db039d90abd591",
    "model.safetensors": "c7c1988aae201f80cf91a5dbbd5866409503b89dcaba877ca6dba7dd0a5167d7",
    "1_Pooling/config.json": "c9bef85e8bbf4b2eab4941b3fb62bd33f88686748b478f2e264d256472d9643b",
    "sentence_bert_config.json": "84e39fda68ccbff05bfa723ae9c0e70e23e2ec373b76e0f8c6e71af72a693cbf",
    "tokenizer.json": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    "tokenizer_config.json": "9261e7d79b44c8195c1cada2b453e55b00aeb81e907a6664974b4d7776172ab3",
    "special_tokens_map.json": "b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3",
    "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
}
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


# The CPU wheel, or the CUDA wheel of the same release pinned by the approved generation lock.
# Inference stays on the CPU in float32 either way; the installed build is recorded in Execution.
TORCH_BUILDS = ("2.15.0+cpu", "2.15.0+cu132")


def check_torch_build(installed: str) -> str:
    if installed not in TORCH_BUILDS:
        raise ValueError("dense inference dependency version mismatch")
    return installed


class BgeEncoder:
    """One verified local snapshot, CPU float32/eager inference; no Hub fallback."""

    def __init__(self, root: Path, *, batch_size: int = 8, threads: int = 4):
        import torch
        from transformers import AutoModel, AutoTokenizer

        from biorag_repro.core.hashing import verify_file_sha256

        for name, digest in ASSETS.items():
            verify_file_sha256(root / name, digest)
        required = {
            "transformers": "4.57.6",
            "tokenizers": "0.22.2",
            "torch": check_torch_build(version("torch")),
            "numpy": "2.4.6",
            "safetensors": "0.8.0",
            "huggingface-hub": "0.36.2",
        }
        if any(version(name) != expected for name, expected in required.items()):
            raise ValueError("dense inference dependency version mismatch")
        if threads < 1 or batch_size < 1:
            raise ValueError("invalid inference batch/thread settings")
        torch.set_num_threads(threads)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        self.tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
            root, local_files_only=True, trust_remote_code=False
        )
        self.model = (
            AutoModel.from_pretrained(
                root,
                local_files_only=True,
                trust_remote_code=False,
                use_safetensors=True,
                dtype=torch.float32,
                attn_implementation="eager",
            )
            .to("cpu")
            .eval()
        )
        cfg = self.model.config
        if (cfg.model_type, cfg.max_position_embeddings, cfg.hidden_size, cfg.vocab_size) != (
            "bert",
            512,
            768,
            30522,
        ):
            raise ValueError("unapproved BGE architecture/capacity/dimension")
        if (
            self.tokenizer.model_max_length != 512
            or self.tokenizer.padding_side != "right"
            or not self.tokenizer.is_fast
        ):
            raise ValueError("unexpected BGE tokenizer configuration")
        if (
            self.tokenizer.get_vocab()
            != json.loads((root / "tokenizer.json").read_bytes())["model"]["vocab"]
        ):
            raise ValueError("loaded tokenizer vocabulary differs from verified assets")
        self.execution = Execution(
            model=Snapshot(
                repository="BAAI/bge-base-en-v1.5",
                state="pinned",
                revision=REVISION,
                file_hashes=ASSETS,
            ),
            tokenizer=Snapshot(
                repository="BAAI/bge-base-en-v1.5",
                state="pinned",
                revision=REVISION,
                file_hashes={k: ASSETS[k] for k in TOKENIZER_FILES},
            ),
            batch_size=batch_size,
            threads=threads,
            interop_threads=torch.get_num_interop_threads(),
            versions={
                **required,
                "python": platform.python_version(),
                "machine": platform.machine(),
                "torch_build_sha256": sha256_bytes(torch.__config__.show().encode()),
                "mkldnn_enabled": str(torch.backends.mkldnn.enabled),
                "matmul_precision": "highest",
            },
        )

    def encode(
        self, texts: list[str], references: list[str], role: Literal["query", "passage"]
    ) -> tuple[FloatArray, list[InputEvidence]]:
        return encode_inputs(
            self.model,
            self.tokenizer,
            texts,
            references,
            role,
            self.execution.capacity,
            self.execution.dimension,
            self.execution.batch_size,
            self.execution.device,
        )


def _selected(
    sources: list[ChunkSelection],
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
) -> tuple[list[ChunkSelection], list[Chunk]]:
    if not sources or len({s.chunk_set.artifact_id for s in sources}) != len(sources):
        raise ValueError("empty/duplicate selected chunk sets")
    selected = sorted(sources, key=lambda s: s.chunk_set.artifact_id)
    settings = load_configuration(configuration, configuration_root)
    if settings.purpose != "synthetic_fixture" and not isinstance(encoder, BgeEncoder):
        raise ValueError("development dense inference requires verified BgeEncoder")
    Execution.model_validate(encoder.execution.model_dump())
    if settings.purpose != "synthetic_fixture" and settings.bge_model != encoder.execution.model:
        raise ValueError("execution differs from selected BGE model configuration")
    chunks: list[Chunk] = []
    papers, releases = [], set()
    for source in selected:
        upstream = load_configuration(source.configuration, source.configuration_root)
        if (
            settings.model_copy(update={"bge_model": upstream.bge_model, "scope": upstream.scope})
            != upstream
        ):
            raise ValueError("dense configuration changes upstream chunk scientific settings")
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
        if not result.chunks:
            raise ValueError("empty chunk input")
        chunks.extend(result.chunks)
        papers.append(result.chunk_set.input.paper_id)
        releases.add(result.chunk_set.input.corpus_release_sha256)
    if (
        len(set(papers)) != len(papers)
        or len(releases) != 1
        or len({c.input.representation for c in chunks}) != 1
    ):
        raise ValueError("invalid paper/release/representation membership")
    chunks.sort(key=lambda c: record_ref(c).artifact_id)
    if len({record_ref(c).artifact_id for c in chunks}) != len(chunks):
        raise ValueError("duplicate chunk identity")
    return selected, chunks


def vector_bytes(vectors: FloatArray) -> bytes:
    stream = io.BytesIO()
    np.save(stream, vectors, allow_pickle=False)
    return stream.getvalue()


def _inputs(value: DenseIndex) -> list[BuildInput]:
    return [
        BuildInput(role=role, artifact=ref)
        for pair in zip(value.chunk_sets, value.chunk_set_manifests, strict=True)
        for role, ref in zip(("chunk_set", "chunk_set_build_manifest"), pair, strict=True)
    ] + [BuildInput(role="configuration", artifact=value.configuration.artifact)]


def _vector_settings(value: DenseIndex) -> str:
    return sha256_bytes(
        canonical_json_bytes(
            {
                "configuration": value.configuration.resolved_sha256,
                "execution": value.execution.model_dump(),
                "ordered_chunk_ids": value.ordered_chunk_ids,
            }
        )
    )


def vector_ref(vectors: FloatArray, value: DenseIndex) -> ArtifactRef:
    digest = sha256_bytes(vector_bytes(vectors))
    identity = derived_artifact_id(
        artifact_kind="dense_vectors",
        inputs=[
            ArtifactInput(i.role, i.artifact.artifact_id, i.artifact.sha256) for i in _inputs(value)
        ],
        transformation_id="bge_cls_l2_v1",
        transformation_settings_sha256=_vector_settings(value),
        output_sha256=digest,
        output_format="npy",
        serialization_version="1",
    )
    return ArtifactRef(
        artifact_id=identity,
        logical_ref=f"corpus/derived/phase5/dense_vectors/{identity}.npy",
        sha256=digest,
    )


def vector_manifest(value: DenseIndex) -> BuildManifest:
    return BuildManifest(
        schema_version="1",
        inputs=_inputs(value),
        transformation_id="bge_cls_l2_v1",
        transformation_settings_sha256=_vector_settings(value),
        status="complete",
        outputs=[
            DerivedOutput(
                artifact=value.vectors,
                artifact_kind="dense_vectors",
                output_format="npy",
                serialization_version="1",
                identity_version="1",
            )
        ],
    )


def _manifest_ref(manifest: BuildManifest, logical: str) -> ArtifactRef:
    digest = sha256_bytes(serialize_project_json(manifest, BuildManifest))
    return ArtifactRef(artifact_id="build_" + digest, logical_ref=logical, sha256=digest)


def _vector_manifest_ref(value: DenseIndex) -> ArtifactRef:
    return _manifest_ref(vector_manifest(value), value.vectors.logical_ref + ".build.json")


def build_index(
    sources: list[ChunkSelection],
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
) -> tuple[DenseIndex, FloatArray]:
    selected, chunks = _selected(sources, configuration, configuration_root, encoder)
    ids = [record_ref(c).artifact_id for c in chunks]
    vectors, evidence = encoder.encode([c.rendered_text for c in chunks], ids, "passage")
    # References are filled from the complete lineage before any publication.
    placeholder = ArtifactRef(artifact_id="pending", logical_ref="pending", sha256="0" * 64)
    value = DenseIndex(
        configuration=configuration,
        representation=chunks[0].input.representation,
        chunk_tokens=load_configuration(configuration, configuration_root).chunk_tokens,
        paper_ids=sorted({c.input.paper_id for c in chunks}),
        chunk_paper_ids=[c.input.paper_id for c in chunks],
        chunk_sets=[s.chunk_set for s in selected],
        chunk_set_manifests=[s.manifest for s in selected],
        ordered_chunk_ids=ids,
        vectors=placeholder,
        vectors_build_manifest=placeholder,
        dimension=encoder.execution.dimension,
        vector_count=len(chunks),
        shape=list(vectors.shape),
        execution=encoder.execution,
        input_token_counts=[e.token_count for e in evidence],
        input_id_hashes=[e.model_input_sha256 for e in evidence],
    )
    validate_vectors(vectors, len(chunks), value.dimension)
    value.vectors = vector_ref(vectors, value)
    value.vectors_build_manifest = _vector_manifest_ref(value)
    return value, vectors


_verified_indexes: ContextVar[dict | None] = ContextVar("verified_dense_indexes", default=None)


def validate_index(
    value: DenseIndex,
    vectors: FloatArray,
    sources: list[ChunkSelection],
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
    *,
    reproduce: bool = False,
) -> list[Chunk]:
    value = DenseIndex.model_validate(value.model_dump())
    load_configuration(configuration, configuration_root)
    cache = _verified_indexes.get()
    key = None
    if cache is not None:
        validate_vectors(vectors, len(value.ordered_chunk_ids), value.dimension)
        key = (
            record_ref(value).sha256,
            vector_ref(vectors, value).sha256,
            _selection_key(sources),
            configuration.model_dump_json(),
            str(configuration_root),
            id(encoder),
            encoder.execution.model_dump_json(),
        )
        if not reproduce and key in cache:
            return list(cache[key][0])
    selected, chunks = _selected(sources, configuration, configuration_root, encoder)
    ids = [record_ref(c).artifact_id for c in chunks]
    rows = prepare_inputs(
        encoder.tokenizer,
        [c.rendered_text for c in chunks],
        ids,
        "passage",
        encoder.execution.capacity,
    )
    evidence = input_evidence(rows, ids)
    if (
        value.configuration != configuration
        or value.execution != encoder.execution
        or value.chunk_sets != [s.chunk_set for s in selected]
        or value.chunk_set_manifests != [s.manifest for s in selected]
        or value.ordered_chunk_ids != ids
        or value.chunk_paper_ids != [c.input.paper_id for c in chunks]
        or value.representation != chunks[0].input.representation
        or value.chunk_tokens != load_configuration(configuration, configuration_root).chunk_tokens
        or value.dimension != encoder.execution.dimension
        or value.input_token_counts != [e.token_count for e in evidence]
        or value.input_id_hashes != [e.model_input_sha256 for e in evidence]
    ):
        raise ValueError("dense index differs from selected scientific inputs/execution")
    validate_vectors(vectors, len(chunks), value.dimension)
    if value.vectors != vector_ref(
        vectors, value
    ) or value.vectors_build_manifest != _vector_manifest_ref(value):
        raise ValueError("dense vector bytes/manifest identity mismatch")
    if reproduce:
        expected, _ = encoder.encode([c.rendered_text for c in chunks], ids, "passage")
        # Same recorded execution settings: require exact retained float32 arrays.
        if not np.array_equal(vectors, expected):
            raise ValueError("vectors differ from independently reproduced model execution")
    if cache is not None:
        cache[key] = (tuple(chunks), tuple(sources), encoder)
    return chunks


def build_manifest(value: DenseIndex | RankedRetrieval) -> BuildManifest:
    if isinstance(value, DenseIndex):
        kind = "dense_index"
        inputs = (
            _inputs(value)[:-1]
            + [
                BuildInput(role="vectors", artifact=value.vectors),
                BuildInput(role="vectors_build_manifest", artifact=value.vectors_build_manifest),
            ]
            + _inputs(value)[-1:]
        )
    else:
        kind = "ranked_retrieval"
        if value.index_build_manifest is None:
            raise ValueError("dense retrieval requires complete index manifest")
        inputs = [
            BuildInput(role="index", artifact=value.index),
            BuildInput(role="index_build_manifest", artifact=value.index_build_manifest),
            BuildInput(role="configuration", artifact=value.configuration.artifact),
        ]
    return BuildManifest(
        schema_version="1",
        inputs=inputs,
        transformation_id=f"{kind}_record_v1",
        transformation_settings_sha256=value.configuration.resolved_sha256,
        status="complete",
        outputs=[
            DerivedOutput(
                artifact=record_ref(value),
                artifact_kind=kind,
                output_format="json",
                serialization_version="1",
                identity_version="1",
            )
        ],
    )


def manifest_ref(value: DenseIndex | RankedRetrieval) -> ArtifactRef:
    return _manifest_ref(build_manifest(value), record_ref(value).logical_ref + ".dense.build.json")


def _publish(
    value: DenseIndex | RankedRetrieval, root: Path, *, reuse_identical: bool = False
) -> tuple[ArtifactRef, ArtifactRef]:
    ref, mr = record_ref(value), manifest_ref(value)
    target = root / ref.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        serialize_project_json(value, type(value)),
        ref.sha256,
        validate_temp=lambda p: read_project_json(p, ref.sha256, type(value)),
        reuse_identical=reuse_identical,
    )
    publish_manifest(
        build_manifest(value),
        root / mr.logical_ref,
        {ref.logical_ref: target},
        reuse_identical=reuse_identical,
    )
    return ref, mr


def publish_index(
    value: DenseIndex,
    vectors: FloatArray,
    sources: list[ChunkSelection],
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
    root: Path,
    *,
    reuse_identical: bool = False,
) -> tuple[ArtifactRef, ArtifactRef]:
    validate_index(value, vectors, sources, configuration, configuration_root, encoder)
    target = root / value.vectors.logical_ref
    target.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        target,
        vector_bytes(vectors),
        value.vectors.sha256,
        validate_temp=lambda p: validate_vectors(
            np.load(p, allow_pickle=False), len(value.ordered_chunk_ids), value.dimension
        ),
        reuse_identical=reuse_identical,
    )
    publish_manifest(
        vector_manifest(value),
        root / value.vectors_build_manifest.logical_ref,
        {value.vectors.logical_ref: target},
        reuse_identical=reuse_identical,
    )
    return _publish(value, root, reuse_identical=reuse_identical)


def _read(ref: ArtifactRef, mr: ArtifactRef, root: Path, model: Any) -> Any:
    value = read_project_json(root / ref.logical_ref, ref.sha256, model)
    if ref != record_ref(value) or mr != manifest_ref(value):
        raise ValueError("dense record/manifest reference mismatch")
    manifest = read_manifest(root / mr.logical_ref, mr.sha256)
    if manifest != build_manifest(value):
        raise ValueError("dense complete manifest mismatch")
    verify_manifest_outputs(manifest, {ref.logical_ref: root / ref.logical_ref})
    return value


def read_index(
    ref: ArtifactRef,
    mr: ArtifactRef,
    sources: list[ChunkSelection],
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
    root: Path,
    *,
    reproduce: bool = False,
) -> tuple[DenseIndex, FloatArray]:
    from biorag_repro.core.hashing import verify_file_sha256

    value: DenseIndex = _read(ref, mr, root, DenseIndex)
    vm = read_manifest(
        root / value.vectors_build_manifest.logical_ref, value.vectors_build_manifest.sha256
    )
    if vm != vector_manifest(value):
        raise ValueError("dense vector manifest mismatch")
    path = root / value.vectors.logical_ref
    verify_manifest_outputs(vm, {value.vectors.logical_ref: path})
    verify_file_sha256(path, value.vectors.sha256)
    vectors: FloatArray = np.load(path, allow_pickle=False)
    validate_index(
        value, vectors, sources, configuration, configuration_root, encoder, reproduce=reproduce
    )
    vectors.flags.writeable = False
    return value, vectors


def retrieve(
    index: DenseIndex,
    vectors: FloatArray,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
) -> RankedRetrieval:
    # The query may select a different scope, but no different scientific execution settings.
    settings = load_configuration(configuration, configuration_root)
    index_settings = load_configuration(index.configuration, configuration_root)
    if (
        settings.scope != query.scope
        or settings.model_copy(update={"scope": index_settings.scope}) != index_settings
    ):
        raise ValueError("dense retrieval scope/configuration mismatch")
    chunks = validate_index(
        index, vectors, sources, index.configuration, configuration_root, encoder
    )
    q, evidence = encoder.encode([query.question], [query.query_id], "query")
    ranked = rank_vectors(
        vectors, q[0], index.ordered_chunk_ids, index.chunk_paper_ids, query.designated_paper_id
    )
    return RankedRetrieval(
        configuration=configuration,
        index=record_ref(index),
        index_build_manifest=manifest_ref(index),
        retriever="bge",
        query_id=query.query_id,
        question=query.question,
        representation=index.representation,
        scope=query.scope,
        designated_paper_id=query.designated_paper_id,
        dense_query=DenseQuery(
            token_count=evidence[0].token_count,
            model_input_sha256=evidence[0].model_input_sha256,
            vector=q[0].tolist(),
        ),
        hits=[
            RankedHit(
                rank=n, chunk=record_ref(chunks[i]), paper_id=chunks[i].input.paper_id, score=s
            )
            for n, (i, s) in enumerate(ranked, 1)
        ],
    )


def validate_retrieval(
    value: RankedRetrieval,
    index: DenseIndex,
    vectors: FloatArray,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
) -> None:
    expected = retrieve(index, vectors, sources, query, configuration, configuration_root, encoder)
    if RankedRetrieval.model_validate(value.model_dump()) != expected:
        raise ValueError("dense retrieval differs from selected query, input or exact ranking")


def publish_retrieval(
    value: RankedRetrieval,
    ir: ArtifactRef,
    im: ArtifactRef,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
    root: Path,
    *,
    reuse_identical: bool = False,
) -> tuple[ArtifactRef, ArtifactRef]:
    index_record = read_project_json(root / ir.logical_ref, ir.sha256, DenseIndex)
    index, vectors = read_index(
        ir, im, sources, index_record.configuration, configuration_root, encoder, root
    )
    validate_retrieval(
        value, index, vectors, sources, query, configuration, configuration_root, encoder
    )
    return _publish(value, root, reuse_identical=reuse_identical)


def read_retrieval(
    ref: ArtifactRef,
    mr: ArtifactRef,
    ir: ArtifactRef,
    im: ArtifactRef,
    sources: list[ChunkSelection],
    query: Query,
    configuration: ConfigurationRef,
    configuration_root: Path,
    encoder: Any,
    root: Path,
) -> RankedRetrieval:
    value: RankedRetrieval = _read(ref, mr, root, RankedRetrieval)
    # Index configuration is independently loaded from the selected index record;
    # validate_index still reconciles all upstream settings and the execution pin.
    index_record = read_project_json(root / ir.logical_ref, ir.sha256, DenseIndex)
    index, vectors = read_index(
        ir, im, sources, index_record.configuration, configuration_root, encoder, root
    )
    validate_retrieval(
        value, index, vectors, sources, query, configuration, configuration_root, encoder
    )
    return value
