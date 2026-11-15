"""Build the generator contexts of a development batch for the five frozen arms.

Candidate builder, development items only, generalised from the Dev35 builder: the frozen
sources are verified, every canonical document is read by hash, chunks are built with the
package's chunking, ranked with its BM25 or BGE code and packed with its packing rule.

    python scripts/build_contexts.py --selection SELECTION.json --out contexts.json

The selection names, by path and SHA-256, everything that is used: the corpus release record,
the development benchmark view, one retrieval configuration per retrieval arm (with the
logical reference that enters every chunk identity), the tokenizer
and BGE snapshots, and the Phase 4 validation record of each (paper, representation). It also
names the items' papers and the pool papers whose chunks give the BM25 statistics; the pool is
the "full corpus" of the scope arm. The output is written once and its SHA-256 is printed for
the runner's --contexts-sha256.
"""

import argparse
import json
from pathlib import Path

from biorag_repro.benchmark.development_view import read_development_view
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.ingestion import Phase4Validation, load_frozen_paper
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.evaluation.analysis_table import ARM_RETRIEVAL
from biorag_repro.retrieval import bm25
from biorag_repro.retrieval.chunking import chunk_document, load_construction_counter
from biorag_repro.retrieval.context import QwenCounter, pack, tokenizer_snapshot
from biorag_repro.retrieval.models import CanonicalInput, ConfigurationRef, Phase5Configuration
from biorag_repro.retrieval.storage import record_ref

REPRESENTATIONS = FROZEN_MATRIX["confirmatory"][0]
CHUNK_TOKENS = {"chunk_robustness": 192}  # every other retrieval arm uses 384


def load_configuration(entry: dict) -> tuple[Phase5Configuration, ConfigurationRef]:
    raw = Path(entry["path"]).read_bytes()
    settings = Phase5Configuration.model_validate_json(raw)
    resolved = sha256_bytes(canonical_json_bytes(settings.model_dump()))
    if resolved != entry["resolved_sha256"]:
        raise SystemExit(f"refused: {entry['path']} does not resolve to the selected hash")
    ref = ConfigurationRef(
        artifact=ArtifactRef(
            artifact_id="configuration_" + resolved,
            logical_ref=entry["logical_ref"],  # part of every chunk identity
            sha256=sha256_bytes(raw),
        ),
        resolved_sha256=resolved,
    )
    return settings, ref


def load_canonical(selection: dict, paper: str, representation: str, raw_sha256: str):
    entry = selection["canonical"][f"{paper}/{representation}"]
    root = Path(entry["root"])
    ref = ArtifactRef(**{k: entry[k] for k in ("artifact_id", "logical_ref", "sha256")})
    validation = read_project_json(root / ref.logical_ref, ref.sha256, Phase4Validation)
    if validation.source.sha256 != raw_sha256:
        raise SystemExit(f"refused: {paper} {representation} was not built from the frozen source")
    canonical = validation.canonical_artifact
    document = read_project_json(root / canonical.logical_ref, canonical.sha256, CanonicalDocument)
    if (document.paper_id, document.representation) != (paper, representation):
        raise SystemExit(f"refused: canonical document is not {paper} {representation}")
    selected = CanonicalInput(
        corpus_version=selection["corpus_version"],
        corpus_release_sha256=selection["release_sha256"],
        paper_id=paper,
        representation=representation,
        canonical=canonical,
        validation=ref,
        build_manifest=validation.build_manifest,
    )
    return document, selected


def ranked_candidates(arm, item, chunks, ids, scorers, dense) -> list:
    """Top candidates of one query, in the frozen order: score, then chunk id."""
    scope, retriever = ARM_RETRIEVAL[arm]
    designated = item.paper_id if scope == "designated_paper" else None
    if retriever == "bm25":
        scores = scorers.get_scores(bm25.tokenize(item.question))
        eligible = [(n, float(s)) for n, s in enumerate(scores)
                    if designated is None or chunks[n].input.paper_id == designated]  # fmt: skip
        eligible.sort(key=lambda pair: (-pair[1], ids[pair[0]]))
        return [chunks[n] for n, _ in eligible[:20]]
    from biorag_repro.retrieval.dense import rank_vectors

    encoder, vectors = dense
    query, _ = encoder.encode([item.question], [item.item_id], "query")
    papers = [c.input.paper_id for c in chunks]
    return [chunks[n] for n, _ in rank_vectors(vectors, query[0], ids, papers, designated)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_bytes())
    arms = selection["arms"]
    if any(arm not in FROZEN_MATRIX for arm in arms):
        raise SystemExit("refused: an arm outside the frozen matrix")

    # 1. frozen sources and canonical documents of every pool paper
    pool, item_papers = selection["pool_papers"], selection["item_papers"]
    if not set(item_papers) <= set(pool):
        raise SystemExit("refused: every item paper must be in the pool")
    documents, provenance = {}, {}
    for paper in pool:
        _, record = load_frozen_paper(
            Path(selection["corpus_root"]), Path(selection["release_record"]),
            selection["release_sha256"], selection["corpus_version"], paper,
        )  # fmt: skip
        for representation, source in zip(REPRESENTATIONS, (record.jats, record.pdf), strict=True):
            document, selected = load_canonical(selection, paper, representation, source.sha256)
            documents[paper, representation] = (document, selected)
            provenance[f"{paper}/{representation}"] = {
                "raw_sha256": source.sha256,
                "canonical_sha256": selected.canonical.sha256,
                "validation_sha256": selected.validation.sha256,
            }

    # 2. development items only
    view, all_items = read_development_view(Path(selection["view_dir"]), selection["view_sha256"])
    if any(item.split != "development" for item in all_items):
        raise SystemExit("refused: the view contains a non-development item")
    items = [item for item in all_items if item.paper_id in item_papers]

    # 3. per retrieval arm: chunks of the pool, a ranking per item, then the packing rule
    qwen = QwenCounter(Path(selection["qwen_tokenizer_root"]), tokenizer_snapshot())
    records, chunk_sets, configurations = [], {}, {}
    for arm in [a for a in arms if a != "closed_book"]:
        settings, ref = load_configuration(selection["retrieval_configs"][arm])
        scope, _retriever = ARM_RETRIEVAL[arm]
        if (settings.chunk_tokens, settings.scope) != (CHUNK_TOKENS.get(arm, 384), scope):
            raise SystemExit(f"refused: the {arm} configuration has the wrong chunk size or scope")
        configurations[arm] = ref.resolved_sha256
        count = load_construction_counter(settings, Path(selection["bge_tokenizer_root"]))
        for representation in REPRESENTATIONS:
            chunks = []
            for paper in pool:
                document, selected = documents[paper, representation]
                result = chunk_document(document, selected, ref, settings, count)
                chunks += result.chunks
                chunk_sets[f"{arm}/{paper}/{representation}"] = record_ref(
                    result.chunk_set
                ).artifact_id
            chunks.sort(key=lambda c: record_ref(c).artifact_id)
            ids = [record_ref(c).artifact_id for c in chunks]
            scorer = bm25.scorer([bm25.tokenize(c.rendered_text) for c in chunks])
            dense = None
            if arm == "dense_robustness":
                from biorag_repro.retrieval.dense import BgeEncoder

                encoder = BgeEncoder(Path(selection["bge_model_root"]))
                dense = (encoder, encoder.encode([c.rendered_text for c in chunks], ids,
                                                 "passage")[0])  # fmt: skip
            for item in items:
                top = ranked_candidates(arm, item, chunks, ids, scorer, dense)
                decisions, text, tokens, stop = pack(top, qwen)
                if not text or tokens != qwen(text) or tokens > settings.evidence_tokens:
                    raise SystemExit(f"refused: no valid context for {item.item_id} in {arm}")
                records.append(record_for(item, view, arm, representation, text, tokens, stop,
                                          decisions, top, ref.resolved_sha256))  # fmt: skip
    if "closed_book" in arms:
        for item in items:
            records.append(record_for(item, view, "closed_book", "CLOSED_BOOK", None, None, None,
                                      [], [], None))  # fmt: skip

    leak = {"evidence_sets", "accepted_answers", "jats_refs", "pdf_refs", "gold", "scoring"}
    if any(set(r) & leak for r in records):
        raise SystemExit("refused: answer-key material in a generator record")
    package = {
        "header": {
            "status": "development_candidate_not_official",
            "benchmark_view_sha256": selection["view_sha256"],
            "benchmark_version": view.benchmark_version,
            "corpus_release_sha256": selection["release_sha256"],
            "pool_papers": pool,
            "item_papers": item_papers,
            "item_ids": [item.item_id for item in items],
            "arms": arms,
            "retrieval_configurations": configurations,
            "paper_provenance": provenance,
            "chunk_set_ids": chunk_sets,
            "selection_sha256": sha256_file(args.selection),
        },
        "records": records,
    }
    data = json.dumps(package, ensure_ascii=False, indent=1).encode("utf-8") + b"\n"
    with open(args.out, "xb") as handle:
        handle.write(data)
    print(f"{len(records)} records; contexts sha256 {sha256_bytes(data)}")


def record_for(item, view, arm, condition, text, tokens, stop, decisions, top, config_sha256):
    return {
        "item_id": item.item_id,
        "paper_id": item.paper_id,
        "benchmark_version": view.benchmark_version,
        "split": item.split,
        "answer_class": item.answer_class,
        "question": item.question,
        "matrix_arm": arm,
        "condition": condition,
        "context": text,
        "context_sha256": None if text is None else sha256_bytes(text.encode("utf-8")),
        "context_tokens": tokens,
        "context_stop": stop,
        "included_chunk_ids": [d.chunk_id for d in decisions if d.decision == "included"],
        "candidate_chunk_ids": [record_ref(c).artifact_id for c in top],
        "retrieval_config_sha256": config_sha256,
    }


if __name__ == "__main__":
    main()
