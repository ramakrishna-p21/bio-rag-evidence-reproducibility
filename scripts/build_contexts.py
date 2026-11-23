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
import os
import time
from datetime import UTC, datetime
from pathlib import Path

from biorag_repro.benchmark.development_view import (
    BENCHMARK_V1_DEVELOPMENT_SHA256,
    read_development_view,
)
from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import read_project_json
from biorag_repro.documents.ingestion import Phase4Validation, load_frozen_paper
from biorag_repro.documents.models import CanonicalDocument
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.evaluation.analysis_table import ARM_RETRIEVAL
from biorag_repro.retrieval import bm25, context, dense
from biorag_repro.retrieval.chunking import (
    chunk_document,
    chunk_set_manifest_ref,
    load_construction_counter,
    publish_chunking,
    read_chunking,
    verified_once,
)
from biorag_repro.retrieval.context import QwenCounter, pack, tokenizer_snapshot
from biorag_repro.retrieval.models import (
    CanonicalInput,
    ChunkSet,
    ConfigurationRef,
    DenseIndex,
    Phase5Configuration,
)
from biorag_repro.retrieval.source_selection import select_pair
from biorag_repro.retrieval.storage import publish_input, record_ref

REPRESENTATIONS = FROZEN_MATRIX["confirmatory"][0]
CHUNK_TOKENS = {"chunk_robustness": 192}  # every other retrieval arm uses 384


def check_scope_pool(arms: list[str], pool: list[str]) -> None:
    """The scope arm searches the frozen corpus, not a development-only pool."""
    if "scope_robustness" in arms and set(pool) != {f"P{i:03d}" for i in range(1, 51)}:
        raise SystemExit("refused: scope_robustness requires all 50 frozen papers")
    if len(pool) != len(set(pool)):
        raise SystemExit("refused: a paper occurs twice in the retrieval pool")


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


def check_final_test_roster(items: list, arms: list[str]) -> None:
    """The complete frozen final-test roster: 305 distinct items on 40 papers, all five arms."""
    if (
        len(items) != 305
        or len({i.item_id for i in items}) != 305
        or any(i.split != "final_test" for i in items)
        or len({i.paper_id for i in items}) != 40
        or sorted(arms) != sorted(FROZEN_MATRIX)
    ):
        raise ValueError("final-test inputs require 305 final-test items, 40 papers and five arms")


def retained_dense_index(root: Path, cfg, representation: str, encoder, progress):
    """The published dense index of this configuration and representation, if one exists."""
    found = None
    for path in sorted((root / "corpus/derived/phase5/dense_index").glob("drv_*.json")):
        if path.name.count(".") != 1:
            continue
        value = read_project_json(path, sha256_file(path), DenseIndex)
        if (value.configuration, value.representation) != (cfg, representation):
            continue
        if value.execution != encoder.execution:
            progress(f"dense {representation} retained index has another execution; not reused")
            continue
        if found is not None:
            raise ValueError("two retained dense indexes for one configuration")
        found = (record_ref(value), dense.manifest_ref(value))
    return found


def build_published(
    selection: dict,
    destination: Path,
    output_root: Path,
    *,
    resume: bool = False,
    final_test: tuple[str, list] | None = None,
) -> None:
    """Publish a declared development subset, or the complete final-test roster.

    `final_test` is (benchmark release SHA-256, items) from `read_final_test_items`, which
    the command line reads only after execution authorization has passed.
    """
    if destination.exists():
        raise FileExistsError(destination)
    if final_test is not None:
        check_final_test_roster(final_test[1], selection["arms"])
    check_scope_pool(selection["arms"], selection["pool_papers"])
    arms = selection["arms"]
    if not arms or len(set(arms)) != len(arms) or set(arms) - set(FROZEN_MATRIX):
        raise ValueError("unknown or duplicate development arms")
    pool = selection["pool_papers"]
    if any(a != "closed_book" for a in arms):
        check_scope_pool(["scope_robustness"], pool)
    expected = {f"{p}/{a}" for p in pool for a in ("JATS", "PDF_DOCLING")}
    if set(selection["canonical"]) != expected:
        raise ValueError(
            f"published construction requires exactly {len(expected)} source selections"
        )
    if final_test is None and selection.get("view_sha256") != BENCHMARK_V1_DEVELOPMENT_SHA256:
        raise ValueError("published inputs require the exact frozen development-only view")
    import subprocess

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def progress(message):
        stamp = datetime.now(UTC).isoformat()
        print(f"{stamp} elapsed={time.perf_counter() - started:.1f}s {message}", flush=True)

    def config(value, name):
        data = json.dumps(value.model_dump(mode="json"), indent=2).encode() + b"\n"
        digest = sha256_bytes(canonical_json_bytes(value.model_dump()))
        path = output_root / f"{name}.json"
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("configuration publication already contains different bytes")
        else:
            path.write_bytes(data)
        return ConfigurationRef(
            artifact=ArtifactRef(
                artifact_id="configuration_" + digest,
                logical_ref=path.name,
                sha256=sha256_bytes(data),
            ),
            resolved_sha256=digest,
        )

    selected = {}
    for p in pool:
        j, d = (selection["canonical"][p + "/" + a] for a in ("JATS", "PDF_DOCLING"))
        root = Path(j["root"])
        if root != Path(d["root"]):
            raise ValueError("paired selections require one artifact root")
        jr, dr = (
            ArtifactRef(**{k: e[k] for k in ("artifact_id", "logical_ref", "sha256")})
            for e in (j, d)
        )
        auth = select_pair(
            corpus_root=Path(selection["corpus_root"]),
            artifact_root=root,
            release_path=Path(selection["release_record"]),
            release_sha256=selection["release_sha256"],
            paper_id=p,
            jats_validation=jr,
            pdf_validation=dr,
            verification_revision=revision,
        )
        for arm, vr in [("JATS", jr), ("PDF_DOCLING", dr)]:
            v = read_project_json(root / vr.logical_ref, vr.sha256, Phase4Validation)
            doc = read_project_json(
                root / v.canonical_artifact.logical_ref,
                v.canonical_artifact.sha256,
                CanonicalDocument,
            )
            ci = CanonicalInput(
                corpus_version="corpus_v1",
                corpus_release_sha256=selection["release_sha256"],
                paper_id=p,
                representation=arm,
                canonical=v.canonical_artifact,
                validation=vr,
                build_manifest=v.build_manifest,
            )
            publish_input(ci, auth, output_root, reuse_identical=resume)
            selected[p, arm] = (doc, ci, auth)
        progress(f"{p} pair scientifically checked; no authentication claimed")
    completed_chunks = {}
    if resume:
        for path in (output_root / "corpus/derived/phase5/chunk_set").glob("drv_*.json"):
            if path.name.count(".") != 1:
                continue
            metadata = read_project_json(path, sha256_file(path), ChunkSet)
            cr, cm = record_ref(metadata), chunk_set_manifest_ref(metadata)
            if path != output_root / cr.logical_ref:
                raise ValueError("retained chunk-set path differs from exact identity")
            if not (output_root / cm.logical_ref).exists():
                continue
            key = (
                metadata.input.paper_id,
                metadata.input.representation,
                metadata.configuration.resolved_sha256,
            )
            if key in completed_chunks and completed_chunks[key] != (metadata, cr, cm):
                raise ValueError("conflicting retained chunk sets")
            completed_chunks[key] = (metadata, cr, cm)
    if final_test is None:
        view, items = read_development_view(Path(selection["view_dir"]), selection["view_sha256"])
        if (
            len(items) != 86
            or any(i.split != "development" for i in items)
            or set(selection["item_papers"]) != {i.paper_id for i in items}
        ):
            raise ValueError("requires exactly 86 development items")
        split, item_papers = "development", selection["item_papers"]
        benchmark_version = view.benchmark_version
    else:
        items = final_test[1]
        split, item_papers = "final_test", sorted({i.paper_id for i in items})
        (benchmark_version,) = {i.benchmark_version for i in items}
    from biorag_repro.generation.config import QUALIFIED_RETRIEVAL_SHA256

    base, base_ref = load_configuration(selection["retrieval_configs"]["confirmatory"])
    if base_ref.resolved_sha256 != QUALIFIED_RETRIEVAL_SHA256:
        raise ValueError("published inputs require the frozen confirmatory retrieval settings")
    records = []
    indexes = []
    encoder = None
    for arm in arms:
        if arm == "closed_book":
            for i in items:
                records.append(
                    {
                        "item_id": i.item_id,
                        "paper_id": i.paper_id,
                        "benchmark_version": benchmark_version,
                        "split": split,
                        "answer_class": i.answer_class,
                        "question": i.question,
                        "matrix_arm": arm,
                        "condition": "CLOSED_BOOK",
                        "context": None,
                        "context_sha256": None,
                        "context_tokens": None,
                        "retrieval_config_sha256": None,
                    }
                )
            continue
        settings = base.model_copy(
            update={
                "chunk_tokens": 192 if arm == "chunk_robustness" else 384,
                "scope": "full_corpus" if arm == "scope_robustness" else "designated_paper",
            }
        )
        if arm == "dense_robustness":
            encoder = dense.BgeEncoder(Path(selection["bge_model_root"]))
            settings = settings.model_copy(update={"bge_model": encoder.execution.model})
        cfg = config(settings, arm)
        count = load_construction_counter(settings, Path(selection["bge_tokenizer_root"]))
        cc = config(
            settings.model_copy(
                update={
                    "packing_tokenizer": context.tokenizer_snapshot(),
                    "context_rendering": context.POLICY,
                }
            ),
            arm + "_context",
        )
        for rep in ("JATS", "PDF_DOCLING"):
            stage_started = time.perf_counter()
            progress(f"{arm} {rep} starting chunks; papers={len(pool)}")
            with verified_once():
                sources = []
                for p in pool:
                    doc, ci, auth = selected[p, rep]
                    retained = completed_chunks.get((p, rep, cfg.resolved_sha256))
                    if retained is not None:
                        metadata, cr, cm = retained
                        if metadata.input != ci or metadata.configuration != cfg:
                            raise ValueError("retained chunk-set selected lineage differs")
                        result = read_chunking(
                            cr,
                            cm,
                            auth,
                            cfg,
                            output_root,
                            output_root,
                            tokenizer_root=Path(selection["bge_tokenizer_root"]),
                        )
                        state = "reused after scientific verification"
                    else:
                        result = chunk_document(doc, ci, cfg, settings, count)
                        cr, cm = publish_chunking(
                            result,
                            auth,
                            cfg,
                            output_root,
                            output_root,
                            tokenizer_root=Path(selection["bge_tokenizer_root"]),
                            reuse_identical=resume,
                        )
                        state = "published"
                    progress(
                        f"{arm} {rep} {p} {state}; chunks={len(result.chunks)} "
                        f"papers={len(sources) + 1}/{len(pool)} "
                        f"stage_elapsed={time.perf_counter() - stage_started:.1f}s"
                    )
                    sources.append(
                        bm25.ChunkSelection(
                            cr,
                            cm,
                            auth,
                            cfg,
                            output_root,
                            output_root,
                            Path(selection["bge_tokenizer_root"]),
                        )
                    )
                retained_index = (
                    retained_dense_index(output_root, cfg, rep, encoder, progress)
                    if resume and arm == "dense_robustness"
                    else None
                )
                if retained_index is not None:
                    # Passage vectors are the costly step; reuse needs the exact recorded
                    # execution and is re-validated against the selected chunks here.
                    ir, im = retained_index
                    index, vectors = dense.read_index(
                        ir, im, sources, cfg, output_root, encoder, output_root
                    )
                elif arm == "dense_robustness":
                    index, vectors = dense.build_index(sources, cfg, output_root, encoder)
                    ir, im = dense.publish_index(
                        index,
                        vectors,
                        sources,
                        cfg,
                        output_root,
                        encoder,
                        output_root,
                        reuse_identical=resume,
                    )
                else:
                    index = bm25.build_index(sources)
                    ir, im = bm25.publish_index(index, sources, output_root, reuse_identical=resume)

                progress(
                    f"{arm} {rep} index verified; papers={len(index.paper_ids)} "
                    f"chunks={len(index.ordered_chunk_ids)}"
                )

                if set(index.paper_ids) != set(pool):
                    raise ValueError("index differs from the selected source universe")
                indexes.append(
                    {
                        "arm": arm,
                        "representation": rep,
                        "index": ir.model_dump(),
                        "manifest": im.model_dump(),
                        "papers": index.paper_ids,
                    }
                )
                for item_number, i in enumerate(items, 1):
                    q = bm25.Query(
                        query_id=i.item_id,
                        question=i.question,
                        scope=settings.scope,
                        designated_paper_id=i.paper_id
                        if settings.scope == "designated_paper"
                        else None,
                    )
                    if arm == "dense_robustness":
                        ranked = dense.retrieve(
                            index, vectors, sources, q, cfg, output_root, encoder
                        )
                        rr, rm = dense.publish_retrieval(
                            ranked,
                            ir,
                            im,
                            sources,
                            q,
                            cfg,
                            output_root,
                            encoder,
                            output_root,
                            reuse_identical=resume,
                        )
                    else:
                        ranked = bm25.retrieve(index, sources, q, cfg, output_root)
                        rr, rm = bm25.publish_retrieval(
                            ranked,
                            ir,
                            im,
                            sources,
                            q,
                            cfg,
                            output_root,
                            output_root,
                            reuse_identical=resume,
                        )
                    sel = context.Selection(
                        rr,
                        rm,
                        ir,
                        im,
                        sources,
                        q,
                        cfg,
                        output_root,
                        output_root,
                        "bge" if arm == "dense_robustness" else "bm25",
                        encoder,
                    )
                    value = context.assemble(
                        sel, cc, output_root, Path(selection["qwen_tokenizer_root"])
                    )
                    vr, vm = context.publish(
                        value,
                        sel,
                        cc,
                        output_root,
                        Path(selection["qwen_tokenizer_root"]),
                        output_root,
                        reuse_identical=resume,
                    )
                    records.append(
                        {
                            "item_id": i.item_id,
                            "paper_id": i.paper_id,
                            "benchmark_version": benchmark_version,
                            "split": split,
                            "answer_class": i.answer_class,
                            "question": i.question,
                            "matrix_arm": arm,
                            "condition": rep,
                            "context": value.exact_context,
                            "context_sha256": sha256_bytes(value.exact_context.encode()),
                            "context_tokens": value.generator_tokens,
                            "retrieval_config_sha256": cfg.resolved_sha256,
                            "context_record": vr.model_dump(),
                            "context_manifest": vm.model_dump(),
                        }
                    )
                    progress(
                        f"{arm} {rep} context={item_number}/{len(items)} "
                        f"total_conditions={len(records)} "
                        f"stage_elapsed={time.perf_counter() - stage_started:.1f}s"
                    )
                progress(f"{arm} {rep} published")
    expected_conditions = len(items) * sum(len(FROZEN_MATRIX[a][0]) for a in arms)
    if len(records) != expected_conditions:
        raise ValueError("incomplete declared development matrix")
    observations = len(items) * sum(
        len(FROZEN_MATRIX[a][0]) * len(FROZEN_MATRIX[a][1]) for a in arms
    )
    out = {
        "header": {
            "benchmark_version": benchmark_version,
            "corpus_release_sha256": selection["release_sha256"],
            "item_arm_cells": len(items) * len(arms),
            "condition_records": len(records),
            "planned_observations": observations,
            "status": (
                "complete_bounded_development_inputs"
                if set(arms) == set(FROZEN_MATRIX)
                else "partial_bounded_development_inputs"
            ),
            "authenticated_complete_gate": False,
            "item_ids": [i.item_id for i in items],
            "item_papers": item_papers,
            "arms": arms,
            "pool_papers": pool,
            "benchmark_view_sha256": selection.get("view_sha256"),
            "indexes": indexes,
        },
        "records": records,
    }
    if final_test is not None:
        del out["header"]["benchmark_view_sha256"]
        out["header"].update(
            status="complete_final_test_inputs",
            split="final_test",
            benchmark_release_sha256=final_test[0],
        )
    with destination.open("x") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    if final_test is not None:
        os.chmod(destination, 0o444)
    print(
        f"{len(records)} conditions; {len(items) * len(arms)} cells; {observations} observations; "
        f"contexts sha256 {sha256_file(destination)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish-root", type=Path)
    parser.add_argument("--resume", action="store_true")
    final = parser.add_argument_group("final test (only after execution authorization)")
    final.add_argument("--final-test", action="store_true")
    final.add_argument("--repo", type=Path)
    final.add_argument("--data-root", type=Path)
    final.add_argument("--expected-head-sha256")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_bytes())
    if args.final_test:
        from biorag_repro.benchmark.release import read_final_test_items
        from biorag_repro.finaltest.authorization import authorize

        if args.publish_root is None or not all(
            (args.repo, args.data_root, args.expected_head_sha256)
        ):
            raise SystemExit("refused: final test requires --publish-root, --repo, --data-root "
                             "and --expected-head-sha256")  # fmt: skip
        # The same campaign authorization as the runner, before any protected item is read.
        authorize(args.repo, args.data_root, args.expected_head_sha256,
                  custody_mode="amended_drive")  # fmt: skip
        final_test = read_final_test_items(args.data_root)
        build_published(
            selection, args.out, args.publish_root, resume=args.resume, final_test=final_test
        )
        return
    if args.publish_root is not None:
        build_published(selection, args.out, args.publish_root, resume=args.resume)
        return
    arms = selection["arms"]
    if any(arm not in FROZEN_MATRIX for arm in arms):
        raise SystemExit("refused: an arm outside the frozen matrix")

    # 1. frozen sources and canonical documents of every pool paper
    pool, item_papers = selection["pool_papers"], selection["item_papers"]
    check_scope_pool(arms, pool)
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
