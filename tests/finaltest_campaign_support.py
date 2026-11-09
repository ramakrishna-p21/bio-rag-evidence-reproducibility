"""Hand-built miniature benchmark: 1 development paper, 2 final-test papers, 5 final items."""

import hashlib
import json

from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.finaltest.campaign import Population

CANARY = "CANARY-QUESTION-TEXT-DO-NOT-LEAK"
BENCH = "bench_v1"
FINAL_ITEMS = {"i1": "p2", "i2": "p2", "i3": "p3", "i4": "p3", "i5": "p3"}
DEV_ITEMS = {"d1": "p1"}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


CONFIGS = {
    "confirmatory": sha("cfg-confirmatory"),
    "closed_book": sha("cfg-closed"),
    "dense_robustness": sha("cfg-dense"),
    "chunk_robustness": sha("cfg-chunk"),
    "scope_robustness": sha("cfg-scope"),
}
SEEDS = sha("seeds")


def population(**changes: object) -> Population:
    values: dict[str, object] = {
        "benchmark_version": BENCH,
        "corpus_version": "corpus_v1",
        "corpus_content_sha256": sha("corpus"),
        "benchmark_release_sha256": sha("release"),
        "split_sha256": sha("split"),
        "final_test_paper_ids": ("p2", "p3"),
        "development_paper_ids": ("p1",),
        "final_test_item_ids": tuple(sorted(FINAL_ITEMS)),
        "item_papers": dict(FINAL_ITEMS),
    }
    return Population(**(values | changes))  # type: ignore[arg-type]


def oracle_key(item: str, arm: str, condition: str, repeat: int, config: str) -> str:
    """Independent observation key: plain json and hashlib, no project code."""
    payload = {
        "identity_version": "1",
        "phase": "final_test",
        "benchmark_version": BENCH,
        "item_id": item,
        "matrix_arm": arm,
        "representation_or_condition": condition,
        "repeat_index": repeat,
        "resolved_config_sha256": config,
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "obs_" + hashlib.sha256(text.encode()).hexdigest()


def oracle_all(items: list[str]) -> dict[str, set[str]]:
    """Hand-written matrix: confirmatory 2x3, closed book 1x3, three robustness arms 2x1."""
    spec = {
        "confirmatory": (["JATS", "PDF_DOCLING"], [1, 2, 3]),
        "closed_book": (["CLOSED_BOOK"], [1, 2, 3]),
        "dense_robustness": (["JATS", "PDF_DOCLING"], [1]),
        "chunk_robustness": (["JATS", "PDF_DOCLING"], [1]),
        "scope_robustness": (["JATS", "PDF_DOCLING"], [1]),
    }
    return {
        arm: {oracle_key(i, arm, c, r, CONFIGS[arm]) for i in items for c in conds for r in reps}
        for arm, (conds, reps) in spec.items()
    }


def release_fixture(items: int) -> BenchmarkRelease:
    ref = {"artifact_id": "a", "logical_ref": "x", "sha256": sha("x")}
    return BenchmarkRelease.model_validate(
        {
            "benchmark_version": BENCH,
            "corpus_version": "corpus_v1",
            "corpus_content_sha256": sha("corpus"),
            "corpus_release": ref,
            "implementation_git_sha": "a" * 40,
            "seed_registry": ref,
            "relatedness": ref,
            "split": ref,
            "items": ref,
            "decisions": ref,
            "dispositions": ref,
            "pre_audit_population": ref,
            "audit": ref,
            "validation_report": ref,
            "item_count": items,
            "development_paper_count": 10,
            "final_test_paper_count": 40,
        }
    )


class StubSplit:
    """The fields of PaperSplit that population_from reads."""

    def __init__(self, dev: list[str], final: list[str], corpus_version: str = "corpus_v1"):
        self.development_ids, self.final_test_ids = dev, final
        self.corpus_version = corpus_version


def full_scale_inputs():
    """10 development papers with 1 item each, 40 final-test papers with 2 items each."""
    dev = [f"dp{n:02d}" for n in range(10)]
    final = [f"fp{n:02d}" for n in range(40)]
    items = [
        {"item_id": f"d{p}", "paper_id": p, "split": "development", "candidate_status": "accepted"}
        for p in dev
    ] + [
        {
            "item_id": f"{p}-{k}",
            "paper_id": p,
            "split": "final_test",
            "candidate_status": "accepted",
        }
        for p in final
        for k in (1, 2)
    ]
    return dev, final, items
