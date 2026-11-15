"""Candidate final-test readiness check over the frozen list (REPRODUCIBILITY_SPEC section 10).

Report only. It authorizes nothing: the existing readiness mechanism stays authoritative until
its replacement is approved, and scripts/run_generation.py refuses final_test regardless.

    python scripts/check_readiness.py --inputs readiness_inputs.json [--report report.json]

Each check verifies live evidence and reduces it to a digest. A human approval is the matching
digest committed to the tracked approvals/policy.json slot, the same twelve slots as before.
A check passes only if its evidence verifies and its slot holds exactly that digest; a check
whose evidence verifies but whose slot is unset is pending. The history check has no slot: it
passes when every batch under the storage root matches its latest sealed anchor and that
anchor's SHA-256 is in the externally recorded list of trusted anchors. READY needs all 13.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from biorag_repro.core.hashing import canonical_json_bytes, sha256_bytes, sha256_file
from biorag_repro.core.storage import read_project_json
from biorag_repro.corpus.manifest import ReleaseRecord, parse_csv
from biorag_repro.documents.ingestion import frozen_source, load_frozen_paper
from biorag_repro.evaluation.analysis_models import FROZEN_MATRIX
from biorag_repro.generation.config import resolve_generator_config
from biorag_repro.generation.model_snapshot import inventory_snapshot
from biorag_repro.generation.seeds import GenerationSeedRegistry, check_seeds_for_configuration
from biorag_repro.retrieval.context import QwenCounter, tokenizer_snapshot

POLICY = Path("approvals/policy.json")
SCOPE = ("src", "scripts", "configs", "analysis", "pyproject.toml")  # approvals are excluded


class CheckFailed(ValueError):
    pass


def file_digest(path: str) -> dict:
    return {"file": Path(path).name, "sha256": sha256_file(Path(path))}


def corpus_identity(i: dict) -> dict:
    root, record = Path(i["corpus_root"]), Path(i["release_record"])
    release = read_project_json(record, i["release_sha256"], ReleaseRecord)
    papers = parse_csv((root / release.manifest_artifact.logical_ref).read_bytes()).papers
    for paper in papers:
        _, row = load_frozen_paper(root, record, i["release_sha256"], i["corpus_version"],
                                   paper.paper_id)  # fmt: skip
        frozen_source(row, "JATS", root)
        frozen_source(row, "PDF_DOCLING", root)
    if len(papers) != 50:
        raise CheckFailed(f"the frozen corpus has {len(papers)} papers, not 50")
    return {"release_sha256": i["release_sha256"], "papers": len(papers)}


def benchmark_identity(i: dict) -> dict:
    return {"release": file_digest(i["benchmark_release"]), "split": file_digest(i["paper_split"])}


def configurations(i: dict) -> dict:
    configs = {}
    for arm in FROZEN_MATRIX:
        config = resolve_generator_config((Path(i["configs"]) / f"{arm}.toml").read_bytes())
        if config.settings.matrix_arm != arm or config.settings.purpose != "development_candidate":
            raise CheckFailed(f"{arm}.toml is not the qualified development configuration")
        configs[arm] = config
    return configs


def final_test_freeze(i: dict) -> dict:
    return {arm: c.configuration_sha256 for arm, c in configurations(i).items()}


def model_parser_identities(i: dict) -> dict:
    configs = configurations(i)
    revisions = {c.settings.model.revision for c in configs.values()}
    parsers = {c.settings.parser_version for c in configs.values()}
    if len(revisions) != 1 or len(parsers) != 1:
        raise CheckFailed("the arms do not share one model revision and parser")
    snapshot = inventory_snapshot(Path(i["model_root"]), revisions.pop())
    QwenCounter(Path(i["tokenizer_root"]), tokenizer_snapshot())  # verifies the tokenizer files
    return {"model_manifest_sha256": snapshot.manifest_sha256, "parser": parsers.pop()}


def python_environment_lock(i: dict) -> dict:
    lock = Path(i["python_lock"]).read_text().splitlines()
    live = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True,
                          check=True).stdout.splitlines()  # fmt: skip
    drift = sorted(set(lock) ^ set(live))
    if drift:
        raise CheckFailed(f"installed packages differ from the lock: {drift[:5]}")
    return file_digest(i["python_lock"])


def analysis_environment(i: dict) -> dict:
    status = subprocess.run(
        ["Rscript", "-e", "quit(status = !isTRUE(renv::status()$synchronized))"],
        cwd="analysis", capture_output=True, text=True, check=False,
    )  # fmt: skip
    if status.returncode != 0:
        raise CheckFailed("the R library is not synchronized with analysis/renv.lock")
    return file_digest("analysis/renv.lock")


def seed_approvals(i: dict) -> dict:
    seeds = GenerationSeedRegistry.model_validate_json(Path(i["seeds"]).read_bytes())
    for config in configurations(i).values():
        check_seeds_for_configuration(config.settings, seeds)
    return file_digest(i["seeds"])


def scientific_source_identity(i: dict) -> dict:
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout

    changed = [line for line in git("status", "--porcelain", "--", *SCOPE).splitlines() if line]
    if changed:
        raise CheckFailed(f"uncommitted scientific source changes: {changed[:5]}")
    return {path: git("rev-parse", f"HEAD:{path}").strip() for path in SCOPE}


def restart_evidence_acceptance(i: dict) -> dict:
    from scripts.run_generation import check_anchor

    check_anchor(Path(i["restart_batch"]), Path(i["restart_anchor"]), i["restart_anchor_sha256"])
    return {"anchor_sha256": i["restart_anchor_sha256"]}


def storage_authorization(i: dict) -> dict:
    root = Path(i["storage_root"])
    if not root.is_absolute() or not root.is_dir():
        raise CheckFailed("the storage root is not an existing absolute directory")
    return {"root": str(root), "corpus_release_sha256": i["release_sha256"],
            "benchmark_release_sha256": sha256_file(Path(i["benchmark_release"]))}  # fmt: skip


def gpu_authorization(i: dict) -> dict:
    return file_digest(i["gpu_guidance"])


def history(i: dict) -> dict:
    from scripts.run_generation import check_anchor

    trusted = set(Path(i["trusted_anchors"]).read_text().split())
    batches = sorted(p.parent for p in Path(i["storage_root"]).glob("*/batch.json"))
    for batch in batches:
        anchors = sorted((batch / "anchors").glob("[0-9]*.json"))
        if not anchors or sha256_file(anchors[-1]) not in trusted:
            raise CheckFailed(f"{batch.name} has no externally recorded anchor")
        check_anchor(batch, anchors[-1], sha256_file(anchors[-1]))
    return {"batches": len(batches)}


CHECKS: dict[str, tuple[str | None, Callable[[dict], dict]]] = {
    "G01_corpus_sources": ("corpus_identity", corpus_identity),
    "G02_benchmark_freeze": ("benchmark_identity", benchmark_identity),
    "G03_final_test_configuration_freeze": ("final_test_freeze", final_test_freeze),
    "G04_generator_qualification": (
        "generator_qualification_acceptance", lambda i: file_digest(i["qualification_report"])
    ),
    "G05_model_parser_identities": ("model_parser_identities", model_parser_identities),
    "G06_python_environment": ("python_environment_lock", python_environment_lock),
    "G07_r_analysis_environment": ("analysis_environment", analysis_environment),
    "G08_seed_registry": ("seed_approvals", seed_approvals),
    "G09_scientific_source_revision": ("scientific_source_identity", scientific_source_identity),
    "G10_development_restart_recovery": ("restart_evidence_acceptance", restart_evidence_acceptance),
    "G11_storage_authorization": ("storage_authorization", storage_authorization),
    "G12_shared_gpu_authorization": ("gpu_authorization", gpu_authorization),
    "G13_final_test_history_integrity": (None, history),
}  # fmt: skip


def evaluate(inputs: dict, slots: dict, checks=CHECKS) -> dict:
    gates = {}
    for gate, (slot, check) in checks.items():
        try:
            evidence = check(inputs)
        except Exception as error:  # noqa: BLE001 - failing to verify is a failed gate, never a pass
            gates[gate] = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
            continue
        digest = sha256_bytes(canonical_json_bytes(evidence))
        if slot is None:
            status = "passed"
        elif slots.get(slot) is None:
            status = "pending"
        else:
            status = "passed" if slots[slot] == digest else "failed"
        gates[gate] = {"status": status, "slot": slot, "evidence": evidence, "digest": digest}
    ready = all(g["status"] == "passed" for g in gates.values())
    return {"kind": "candidate_readiness_report", "authoritative": False,
            "decision": "READY" if ready else "NOT_READY", "gates": gates}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = evaluate(
        json.loads(args.inputs.read_bytes()), json.loads(POLICY.read_bytes())["slots"]
    )
    for gate, result in report["gates"].items():
        print(f"{gate}: {result['status']} {result.get('reason', '')}".rstrip())
    print(f"{report['decision']} (candidate check; not authoritative)")
    if args.report is not None:
        data = json.dumps(report, indent=1, sort_keys=True).encode() + b"\n"
        with open(args.report, "xb") as handle:
            handle.write(data)
        print(f"report sha256 {sha256_bytes(data)}")


if __name__ == "__main__":
    main()
