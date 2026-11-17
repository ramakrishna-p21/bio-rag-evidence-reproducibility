"""A tiny synthetic checkout and data root with the shapes the real ones have."""

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from synthetic_environment import inventory
from test_corpus import paper
from test_development_view import DEV, FINAL, SPLIT, make_item

from biorag_repro.benchmark.release import BenchmarkRelease
from biorag_repro.benchmark.seeds import prospective_seeds
from biorag_repro.benchmark.storage import encode_jsonl
from biorag_repro.core.hashing import sha256_bytes
from biorag_repro.core.manifest import ArtifactRef
from biorag_repro.core.storage import serialize_project_json
from biorag_repro.corpus.manifest import (
    CorpusManifest,
    PaperRecord,
    ReleaseRecord,
    ValidationReport,
    corpus_content_hash,
    manifest_semantic_hash,
    serialize_csv,
)
from biorag_repro.readiness.evidence import Context, Probes

REAL_REPO = Path(__file__).resolve().parent.parent
R_OUTPUT = (
    "noise from renv startup\n"
    "BIORAG\tlibpaths\t{lib}\t/system\n"
    "BIORAG\trversion\t4.6.1\n"
    "BIORAG\tpkg\tR6\t2.6.1\t1\n"
)
LOCK = '{"R":{"Version":"4.6.1"},"Packages":{"R6":{"Package":"R6","Version":"2.6.1"}}}\n'
SPECS = (
    "SCIENTIFIC_CONTRACT",
    "ARCHITECTURE",
    "BENCHMARK_SPEC",
    "EXPERIMENT_PLAN",
    "REPRODUCIBILITY_SPEC",
    "CORPUS_MANIFEST_SPEC",
    "IMPLEMENTATION_PLAN",
)


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.org", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def write(path: Path, content: str | bytes) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = content.encode() if isinstance(content, str) else content
    path.write_bytes(data)
    return data


@dataclass
class World:
    repo: Path
    data: Path
    commit: str
    ctx: Context

    def probes(self, **changes) -> Context:
        values = {
            "inventory": lambda: inventory(),
            "r_runtime": lambda d, p: R_OUTPUT.format(lib=d / "renv/library/x"),
        }
        return Context(self.repo, self.data, Probes(**(values | changes)))


def build_repo(root: Path) -> tuple[Path, str]:
    repo = root / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    write(repo / ".gitignore", "docs/frozen/\nout/\n")
    write(repo / "src/code.py", "x = 1\n")
    write(repo / "schemas/s.json", "{}\n")
    write(repo / "scripts/s.py", "y = 1\n")
    write(repo / "pyproject.toml", "[project]\n")
    (repo / "configs").mkdir()
    shutil.copy(REAL_REPO / "configs/environment.toml", repo / "configs/environment.toml")
    write(repo / "analysis/renv.lock", LOCK)
    write(repo / "analysis/R/a.R", "a <- 1\n")
    write(repo / "analysis/run_analysis.R", "b <- 1\n")
    (repo / "approvals").mkdir()
    shutil.copy(REAL_REPO / "approvals/policy.json", repo / "approvals/policy.json")
    shutil.copy(
        REAL_REPO / "approvals/researcher_decisions.json",
        repo / "approvals/researcher_decisions.json",
    )
    for name in SPECS:
        write(repo / f"docs/frozen/{name}.md", f"frozen {name}\n")
    write(
        repo / "docs/frozen/RESEARCHER_AUTHORITY_AMENDMENT_V1.md", "prospective authority fixture\n"
    )
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "initial")
    return repo, git(repo, "rev-parse", "HEAD")


def build_data(root: Path, commit: str) -> Path:
    data = root / "data"
    rows = []
    for i in range(1, 51):
        row = paper(i)
        row["corpus_version"] = "corpus_v1"
        for kind, folder in (("jats", "JATS"), ("pdf", "PDF")):
            row[kind]["logical_ref"] = f"{row['paper_id']}/{folder}/{kind}_{i}.bin"
        rows.append(row)
    manifest = CorpusManifest(papers=[PaperRecord.model_validate(r) for r in rows])
    for p in manifest.papers:
        write(data / "corpus/raw" / p.jats.logical_ref, b"<article/>")
        write(data / "corpus/raw" / p.pdf.logical_ref, b"%PDF-1.4\n%%EOF\n")
    csv = serialize_csv(manifest)
    base = "corpus/releases/corpus_v1/manifest_revision_1"
    write(data / base / "corpus_manifest.csv", csv)
    report = serialize_project_json(
        ValidationReport(schema_version="1", findings=[]), ValidationReport
    )
    write(data / base / "validation_report.json", report)

    def ref(name: str, logical: str, content: bytes) -> ArtifactRef:
        return ArtifactRef(artifact_id=name, logical_ref=logical, sha256=sha256_bytes(content))

    content_hash = corpus_content_hash(manifest)
    record = ReleaseRecord(
        record_schema_version="1",
        corpus_version="corpus_v1",
        manifest_revision="1",
        schema_version="1",
        corpus_content_sha256=content_hash,
        manifest_semantic_sha256=manifest_semantic_hash(manifest),
        csv_sha256=sha256_bytes(csv),
        manifest_artifact=ref("manifest", f"{base}/corpus_manifest.csv", csv),
        validation_report_artifact=ref("report", f"{base}/validation_report.json", report),
        git_sha=commit,
    )
    record_bytes = write(
        data / base / "release.json", serialize_project_json(record, ReleaseRecord)
    )
    bench = "benchmark/releases/benchmark_v1"
    items = [
        make_item(
            f"I{n}", (DEV if n % 2 else FINAL)[n % 10], "development" if n % 2 else "final_test"
        )
        for n in range(5)
    ]
    pieces = {
        "seeds": serialize_project_json(prospective_seeds(), type(prospective_seeds())),
        "relatedness": b"{}\n",
        "split": serialize_project_json(SPLIT, type(SPLIT)),
        "items": encode_jsonl(items),
        "decisions": b"",
        "dispositions": b"",
        "pre": b"",
        "audit": b"{}\n",
        "validation": b"{}\n",
    }
    refs = {}
    for name, content in pieces.items():
        write(data / "benchmark/working" / name, content)
        refs[name] = ref(f"b_{name}", f"benchmark/working/{name}", content)
    refs["items"] = ref("b_items", f"{bench}/items.jsonl", pieces["items"])
    write(data / bench / "items.jsonl", pieces["items"])
    release = BenchmarkRelease(
        benchmark_version="benchmark_v1",
        corpus_version="corpus_v1",
        corpus_content_sha256=content_hash,
        corpus_release=ref("corpus_release", f"{base}/release.json", record_bytes),
        implementation_git_sha=commit,
        seed_registry=refs["seeds"],
        relatedness=refs["relatedness"],
        split=refs["split"],
        items=refs["items"],
        decisions=refs["decisions"],
        dispositions=refs["dispositions"],
        pre_audit_population=refs["pre"],
        audit=refs["audit"],
        validation_report=refs["validation"],
        item_count=len(items),
        development_paper_count=10,
        final_test_paper_count=40,
    )
    write(data / bench / "release_record.json", serialize_project_json(release, BenchmarkRelease))
    return data


def build_world(root: Path, monkeypatch) -> World:
    repo, commit = build_repo(root)
    data = build_data(root, commit)
    monkeypatch.setattr("biorag_repro.readiness.evidence.verify_release_inputs", lambda r, d: None)
    world = World(repo, data, commit, None)  # type: ignore[arg-type]
    world.ctx = world.probes()
    return world
