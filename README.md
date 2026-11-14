# Evidence reproducibility in retrieval-augmented scientific question answering

Research software for studying how alternative representations of the same scientific publication
affect retrieval-augmented question answering in bioinformatics.

## Research objective

A retrieval-augmented system never sees a publication itself. It sees one representation of it:
structured text from the publisher's JATS XML, or text recovered from the published PDF by a layout
parser. This project asks whether the choice of representation changes

- whether the evidence needed to answer a question is preserved in the representation;
- whether retrieval delivers that evidence in the context supplied to the generator;
- whether the generated answer is scientifically correct;
- whether those answers are reproducible across repeated runs.

Each question is graded against the original source, and each result stays traceable through the chain

```
source file → ingestion → canonical document → chunks → ranked retrieval
            → supplied context → generated answer → evaluation → statistical analysis
```

by content hashes recorded in build manifests.

## Design

| Stage | What it does |
| --- | --- |
| Ingestion | Converts JATS XML and PDF (via Docling) into one canonical document schema, keeping source locations. |
| Chunking | Splits canonical documents into 192- or 384-token chunks, counted with a pinned construction tokenizer. |
| Retrieval | BM25 and dense retrieval (`BAAI/bge-base-en-v1.5`) over the chunks. |
| Context | Assembles whole chunks from the top 20 ranked results into the generator input, stopping before a 2,048-token (Qwen3 tokenizer) evidence budget would be exceeded. |
| Generation | Runs a Qwen3-8B generator with recorded attempts, seeds and exact supplied context. |
| Evaluation | Stage A: evidence preserved in the representation. Stage B: evidence delivered in the supplied context. Stage C: answer correctness. |
| Analysis | An R analysis (whole-paper bootstrap, cluster-robust inference) reads a validated table exported by Python. Python remains authoritative for scoring. |

### Repository layout

| Path | Contents |
| --- | --- |
| `src/biorag_repro/core` | Hashing, identities, build manifests, immutable file publication. |
| `src/biorag_repro/corpus`, `benchmark` | Corpus manifest and benchmark construction and validation. |
| `src/biorag_repro/documents` | JATS and PDF ingestion, canonical documents, coverage and lineage validation. |
| `src/biorag_repro/retrieval` | Chunking, BM25, dense retrieval, context assembly. |
| `src/biorag_repro/generation` | Generator configuration, prompts, run and attempt records, environment checks. |
| `src/biorag_repro/evaluation` | Stage A, B and C evaluation and the analysis handoff. |
| `src/biorag_repro/readiness`, `finaltest` | Checks that gate protected evaluation runs. |
| `analysis/` | R analysis, locked with `renv`. |
| `configs/` | Ingestion and environment configuration. |
| `scripts/` | Command-line entry points for corpus, benchmark, ingestion and review work. |
| `tests/` | Python test suite. |

## Installation

Python 3.11 is required.

```bash
python -m pip install -e '.[dev,benchmark,bm25,dense]'
```

Optional extras: `context` (tokenizer support for context assembly), `generation` (generator
backend), `chunking`, `pdf` (pinned Docling stack for PDF ingestion). For a CPU-only dense build,
install `torch==2.15.0` from `https://download.pytorch.org/whl/cpu` before the project extras.

Model files and source data are not part of the repository. Retrieval and generation read local,
hash-verified snapshots: set `BIORAG_BGE_ROOT` to the pinned BGE snapshot and `BIORAG_QWEN_ROOT` to
the pinned Qwen3-8B tokenizer files to include the checks that use them. Without them those tests
are skipped.

The R analysis needs R 4.6.1. From `analysis/`:

```bash
Rscript -e 'renv::restore()'
```

PyMuPDF (`benchmark` extra) is used only to locate evidence in original PDFs for human review. It
is not an experimental parser.

## Usage

Installed console scripts, each also runnable as `python -m <module> --help`:

| Command | Module | Purpose |
| --- | --- | --- |
| `biorag-context` | `biorag_repro.retrieval.cli` | Assemble generator contexts from selected chunks and re-read them with full validation. |
| `biorag-analysis` | `biorag_repro.evaluation.analysis_cli` | Verify the analysis table and handoff, and check R results against them. |
| `biorag-checkpoint` | `biorag_repro.generation.checkpoint` | Inventory and verify a local model snapshot. |
| `biorag-environment` | `biorag_repro.generation.environment_lock` | Compare the intended execution environment with the installed one. |
| `biorag-readiness` | `biorag_repro.readiness.cli` | Evaluate and verify readiness for protected evaluation runs. |

Scripts in `scripts/`:

- `benchmark.py` — `split`, `discover`, `review-pack`, `pdf-search`, `validate`, `audit-draw`,
  `derive-development`. Every command takes an external data root and the explicit path and SHA-256
  of the corpus release record. Candidate evidence found automatically is a lead for a human
  reviewer; it is never accepted as gold without inspection of the original PDF.
- `ingest_paper.py`, `reproduce_pair.py`, `stage_a_review.py` — single-paper ingestion,
  reproduction of a source pair, and masked Stage A review packets. Run each with `--help`.

R analysis (from `analysis/`): `Rscript run_analysis.R run|verify|derive|supplement ...`; see the
header of `analysis/run_analysis.R` for the arguments.

## Reproducibility

- Inputs are selected by explicit path and SHA-256; files are published without overwriting and
  re-read against their recorded hashes.
- Build manifests link every derived artifact to its inputs and to the transformation that made it.
- Continuous integration (`.github/workflows/ci.yml`) runs on Ubuntu 22.04 with Python 3.11.15 and
  R 4.6.1 and requires the R tests to run. The primary R result is compared byte for byte, so it is
  tied to that platform's numerics.

### Checks

```bash
python -m pytest -q
python -m ruff format --check .
python -m ruff check .
(cd analysis && Rscript -e 'testthat::test_dir("tests/testthat")')
```

Several Python tests call `Rscript` and skip when R or the `renv` library is absent; set
`BIORAG_REQUIRE_R=1` to make that a failure.

## Scope and limitations

The repository contains the pipeline, its contracts and its tests, exercised on synthetic fixtures.
It does not contain the corpus, benchmark gold, model weights or experimental results; those are
kept outside version control. Dense retrieval and generation behaviour is verified only for the
pinned snapshots and the platform noted above.
