# Reproducing and inspecting the workflows

This guide separates what anyone can run from what needs controlled study data. Only the first tier
works from this repository alone. Nothing here claims that the full study can be re-executed from
public material.

## 1. Workflow tiers

| Tier | What you can do | Needs | Where it works |
| --- | --- | --- | --- |
| 1. Install and validate on synthetic data | Python lint and tests, R tests, synthetic end-to-end analysis chain | This repository, Python 3.11, R 4.6.1 | Any machine that meets the requirements |
| 2. Development tooling | Ingest a paper, build and inspect contexts, run development generation and evaluation | Development corpus, benchmark and model files (not distributed) | Controlled |
| 3. Figures from analysis outputs | Redraw the five statistical figures | The two result directories of the official R analysis | Controlled until aggregate results are released |
| 4. Official analysis verification | Re-verify the official R analysis and its handoff | Analysis table, handoff and result directories; the historical checkout | Controlled |
| 5. Full experimental reproduction | Regenerate contexts and answers, redo Stage A/B/C evaluation | Corpus, benchmark gold, model weights, GPU, the historical checkouts, reviewer decisions | Controlled; reviewer judgments are human records, not re-computable |

## 2. Requirements

| Item | Requirement |
| --- | --- |
| Python | 3.11 (`requires-python >=3.11,<3.12`). CI uses 3.11.15; some test goldens embed the patch version |
| R | 4.6.1 with the locked library in `analysis/renv.lock` |
| OS for the official R result | Ubuntu 22.04 class platform. The primary result is compared byte for byte and depends on that platform's numerics. The analysis runner pins `OPENBLAS_CORETYPE=HASWELL` and one thread |
| Embedding model | `BAAI/bge-base-en-v1.5`, snapshot `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`, via `BIORAG_BGE_ROOT` |
| Tokenizer | `Qwen/Qwen3-8B` tokenizer files at revision `b968826d9c46dd6066d109eabc6255188de91218`, via `BIORAG_QWEN_ROOT` (files only; each is hash-checked before use) |
| Generator | `Qwen/Qwen3-8B` at the same revision, bfloat16, one GPU of a qualified host class (the final run used an NVIDIA L4). Not needed below tier 5 |
| PDF ingestion | Pinned Docling stack (`pdf` extra) |
| Disk, RAM | Tier 1 is light. Evaluation of the full final-test batch used about 5 GB of peak memory; generation memory is set by the GPU |

Install:

```bash
python -m pip install -e '.[dev,benchmark,bm25,dense]'
# optional extras: context, generation, chunking, pdf
```

For CPU-only dense retrieval install `torch==2.15.0` from `https://download.pytorch.org/whl/cpu`
first. PyMuPDF (`benchmark` extra) only helps a human locate evidence in a PDF; it is not an
experimental parser and is optional.

## 3. Tier 1: installation and synthetic validation

```bash
python -m ruff check .
python -m ruff format --check .
BIORAG_QWEN_ROOT=... BIORAG_BGE_ROOT=... BIORAG_REQUIRE_R=1 python -m pytest -q -rs
cd analysis
Rscript -e 'renv::restore()'
OPENBLAS_CORETYPE=HASWELL OPENBLAS_NUM_THREADS=1 \
  Rscript -e 'testthat::test_dir("tests/testthat", reporter = "summary", stop_on_failure = TRUE)'
```

`BIORAG_REQUIRE_R=1` makes Python tests that call R fail instead of skipping when R or the library
is missing. Without the model roots, tests that use the tokenizer or embeddings are skipped; the
skip reasons are printed by `-rs`. Two test groups are opt-in: `BIORAG_MODEL_SMOKE=1` runs a tiny real
model, and `BIORAG_REAL_DATA=<manifest>` reads local development artifacts.

### Public CI and complete private validation

The repository's CI runs on a clean public checkout. Thirteen readiness acceptance test functions
(22 test cases, listed in `.github/ci_deselected_tests.txt`) read private research records that are
not in the repository: governance records under the git-ignored `docs/reports/` and local
artifacts. In a clean clone they report that evidence is absent rather than exercising the gate,
so CI deselects exactly those cases, prints the list, and fails if the number deselected is not 22.
Everything else runs, including the Python lint, format and regression tests, the R statistical
tests, and the documentation and citation checks.

This is a separation of validation environments, not a reduction in what the project validates.
On the author's machine, with the private records present, the complete suite is run without any
deselection (`python -m pytest -q`); those cases pass there. Outside readers do not have the
records and can run only the public selection:

```bash
python -m pytest -q $(grep -v '^#' .github/ci_deselected_tests.txt | sed 's/^/--deselect=/')
```

### Revision-bound readiness checks

Some readiness and authorization tests check the repository against approval records that name one
historical source revision. On a descendant revision such as this one, such checks can fail by
design. This is a revision-binding effect, not evidence of a regression. See
[Historical revisions](#6-historical-revisions-and-checkouts) for the revision each record binds.

## 4. Command-line entry points

Console scripts installed by `pip install -e .` (each is also `python -m <module> --help`):

| Command | Module | Purpose |
| --- | --- | --- |
| `biorag-context` | `biorag_repro.retrieval.cli` | `assemble` or `validate` evidence contexts from selected chunks |
| `biorag-analysis` | `biorag_repro.evaluation.analysis_cli` | `verify` an analysis table, `verify-handoff` (rehearsal handoffs only), `verify-supplement` |
| `biorag-environment` | `biorag_repro.generation.environment_lock` | `capture` an environment inventory and `verify` it against `configs/environment.toml` |
| `biorag-readiness` | `biorag_repro.readiness.cli` | `evaluate` and `verify` final-test readiness; exit 0 READY, 2 NOT_READY, 1 failure |

Programs in `scripts/` (run from the repository root with `--help`):

| Script | Purpose | Data needed |
| --- | --- | --- |
| `benchmark.py` | Benchmark construction: `split`, `discover`, `review-pack`, `pdf-search`, `audit-draw`, `validate`, `derive-development`. Automatically found evidence is a lead for a human reviewer, never gold | Controlled |
| `ingest_paper.py`, `reproduce_pair.py` | Build or reproduce one verified source pair (JATS plus PDF) | Controlled |
| `build_contexts.py` | Build generator contexts for the five arms (development, or `--final-test` with the approved clean checkout) | Controlled |
| `run_generation.py` | Generation batches: `run`, `preflight`, `verify-official`, `probe`, `inventory`, `propose-environment-identity`, `seal`, `verify`. Write-once, append-only, resumable | Controlled, GPU |
| `initialize_final_test_root.py` | Create the guarded final-test root | Controlled |
| `stage_a_review.py` | Masked Stage A review packets: `queue`, `final-queue`, `start`, `finish`, `guess`, `inspect`, `check`, `summary`, `publish` | Controlled |
| `transcribe_stage_a.py`, `account_stage_a_v2.py` | Stage A transcription and a development-only accounting dry run (run as modules from the root, for example `python -m scripts.transcribe_stage_a`) | Controlled |
| `stage_b_sheet.py` | Stage B verification sheets: `plan`, `build`, `sheet`, `apply`. A sheet is a reading aid and never records a label | Controlled |
| `evaluate_development.py` | Development evaluation run | Controlled |
| `evaluate_final_test.py` | Final-test evaluation: `stage-b-queue`, `stage-b-ingest`, `stage-c-queue`, `evaluate`. Needs `approvals/final_test_evaluation.json` and a request file | Controlled, approved revision |
| `export_analysis_host_evidence.py` | Export retained evidence for the analysis host | Controlled |

R analysis, run from `analysis/` so that the project library is active:

```bash
Rscript run_analysis.R run|verify|derive|verify-derived|supplement|verify-supplement ...
```

The header of `analysis/run_analysis.R` lists the arguments of each command.

## 5. Tiers 3 to 5

**Figures (tier 3).** See [../figures/README.md](../figures/README.md). Set `FIG_PRIMARY` and
`FIG_SECONDARY` to the two result directories and run `figures/make_figures.sh`. The scripts read only
tables whose hashes match the run records.

**Official analysis verification (tier 4).** The R analysis records the exact code revision it ran
at, and the Python authorization checks require the approved clean revision. To verify an official
run, check out that revision (section 6) and run `Rscript run_analysis.R verify --result DIR ...` from
`analysis/` with the recorded hashes. The consolidated revision is not that revision and
official-origin commands will not accept it.

**Full reproduction (tier 5).** Order of work, each step reading the previous step's published
artifacts by hash: ingestion (`ingest_paper.py`), contexts (`build_contexts.py`), generation
(`run_generation.py`), Stage A/B/C review and evaluation (`stage_a_review.py`,
`stage_b_sheet.py`, `evaluate_final_test.py`), analysis (`run_analysis.R`). Reviewer judgments for
Stages A and B are human records and are read, not recomputed. Generation requires the pinned Qwen3-8B
checkpoint on one GPU; the final run used an L4 and a recorded environment lock
(`configs/environment.toml`).

## 6. Historical revisions and checkouts

| Step | Full revision | Local tag in the author's clone |
| --- | --- | --- |
| Generation | `92af41e840b6667c5bf967a08e7ed61da078640a` | `exec/generation-92af41e` |
| Stage A packet and review | `04eb08499f518c51a223019795cc20c19b6d1ca7` | `exec/stage-a-04eb084` |
| Stage B review records | `8c0a8255d85409e717cde0c7757ee68efeb0002c` | `exec/stage-b-review-8c0a825` |
| Stage B ingestion, Stage C queue, first evaluation | `71cc332766204c33af785b5b317c93eff3c394a6` | `exec/stage-b-ingest-stage-c-eval-71cc332` |
| Final evaluation run, official R primary and secondary | `cfa2ac026ed8f5e354bd45148774446ce07287cf` | `exec/final-eval-and-analysis-cfa2ac0` |

The tags exist only in the author's clone. Use the full revision hash:

```bash
git worktree add ../biorag-cfa2ac0 cfa2ac026ed8f5e354bd45148774446ce07287cf
```

`approvals/final_test_evaluation.json` authorizes evaluation at `cfa2ac0`. It does not authorize
this revision. No approval or readiness record was created for the consolidated revision.

## 7. Colab generation workflow (documented, not a turnkey notebook)

The final generation ran in Google Colab on an NVIDIA L4 GPU. The original notebooks are kept
privately and unmodified; they carry run-specific records and were not re-executed or replaced. This
section describes the sequence from the production scripts and the notebooks' command cells. It is
a guide, not a newly validated notebook.

1. **Runtime.** Select a GPU runtime and confirm the GPU model reported by `nvidia-smi`; the
   qualified host class is defined by GPU model plus an approved environment lock.
2. **Storage.** Mount Google Drive for write-once backups of the batch and for transfer back. Keep
   Drive locations and account details out of notebooks you share.
3. **Controlled inputs.** Place the context package, configuration files, prompt, seed registry,
   tokenizer files and model snapshot in a local data root, with a `sha256sum` manifest for each.
   Verify every manifest before use.
4. **Python environment.** Create a Python 3.11 environment and install the project with the
   `generation` extra plus the CUDA build of torch described in `configs/environment.toml`.
5. **Source.** Fetch the repository, check out the generation revision above in detached state and
   confirm that `git status --porcelain --untracked-files=normal` is empty. The runner refuses a
   dirty or unapproved tree.
6. **Environment check.** `biorag-environment capture` then `biorag-environment verify` against
   `configs/environment.toml`.
7. **Contexts.** `scripts/build_contexts.py --final-test --repo ... --data-root ...
   --expected-head-sha256 ...` with the approved selection.
8. **Generation.** `scripts/run_generation.py run --official --phase final_test --custody-mode
   amended_drive --device cuda:0 ...`, which appends attempt records, resumes after interruption
   and is sealed with `seal` and checked with `verify`. A Colab session can end mid-run; rerunning
   continues from the recorded attempts.
9. **Transfer.** Download the batch, verify every file against its manifest, and keep an unmodified
   copy before any evaluation reads it.

Custody for the final batch was recorded under an approved exception; see
[DATA_AND_LIMITATIONS.md](DATA_AND_LIMITATIONS.md).

## 8. Expected outputs

| Command | Output |
| --- | --- |
| `pytest` | Test summary; skipped tests are listed by `-rs` with the missing prerequisite |
| `testthat::test_dir` | Summary reporter; any failure stops the run |
| `run_generation.py run` | A write-once batch directory: results, receipts, inputs, raw outputs, attempts, environment records |
| `evaluate_final_test.py evaluate` | Published analysis table, handoff and build record, each with a SHA-256 |
| `run_analysis.R run` | A run directory with `run_record.json`, result tables and an environment inventory |
| `figures/make_figures.sh` | `FIG_OUT/figures/{pdf,svg,png}`, previews, `validation/` logs |
