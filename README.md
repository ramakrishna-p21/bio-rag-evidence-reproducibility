# Bioinformatics RAG evidence reproducibility

Research software for evaluating how two representations of the same frozen publication affect scientific evidence delivery and answer correctness. The source corpus, benchmark, and later run artifacts are versioned separately. The benchmark is built from original JATS and original PDFs before experimental PDF ingestion.

## Local setup

Python 3.11 is required. Install the project and development tools with `python -m pip install -e '.[dev,benchmark,bm25,dense]'`. PyMuPDF is used only to locate evidence in original PDFs for human review.

## Benchmark construction

`scripts/benchmark.py` has `split`, `discover`, `review-pack`, `pdf-search`, `validate`, and `audit-draw` commands. Each command requires an external data root and the **explicit path and SHA-256** of the selected corpus release record. `split` also requires a hash-checked related-paper record. `discover` accepts a paper in the frozen split and writes source-located leads and an unfilled review worksheet. `review-pack` combines selected JATS leads, flags, and PDF search hints. `pdf-search` returns page and region hints; a reviewer must inspect the original PDF before an item is accepted. `validate` checks typed items and their original-source locations. `audit-draw` samples accepted item IDs and is intended for the full proposed benchmark before final freeze.

The frozen source files and production benchmark working artifacts remain outside Git. Candidate leads and search hits are never gold answers or source-equivalence decisions. The 10/40 paper split remains fixed after its prospective seeded assignment. No Docling, retrieval, or generator output is used to select benchmark items.

## Checks

Run `python -m pytest -q`, `python -m ruff format --check .`, `python -m ruff check .`, `python -m mypy src/biorag_repro`, and `python scripts/export_schemas.py --check` for the current code and schema checks.

Dense inference uses a verified local `BAAI/bge-base-en-v1.5` snapshot and CPU float32 execution. For the CPU dependency build, install `torch==2.15.0` from `https://download.pytorch.org/whl/cpu` before installing the project extras. Select the pinned snapshot directory with `BIORAG_BGE_ROOT` when running tests to include actual model checks. Without that setting, the model test is skipped and model behavior remains unverified by that test run. Model files and scientific data are acquired separately and kept outside Git.

Evidence contexts use whole chunks from the original top 20 ranked results and
stop at the first eligible chunk that exceeds 2,048 Qwen evidence tokens.
Install the `context` extra for tokenizer support; assets must be acquired
separately at the approved exact revision. No generator weights are required.

`python -m biorag_repro.retrieval.cli --help` (or `biorag-context --help` after installation) exposes context assembly and scientific
read-back. Both operations require explicit selection JSON (the `Selection`
fields in `retrieval/context.py`), configuration reference/root, local tokenizer
assets, and output root. BGE read-back additionally requires its verified model
assets. Selected chunk, index and ranking artifacts pass their existing scientific
readers; this command does not construct or replace those upstream artifacts.
Output publication refuses to overwrite existing files. These development
operations do not establish generator qualification or final-test readiness.

## R analysis checks

The pre-specified inferential analysis is in `analysis/` and uses a project-local renv library locked in `analysis/renv.lock` (R 4.6.1). From `analysis/`, run `Rscript -e 'renv::restore()'`, then `Rscript -e 'testthat::test_dir("tests/testthat")'`. Several Python tests (Python/R handoff, rehearsal, result verification, supplement chain) call `Rscript` and skip when R or the renv library is absent; set `BIORAG_REQUIRE_R=1` to make that a failure instead. CI sets it. Python remains authoritative for scientific scoring; R only consumes the validated analysis table.

## Project status (checkpoint)

Implemented and tested on synthetic fixtures only: evidence preservation, delivery and answer scoring (Stages A, B, C); generation run and attempt lifecycle with exact supplied-context records; the validated Python-to-R analysis handoff; the R estimators, whole-paper bootstrap and CR2 inference with verified result publication; secondary tables, a derived report and a records-only release index; the readiness gate and the protected final-test execution architecture with rehearsal start, resume and completion.

`analysis_supplement` (answer-identity and evidence-form labels) is a separate, versioned, hash-linked artifact. The frozen primary analysis table is unchanged by it.

What this checkpoint establishes: the infrastructure behaves as specified on synthetic data, and its Python and R tests run in CI. What it does not establish: any scientific result. No official model qualification, development execution or final-test observation has been run, readiness is not approved (production final-test execution is disabled), and the real Phase 11 analysis gate has not been passed. Benchmark gold, frozen sources and final-test material stay outside Git.

Next planned work: Phase 7 generator environment and GPU qualification, then Phase 8 official development execution.
