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
