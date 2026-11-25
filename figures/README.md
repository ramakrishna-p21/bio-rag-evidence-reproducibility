# Statistical figures

R/ggplot2 code that draws the five statistical figures of the study from the result files of the
official R analysis. The code reads published, hash-verified result tables; it recomputes no
statistics. The result tables and the generated figures are not part of this repository.

This directory is separate from `analysis/`: it has its own package list and its own R library and
never touches `analysis/renv.lock` or the analysis `renv` library.

## Inputs

The official R analysis writes two result directories. Point the scripts at them:

| Variable | Directory |
| --- | --- |
| `FIG_PRIMARY` | primary result directory (contains `run_record.json`) |
| `FIG_SECONDARY` | secondary result directory (contains `secondary_record.json`) |
| `FIG_OUT` | where figures and logs are written (default `figures/output/`, git-ignored) |
| `FIG_LIB` | private R library for the plotting packages (default `figures/environment/library/`, git-ignored) |

Every table is checked against the SHA-256 recorded in these run records before it is read, and the
primary run record is checked against the hash stored in the secondary record. A mismatch stops the
run.

## Regenerate

From this directory:

```bash
export FIG_PRIMARY=/path/to/primary_result_dir
export FIG_SECONDARY=/path/to/secondary_result_dir
./make_figures.sh
```

On first use the script installs the plotting packages into `FIG_LIB` (see Environment), then runs
`scripts/make_all.R`. Outputs under `FIG_OUT`: `figures/{pdf,svg,png}`, 200 dpi `preview/`, and
`validation/` (source-hash log and numeric checks). `make_all.R` also writes the installed package
list and session information to `FIG_OUT/environment/`.

## Layout

| File | Role |
| --- | --- |
| `scripts/common.R` | theme, hash-verified source loading, check helpers, export |
| `scripts/fig_coprimary_effects.R` | co-primary effect estimates |
| `scripts/fig_stage_diagnostics.R` | Stage B delivery status and earliest failure stage |
| `scripts/fig_paired_discordance.R` | paired item-level agreement between the two representations |
| `scripts/fig_answer_class_outcomes.R` | outcomes by answer class |
| `scripts/fig_sensitivity_analysis.R` | sensitivity to unresolved outcomes |
| `scripts/make_all.R` | runs the five scripts, builds previews, writes the logs |
| `make_figures.sh` | wrapper that sets the private library and calls `make_all.R` |
| `environment/packages.csv`, `session_info.txt` | package versions and R session of the rendering environment |
| `environment/install_packages.R` | installs the plotting packages into `FIG_LIB` |

## Sources used by each figure

| Figure | Result tables read |
| --- | --- |
| Co-primary effects | `coprimary_results.csv`; `state_counts.csv` (consistency check only) |
| Stage diagnostics | `secondary_stage_b_failure.csv`, `secondary_stage_a.csv`; `state_counts.csv` (check) |
| Paired discordance | `secondary_discordance.csv`; `coprimary_results.csv` (check) |
| Answer-class outcomes | `secondary_answer_class.csv`; `state_counts.csv` (check) |
| Sensitivity analysis | `coprimary_results.csv`, `secondary_robustness.csv` |

Conventions: effects are JATS minus PDF_DOCLING, so a negative value means PDF_DOCLING is higher.
Figures use the first repeat of the confirmatory arm. Percentages are taken over all eligible
item-conditions, with unresolved outcomes kept as their own category. Intervals are the registered
whole-paper percentile bootstrap intervals as published; none are recomputed. Sensitivity bounds
assign best-case and worst-case values to unresolved outcomes and are not confidence intervals.
The paired-discordance figure compares correctness states only, not answer text.

## Checks run by the scripts

Each script stops on a failed check. The checks confirm that source hashes match, that state counts
sum to the number of eligible items in every condition, that the published rate difference equals
the difference of the published rates, that the Holm-adjusted p-values follow from the published raw
values, that interval bounds bracket their estimates, and that the values drawn (read back from the
built ggplot objects) equal the published values. A summary log is written to `FIG_OUT/validation/`.

## Environment

Rendered with R 4.6.1 on Ubuntu 22.04 and the 26 package versions in `environment/packages.csv`
(base packages excluded); the font is Liberation Sans, an Arial-metric font that must be installed
for the SVG and PDF text to render as designed.

`install_packages.R` installs those exact versions with `renv::install()` into `FIG_LIB`.
`FIG_PINNED=0 ./make_figures.sh` installs current CRAN versions of the six direct dependencies
instead. The pinned installation route is a restore from the recorded inventory; it has not been
re-run from an empty library, so it is not a verified reproduction of the original rendering
environment. The same scripts, run against the original library, reproduced the original PNG and SVG
files byte for byte; PDFs differ only in the embedded creation date.
