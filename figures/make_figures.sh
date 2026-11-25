#!/usr/bin/env bash
# Regenerates the statistical figures from the official R analysis result directories.
#   FIG_PRIMARY    primary result directory (contains run_record.json)
#   FIG_SECONDARY  secondary result directory (contains secondary_record.json)
#   FIG_OUT        output directory (default: figures/output, which is git-ignored)
#   FIG_LIB        private R library for the plotting packages (default: figures/environment/library)
# The analysis environment (analysis/renv) is not used or modified.
set -euo pipefail
cd "$(dirname "$0")"
export FIG_ROOT="$PWD"
export FIG_LIB="${FIG_LIB:-$PWD/environment/library}"
export R_LIBS_USER="$FIG_LIB" R_LIBS_SITE=/nonexistent
[ -d "$FIG_LIB/ggplot2" ] || Rscript environment/install_packages.R
Rscript --no-init-file scripts/make_all.R
