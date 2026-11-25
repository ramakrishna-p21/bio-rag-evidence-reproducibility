# Installs the plotting packages into the private library FIG_LIB (never into analysis/renv).
#
# Default: the exact package versions of the rendering environment recorded in packages.csv,
# resolved with renv::install(). FIG_PINNED=0 installs the six direct dependencies
# (ggplot2, patchwork, svglite, ragg, png, jsonlite) at their current CRAN versions instead.
lib <- Sys.getenv("FIG_LIB")
stopifnot(nzchar(lib))
dir.create(lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(lib, .Library))
repos <- "https://cloud.r-project.org"

if (identical(Sys.getenv("FIG_PINNED", "1"), "0")) {
  install.packages(c("ggplot2", "patchwork", "svglite", "ragg", "png", "jsonlite"),
                   lib = lib, repos = repos, Ncpus = 4)
} else {
  if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv", lib = lib, repos = repos)
  pins <- read.csv(file.path(Sys.getenv("FIG_ROOT", getwd()), "environment", "packages.csv"))
  base <- rownames(installed.packages(priority = "base"))
  pins <- pins[!pins$Package %in% base, ]
  renv::install(sprintf("%s@%s", pins$Package, pins$Version), library = lib,
                repos = repos, prompt = FALSE)
}
