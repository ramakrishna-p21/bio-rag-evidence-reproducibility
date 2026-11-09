# Whole-paper percentile bootstrap of the item-weighted paired risk difference.
#
# The unit of resampling is the paper. A drawn paper brings every one of its complete-case paired
# differences, and a paper drawn twice contributes them twice. Each replicate is the pooled ratio
# sum(d) / n_items over the drawn papers (item-weighted); the equal-paper mean is never used here.
# The frame is, per outcome, the papers with at least one complete pair, so every resample has at
# least one item. Everything is deterministic for a given integer seed under the RNG below.

PRIMARY_RESAMPLES <- 10000L
BOOTSTRAP_RNG <- list(kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
BOOTSTRAP_QUANTILE_TYPE <- 7L
BOOTSTRAP_CONF_LEVEL <- 0.95

bootstrap_error <- function(message) input_error("bootstrap_input", message)

# Validate item-level paired differences and their paper ids. Each difference must belong to
# exactly one non-missing paper id; anything else would silently change the clustering.
check_pairs <- function(d, paper_id) {
  if (!is.numeric(d) || anyNA(d) || !all(d %in% c(-1, 0, 1))) {
    input_error("estimate_input", "paired differences must be non-missing values in {-1, 0, 1}")
  }
  if (!is.character(paper_id) || anyNA(paper_id) || any(paper_id == "")) {
    input_error("estimate_input", "paper ids must be non-missing, non-empty character values")
  }
  if (length(d) != length(paper_id)) {
    input_error("estimate_input", "each paired difference needs exactly one paper id")
  }
  invisible(TRUE)
}

# Per-paper integer item counts and difference sums, papers in radix order (independent of row
# order and of the locale).
paper_totals <- function(d, paper_id) {
  check_pairs(d, paper_id)
  papers <- sort(unique(paper_id), method = "radix")
  n_items <- as.integer(tabulate(match(paper_id, papers), nbins = length(papers)))
  sum_d <- as.integer(vapply(papers, function(p) sum(d[paper_id == p]), numeric(1)))
  data.frame(paper_id = papers, n_items = n_items, sum_d = sum_d, stringsAsFactors = FALSE)
}

# Run `expr` under the documented RNG and put the caller's RNG state back afterwards, whether
# `expr` returns or fails. The seed vector also encodes the RNG kinds, so restoring it restores them.
with_bootstrap_rng <- function(seed, expr) {
  env <- globalenv()
  had_seed <- exists(".Random.seed", envir = env, inherits = FALSE)
  old_seed <- if (had_seed) get(".Random.seed", envir = env, inherits = FALSE)
  on.exit({
    if (had_seed) {
      assign(".Random.seed", old_seed, envir = env)
    } else if (exists(".Random.seed", envir = env, inherits = FALSE)) {
      rm(".Random.seed", envir = env)
    }
  }, add = TRUE)
  set.seed(seed, kind = BOOTSTRAP_RNG$kind, normal.kind = BOOTSTRAP_RNG$normal.kind,
           sample.kind = BOOTSTRAP_RNG$sample.kind)
  expr
}

# B x P matrix of paper indices (1..P into the sorted frame). One sample.int call fills the
# matrix row by row, so row i is the i-th block of P consecutive draws.
draw_papers <- function(n_papers, n_resamples, seed) {
  whole <- function(x) is.numeric(x) && length(x) == 1 && is.finite(x) && x == round(x)
  if (!whole(n_papers) || n_papers < 1) bootstrap_error("n_papers must be a positive whole number")
  if (!whole(n_resamples) || n_resamples < 1) bootstrap_error("n_resamples must be a positive whole number")
  if (!is.numeric(seed) || length(seed) != 1 || !is.finite(seed) || seed != round(seed) ||
      abs(seed) > .Machine$integer.max) {
    bootstrap_error("seed must be one finite integer")
  }
  flat <- with_bootstrap_rng(as.integer(seed), sample.int(as.integer(n_papers), n_resamples * n_papers, replace = TRUE))
  matrix(flat, nrow = as.integer(n_resamples), ncol = as.integer(n_papers), byrow = TRUE)
}

# Item-weighted replicate for each row of `draws`. Sums are over integers, so each replicate is a
# single exact division. A resample with no items would be NA; it is counted, not hidden.
bootstrap_replicates <- function(totals, draws) {
  p <- nrow(totals)
  if (!is.matrix(draws) || ncol(draws) != p || anyNA(draws) || any(draws < 1) || any(draws > p)) {
    bootstrap_error("draws must be a matrix of paper indices with one column per frame paper")
  }
  num <- rowSums(matrix(totals$sum_d[draws], nrow = nrow(draws)))
  den <- rowSums(matrix(totals$n_items[draws], nrow = nrow(draws)))
  empty <- den == 0
  list(estimates = ifelse(empty, NA_real_, num / ifelse(empty, 1, den)), n_resamples = nrow(draws),
       n_empty = as.integer(sum(empty)))
}

percentile_interval <- function(estimates) {
  est <- estimates[!is.na(estimates)]
  tail <- (1 - BOOTSTRAP_CONF_LEVEL) / 2
  unname(stats::quantile(est, c(tail, 1 - tail), type = BOOTSTRAP_QUANTILE_TYPE, names = FALSE))
}

percentile_bootstrap <- function(d, paper_id, seed, n_resamples) {
  totals <- paper_totals(d, paper_id)
  base <- list(method = "whole_paper_percentile", n_papers_frame = nrow(totals), seed = seed, rng = BOOTSTRAP_RNG,
               quantile_type = BOOTSTRAP_QUANTILE_TYPE, conf_level = BOOTSTRAP_CONF_LEVEL)
  if (nrow(totals) == 0) {
    return(c(list(status = "no_complete_pairs", n_resamples = 0L, n_empty = 0L, ci_lower = NA_real_, ci_upper = NA_real_), base))
  }
  rep <- bootstrap_replicates(totals, draw_papers(nrow(totals), n_resamples, seed))
  ci <- percentile_interval(rep$estimates)
  # One observed paper: every resample is that paper, so the interval is a mechanical artefact.
  status <- if (nrow(totals) == 1) "single_cluster_degenerate" else "ok"
  # Report the number of replicates actually computed, not the number asked for.
  c(list(status = status, n_resamples = rep$n_resamples, n_empty = rep$n_empty, ci_lower = ci[1], ci_upper = ci[2]), base)
}

# The resample count of the primary analysis, from the handoff's frozen method.
primary_resamples <- function(handoff) {
  b <- handoff$analysis_spec$bootstrap_resamples
  if (!identical(b, PRIMARY_RESAMPLES)) bootstrap_error("the primary analysis uses exactly 10000 resamples")
  b
}
