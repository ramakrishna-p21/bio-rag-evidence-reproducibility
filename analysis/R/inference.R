# Formal inference for one co-primary outcome and for the pair of them.
#
# The test is fixed by the frozen plan: an intercept-only linear model on the item-level paired
# differences (JATS - PDF_DOCLING), paper-clustered CR2 variance, and the Satterthwaite two-sided
# Wald test from clubSandwich. When that test cannot give a meaningful result the answer is an
# explicit `inference_undefined` with a reason. No other test is substituted and no p-value is
# made up. Unexpected failures of the package propagate.

CR2_METHOD <- "CR2_Satterthwaite_intercept_only"
INTERCEPT_TOLERANCE <- 1e-12   # d is integers in {-1, 0, 1}; means differ from the fit by ~1e-16

undefined_inference <- function(reason, estimate, n_items, n_papers, warnings = character(0)) {
  list(status = "inference_undefined", reason = reason, estimate = estimate, se = NA_real_,
       df = NA_real_, t = NA_real_, p = NA_real_, n_items = n_items, n_papers = n_papers,
       method = CR2_METHOD, package = "clubSandwich",
       package_version = as.character(utils::packageVersion("clubSandwich")), warnings = warnings)
}

# CR2 residual cluster sums are n * S_g - n_g * S_total over integers. They are all exactly zero
# exactly when the robust variance is zero, so this decision uses no floating-point tolerance.
# When any is nonzero the variance is at least 1 / n^4 (about 1e-16 at n = 10^4), far above the
# rounding noise of the fit, and the package's own result is still range-checked below.
all_residual_sums_zero <- function(d, paper_id) {
  n <- length(d)
  s_total <- sum(d)
  sums <- tapply(d, paper_id, sum)
  sizes <- tapply(d, paper_id, length)
  all(n * sums - sizes * s_total == 0)
}

cr2_satterthwaite <- function(d, paper_id) {
  check_pairs(d, paper_id)
  n <- length(d)
  n_papers <- length(unique(paper_id))
  if (n == 0) return(undefined_inference("no_complete_pairs", NA_real_, 0L, 0L))
  estimate <- mean(d)
  gone <- function(reason) undefined_inference(reason, estimate, n, n_papers)
  if (n_papers < 2) return(gone("single_cluster"))
  if (length(unique(d)) == 1) return(gone("zero_variance"))
  if (all_residual_sums_zero(d, paper_id)) return(gone("zero_robust_variance"))

  warnings <- character(0)
  frame <- data.frame(d = d, paper = factor(paper_id))
  ct <- withCallingHandlers(
    clubSandwich::coef_test(stats::lm(d ~ 1, data = frame), vcov = "CR2", cluster = frame$paper,
                            test = "Satterthwaite"),
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  needed <- c("beta", "SE", "tstat", "df_Satt", "p_Satt")
  missing_cols <- setdiff(needed, names(ct))
  if (length(missing_cols) || nrow(ct) != 1) {
    stop(sprintf("clubSandwich result lacks expected field(s): %s", paste(missing_cols, collapse = ", ")))
  }
  if (!is.finite(ct$beta) || abs(ct$beta - estimate) > INTERCEPT_TOLERANCE) {
    stop("intercept differs from the item-weighted paired risk difference")
  }
  usable <- is.finite(ct$SE) && ct$SE > 0 && is.finite(ct$df_Satt) && ct$df_Satt > 0 &&
    is.finite(ct$tstat) && is.finite(ct$p_Satt) && ct$p_Satt >= 0 && ct$p_Satt <= 1
  if (!usable) return(undefined_inference("nonfinite_result", estimate, n, n_papers, warnings))
  list(status = "ok", reason = NA_character_, estimate = estimate, se = ct$SE, df = ct$df_Satt,
       t = ct$tstat, p = ct$p_Satt, n_items = n, n_papers = n_papers, method = CR2_METHOD,
       package = "clubSandwich", package_version = as.character(utils::packageVersion("clubSandwich")),
       warnings = warnings)
}

# One co-primary outcome from its Python-selected complete cases (accounting$items).
# Statistical results only; evaluability and complete-case membership are Python's.
outcome_inference <- function(items, handoff, seed) {
  cc <- items[items$complete, , drop = FALSE]
  d <- paired_differences(cc$y_jats, cc$y_pdf)       # JATS - PDF_DOCLING
  paper <- cc$paper_id
  list(summary = summarize_pairs(cc$y_jats, cc$y_pdf, paper),
       bootstrap = percentile_bootstrap(d, paper, seed, primary_resamples(handoff)),
       test = cr2_satterthwaite(d, paper))
}

# Both co-primary outcomes and the Holm family over exactly their two p-values. The seed is
# supplied by the caller and initialised afresh for each outcome. No production seed exists yet.
coprimary_inference <- function(accounting, handoff, seed) {
  family <- unlist(handoff$analysis_spec$multiplicity_family)
  if (!identical(family, EXPECTED_FAMILY) || !identical(names(accounting), EXPECTED_FAMILY)) {
    input_error("inference_input", "accounting must hold exactly the two co-primary outcomes")
  }
  outcomes <- lapply(accounting, function(a) outcome_inference(a$items, handoff, seed))
  p <- vapply(outcomes, function(o) o$test$p, numeric(1))
  holm <- holm_adjust(unname(p))
  holm$adjusted <- stats::setNames(holm$adjusted, family)
  list(outcomes = outcomes, holm = holm, seed = seed,
       seed_name = handoff$analysis_spec$bootstrap_seed_name, rng = BOOTSTRAP_RNG,
       r_version = R.version.string, package_version = as.character(utils::packageVersion("clubSandwich")))
}
