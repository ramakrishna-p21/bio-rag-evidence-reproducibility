# Pure estimators on already-validated 0/1 outcomes. No inference (bootstrap, CR2) lives here.

estimate_error <- function(message) input_error("estimate_input", message)

paired_differences <- function(y_jats, y_pdf) {
  ok01 <- function(y) is.numeric(y) && !anyNA(y) && all(y %in% c(0, 1))
  if (length(y_jats) != length(y_pdf)) estimate_error("outcome vectors differ in length")
  if (!ok01(y_jats) || !ok01(y_pdf)) estimate_error("outcomes must be non-missing 0/1")
  y_jats - y_pdf
}

# Item-weighted paired risk difference (headline) and the equal-paper-weighted sensitivity,
# from the complete pairs of one outcome. No complete pair is its own status.
summarize_pairs <- function(y_jats, y_pdf, paper_id) {
  d <- paired_differences(y_jats, y_pdf)
  if (length(paper_id) != length(d) || anyNA(paper_id)) estimate_error("paper ids must match the pairs")
  if (length(d) == 0) {
    return(list(status = "no_complete_pairs", n_items = 0L, n_papers = 0L, rate_jats = NA_real_,
                rate_pdf = NA_real_, rd = NA_real_, rd_pp = NA_real_, equal_paper_rd = NA_real_))
  }
  per_paper <- tapply(d, paper_id, mean)
  rd <- mean(d)
  list(status = "ok", n_items = length(d), n_papers = length(per_paper),
       rate_jats = mean(y_jats), rate_pdf = mean(y_pdf), rd = rd, rd_pp = 100 * rd,
       equal_paper_rd = mean(per_paper))
}

# Holm over exactly the two co-primary p-values, returned in input order. An undefined p-value
# makes the family undefined; a one-p Holm is never reported.
holm_adjust <- function(p) {
  if (!is.numeric(p) || length(p) != 2) estimate_error("Holm is applied to exactly two p-values")
  if (any(!is.na(p) & (p < 0 | p > 1))) estimate_error("p-values must lie in [0, 1]")
  if (anyNA(p)) return(list(status = "holm_undefined", adjusted = c(NA_real_, NA_real_)))
  list(status = "ok", adjusted = stats::p.adjust(p, method = "holm"))
}

# Operational states: an execution that failed or never ran is missing, not a scientific outcome
# (EXPERIMENT_PLAN section 9), so it is never assigned a favourable or unfavourable value.
OPERATIONAL_STATES <- c("missing", "not_executed")

# Best/worst-case bounds (EXPERIMENT_PLAN section 9): a known success/failure keeps its value; an
# unresolved scientific cell is assigned the value most favourable (upper) or least favourable
# (lower) to JATS - PDF_DOCLING. Items with an operational state in either representation are
# left out of the bounds and counted; the denominator is every other eligible item. The distinct
# unresolved states are counted, not merged.
missingness_bounds <- function(jats_state, pdf_state, rule) {
  if (length(jats_state) == 0 || length(pdf_state) != length(jats_state)) {
    estimate_error("bounds need equal, non-empty state vectors")
  }
  known <- c(rule$success, rule$failure)
  if (!all(c(jats_state, pdf_state) %in% c(known, rule$non_evaluable))) estimate_error("state outside the outcome's vocabulary")
  operational <- jats_state %in% OPERATIONAL_STATES | pdf_state %in% OPERATIONAL_STATES
  jats_state <- jats_state[!operational]
  pdf_state <- pdf_state[!operational]
  n <- length(jats_state)
  if (n == 0) estimate_error("bounds need at least one item without an operational state")
  value <- function(state) ifelse(state == rule$success, 1, ifelse(state == rule$failure, 0, NA_real_))
  j <- value(jats_state)
  p <- value(pdf_state)
  upper <- ifelse(is.na(j), 1, j) - ifelse(is.na(p), 0, p)
  lower <- ifelse(is.na(j), 0, j) - ifelse(is.na(p), 1, p)
  counts <- table(c(sprintf("JATS|%s", jats_state[is.na(j)]), sprintf("PDF_DOCLING|%s", pdf_state[is.na(p)])))
  counts <- stats::setNames(as.integer(counts), as.character(names(counts)))
  list(n_eligible = n, n_operational_excluded = sum(operational),
       lower = sum(lower) / n, upper = sum(upper) / n,
       n_unknown_jats = sum(is.na(j)), n_unknown_pdf = sum(is.na(p)),
       unknown_states = counts[order(names(counts), method = "radix")])
}
