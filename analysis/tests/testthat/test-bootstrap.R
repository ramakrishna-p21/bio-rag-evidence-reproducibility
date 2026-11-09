# Whole-paper percentile bootstrap. Fixtures are synthetic. Seeds are test-only values and are
# not the production `primary_bootstrap` seed (which is not approved).
#
# Hand fixture. Paper A: 3 items, d = +1,+1,+1 (sum 3).  Paper B: 1 item, d = -1.
# Paper C: 2 items, d = +1,-1 (sum 0).  Sorted paper order A, B, C = 1, 2, 3.

TEST_SEED <- 20260101L
# sample.int(4, 12, replace = TRUE) after set.seed(20260101, "Mersenne-Twister", "Inversion", "Rejection"),
# R 4.6.1. Pinned so an R change to the RNG or to sample() is noticed.
PINNED_DRAWS <- c(4L, 4L, 2L, 4L, 1L, 1L, 1L, 2L, 4L, 1L, 1L, 2L)

hand_pairs <- function() {
  list(d = c(1, 1, 1, -1, 1, -1), paper = c("A", "A", "A", "B", "C", "C"))
}

test_that("paper totals are integer item counts and difference sums in sorted paper order", {
  h <- hand_pairs()
  t <- paper_totals(h$d, h$paper)
  expect_identical(t$paper_id, c("A", "B", "C"))
  expect_identical(t$n_items, c(3L, 1L, 2L))
  expect_identical(t$sum_d, c(3L, -1L, 0L))
  # row order of the input must not matter
  o <- c(5, 2, 6, 1, 4, 3)
  expect_identical(paper_totals(h$d[o], h$paper[o]), t)
})

test_that("pairs are refused when ids or differences are not usable", {
  expect_input_error(paper_totals(c(1, 0), "A"), "estimate_input")            # length mismatch
  expect_input_error(paper_totals(c(1, 0), c("A", NA)), "estimate_input")      # missing id
  expect_input_error(paper_totals(c(1, 0), c("A", "")), "estimate_input")      # empty id
  expect_input_error(paper_totals(c(1, NA), c("A", "B")), "estimate_input")    # missing difference
  expect_input_error(paper_totals(c(1, 2), c("A", "B")), "estimate_input")     # not a 0/1 difference
  expect_input_error(paper_totals(c(1, 0), factor(c("A", "B"))), "estimate_input")  # ids must be character
  expect_input_error(paper_totals(c("1", "0"), c("A", "B")), "estimate_input") # not numeric
})

test_that("replicates are the pooled item-weighted ratio of the drawn papers, by hand", {
  t <- paper_totals(hand_pairs()$d, hand_pairs()$paper)
  draws <- rbind(
    c(1L, 1L, 2L),   # A,A,B : (3+3-1)/(3+3+1) = 5/7   (A counted twice, with all 3 items each time)
    c(2L, 2L, 2L),   # B,B,B : -1
    c(3L, 3L, 1L),   # C,C,A : (0+0+3)/(2+2+3) = 3/7
    c(1L, 2L, 3L),   # A,B,C : (3-1+0)/6 = 1/3
    c(1L, 1L, 1L),   # A,A,A : 1
    c(1L, 2L, 2L)    # A,B,B : (3-1-1)/(3+1+1) = 1/5
  )
  r <- bootstrap_replicates(t, draws)
  expect_equal(r$estimates, c(5 / 7, -1, 3 / 7, 1 / 3, 1, 1 / 5), tolerance = 1e-15)
  expect_identical(r$n_empty, 0L)
  # an equal-paper-weighted replicate would be mean of paper means: row 1 = (1+1-1)/3 = 1/3, not 5/7
  expect_false(isTRUE(all.equal(r$estimates[1], 1 / 3)))
})

test_that("replicates are exact integer ratios (no accumulated floating-point error)", {
  t <- paper_totals(rep(c(1, -1, 0), 40), rep(c("A", "B", "C"), 40))
  draws <- matrix(c(1L, 2L, 3L, 3L, 3L, 1L), nrow = 2, byrow = TRUE)
  r <- bootstrap_replicates(t, draws)
  expect_identical(r$estimates, c(0, 40 / 120))   # (40-40+0)/120 ; (0+0+40)/120
})

test_that("draws must index the frame", {
  t <- paper_totals(hand_pairs()$d, hand_pairs()$paper)
  expect_input_error(bootstrap_replicates(t, matrix(c(1L, 2L, 4L), 1)), "bootstrap_input")
  expect_input_error(bootstrap_replicates(t, matrix(c(1L, 2L, 0L), 1)), "bootstrap_input")
  expect_input_error(bootstrap_replicates(t, matrix(c(1L, 2L), 1)), "bootstrap_input")   # not P columns
  expect_input_error(bootstrap_replicates(t, matrix(c(1L, NA, 2L), 1)), "bootstrap_input")
})

test_that("resamples with no evaluable items are counted and excluded", {
  # Not reachable on the contributing-paper frame; proves the counter and the exclusion.
  t <- data.frame(paper_id = c("A", "Z"), n_items = c(2L, 0L), sum_d = c(2L, 0L), stringsAsFactors = FALSE)
  r <- bootstrap_replicates(t, rbind(c(1L, 1L), c(2L, 2L), c(1L, 2L)))
  expect_identical(r$n_empty, 1L)
  expect_equal(r$estimates, c(1, NA, 1))
  expect_equal(percentile_interval(r$estimates), c(1, 1))
})

test_that("percentile interval is quantile type 7 (hand interpolation)", {
  # n = 41: h = 1 + 40p is exactly 2 and 40 -> the 2nd and 40th order statistics.
  v <- as.numeric(1:41)
  expect_equal(percentile_interval(rev(v)), c(2, 40), tolerance = 1e-12)
  expect_equal(percentile_interval(v[c(seq(1, 41, 2), seq(2, 40, 2))]), c(2, 40), tolerance = 1e-12)
  # n = 10: h = 1.225 and 9.775.  x = i^2: 1 + 0.225*(4-1) and 81 + 0.775*(100-81)
  x <- (1:10)^2
  expect_equal(percentile_interval(rev(x)), c(1 + 0.225 * 3, 81 + 0.775 * 19), tolerance = 1e-14)
})

test_that("the draw matrix is B x P, in 1..P, and fully determined by the seed", {
  a <- draw_papers(5L, 200L, TEST_SEED)
  expect_identical(dim(a), c(200L, 5L))
  expect_true(is.integer(a))
  expect_true(all(a >= 1L & a <= 5L))
  expect_identical(a, draw_papers(5L, 200L, TEST_SEED))
  b <- draw_papers(5L, 200L, TEST_SEED + 1L)     # fixed seeds, fixed outcome: not a chance assertion
  expect_false(identical(a, b))
  expect_true(any(duplicated(a[1, ])) || any(duplicated(a[2, ])))   # papers are drawn with replacement
})

test_that("RNG regression vector: Mersenne-Twister, Inversion, Rejection, single sample.int call", {
  expect_identical(BOOTSTRAP_RNG, list(kind = "Mersenne-Twister", normal.kind = "Inversion",
                                       sample.kind = "Rejection"))
  # Same stream the documented algorithm defines, row-wise.
  set.seed(TEST_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  flat <- sample.int(4L, 12L, replace = TRUE)
  expect_identical(draw_papers(4L, 3L, TEST_SEED), matrix(flat, nrow = 3, byrow = TRUE))
  expect_identical(as.vector(t(draw_papers(4L, 3L, TEST_SEED))), PINNED_DRAWS)
})

test_that("draw_papers leaves the caller's RNG state untouched", {
  # 1. an existing default-kind seed
  set.seed(7)
  before <- get(".Random.seed", envir = globalenv())
  invisible(draw_papers(4L, 10L, TEST_SEED))
  expect_identical(get(".Random.seed", envir = globalenv()), before)
  # 2. a different kind in force
  on.exit(RNGkind("Mersenne-Twister", "Inversion", "Rejection"), add = TRUE)
  suppressWarnings(RNGkind("L'Ecuyer-CMRG", "Box-Muller", "Rounding"))
  set.seed(7)
  before <- get(".Random.seed", envir = globalenv())
  kinds <- RNGkind()
  invisible(draw_papers(4L, 10L, TEST_SEED))
  expect_identical(get(".Random.seed", envir = globalenv()), before)
  expect_identical(RNGkind(), kinds)
  # the draws themselves do not depend on the caller's RNG kind
  expect_identical(as.vector(t(draw_papers(4L, 3L, TEST_SEED))), PINNED_DRAWS)
})

test_that("an initially absent .Random.seed stays absent", {
  rscript <- file.path(R.home("bin"), "Rscript")
  code <- c(
    sprintf("source('%s')", file.path(r_dir, "read_inputs.R")),
    sprintf("source('%s')", file.path(r_dir, "bootstrap.R")),
    "stopifnot(!exists('.Random.seed', envir = globalenv(), inherits = FALSE))",
    sprintf("invisible(draw_papers(4L, 10L, %dL))", TEST_SEED),
    "cat(exists('.Random.seed', envir = globalenv(), inherits = FALSE), paste(RNGkind(), collapse = '|'))"
  )
  f <- tempfile(fileext = ".R")
  writeLines(code, f)
  out <- system2(rscript, c("--vanilla", f), stdout = TRUE, stderr = FALSE)
  expect_identical(out, "FALSE Mersenne-Twister|Inversion|Rejection")
})

test_that("RNG state is restored when the sampling code fails", {
  set.seed(11)
  before <- get(".Random.seed", envir = globalenv())
  expect_error(with_bootstrap_rng(TEST_SEED, { runif(3); stop("boom") }), "boom")
  expect_identical(get(".Random.seed", envir = globalenv()), before)
  # failure during validation, after nothing was drawn
  expect_input_error(draw_papers(0L, 10L, TEST_SEED), "bootstrap_input")
  expect_identical(get(".Random.seed", envir = globalenv()), before)
})

test_that("seeds and sizes are validated; there is no default seed", {
  expect_error(draw_papers(4L, 10L), "seed")
  expect_input_error(draw_papers(4L, 10L, NA_integer_), "bootstrap_input")
  expect_input_error(draw_papers(4L, 10L, 1.5), "bootstrap_input")
  expect_input_error(draw_papers(4L, 10L, c(1L, 2L)), "bootstrap_input")
  expect_input_error(draw_papers(4L, 10L, "7"), "bootstrap_input")
  expect_input_error(draw_papers(4L, 10L, 2^31), "bootstrap_input")        # not representable as an R integer seed
  expect_input_error(draw_papers(4L, 0L, TEST_SEED), "bootstrap_input")
  expect_input_error(draw_papers(0L, 10L, TEST_SEED), "bootstrap_input")
})

test_that("percentile_bootstrap equals the documented procedure computed independently", {
  h <- hand_pairs()
  r <- percentile_bootstrap(h$d, h$paper, TEST_SEED, 500L)
  # Independent recomputation straight from the item rows, one resample at a time.
  set.seed(TEST_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  flat <- sample.int(3L, 500L * 3L, replace = TRUE)
  m <- matrix(flat, nrow = 500, byrow = TRUE)
  papers <- c("A", "B", "C")
  est <- vapply(seq_len(500), function(i) {
    rows <- unlist(lapply(papers[m[i, ]], function(p) which(h$paper == p)))
    mean(h$d[rows])
  }, 0)
  q <- unname(stats::quantile(est, c(0.025, 0.975), type = 7))
  expect_identical(r$status, "ok")
  expect_equal(c(r$ci_lower, r$ci_upper), q, tolerance = 1e-12)
  expect_identical(r$n_resamples, 500L)           # replicates actually computed
  expect_identical(r$n_empty, 0L)
  expect_identical(r$n_papers_frame, 3L)
  expect_identical(r$method, "whole_paper_percentile")
  expect_identical(r$seed, TEST_SEED)
  expect_identical(r$rng, BOOTSTRAP_RNG)
  expect_identical(r$quantile_type, 7L)
})

test_that("bootstrap is invariant to input row order and paper labels' sort position", {
  h <- hand_pairs()
  base <- percentile_bootstrap(h$d, h$paper, TEST_SEED, 300L)
  o <- c(4, 6, 1, 3, 5, 2)
  expect_identical(percentile_bootstrap(h$d[o], h$paper[o], TEST_SEED, 300L), base)
})

test_that("unequal paper sizes: item weighting differs from equal-paper weighting in the interval", {
  # One large all-positive paper and many single-item negative papers.
  d <- c(rep(1, 10), rep(-1, 4))
  paper <- c(rep("BIG", 10), paste0("S", 1:4))
  r <- percentile_bootstrap(d, paper, TEST_SEED, 400L)
  set.seed(TEST_SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
  m <- matrix(sample.int(5L, 400L * 5L, replace = TRUE), nrow = 400, byrow = TRUE)
  ord <- c("BIG", "S1", "S2", "S3", "S4")          # radix order of the labels
  expect_identical(sort(unique(paper), method = "radix"), ord)
  n <- c(10, 1, 1, 1, 1); s <- c(10, -1, -1, -1, -1)
  est <- vapply(seq_len(400), function(i) sum(s[m[i, ]]) / sum(n[m[i, ]]), 0)
  expect_equal(c(r$ci_lower, r$ci_upper), unname(stats::quantile(est, c(.025, .975), type = 7)), tolerance = 1e-12)
})

test_that("zero complete pairs: explicit status, no interval, no RNG consumed", {
  set.seed(5)
  before <- get(".Random.seed", envir = globalenv())
  r <- percentile_bootstrap(numeric(0), character(0), TEST_SEED, 100L)
  expect_identical(r$status, "no_complete_pairs")
  expect_true(is.na(r$ci_lower) && is.na(r$ci_upper))
  expect_identical(r$n_papers_frame, 0L)
  expect_identical(get(".Random.seed", envir = globalenv()), before)
})

test_that("a single paper gives a degenerate interval labelled single_cluster_degenerate", {
  r <- percentile_bootstrap(c(1, 0, 1), c("A", "A", "A"), TEST_SEED, 50L)
  expect_identical(r$status, "single_cluster_degenerate")
  expect_identical(r$n_papers_frame, 1L)
  expect_equal(c(r$ci_lower, r$ci_upper), c(2 / 3, 2 / 3))
})

test_that("all-identical differences give a valid degenerate interval at the extreme", {
  r <- percentile_bootstrap(c(1, 1, 1, 1), c("A", "A", "B", "C"), TEST_SEED, 100L)
  expect_identical(r$status, "ok")
  expect_equal(c(r$ci_lower, r$ci_upper), c(1, 1))
})

test_that("the frame is per outcome: papers without a complete pair for that outcome are absent", {
  # outcome 1 has papers A,B,C ; outcome 2 has only A,B
  r1 <- percentile_bootstrap(c(1, 0, -1), c("A", "B", "C"), TEST_SEED, 50L)
  r2 <- percentile_bootstrap(c(1, 0), c("A", "B"), TEST_SEED, 50L)
  expect_identical(c(r1$n_papers_frame, r2$n_papers_frame), c(3L, 2L))
})

test_that("the primary path uses exactly the handoff's 10,000 resamples", {
  read <- fixture()
  expect_identical(read$handoff$analysis_spec$bootstrap_resamples, 10000L)
  expect_identical(primary_resamples(read$handoff), 10000L)
  bad <- read$handoff
  bad$analysis_spec$bootstrap_resamples <- 9999L
  expect_input_error(primary_resamples(bad), "bootstrap_input")
  expect_identical(PRIMARY_RESAMPLES, 10000L)
})
