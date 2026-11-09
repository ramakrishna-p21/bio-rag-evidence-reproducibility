# Phase 11 audit: oracles that do not share code with the implementation under test.

test_that("bounds equal the extremes found by enumerating every assignment of the unresolved cells", {
  rule <- list(success = "yes", failure = "no", non_evaluable = c("non_evaluable", "missing"))
  cases <- list(
    list(j = c("yes", "no", "missing", "yes"), p = c("no", "non_evaluable", "yes", "no")),
    list(j = c("missing", "missing", "no"), p = c("yes", "no", "non_evaluable")),
    list(j = c("yes", "yes", "no"), p = c("no", "yes", "no")),
    list(j = c("non_evaluable", "yes"), p = c("non_evaluable", "missing"))
  )
  for (cs in cases) {
    n <- length(cs$j)
    unknown <- c(which(!cs$j %in% c("yes", "no")), n + which(!cs$p %in% c("yes", "no")))
    known <- c(ifelse(cs$j == "yes", 1, ifelse(cs$j == "no", 0, NA)), ifelse(cs$p == "yes", 1, ifelse(cs$p == "no", 0, NA)))
    grid <- if (length(unknown)) expand.grid(rep(list(c(0, 1)), length(unknown))) else data.frame(none = 0)
    rd <- apply(grid, 1, function(v) {
      cells <- known
      if (length(unknown)) cells[unknown] <- v
      mean(cells[seq_len(n)] - cells[n + seq_len(n)])
    })
    b <- missingness_bounds(cs$j, cs$p, rule)
    expect_equal(b$lower, min(rd), tolerance = 1e-15)
    expect_equal(b$upper, max(rd), tolerance = 1e-15)
  }
})

test_that("the whole-paper bootstrap matches the exact resampling distribution of a three-paper frame", {
  d <- c(1, 0, 1, -1, 0, 0)                  # papers A (2 items), B (3 items), C (1 item)
  paper <- c("A", "A", "B", "B", "B", "C")
  sizes <- c(2, 3, 1); sums <- c(1, 0, 0)
  all_draws <- as.matrix(expand.grid(1:3, 1:3, 1:3))     # all 27 equally likely paper draws
  exact_values <- apply(all_draws, 1, function(i) sum(sums[i]) / sum(sizes[i]))
  exact <- tapply(rep(1 / 27, 27), exact_values, sum)
  b <- bootstrap_replicates(paper_totals(d, paper), draw_papers(3, 20000, 11L))
  observed <- table(factor(round(b$estimates, 12), levels = round(as.numeric(names(exact)), 12))) / 20000
  expect_equal(sum(observed), 1)                         # no replicate outside the exact support
  expect_true(all(abs(as.numeric(observed) - as.numeric(exact)) < 0.012))
  expect_equal(mean(b$estimates), sum(as.numeric(names(exact)) * exact), tolerance = 0.01)
})

test_that("a paper drawn twice counts twice in an expanded-item resample", {
  d <- c(1, 1, -1, 0, 1)
  paper <- c("A", "A", "B", "C", "C")
  totals <- paper_totals(d, paper)
  draws <- rbind(c(1, 1, 2), c(3, 3, 3), c(2, 2, 2))
  by_loop <- apply(draws, 1, function(i) mean(unlist(lapply(c("A", "B", "C")[i], function(p) d[paper == p]))))
  expect_equal(bootstrap_replicates(totals, draws)$estimates, by_loop, tolerance = 1e-15)
})

test_that("Holm adjustment on exact two-p cases", {
  expect_equal(holm_adjust(c(0.01, 0.04))$adjusted, c(0.02, 0.04))
  expect_equal(holm_adjust(c(0.04, 0.01))$adjusted, c(0.04, 0.02))
  expect_equal(holm_adjust(c(0.03, 0.03))$adjusted, c(0.06, 0.06))
  expect_equal(holm_adjust(c(0.6, 0.7))$adjusted, c(1, 1))
  expect_equal(holm_adjust(c(0, 0.5))$adjusted, c(0, 0.5))
})

test_that("extra repeats and robustness rows leave the primary accounting unchanged", {
  a <- fixture_args()
  inputs <- read_inputs(a$handoff_path, a$handoff_sha256, a$csv_path)
  base <- derive_accounting(inputs$rows, inputs$handoff)
  extra <- inputs$rows[inputs$rows$matrix_arm == "confirmatory", ]
  r2 <- extra; r2$repeat_index <- 2L; r2$primary_row <- FALSE
  r2$observation_key <- paste0(r2$observation_key, "_r2")
  flip <- function(x) ifelse(x == "yes", "no", ifelse(x == "no", "yes", x))
  r2$answer_correct <- flip(r2$answer_correct); r2$evidence_delivered <- flip(r2$evidence_delivered)
  dense <- extra; dense$matrix_arm <- "dense_robustness"
  dense$answer_correct <- flip(dense$answer_correct)
  expect_identical(derive_accounting(rbind(inputs$rows, r2, dense), inputs$handoff), base)
})
