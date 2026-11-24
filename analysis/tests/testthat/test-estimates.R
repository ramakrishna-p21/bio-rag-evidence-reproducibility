# Pure estimators. Every expected number is computed by hand in the comment next to it.

test_that("paired differences are JATS minus PDF_DOCLING on 0/1 outcomes", {
  expect_identical(paired_differences(c(1, 1, 1), c(0, 0, 0)), c(1, 1, 1))     # all positive
  expect_identical(paired_differences(c(0, 0), c(1, 1)), c(-1, -1))            # all negative
  expect_identical(paired_differences(c(1, 0, 1, 0), c(0, 0, 1, 1)), c(1, 0, 0, -1))  # mixed
  expect_input_error(paired_differences(c(1, 0), c(1)), "estimate_input")
  expect_input_error(paired_differences(c(1, NA), c(1, 0)), "estimate_input")
  expect_input_error(paired_differences(c(1, 2), c(1, 0)), "estimate_input")
})

test_that("summary: item-weighted headline, rates, counts and the equal-paper sensitivity", {
  # Paper A: three items, d = +1 each (J=1,P=0). Paper B: one item, d = -1 (J=0,P=1).
  s <- summarize_pairs(y_jats = c(1, 1, 1, 0), y_pdf = c(0, 0, 0, 1), paper_id = c("A", "A", "A", "B"))
  expect_identical(s$status, "ok")
  expect_identical(s$n_items, 4L)
  expect_identical(s$n_papers, 2L)
  expect_equal(s$rate_jats, 3 / 4)                 # (1+1+1+0)/4
  expect_equal(s$rate_pdf, 1 / 4)                  # (0+0+0+1)/4
  expect_equal(s$rd, 2 / 4)                        # (1+1+1-1)/4 : item-weighted, paper A dominates
  expect_equal(s$rd_pp, 50)
  expect_equal(s$equal_paper_rd, 0)                # mean(paper A = 1, paper B = -1)
  expect_equal(s$rd, s$rate_jats - s$rate_pdf)
})

test_that("summary of all-positive and all-negative samples", {
  pos <- summarize_pairs(c(1, 1), c(0, 0), c("A", "B"))
  expect_equal(c(pos$rd, pos$equal_paper_rd, pos$rate_jats, pos$rate_pdf), c(1, 1, 1, 0))
  neg <- summarize_pairs(c(0, 0, 0), c(1, 1, 1), c("A", "A", "B"))
  expect_equal(c(neg$rd, neg$equal_paper_rd), c(-1, -1))
})

test_that("no complete pairs is an explicit status, not a zero effect", {
  s <- summarize_pairs(numeric(0), numeric(0), character(0))
  expect_identical(s$status, "no_complete_pairs")
  expect_identical(s$n_items, 0L)
  expect_true(is.na(s$rd) && is.na(s$equal_paper_rd) && is.na(s$rate_jats))
})

test_that("Holm over exactly two p-values", {
  # Sorted ascending: p_(1)*2, then max(previous, p_(2)*1), capped at 1.
  h <- holm_adjust(c(0.01, 0.04));  expect_identical(h$status, "ok"); expect_equal(h$adjusted, c(0.02, 0.04))
  expect_equal(holm_adjust(c(0.04, 0.01))$adjusted, c(0.04, 0.02))      # order of input preserved
  expect_equal(holm_adjust(c(0.03, 0.03))$adjusted, c(0.06, 0.06))
  expect_equal(holm_adjust(c(0.6, 0.7))$adjusted, c(1, 1))              # 1.2 capped; 0.7 raised to 1.2 -> 1
  expect_equal(holm_adjust(c(0.4, 0.01))$adjusted, c(0.4, 0.02))
  expect_equal(holm_adjust(c(0, 0.5))$adjusted, c(0, 0.5))
  expect_input_error(holm_adjust(0.01), "estimate_input")                # not exactly two
  expect_input_error(holm_adjust(c(0.01, 0.02, 0.03)), "estimate_input")
  expect_input_error(holm_adjust(c(-0.1, 0.5)), "estimate_input")
  expect_input_error(holm_adjust(c(0.5, 1.1)), "estimate_input")
  und <- holm_adjust(c(0.01, NA))                                        # never a one-p Holm
  expect_identical(und$status, "holm_undefined")
  expect_true(all(is.na(und$adjusted)))
})

rule <- list(success = "yes", failure = "no", non_evaluable = c("non_evaluable", "missing"))

test_that("bounds: exact lower/upper over the eligible set (hand-computed)", {
  # Items (JATS, PDF): 1:(yes,no)  2:(yes,non_evaluable)  3:(non_evaluable,yes)  4:(no,no)
  # d known: item1 = +1, item4 = 0.
  # item2 PDF unknown: best (upper) PDF=0 -> d=+1 ; worst (lower) PDF=1 -> d=0.
  # item3 JATS unknown: best JATS=1 -> d=0      ; worst JATS=0 -> d=-1.
  # upper sum = 1+1+0+0 = 2, lower sum = 1+0-1+0 = 0 ; denominator = 4 eligible items.
  b <- missingness_bounds(
    jats_state = c("yes", "yes", "non_evaluable", "no"),
    pdf_state = c("no", "non_evaluable", "yes", "no"), rule = rule
  )
  expect_identical(b$n_eligible, 4L)
  expect_identical(b$n_operational_excluded, 0L)
  expect_equal(b$upper, 2 / 4)
  expect_equal(b$lower, 0)
  expect_identical(b$n_unknown_jats, 1L)
  expect_identical(b$n_unknown_pdf, 1L)
  expect_identical(b$unknown_states, c("JATS|non_evaluable" = 1L, "PDF_DOCLING|non_evaluable" = 1L))
})

test_that("operational missing is never bounded as a scientific outcome (EXPERIMENT_PLAN 9)", {
  # Items: 1:(yes,no) d=+1  2:(yes,missing) operational  3:(non_evaluable,yes) unresolved
  # 4:(not_executed,no) operational. Item 2 and 4 leave the bounds and are counted; item 3 is
  # bounded: upper d=0, lower d=-1. Denominator 2: upper (1+0)/2, lower (1-1)/2.
  r <- list(success = "yes", failure = "no", non_evaluable = c("non_evaluable", "missing", "not_executed"))
  b <- missingness_bounds(c("yes", "yes", "non_evaluable", "not_executed"),
                          c("no", "missing", "yes", "no"), r)
  expect_identical(c(b$n_eligible, b$n_operational_excluded), c(2L, 2L))
  expect_equal(c(b$lower, b$upper), c(0, 0.5))
  expect_identical(b$unknown_states, c("JATS|non_evaluable" = 1L))
  expect_input_error(missingness_bounds("missing", "yes", r), "estimate_input")
})

test_that("bounds do not necessarily bracket the complete-case estimate", {
  # item1 (yes,no): d=+1 ; item2 (no, non_evaluable): JATS=0 known, PDF unknown.
  # complete-case estimate = 1 (item1 only). Upper: PDF=0 -> d=0 ; lower: PDF=1 -> d=-1.
  # upper = (1+0)/2 = 0.5 ; lower = (1-1)/2 = 0 ; the estimate 1 lies outside [0, 0.5].
  b <- missingness_bounds(c("yes", "no"), c("no", "non_evaluable"), rule)
  expect_equal(c(b$lower, b$upper), c(0, 0.5))
  s <- summarize_pairs(1, 0, "A")
  expect_equal(s$rd, 1)
  expect_true(s$rd > b$upper)
})

test_that("with no unknown cell the bounds equal the item-weighted estimate", {
  b <- missingness_bounds(c("yes", "no", "yes"), c("no", "no", "yes"), rule)   # d = 1, 0, 0
  expect_equal(c(b$lower, b$upper), c(1 / 3, 1 / 3))
  expect_identical(length(b$unknown_states), 0L)
})

test_that("bounds on the frozen fixture, per outcome (hand-computed over 9 eligible items)", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  rules <- outcome_rules(read$handoff)
  d <- acc$evidence_delivered$items
  bd <- missingness_bounds(d$jats_state, d$pdf_state, rules$evidence_delivered)
  # known d sum over complete pairs = 1 (q_001 +1, q_006 -1, q_007 +1); q_005 JATS unknown
  # (PDF yes): upper 0, lower -1; q_009 PDF unknown (JATS yes): upper +1, lower 0.
  expect_identical(bd$n_eligible, 9L)
  expect_equal(bd$upper, (1 + 0 + 1) / 9)
  expect_equal(bd$lower, (1 - 1 + 0) / 9)
  a <- acc$answer_correct$items
  ba <- missingness_bounds(a$jats_state, a$pdf_state, rules$answer_correct)
  # known sum = 0 (q_001 +1, q_003 -1, q_006 -1, q_009 +1); q_004 PDF unknown (JATS yes):
  # upper +1, lower 0; q_008 both unknown: upper +1, lower -1.
  # q_004 (PDF operational failure) and q_008 (both operational failures) are missing, not
  # unresolved: they leave the bounds, which equal the complete-case estimate 0 over 7 items.
  expect_identical(c(ba$n_eligible, ba$n_operational_excluded), c(7L, 2L))
  expect_equal(c(ba$lower, ba$upper), c(0, 0))
  expect_identical(length(ba$unknown_states), 0L)
})

test_that("estimates on the frozen fixture match the hand-computed oracle", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  est <- function(name) {
    items <- acc[[name]]$items
    done <- items[items$complete, ]
    summarize_pairs(done$y_jats, done$y_pdf, done$paper_id)
  }
  d <- est("evidence_delivered")   # 7 pairs: d = +1,0,0,0,-1,+1,0 -> sum 1
  expect_identical(c(d$n_items, d$n_papers), c(7L, 5L))
  expect_equal(d$rd, 1 / 7)
  expect_equal(c(d$rate_jats, d$rate_pdf), c(6 / 7, 5 / 7))
  expect_equal(d$equal_paper_rd, (1 + 0 - 0.5 + 1 + 0) / 5)    # P001 1, P002 0, P003 -.5, P004 1, P005 0
  a <- est("answer_correct")       # 7 pairs: d = +1,0,-1,0,-1,0,+1 -> sum 0
  expect_identical(c(a$n_items, a$n_papers), c(7L, 5L))
  expect_equal(a$rd, 0)
  expect_equal(c(a$rate_jats, a$rate_pdf), c(5 / 7, 5 / 7))
  expect_equal(a$equal_paper_rd, (1 - 0.5 - 0.5 + 0 + 1) / 5)  # P001 1, P002 -.5, P003 -.5, P004 0, P005 1
  expect_false(isTRUE(all.equal(d$rd, d$equal_paper_rd)))      # weighting differs
})

test_that("unexpected or non-confirmatory states are rejected by the bounds", {
  expect_input_error(missingness_bounds(c("yes", "maybe"), c("no", "no"), rule), "estimate_input")
  expect_input_error(missingness_bounds(c("yes"), c("no", "no"), rule), "estimate_input")
  expect_input_error(missingness_bounds(character(0), character(0), rule), "estimate_input")
})
