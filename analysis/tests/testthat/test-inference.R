# CR2 / Satterthwaite inference on the paired differences. Fixtures are synthetic.
#
# Two independent oracles that do not call clubSandwich:
#  * closed form, intercept-only: with n items, paper sizes n_g and paper sums S_g,
#      SE^2 = sum_g (S_g - n_g * mean)^2 / (n^2 * (1 - n_g / n))
#  * matrix form under the identity working model: Q = M C M with M = I - 11'/n and
#      C[i,j] = sqrt(w_i w_j) / n^2 for items in the same paper (w = 1 / (1 - n_g / n)),
#      SE^2 = d'Qd, df = (tr Q)^2 / tr(Q^2), p = 2 pt(-|t|, df).

D1 <- c(1, 1, 0, -1, 0, 1, -1, 0, 0, 1); G1 <- c("A", "A", "A", "B", "B", "C", "C", "C", "D", "D")
D2 <- c(1, 1, -1, 1, 1, 0, -1, 0);       G2 <- c("X", "Y", "Y", "Z", "Z", "Z", "Z", "Z")   # sizes 1, 2, 5

test_that("closed-form variance reproduces the hand value for the 4-paper fixture", {
  # S = (2, -1, 0, 1), n_g = (3, 2, 3, 2), mean 0.2, n = 10
  # (1.4^2)/0.7 + (1.4^2)/0.8 + (0.6^2)/0.7 + (0.6^2)/0.8 = 6.2142857 ; SE = sqrt(6.2142857)/10
  expect_equal(ref_closed_se(D1, G1)^2 * 100, 2.8 + 2.45 + 0.36 / 0.7 + 0.45, tolerance = 1e-12)
  expect_equal(ref_closed_se(D1, G1), 0.2492847, tolerance = 1e-6)
})

test_that("independent matrix reference agrees with the closed form (oracle self-check)", {
  for (case in list(list(D1, G1), list(D2, G2))) {
    expect_equal(ref_matrix(case[[1]], case[[2]])[["se"]], ref_closed_se(case[[1]], case[[2]]), tolerance = 1e-12)
  }
})

test_that("implementation matches the independent mathematical reference", {
  for (case in list(list(D1, G1), list(D2, G2))) {
    r <- cr2_satterthwaite(case[[1]], case[[2]])
    expect_identical(r$status, "ok")
    expect_equal(fields(r), ref_matrix(case[[1]], case[[2]]), tolerance = 1e-10)
  }
  r <- cr2_satterthwaite(D1, G1)        # literal values observed from the package for this fixture
  expect_equal(c(r$se, r$df, r$p), c(0.2492847, 2.861314, 0.4836377), tolerance = 1e-6)
})

test_that("implementation matches a separate direct clubSandwich call", {
  for (case in list(list(D1, G1), list(D2, G2))) {
    frame <- data.frame(y = case[[1]], g = factor(case[[2]]))
    ct <- clubSandwich::coef_test(stats::lm(y ~ 1, data = frame), vcov = "CR2",
                                  cluster = frame$g, test = "Satterthwaite")
    r <- cr2_satterthwaite(case[[1]], case[[2]])
    expect_equal(fields(r), c(estimate = ct$beta, se = ct$SE, df = ct$df_Satt, t = ct$tstat, p = ct$p_Satt),
                 tolerance = 1e-12)
  }
})

test_that("estimate is the item-weighted paired risk difference and counts are reported", {
  r <- cr2_satterthwaite(D1, G1)
  expect_equal(r$estimate, mean(D1), tolerance = 1e-12)
  expect_identical(c(r$n_items, r$n_papers), c(10L, 4L))
  expect_identical(r$method, "CR2_Satterthwaite_intercept_only")
  expect_identical(r$package, "clubSandwich")
  expect_identical(r$package_version, as.character(utils::packageVersion("clubSandwich")))
  expect_identical(r$warnings, character(0))
})

test_that("results do not depend on row order or on the paper labels", {
  base <- cr2_satterthwaite(D1, G1)
  o <- c(7, 2, 9, 1, 10, 4, 3, 8, 5, 6)
  expect_equal(fields(cr2_satterthwaite(D1[o], G1[o])), fields(base), tolerance = 1e-12)
  relabel <- c(A = "q9", B = "a1", C = "m5", D = "b2")
  expect_equal(fields(cr2_satterthwaite(D1, unname(relabel[G1]))), fields(base), tolerance = 1e-12)
})

test_that("reversing the contrast flips the estimate and t only", {
  a <- cr2_satterthwaite(D1, G1); b <- cr2_satterthwaite(-D1, G1)
  expect_equal(c(b$estimate, b$t), -c(a$estimate, a$t), tolerance = 1e-12)
  expect_equal(c(b$se, b$df, b$p), c(a$se, a$df, a$p), tolerance = 1e-12)
})

test_that("the paper-clustered SE is neither the naive nor the item-clustered SE", {
  # Strong within-paper agreement, so clustering matters.
  d <- c(1, 1, 1, 1, -1, -1, -1, 0, 0, 1, 1); g <- rep(c("A", "B", "C", "D"), c(4, 3, 2, 2))
  r <- cr2_satterthwaite(d, g)
  naive <- stats::sd(d) / sqrt(length(d))
  by_item <- ref_closed_se(d, as.character(seq_along(d)))
  expect_gt(abs(r$se - naive), 0.05)
  expect_gt(abs(r$se - by_item), 0.05)
  expect_equal(r$se, ref_closed_se(d, g), tolerance = 1e-12)
})

test_that("an extreme but valid estimate is not undefined", {
  d <- c(rep(1, 9), -1); g <- c("A", "A", "B", "B", "C", "C", "D", "D", "E", "E")
  r <- cr2_satterthwaite(d, g)
  expect_identical(r$status, "ok")
  expect_equal(r$estimate, 0.8)
  expect_equal(fields(r), ref_matrix(d, g), tolerance = 1e-10)
})

test_that("two papers are enough; low degrees of freedom are reported, not hidden", {
  d <- c(1, 0, -1, 1); g <- c("A", "A", "B", "B")
  r <- cr2_satterthwaite(d, g)
  expect_identical(r$status, "ok")
  expect_equal(fields(r), ref_matrix(d, g), tolerance = 1e-10)
  expect_equal(r$df, 1)
  expect_identical(r$n_papers, 2L)
})

expect_undefined <- function(r, reason) {
  expect_identical(r$status, "inference_undefined")
  expect_identical(r$reason, reason)
  expect_true(all(is.na(c(r$se, r$df, r$t, r$p))))
}

test_that("zero complete pairs is undefined and has no estimate", {
  r <- cr2_satterthwaite(numeric(0), character(0))
  expect_undefined(r, "no_complete_pairs")
  expect_true(is.na(r$estimate))
  expect_identical(c(r$n_items, r$n_papers), c(0L, 0L))
})

test_that("one paper is undefined, but the estimate stays", {
  r <- cr2_satterthwaite(c(1, 0, 1), c("A", "A", "A"))
  expect_undefined(r, "single_cluster")
  expect_equal(r$estimate, 2 / 3)
})

test_that("all-zero and identical nonzero differences are zero_variance with their valid estimate", {
  r0 <- cr2_satterthwaite(rep(0, 6), rep(c("A", "B", "C"), each = 2))
  expect_undefined(r0, "zero_variance"); expect_equal(r0$estimate, 0)
  r1 <- cr2_satterthwaite(rep(1, 6), rep(c("A", "B", "C"), each = 2))
  expect_undefined(r1, "zero_variance"); expect_equal(r1$estimate, 1)
  rm1 <- cr2_satterthwaite(rep(-1, 4), c("A", "B", "C", "C"))
  expect_undefined(rm1, "zero_variance"); expect_equal(rm1$estimate, -1)
})

test_that("zero robust variance with varying differences is undefined, decided in exact integers", {
  # Every paper sum equals n_g * grand mean: residual cluster sums are exactly 0.
  expect_undefined(cr2_satterthwaite(c(1, -1, 1, -1), c("A", "A", "B", "B")), "zero_robust_variance")
  r <- cr2_satterthwaite(c(1, 0, 1, 0, 1, 0), c("A", "A", "B", "B", "C", "C"))
  expect_undefined(r, "zero_robust_variance")
  expect_equal(r$estimate, 0.5)
  # one residual sum nonzero by a single unit stays defined and matches the reference
  d <- c(1, 0, 1, 0, 1, 1); g <- c("A", "A", "B", "B", "C", "C")
  expect_identical(cr2_satterthwaite(d, g)$status, "ok")
  expect_equal(fields(cr2_satterthwaite(d, g)), ref_matrix(d, g), tolerance = 1e-10)
})

test_that("a small but genuine variance is not mistaken for degenerate", {
  n_pairs <- 200
  d <- rep(0, n_pairs); d[1] <- 1                      # one nonzero item among 200, 100 papers
  g <- paste0("P", rep(1:100, each = 2))
  r <- cr2_satterthwaite(d, g)
  expect_identical(r$status, "ok")
  expect_equal(fields(r), ref_matrix(d, g), tolerance = 1e-8)
  expect_gt(r$se, 0)
})

test_that("unusable pairs are refused rather than clustered wrongly", {
  expect_input_error(cr2_satterthwaite(c(1, 0), "A"), "estimate_input")
  expect_input_error(cr2_satterthwaite(c(1, 0), c("A", NA)), "estimate_input")
  expect_input_error(cr2_satterthwaite(c(1, NA), c("A", "B")), "estimate_input")
  expect_input_error(cr2_satterthwaite(c(1, 0.5), c("A", "B")), "estimate_input")
  expect_input_error(cr2_satterthwaite(c(1, 0), factor(c("A", "B"))), "estimate_input")
})

test_that("unexpected package failures are not turned into undefined inference", {
  testthat::local_mocked_bindings(coef_test = function(...) stop("package exploded"), .package = "clubSandwich")
  expect_error(cr2_satterthwaite(D1, G1), "package exploded")
})

test_that("a package result with an unusable field is flagged, not reported", {
  bad <- function(se = 0.25, df = 2.8, tstat = 0.8, p = 0.48, beta = 0.2) {
    function(...) data.frame(Coef = "(Intercept)", beta = beta, SE = se, null_value = 0, tstat = tstat,
                             df_Satt = df, p_Satt = p)
  }
  testthat::local_mocked_bindings(coef_test = bad(se = NaN), .package = "clubSandwich")
  expect_undefined(cr2_satterthwaite(D1, G1), "nonfinite_result")
  testthat::local_mocked_bindings(coef_test = bad(df = Inf), .package = "clubSandwich")
  expect_undefined(cr2_satterthwaite(D1, G1), "nonfinite_result")
  testthat::local_mocked_bindings(coef_test = bad(p = NA_real_), .package = "clubSandwich")
  expect_undefined(cr2_satterthwaite(D1, G1), "nonfinite_result")
  testthat::local_mocked_bindings(coef_test = bad(p = 1.2), .package = "clubSandwich")
  expect_undefined(cr2_satterthwaite(D1, G1), "nonfinite_result")
  testthat::local_mocked_bindings(coef_test = bad(se = 0), .package = "clubSandwich")
  expect_undefined(cr2_satterthwaite(D1, G1), "nonfinite_result")
})

test_that("a package result that disagrees with the mean, or lacks a field, is a defect", {
  testthat::local_mocked_bindings(
    coef_test = function(...) data.frame(Coef = "(Intercept)", beta = 0.2001, SE = 0.25, null_value = 0,
                                         tstat = 0.8, df_Satt = 2.8, p_Satt = 0.48),
    .package = "clubSandwich")
  expect_error(cr2_satterthwaite(D1, G1), "intercept")
  testthat::local_mocked_bindings(
    coef_test = function(...) data.frame(Coef = "(Intercept)", beta = 0.2, SE = 0.25),
    .package = "clubSandwich")
  expect_error(cr2_satterthwaite(D1, G1), "df_Satt")
})

test_that("the estimator is CR2 with Satterthwaite, not CR0/CR1 or a z test", {
  frame <- data.frame(y = D1, g = factor(G1))
  fit <- stats::lm(y ~ 1, data = frame)
  other <- function(v) clubSandwich::coef_test(fit, vcov = v, cluster = frame$g, test = "Satterthwaite")$SE
  r <- cr2_satterthwaite(D1, G1)
  for (v in c("CR0", "CR1", "CR3")) expect_gt(abs(r$se - other(v)), 1e-4)
  expect_gt(abs(r$p - 2 * stats::pnorm(-abs(r$t))), 1e-3)
})
