# Both co-primary outcomes through the frozen synthetic fixture (see test-accounting.R for the
# per-item construction). Hand-derived paired differences JATS - PDF_DOCLING, complete cases only:
#   evidence_delivered: q_001 P001 +1, q_002 P002 0, q_003 P002 0, q_004 P003 0, q_006 P003 -1,
#                       q_007 P004 +1, q_008 P005 0           sum 1, n 7, rd 1/7, 5 papers
#   answer_correct:     q_001 P001 +1, q_002 P002 0, q_003 P002 -1, q_005 P003 0, q_006 P003 -1,
#                       q_007 P004 0, q_009 P005 +1           sum 0, n 7, rd 0,   5 papers
# Test-only seed; the production `primary_bootstrap` seed is not approved.

SEED <- 20260101L
EV_D <- c(1, 0, 0, 0, -1, 1, 0);  EV_G <- c("P001", "P002", "P002", "P003", "P003", "P004", "P005")
AN_D <- c(1, 0, -1, 0, -1, 0, 1); AN_G <- c("P001", "P002", "P002", "P003", "P003", "P004", "P005")

run_fixture <- function(rows = NULL, seed = SEED) {
  read <- fixture()
  acc <- derive_accounting(if (is.null(rows)) read$rows else rows, read$handoff)
  coprimary_inference(acc, read$handoff, seed)
}

test_that("fixture counts and estimates match the hand derivation", {
  r <- run_fixture()
  ev <- r$outcomes$evidence_delivered; an <- r$outcomes$answer_correct
  expect_identical(c(ev$summary$n_items, ev$summary$n_papers), c(7L, 5L))
  expect_identical(c(an$summary$n_items, an$summary$n_papers), c(7L, 5L))
  expect_equal(ev$summary$rd, 1 / 7)
  expect_equal(an$summary$rd, 0)
  expect_equal(ev$test$estimate, 1 / 7); expect_equal(an$test$estimate, 0)
  expect_equal(ev$summary$equal_paper_rd, mean(c(1, 0, -0.5, 1, 0)))   # paper means, sensitivity only
})

test_that("CR2 results equal the independent references on the fixture", {
  r <- run_fixture()
  ev <- r$outcomes$evidence_delivered$test; an <- r$outcomes$answer_correct$test
  expect_equal(fields(ev), ref_matrix(EV_D, EV_G), tolerance = 1e-10)
  expect_equal(fields(an), ref_matrix(AN_D, AN_G), tolerance = 1e-10)
  # answer_correct by hand: S = (1,-1,-1,0,1), n_g = (1,2,2,1,1), mean 0, n = 7
  #   SE^2 = (1 + 0 + 1)/(49 (6/7)) + (1 + 1)/(49 (5/7)) = 77/735
  expect_equal(an$se, sqrt(77 / 735), tolerance = 1e-12)
  expect_equal(an$t, 0)
  expect_equal(an$p, 1)
})

test_that("bootstrap per outcome re-initialises the same seed on its own frame", {
  r <- run_fixture()
  boot_by_hand <- function(d, g) {
    papers <- sort(unique(g), method = "radix")
    n <- vapply(papers, function(p) sum(g == p), 0); s <- vapply(papers, function(p) sum(d[g == p]), 0)
    set.seed(SEED, kind = "Mersenne-Twister", normal.kind = "Inversion", sample.kind = "Rejection")
    m <- matrix(sample.int(5L, 10000L * 5L, replace = TRUE), nrow = 10000, byrow = TRUE)
    est <- vapply(seq_len(10000), function(i) sum(s[m[i, ]]) / sum(n[m[i, ]]), 0)
    unname(stats::quantile(est, c(0.025, 0.975), type = 7))
  }
  for (o in c("evidence_delivered", "answer_correct")) {
    b <- r$outcomes[[o]]$bootstrap
    d <- if (o == "evidence_delivered") EV_D else AN_D
    g <- if (o == "evidence_delivered") EV_G else AN_G
    expect_identical(b$n_resamples, 10000L)
    expect_identical(b$n_empty, 0L)
    expect_identical(b$status, "ok")
    expect_equal(c(b$ci_lower, b$ci_upper), boot_by_hand(d, g), tolerance = 1e-12)
  }
})

test_that("Holm is over the two co-primary p-values in family order", {
  r <- run_fixture()
  p <- c(r$outcomes$evidence_delivered$test$p, r$outcomes$answer_correct$test$p)
  expect_identical(r$holm$status, "ok")
  expect_identical(names(r$holm$adjusted), c("evidence_delivered", "answer_correct"))
  lo <- min(p); hi <- max(p)                       # Holm: min(2*p_min, 1) then max with previous
  adj_lo <- min(1, 2 * lo); adj_hi <- max(adj_lo, min(1, hi))
  expect_equal(unname(r$holm$adjusted[which.min(p)]), adj_lo)
  expect_equal(unname(r$holm$adjusted[which.max(p)]), adj_hi)
})

test_that("seed, RNG and package provenance are recorded", {
  r <- run_fixture()
  expect_identical(r$seed, SEED)
  expect_identical(r$seed_name, "primary_bootstrap")
  expect_identical(r$rng, BOOTSTRAP_RNG)
  expect_identical(r$r_version, R.version.string)
  expect_identical(r$package_version, as.character(utils::packageVersion("clubSandwich")))
})

test_that("only confirmatory repeat-1 JATS/PDF_DOCLING rows reach the inference", {
  read <- fixture()
  base <- run_fixture()
  extra <- read$rows
  later <- extra; later$repeat_index <- 2L; later$evidence_delivered <- "no"; later$answer_correct <- "no"
  robust <- extra; robust$matrix_arm <- "dense_robustness"; robust$evidence_delivered <- "no"
  expect_identical(run_fixture(rbind(read$rows, later, robust)), base)
  # editing a repeat-1 confirmatory cell does change the answer (JATS q_002 evidence yes -> no)
  edited <- read$rows
  hit <- edited$item_id == "q_002" & edited$condition == "JATS"
  edited$evidence_delivered[hit] <- "no"
  edited$evidence_reason[hit] <- NA_character_
  expect_input_error(run_fixture(edited), "accounting")   # no longer matches Python's cell: refused, not re-judged
})

test_that("the outcome frames follow each outcome's complete cases", {
  r <- run_fixture()
  # P003 has an incomplete evidence pair for q_005 but it still has q_004 and q_006
  expect_identical(r$outcomes$evidence_delivered$bootstrap$n_papers_frame, 5L)
  # drop a paper from the answer outcome only
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  acc$answer_correct$items$complete[acc$answer_correct$items$paper_id == "P004"] <- FALSE
  r2 <- coprimary_inference(acc, read$handoff, SEED)
  expect_identical(c(r2$outcomes$evidence_delivered$bootstrap$n_papers_frame,
                     r2$outcomes$answer_correct$bootstrap$n_papers_frame), c(5L, 4L))
})

test_that("an undefined p-value makes the Holm family undefined; raw p-values stay visible", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  keep <- acc$answer_correct$items$paper_id == "P001"
  acc$answer_correct$items$complete <- keep & acc$answer_correct$items$complete   # one paper only
  r <- coprimary_inference(acc, read$handoff, SEED)
  expect_identical(r$outcomes$answer_correct$test$reason, "single_cluster")
  expect_identical(r$outcomes$answer_correct$bootstrap$status, "single_cluster_degenerate")
  expect_identical(r$holm$status, "holm_undefined")
  expect_true(all(is.na(r$holm$adjusted)))
  expect_identical(r$outcomes$evidence_delivered$test$status, "ok")
  expect_true(is.finite(r$outcomes$evidence_delivered$test$p))
})

test_that("coprimary_inference refuses a family that is not exactly the two outcomes", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  expect_input_error(coprimary_inference(acc[1], read$handoff, SEED), "inference_input")
  expect_input_error(coprimary_inference(rev(acc), read$handoff, SEED), "inference_input")
})

test_that("the formal result does not depend on the order of the complete items", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  base <- coprimary_inference(acc, read$handoff, SEED)
  shuffled <- acc
  for (o in names(shuffled)) {
    it <- shuffled[[o]]$items
    shuffled[[o]]$items <- it[rev(seq_len(nrow(it))), , drop = FALSE]
  }
  again <- coprimary_inference(shuffled, read$handoff, SEED)
  for (o in names(base$outcomes)) {
    expect_identical(again$outcomes[[o]]$bootstrap, base$outcomes[[o]]$bootstrap)   # integer sums: exact
    expect_equal(fields(again$outcomes[[o]]$test), fields(base$outcomes[[o]]$test), tolerance = 1e-12)
  }
})

test_that("Holm adjusted values are attached to the right outcome (distinct p-values)", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  mk <- function(y_jats, y_pdf, paper) data.frame(item_id = paste0("i", seq_along(paper)), paper_id = paper,
                                                  complete = TRUE, y_jats = y_jats, y_pdf = y_pdf,
                                                  stringsAsFactors = FALSE)
  paper <- rep(c("P1", "P2", "P3", "P4", "P5"), each = 2)
  # evidence: JATS better in almost every item (small p); answer: mixed (large p)
  acc$evidence_delivered$items <- mk(rep(1, 10), c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0), paper)
  acc$answer_correct$items <- mk(c(1, 0, 1, 0, 0, 1, 1, 0, 1, 0), c(0, 0, 1, 1, 1, 0, 1, 1, 0, 0), paper)
  r <- coprimary_inference(acc, read$handoff, SEED)
  p_ev <- ref_matrix(rep(1, 10) - c(0, 0, 0, 0, 0, 0, 0, 0, 1, 0), paper)[["p"]]
  p_an <- ref_matrix(c(1, 0, 1, 0, 0, 1, 1, 0, 1, 0) - c(0, 0, 1, 1, 1, 0, 1, 1, 0, 0), paper)[["p"]]
  expect_lt(p_ev, 0.05); expect_gt(p_an, p_ev)
  expect_equal(r$outcomes$evidence_delivered$test$p, p_ev, tolerance = 1e-9)
  expect_equal(r$outcomes$answer_correct$test$p, p_an, tolerance = 1e-9)
  expected <- stats::p.adjust(c(p_ev, p_an), "holm")
  expect_gt(abs(expected[1] - expected[2]), 1e-3)           # the two adjusted values differ
  expect_equal(unname(r$holm$adjusted), expected, tolerance = 1e-9)
  expect_identical(names(r$holm$adjusted), c("evidence_delivered", "answer_correct"))
})

test_that("a handoff whose family is not the two co-primary outcomes is refused", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  h <- read$handoff
  h$analysis_spec$multiplicity_family <- rev(h$analysis_spec$multiplicity_family)
  expect_input_error(coprimary_inference(acc, h, SEED), "inference_input")
})
