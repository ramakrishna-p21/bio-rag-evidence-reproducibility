# Secondary tables on a hand-built row table (helper-secondary.R). Expected values were derived
# by hand from the rows; no expectation is produced by the code under test.

rows <- syn_rows()
tabs <- compute_secondary_tables(rows, SYN_RULES)
get_row <- function(df, ...) {
  keep <- rep(TRUE, nrow(df))
  args <- list(...)
  for (nm in names(args)) keep <- keep & df[[nm]] == args[[nm]]
  df[keep, , drop = FALSE]
}

test_that("every table has its schema, unique rows and a population entry", {
  expect_identical(names(tabs), names(SECONDARY_SCHEMAS))
  for (nm in names(tabs)) {
    expect_identical(names(tabs[[nm]]), names(SECONDARY_SCHEMAS[[nm]]))
    expect_false(anyDuplicated(tabs[[nm]]) > 0)
    expect_type(csv_bytes(tabs[[nm]], SECONDARY_SCHEMAS[[nm]]), "raw")
  }
  expect_setequal(tabs[["secondary_populations.csv"]]$table, names(SECONDARY_SCHEMAS))
})

test_that("Stage A counts each primary item once per representation", {
  a <- tabs[["secondary_stage_a.csv"]]
  expect_identical(n_of(a, condition = "JATS", stage_a_label = "preserved"), 5L)
  expect_identical(n_of(a, condition = "JATS", stage_a_label = NO_STAGE_A), 1L)
  expect_identical(n_of(a, condition = "PDF_DOCLING", stage_a_label = "preserved"), 4L)
  expect_identical(n_of(a, condition = "PDF_DOCLING", stage_a_label = "absent"), 1L)
  expect_identical(n_of(a, condition = "PDF_DOCLING", stage_a_label = "altered_but_sufficient"), 1L)
  expect_true(all(a$denominator == 6L))
  # Repeats and other arms are not extra Stage A judgments: the table is identical without them.
  only_primary <- rows[rows$matrix_arm == "confirmatory" & rows$repeat_index == 1L, ]
  expect_identical(secondary_stage_a(only_primary), tabs[["secondary_stage_a.csv"]])
})

test_that("Stage B and earliest failure stay separate states and exclude closed book", {
  b <- tabs[["secondary_stage_b_failure.csv"]]
  expect_false("closed_book" %in% b$matrix_arm)
  s <- get_row(b, matrix_arm = "confirmatory", condition = "JATS", measure = "stage_b_status")
  expect_identical(setNames(s$n, s$value), c(delivered = 4L, no_stage_b_record = 1L, not_delivered = 1L))
  p <- get_row(b, matrix_arm = "confirmatory", condition = "PDF_DOCLING", measure = "stage_b_status")
  expect_identical(setNames(p$n, p$value), c(delivered = 4L, not_delivered = 1L, not_reachable_stage_a = 1L))
  e <- get_row(b, matrix_arm = "confirmatory", condition = "JATS", measure = "earliest_failure")
  expect_identical(setNames(e$n, e$value), c(incomplete = 1L, none = 3L, stage_b = 1L, stage_c = 1L))
  expect_true(all(b$denominator[b$matrix_arm == "confirmatory" & b$condition == "JATS"] == 6L))
  expect_identical(secondary_stage_b_failure(rows[rows$repeat_index == 1L, ]), b)
})

test_that("robustness keeps pairing, the primary rule and explicit absence", {
  r <- tabs[["secondary_robustness.csv"]]
  expect_identical(nrow(r), 8L)
  ev <- get_row(r, matrix_arm = "confirmatory", outcome = "evidence_delivered")
  expect_identical(c(ev$n_items, ev$n_paired, ev$n_complete, ev$n_papers), c(6L, 6L, 5L, 2L))
  expect_equal(c(ev$rate_jats, ev$rate_pdf, ev$rd, ev$rd_pp, ev$equal_paper_rd), c(0.8, 0.6, 0.2, 20, 0.25))
  ac <- get_row(r, matrix_arm = "confirmatory", outcome = "answer_correct")
  expect_equal(c(ac$rate_jats, ac$rate_pdf, ac$rd, ac$equal_paper_rd), c(0.6, 0.6, 0, 0))
  expect_identical(ac$n_complete, 5L)
  dn <- get_row(r, matrix_arm = "dense_robustness", outcome = "evidence_delivered")
  expect_equal(c(dn$rate_jats, dn$rate_pdf, dn$rd, dn$equal_paper_rd), c(0.5, 1, -0.5, -0.5))
  expect_equal(get_row(r, matrix_arm = "dense_robustness", outcome = "answer_correct")$rd, 0.5)
  ch <- get_row(r, matrix_arm = "chunk_robustness", outcome = "answer_correct")
  expect_identical(c(ch$n_items, ch$n_paired, ch$n_complete), c(2L, 1L, 1L))
  expect_equal(ch$rd, -1)
  sc <- get_row(r, matrix_arm = "scope_robustness", outcome = "evidence_delivered")
  expect_equal(c(sc$rd, sc$n_complete), c(0, 1))
  expect_true(all(r$role[r$matrix_arm == "confirmatory"] == "reference_primary_arm"))
  gone <- compute_secondary_tables(rows[rows$matrix_arm != "dense_robustness", ], SYN_RULES)[["secondary_robustness.csv"]]
  d <- get_row(gone, matrix_arm = "dense_robustness")
  expect_true(all(d$status == "arm_absent") && all(d$n_items == 0L) && all(is.na(d$rd)))
})

test_that("a robustness cell with no complete pair is its own status, not a zero effect", {
  none <- rows[rows$matrix_arm == "dense_robustness", ]
  none$answer_correct <- "missing"
  r <- secondary_robustness(rbind(rows[rows$matrix_arm != "dense_robustness", ], none), SYN_RULES)
  d <- get_row(r, matrix_arm = "dense_robustness", outcome = "answer_correct")
  expect_identical(d$status, "no_complete_pairs")
  expect_true(is.na(d$rd) && d$n_complete == 0L && d$n_paired == 2L)
})

test_that("scope table cross-tabulates composition and delivery per representation", {
  s <- tabs[["secondary_scope.csv"]]
  expect_identical(nrow(s), 2L)
  expect_identical(get_row(s, condition = "JATS")$context_composition, "other_only")
  expect_identical(get_row(s, condition = "PDF_DOCLING")$context_composition, "mixed")
  expect_true(all(s$denominator == 1L))
})

test_that("repeat outcomes use the three prespecified repeats and keep closed book separate", {
  o <- tabs[["secondary_repeat_outcomes.csv"]]
  expect_setequal(paste(o$matrix_arm, o$condition), c("confirmatory JATS", "confirmatory PDF_DOCLING", "closed_book CLOSED_BOOK"))
  j1 <- get_row(o, matrix_arm = "confirmatory", condition = "JATS", repeat_index = 1L)
  expect_identical(setNames(j1$n, j1$answer_correct), c(missing = 1L, no = 2L, yes = 3L))
  expect_true(all(j1$denominator == 6L))
  j3 <- get_row(o, matrix_arm = "confirmatory", condition = "JATS", repeat_index = 3L)
  expect_identical(setNames(j3$n, j3$answer_correct), c(no = 2L))
  c3 <- get_row(o, matrix_arm = "closed_book", repeat_index = 3L)
  expect_identical(setNames(c3$n, c3$answer_correct), c(missing = 1L, no = 1L, yes = 1L))
  c2 <- get_row(o, matrix_arm = "closed_book", repeat_index = 2L)
  expect_identical(n_of(c2, answer_correct = "yes"), 1L)
  expect_identical(c2$denominator[1], 4L)
})

test_that("repeat variability counts correctness agreement and separates missing and unevaluable repeats", {
  v <- tabs[["secondary_repeat_variability.csv"]]
  jats <- get_row(v, matrix_arm = "confirmatory", condition = "JATS")
  expect_identical(unlist(jats[c("n_items", "n_all_repeats_evaluable", "n_correct_0", "n_correct_1", "n_correct_2",
                                 "n_correct_3", "n_missing_repeat_rows", "n_with_unevaluable_repeat")], use.names = FALSE),
                   c(6L, 2L, 1L, 0L, 1L, 0L, 4L, 0L))
  pdf <- get_row(v, matrix_arm = "confirmatory", condition = "PDF_DOCLING")
  expect_identical(c(pdf$n_all_repeats_evaluable, pdf$n_missing_repeat_rows), c(0L, 6L))
  cb <- get_row(v, matrix_arm = "closed_book")
  expect_identical(unlist(cb[c("n_items", "n_all_repeats_evaluable", "n_correct_0", "n_correct_3",
                               "n_missing_repeat_rows", "n_with_unevaluable_repeat")], use.names = FALSE),
                   c(4L, 2L, 1L, 1L, 1L, 1L))
  parts <- v[c("n_all_repeats_evaluable", "n_missing_repeat_rows", "n_with_unevaluable_repeat")]
  expect_equal(unname(rowSums(parts)), v$n_items)
  expect_equal(unname(rowSums(v[c("n_correct_0", "n_correct_1", "n_correct_2", "n_correct_3")])), v$n_all_repeats_evaluable)
})

test_that("discordance lists every state pair and is not inflated by repeats or arms", {
  d <- tabs[["secondary_discordance.csv"]]
  ev <- d[d$outcome == "evidence_delivered", ]
  expect_identical(as.list(tapply(ev$n, ev$pair_class, sum)), list(both_failure = 1L, both_success = 3L, jats_only_success = 1L, not_complete = 1L))
  ac <- d[d$outcome == "answer_correct", ]
  cls <- tapply(ac$n, ac$pair_class, sum)
  expect_identical(as.list(cls), list(both_success = 1L, jats_only_success = 2L, not_complete = 1L, pdf_only_success = 2L))
  expect_true(all(d$denominator == 6L))
  expect_identical(n_of(d, outcome = "answer_correct", jats_state = "missing", pdf_state = "yes"), 1L)
})

test_that("correct_without_evidence keeps the tri-state NA as its own value", {
  x <- tabs[["secondary_correct_without_evidence.csv"]]
  expect_identical(n_of(x, condition = "JATS", correct_without_evidence = NOT_DETERMINED), 1L)
  expect_identical(n_of(x, condition = "PDF_DOCLING", correct_without_evidence = "TRUE"), 1L)
  expect_false("NA" %in% x$correct_without_evidence)
})

test_that("completion separates operational failure, abstention and scored answers", {
  cmp <- tabs[["secondary_completion.csv"]]
  expect_identical(n_of(cmp, condition = "JATS", execution_status = "operational_failure", completion_kind = NO_COMPLETION, answer_correct = "missing"), 1L)
  expect_identical(n_of(cmp, condition = "PDF_DOCLING", completion_kind = "abstained", answer_correct = "no"), 1L)
  expect_identical(n_of(cmp, condition = "PDF_DOCLING", completion_kind = "answer", answer_correct = "yes"), 4L)
  expect_true(all(cmp$denominator == 6L))
})

test_that("answer class reports each evaluated population with the other states beside it", {
  a <- tabs[["secondary_answer_class.csv"]]
  j <- get_row(a, condition = "JATS", outcome = "answer_correct")
  expect_identical(setNames(j$n, j$state), c(missing = 1L, no = 2L, yes = 3L))
  expect_true(all(j$denominator == 6L))
  e <- get_row(a, condition = "JATS", outcome = "evidence_delivered")
  expect_identical(n_of(e, state = "non_evaluable"), 1L)
})

test_that("unavailable analyses are explicit and answer disagreement is not claimed", {
  u <- tabs[["secondary_unavailable.csv"]]
  expect_setequal(u$analysis, c("answer_identity_disagreement", "identical_incorrect_answer_stability", "table_vs_prose", "evidence_location"))
  expect_true(all(u$status == "deferred_data_not_in_analysis_table"))
  expect_false(any(grepl("answer_identity|stable_incorrect", names(tabs[["secondary_discordance.csv"]]))))
})

test_that("populations count the source rows and expose an absent arm", {
  p <- tabs[["secondary_populations.csv"]]
  expect_identical(p$n_rows[p$table == "secondary_stage_a.csv"], 12L)
  expect_identical(p$n_rows[p$table == "secondary_scope.csv"], 2L)
  none <- compute_secondary_tables(rows[rows$matrix_arm != "scope_robustness", ], SYN_RULES)
  expect_identical(none[["secondary_populations.csv"]]$n_rows[none[["secondary_populations.csv"]]$table == "secondary_scope.csv"], 0L)
  expect_identical(nrow(none[["secondary_scope.csv"]]), 0L)
})

test_that("results do not depend on row order", {
  shuffled <- rows[c(rev(seq_len(nrow(rows)))), ]
  expect_identical(compute_secondary_tables(shuffled, SYN_RULES), tabs)
})

test_that("duplicate observations and missing columns are refused", {
  expect_input_error(compute_secondary_tables(rbind(rows, rows[1, ]), SYN_RULES), "secondary_input")
  expect_input_error(compute_secondary_tables(rows[setdiff(names(rows), "stage_a_label")], SYN_RULES), "secondary_input")
})

test_that("an input with no rows yields typed empty tables, not errors", {
  t0 <- compute_secondary_tables(rows[0, ], SYN_RULES)
  expect_identical(names(t0), names(SECONDARY_SCHEMAS))
  for (nm in setdiff(names(t0), c("secondary_robustness.csv", "secondary_unavailable.csv", "secondary_populations.csv"))) {
    expect_identical(nrow(t0[[nm]]), 0L)
  }
  expect_true(all(t0[["secondary_robustness.csv"]]$status == "arm_absent"))
})
