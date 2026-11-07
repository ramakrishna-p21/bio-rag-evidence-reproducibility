# Complete-case reconciliation with the Python handoff. Expectations are hand-derived from the
# construction in tests/test_r_input_fixture.py (JATS cell, PDF_DOCLING cell per item):
#   q_001 P001 (yes,yes)(no,no)    q_002 P002 (yes,yes)(yes,yes)   q_003 P002 (yes,no)(yes,yes)
#   q_004 P003 (yes,yes)(yes,missing)  q_005 P003 (non_eval,yes)(yes,yes)
#   q_006 P003 (no,no)(yes,yes)    q_007 P004 (yes,yes)(no,yes)
#   q_008 P005 (yes,missing)(yes,missing)  q_009 P005 (yes,yes)(non_eval,no)
# where each cell is (evidence_delivered, answer_correct).

test_that("R reproduces Python's complete cases for each outcome", {
  read <- fixture()
  acc <- derive_accounting(read$rows, read$handoff)
  expect_identical(names(acc), c("evidence_delivered", "answer_correct"))
  d <- acc$evidence_delivered
  expect_identical(d$complete_item_ids, c("q_001", "q_002", "q_003", "q_004", "q_006", "q_007", "q_008"))
  expect_identical(nrow(d$items), 9L)                       # all included items are eligible
  expect_identical(d$state_counts, c(
    "JATS|no" = 1L, "JATS|non_evaluable" = 1L, "JATS|yes" = 7L,
    "PDF_DOCLING|no" = 2L, "PDF_DOCLING|non_evaluable" = 1L, "PDF_DOCLING|yes" = 6L
  ))
  expect_identical(d$papers$paper_id, c("P001", "P002", "P003", "P004", "P005"))
  expect_identical(d$papers$eligible, c(1L, 2L, 3L, 1L, 2L))
  expect_identical(d$papers$complete, c(1L, 2L, 2L, 1L, 1L))
  a <- acc$answer_correct
  expect_identical(a$complete_item_ids, c("q_001", "q_002", "q_003", "q_005", "q_006", "q_007", "q_009"))
  expect_identical(a$state_counts, c(
    "JATS|missing" = 1L, "JATS|no" = 2L, "JATS|yes" = 6L,
    "PDF_DOCLING|missing" = 2L, "PDF_DOCLING|no" = 2L, "PDF_DOCLING|yes" = 5L
  ))
  expect_identical(a$papers$eligible, c(1L, 2L, 3L, 1L, 2L))
  expect_identical(a$papers$complete, c(1L, 2L, 2L, 1L, 1L))
  # The two outcomes keep different complete-case sets; missingness is outcome-specific.
  expect_false(identical(d$complete_item_ids, a$complete_item_ids))
})

test_that("an operationally failed generation stays missing and keeps its status and reason", {
  read <- fixture()
  a <- derive_accounting(read$rows, read$handoff)$answer_correct$items
  q4 <- a[a$item_id == "q_004", ]
  expect_identical(c(q4$jats_state, q4$pdf_state), c("yes", "missing"))
  expect_identical(q4$pdf_status, "operational_failure")
  expect_identical(q4$pdf_reason, "operational_failure")
  expect_false(q4$complete)
  expect_true(is.na(q4$y_pdf))
})

test_that("repeats 2-3, robustness arms and closed-book rows never enter primary accounting", {
  read <- fixture()
  base <- derive_accounting(read$rows, read$handoff)
  rows <- read$rows
  template <- rows[rows$item_id == "q_001" & rows$condition == "JATS", ]
  repeat2 <- transform(template, repeat_index = 2L, primary_row = FALSE, observation_key = "obs_r2",
                       answer_correct = "no", evidence_delivered = "no")
  dense <- transform(template, matrix_arm = "dense_robustness", observation_key = "obs_dense",
                     answer_correct = "no", evidence_delivered = "no")
  closed <- transform(template, matrix_arm = "closed_book", condition = "CLOSED_BOOK",
                      observation_key = "obs_cb", evidence_delivered = "not_applicable")
  extended <- rbind(rows, repeat2, dense, closed)
  expect_identical(derive_accounting(extended, read$handoff), base)
})

test_that("a confirmatory primary cell with not_applicable delivery is a contract error", {
  read <- fixture()
  rows <- read$rows
  rows$evidence_delivered[rows$item_id == "q_001" & rows$condition == "JATS"] <- "not_applicable"
  expect_input_error(derive_accounting(rows, read$handoff), "contract")
})

test_that("rows outside, or missing from, the declared eligible set are refused", {
  read <- fixture()
  rows <- read$rows
  stray <- rows[rows$item_id == "q_009", ]
  stray$item_id <- "q_011"
  stray$observation_key <- paste0(stray$observation_key, "x")
  expect_input_error(derive_accounting(rbind(rows, stray), read$handoff), "unexpected_rows")
  # A representation never planned for an eligible item is a roster/contract error.
  no_pdf <- rows[!(rows$item_id == "q_002" & rows$condition == "PDF_DOCLING"), ]
  expect_input_error(derive_accounting(no_pdf, read$handoff), "roster_contract")
  h <- read$handoff
  h$accounting[[1]]$eligible_items[[1]]["pdf_docling"] <- list(NULL)
  expect_input_error(derive_accounting(rows, h), "roster_contract")
  # The paper of an item must be the paper the handoff declares.
  moved <- rows
  moved$paper_id[moved$item_id == "q_001"] <- "P009"
  expect_input_error(derive_accounting(moved, read$handoff), "accounting")
})

test_that("any disagreement with Python's accounting stops the run", {
  edit <- function(old, new, n = 1) function(t) replace_nth(t, old, new, n)
  disagree <- function(handoff_edit) {
    read <- load_fixture(handoff_edit = handoff_edit)  # structurally valid handoff text
    derive_accounting(read$rows, read$handoff)
  }
  # complete_case flag of a first-outcome item, the complete-id list, counts and paper counts
  expect_input_error(disagree(edit('"complete_case": false', '"complete_case": true')), "accounting")
  expect_input_error(disagree(edit('"complete_case": true', '"complete_case": false')), "accounting")
  expect_input_error(disagree(edit('        "q_003",\n', '', 1)), "accounting")
  expect_input_error(disagree(edit('"JATS|yes": 7', '"JATS|yes": 6')), "accounting")
  expect_input_error(disagree(edit('"complete": 2,\n          "eligible": 3', '"complete": 3,\n          "eligible": 3')), "accounting")
  expect_input_error(disagree(edit('"state": "no"', '"state": "yes"')), "accounting")
  expect_input_error(disagree(edit('"execution_status": "completed"', '"execution_status": "operational_failure"')), "accounting")
  # observation key of a cell
  key <- fixture()$rows$observation_key[1]
  expect_input_error(disagree(edit(paste0('"', key, '"'), '"obs_other"')), "accounting")
})

# An outcome with no paired complete case. Python publishes `complete_item_ids: []`, which R reads
# as list(); a plain unlist() turned that into NULL and the comparison with character(0) failed.
# Every PDF_DOCLING cell of the chosen outcome becomes non_evaluable, in the rows and in the handoff alike.
no_complete <- function(read, outcome = "evidence_delivered") {
  rows <- read$rows
  pdf <- rows$matrix_arm == "confirmatory" & rows$repeat_index == 1L & rows$condition == "PDF_DOCLING"
  rows[[outcome]][pdf] <- "non_evaluable"
  handoff <- read$handoff
  for (k in seq_along(handoff$accounting)) {
    if (!identical(handoff$accounting[[k]]$outcome, outcome)) next
    a <- handoff$accounting[[k]]
    for (i in seq_along(a$eligible_items)) {
      a$eligible_items[[i]]$pdf_docling$state <- "non_evaluable"
      a$eligible_items[[i]]$complete_case <- FALSE
    }
    a$complete_item_ids <- list()
    a$state_counts <- a$state_counts[!startsWith(names(a$state_counts), "PDF_DOCLING|")]
    a$state_counts[["PDF_DOCLING|non_evaluable"]] <- length(a$eligible_items)
    for (j in seq_along(a$papers)) a$papers[[j]]$complete <- 0L
    handoff$accounting[[k]] <- a
  }
  list(rows = rows, handoff = handoff)
}

test_that("a valid empty complete-case list reconciles as an empty character vector", {
  z <- no_complete(fixture())
  acc <- derive_accounting(z$rows, z$handoff)
  expect_identical(acc$evidence_delivered$complete_item_ids, character(0))
  expect_identical(sum(acc$evidence_delivered$items$complete), 0L)
  expect_identical(acc$evidence_delivered$papers$complete, rep(0L, 5L))
  # The other outcome is unaffected.
  expect_identical(acc$answer_correct$complete_item_ids, derive_accounting(fixture()$rows, fixture()$handoff)$answer_correct$complete_item_ids)
})

test_that("empty-list handling does not let wrong complete-case ids through", {
  z <- no_complete(fixture())
  claim <- function(ids) {
    h <- z$handoff
    k <- which(vapply(h$accounting, function(a) identical(a$outcome, "evidence_delivered"), logical(1)))
    h$accounting[[k]]$complete_item_ids <- ids
    h
  }
  expect_input_error(derive_accounting(z$rows, claim(list("q_001"))), "accounting")
  expect_input_error(derive_accounting(z$rows, claim(list(1L))), "accounting")
  expect_input_error(derive_accounting(z$rows, claim(NULL)), "accounting")
  # A handoff that lists nothing while the rows hold a complete case is also refused.
  r <- fixture()
  h <- r$handoff
  k <- which(vapply(h$accounting, function(a) identical(a$outcome, "evidence_delivered"), logical(1)))
  h$accounting[[k]]$complete_item_ids <- list()
  expect_input_error(derive_accounting(r$rows, h), "accounting")
})

test_that("a zero-complete outcome reaches inference with its existing undefined statuses", {
  read <- fixture()
  z <- no_complete(read)
  acc <- derive_accounting(z$rows, z$handoff)
  res <- coprimary_inference(acc, z$handoff, 20260101L)
  ev <- res$outcomes$evidence_delivered
  expect_identical(ev$summary$status, "no_complete_pairs")
  expect_true(is.na(ev$summary$rd))
  expect_identical(ev$bootstrap$status, "no_complete_pairs")
  expect_identical(ev$test$status, "inference_undefined")
  expect_identical(ev$test$reason, "no_complete_pairs")
  expect_true(is.null(ev$test$p_value) || is.na(ev$test$p_value))
  expect_identical(res$holm$status, "holm_undefined")
  expect_true(all(is.na(res$holm$adjusted)))
  # The other outcome is estimated exactly as before.
  base <- coprimary_inference(derive_accounting(read$rows, read$handoff), read$handoff, 20260101L)
  expect_identical(res$outcomes$answer_correct$summary, base$outcomes$answer_correct$summary)
})

test_that("either outcome alone can have no complete case", {
  read <- fixture()
  z <- no_complete(read, "answer_correct")
  acc <- derive_accounting(z$rows, z$handoff)
  expect_identical(acc$answer_correct$complete_item_ids, character(0))
  expect_identical(acc$evidence_delivered$complete_item_ids,
                   derive_accounting(read$rows, read$handoff)$evidence_delivered$complete_item_ids)
  res <- coprimary_inference(acc, z$handoff, 20260101L)
  expect_identical(res$outcomes$answer_correct$summary$status, "no_complete_pairs")
  expect_identical(res$holm$status, "holm_undefined")
  expect_false(is.na(res$outcomes$evidence_delivered$summary$rd))
})
