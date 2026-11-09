# A hand-built row table for the secondary tables. Each helper call is one table row.
SYN_RULES <- list(
  evidence_delivered = list(column = "evidence_delivered", success = "yes", failure = "no",
                            non_evaluable = c("non_evaluable", "not_applicable")),
  answer_correct = list(column = "answer_correct", success = "yes", failure = "no",
                        non_evaluable = c("non_evaluable", "missing", "not_executed"))
)

syn_row <- function(arm, cond, item, paper, rep = 1L, delivered = "yes", correct = "yes", a = "preserved",
                    b = "delivered", comp = "designated_only", earliest = "none", cwe = "FALSE",
                    ck = "answer", exec = "completed", cls = "quantitative_value") {
  data.frame(matrix_arm = arm, condition = cond, item_id = item, paper_id = paper, repeat_index = as.integer(rep),
             evidence_delivered = delivered, answer_correct = correct, stage_a_label = a, stage_b_status = b,
             context_composition = comp, earliest_failure = earliest, correct_without_evidence = cwe,
             completion_kind = ck, execution_status = exec, answer_class = cls, stringsAsFactors = FALSE)
}

# Six items in three papers; see test-secondary.R for the hand-derived expectations.
syn_rows <- function() {
  r <- function(...) syn_row(...)
  rows <- rbind(
    r("confirmatory", "JATS", "a", "P1", delivered = "yes", correct = "yes"),
    r("confirmatory", "PDF_DOCLING", "a", "P1", delivered = "no", correct = "no", a = "absent",
      b = "not_reachable_stage_a", earliest = "stage_a"),
    r("confirmatory", "JATS", "b", "P1", delivered = "yes", correct = "no", earliest = "stage_c"),
    r("confirmatory", "PDF_DOCLING", "b", "P1", delivered = "yes", correct = "yes"),
    r("confirmatory", "JATS", "c", "P2", delivered = "non_evaluable", correct = "missing", a = NA_character_,
      b = NA_character_, comp = NA_character_, earliest = "incomplete", cwe = "NA", ck = NA_character_,
      exec = "operational_failure"),
    r("confirmatory", "PDF_DOCLING", "c", "P2", delivered = "yes", correct = "yes"),
    r("confirmatory", "JATS", "d", "P3", delivered = "no", correct = "no", b = "not_delivered", earliest = "stage_b"),
    r("confirmatory", "PDF_DOCLING", "d", "P3", delivered = "no", correct = "yes", a = "altered_but_sufficient",
      b = "not_delivered", earliest = "stage_b", cwe = "TRUE"),
    r("confirmatory", "JATS", "e", "P3", delivered = "yes", correct = "yes"),
    r("confirmatory", "PDF_DOCLING", "e", "P3", delivered = "yes", correct = "no", ck = "abstained", earliest = "stage_c"),
    r("confirmatory", "JATS", "f", "P3", delivered = "yes", correct = "yes"),
    r("confirmatory", "PDF_DOCLING", "f", "P3", delivered = "yes", correct = "yes"),
    # confirmatory repeats 2 and 3 for items a and b, JATS only; correctness differs by repeat
    r("confirmatory", "JATS", "a", "P1", rep = 2, correct = "yes"), r("confirmatory", "JATS", "a", "P1", rep = 3, correct = "no"),
    r("confirmatory", "JATS", "b", "P1", rep = 2, correct = "no"), r("confirmatory", "JATS", "b", "P1", rep = 3, correct = "no"),
    # dense robustness: items a, b
    r("dense_robustness", "JATS", "a", "P1", delivered = "yes", correct = "yes"),
    r("dense_robustness", "PDF_DOCLING", "a", "P1", delivered = "yes", correct = "yes"),
    r("dense_robustness", "JATS", "b", "P1", delivered = "no", correct = "yes"),
    r("dense_robustness", "PDF_DOCLING", "b", "P1", delivered = "yes", correct = "no"),
    # chunk robustness: item a both, item b JATS only
    r("chunk_robustness", "JATS", "a", "P1", delivered = "yes", correct = "no"),
    r("chunk_robustness", "PDF_DOCLING", "a", "P1", delivered = "yes", correct = "yes"),
    r("chunk_robustness", "JATS", "b", "P1", delivered = "yes", correct = "yes"),
    # scope robustness: item a
    r("scope_robustness", "JATS", "a", "P1", delivered = "no", correct = "no", b = "not_delivered", comp = "other_only", earliest = "stage_b"),
    r("scope_robustness", "PDF_DOCLING", "a", "P1", delivered = "no", correct = "no", b = "not_delivered", comp = "mixed", earliest = "stage_b"),
    # closed book: a, b, c with three repeats; d with two
    r("closed_book", "CLOSED_BOOK", "a", "P1", rep = 1, delivered = "not_applicable", correct = "yes", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "a", "P1", rep = 2, delivered = "not_applicable", correct = "yes", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "a", "P1", rep = 3, delivered = "not_applicable", correct = "yes", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "b", "P1", rep = 1, delivered = "not_applicable", correct = "no", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "b", "P1", rep = 2, delivered = "not_applicable", correct = "no", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "b", "P1", rep = 3, delivered = "not_applicable", correct = "no", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "c", "P2", rep = 1, delivered = "not_applicable", correct = "yes", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "c", "P2", rep = 2, delivered = "not_applicable", correct = "no", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "c", "P2", rep = 3, delivered = "not_applicable", correct = "missing", a = NA_character_, b = NA_character_, comp = NA_character_, exec = "operational_failure", ck = NA_character_),
    r("closed_book", "CLOSED_BOOK", "d", "P3", rep = 1, delivered = "not_applicable", correct = "yes", a = NA_character_, b = NA_character_, comp = NA_character_),
    r("closed_book", "CLOSED_BOOK", "d", "P3", rep = 2, delivered = "not_applicable", correct = "no", a = NA_character_, b = NA_character_, comp = NA_character_)
  )
  rownames(rows) <- NULL
  rows
}

n_of <- function(df, ...) {
  keep <- rep(TRUE, nrow(df))
  for (nm in names(list(...))) keep <- keep & df[[nm]] == list(...)[[nm]]
  sum(df$n[keep])
}
