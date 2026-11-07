# Reproduce Python's primary complete-case accounting from the table rows and require it to
# equal the handoff's. R does not decide evaluability: the only states that count as a complete
# case are the handoff's declared success and failure values, and any disagreement stops the run.

REPRESENTATIONS <- c("JATS", "PDF_DOCLING")
# The reason column that accompanies each outcome column (same pairing as Python's accounting).
REASON_COLUMN <- c(evidence_delivered = "evidence_reason", answer_correct = "stage_c_reason")

outcome_rules <- function(handoff) {
  rules <- lapply(handoff$analysis_spec$outcomes, function(o) {
    list(column = o$column, success = o$success, failure = o$failure,
         non_evaluable = unlist(o$non_evaluable))
  })
  names(rules) <- vapply(handoff$analysis_spec$outcomes, `[[`, "", "name")
  rules
}

as_state_counts <- function(x) {
  counts <- unlist(x)
  storage.mode(counts) <- "integer"
  counts[order(names(counts), method = "radix")]
}

# The handoff's complete-case ids as a character vector. A JSON [] is list() and unlist() would turn
# it into NULL, so an empty list is mapped to character(0); a missing field, a non-list or
# non-string ids are returned unchanged and so fail the comparison.
stated_ids <- function(x) {
  if (is.list(x) && length(x) == 0) character(0) else unname(unlist(x))
}

cell_matches <- function(cell, key, state, status, reason) {
  identical(cell$observation_key, key) && identical(cell$state, state) &&
    identical(cell$execution_status, status) &&
    identical(if (is.null(cell$reason)) NA_character_ else cell$reason, reason)
}

derive_accounting <- function(rows, handoff) {
  rules <- outcome_rules(handoff)
  primary <- rows[rows$matrix_arm == "confirmatory" & rows$repeat_index == 1L &
                    rows$condition %in% REPRESENTATIONS, , drop = FALSE]
  result <- list()
  for (outcome in names(rules)) {
    rule <- rules[[outcome]]
    stated <- Filter(function(a) identical(a$outcome, outcome), handoff$accounting)
    if (length(stated) != 1) input_error("accounting", sprintf("handoff has no single %s accounting", outcome))
    stated <- stated[[1]]
    eligible <- stated$eligible_items
    ids <- vapply(eligible, `[[`, "", "item_id")
    if (anyDuplicated(ids)) input_error("accounting", "duplicate eligible item")
    for (e in eligible) {
      if (is.null(e$jats) || is.null(e$pdf_docling)) {
        input_error("roster_contract", sprintf(
          "eligible item %s has a representation that was never planned", e$item_id))
      }
    }
    stray <- setdiff(unique(primary$item_id), ids)
    if (length(stray)) {
      input_error("unexpected_rows", sprintf("primary rows for items outside the eligible set: %s",
                                             paste(utils::head(sort(stray), 3), collapse = ", ")))
    }
    items <- data.frame(
      item_id = ids, paper_id = vapply(eligible, `[[`, "", "paper_id"),
      jats_state = NA_character_, pdf_state = NA_character_,
      jats_reason = NA_character_, pdf_reason = NA_character_,
      jats_status = NA_character_, pdf_status = NA_character_,
      stringsAsFactors = FALSE
    )
    for (i in seq_along(eligible)) {
      for (condition in REPRESENTATIONS) {
        hit <- primary[primary$item_id == ids[i] & primary$condition == condition, , drop = FALSE]
        if (nrow(hit) != 1) {
          input_error("roster_contract", sprintf("eligible item %s has %d primary rows for %s",
                                                 ids[i], nrow(hit), condition))
        }
        if (hit$paper_id != items$paper_id[i]) {
          input_error("accounting", sprintf("item %s is in paper %s, the handoff says %s",
                                            ids[i], hit$paper_id, items$paper_id[i]))
        }
        state <- hit[[rule$column]]
        if (identical(state, "not_applicable") || !state %in% c(rule$success, rule$failure, rule$non_evaluable)) {
          input_error("contract", sprintf("item %s %s has state '%s' in a confirmatory cell",
                                          ids[i], condition, state))
        }
        prefix <- if (condition == "JATS") "jats_" else "pdf_"
        items[[paste0(prefix, "state")]][i] <- state
        items[[paste0(prefix, "reason")]][i] <- hit[[REASON_COLUMN[[outcome]]]]
        items[[paste0(prefix, "status")]][i] <- hit$execution_status
        cell <- if (condition == "JATS") eligible[[i]]$jats else eligible[[i]]$pdf_docling
        if (!cell_matches(cell, hit$observation_key, state, hit$execution_status, hit[[REASON_COLUMN[[outcome]]]])) {
          input_error("accounting", sprintf("item %s %s differs from the handoff's cell", ids[i], condition))
        }
      }
    }
    evaluable <- c(rule$success, rule$failure)
    items$complete <- items$jats_state %in% evaluable & items$pdf_state %in% evaluable
    code <- function(state) ifelse(state == rule$success, 1, ifelse(state == rule$failure, 0, NA_real_))
    items$y_jats <- ifelse(items$complete, code(items$jats_state), NA_real_)
    items$y_pdf <- ifelse(items$complete, code(items$pdf_state), NA_real_)
    flags <- vapply(eligible, function(e) isTRUE(e$complete_case), logical(1))
    if (!identical(flags, items$complete)) {
      input_error("accounting", sprintf("%s: complete-case flags differ from the handoff", outcome))
    }
    complete_ids <- items$item_id[items$complete]
    if (!identical(complete_ids, stated_ids(stated$complete_item_ids))) {
      input_error("accounting", sprintf("%s: complete item ids differ from the handoff", outcome))
    }
    counts <- table(c(sprintf("JATS|%s", items$jats_state), sprintf("PDF_DOCLING|%s", items$pdf_state)))
    counts <- as_state_counts(as.list(setNames(as.integer(counts), names(counts))))
    if (!identical(counts, as_state_counts(stated$state_counts))) {
      input_error("accounting", sprintf("%s: state counts differ from the handoff", outcome))
    }
    papers <- sort(unique(items$paper_id), method = "radix")
    by_paper <- data.frame(
      paper_id = papers,
      eligible = as.integer(vapply(papers, function(p) sum(items$paper_id == p), numeric(1))),
      complete = as.integer(vapply(papers, function(p) sum(items$complete[items$paper_id == p]), numeric(1))),
      stringsAsFactors = FALSE
    )
    stated_papers <- data.frame(
      paper_id = vapply(stated$papers, `[[`, "", "paper_id"),
      eligible = as.integer(vapply(stated$papers, `[[`, 0, "eligible")),
      complete = as.integer(vapply(stated$papers, `[[`, 0, "complete")),
      stringsAsFactors = FALSE
    )
    if (!identical(by_paper, stated_papers)) {
      input_error("accounting", sprintf("%s: per-paper counts differ from the handoff", outcome))
    }
    result[[outcome]] <- list(items = items, complete_item_ids = complete_ids,
                              state_counts = counts, papers = by_paper)
  }
  result
}
