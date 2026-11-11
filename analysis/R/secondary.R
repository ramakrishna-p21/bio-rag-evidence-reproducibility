# Secondary and exploratory descriptive tables (EXPERIMENT_PLAN section 9).
#
# Counts and proportions only. Nothing here resamples, tests or adjusts anything, and none of it
# is a confirmatory result. Every table states its population in secondary_populations(): the rows
# it uses, the unit, the denominator, how evaluability is decided and how missingness is shown.
# Distinct states are never merged: operational failure, not executed, non-evaluable, abstention,
# unresolved and a record that does not exist each keep their own value. Input is the validated
# row table from read_inputs(); R does not score or infer a label.

SECONDARY_LABEL <- "secondary_exploratory"

NO_STAGE_A <- "no_stage_a_record"
NO_STAGE_B <- "no_stage_b_record"
NO_CONTEXT <- "no_context_record"
NO_COMPLETION <- "no_completion"
NOT_DETERMINED <- "not_determined"   # the tri-state value NA of correct_without_evidence

ROBUSTNESS_ARMS <- c("dense_robustness", "chunk_robustness", "scope_robustness")
REPEATS <- 1:3

STAGE_A_SCHEMA <- c(condition = "chr", stage_a_label = "chr", n = "int", denominator = "int")
STAGE_B_FAILURE_SCHEMA <- c(matrix_arm = "chr", condition = "chr", measure = "chr", value = "chr",
                            n = "int", denominator = "int")
ROBUSTNESS_SCHEMA <- c(matrix_arm = "chr", outcome = "chr", role = "chr", status = "chr", n_items = "int",
                       n_paired = "int", n_complete = "int", n_papers = "int", rate_jats = "dbl",
                       rate_pdf = "dbl", rd = "dbl", rd_pp = "dbl", equal_paper_rd = "dbl")
SCOPE_SCHEMA <- c(condition = "chr", context_composition = "chr", evidence_delivered = "chr",
                  n = "int", denominator = "int")
REPEAT_OUTCOMES_SCHEMA <- c(matrix_arm = "chr", condition = "chr", repeat_index = "int",
                            answer_correct = "chr", n = "int", denominator = "int")
REPEAT_VARIABILITY_SCHEMA <- c(matrix_arm = "chr", condition = "chr", n_items = "int",
                               n_all_repeats_evaluable = "int", n_correct_0 = "int", n_correct_1 = "int",
                               n_correct_2 = "int", n_correct_3 = "int", n_missing_repeat_rows = "int",
                               n_with_unevaluable_repeat = "int")
DISCORDANCE_SCHEMA <- c(outcome = "chr", jats_state = "chr", pdf_state = "chr", pair_class = "chr",
                        n = "int", denominator = "int")
CORRECT_WITHOUT_EVIDENCE_SCHEMA <- c(condition = "chr", correct_without_evidence = "chr", n = "int",
                                     denominator = "int")
COMPLETION_SCHEMA <- c(condition = "chr", execution_status = "chr", completion_kind = "chr",
                       answer_correct = "chr", n = "int", denominator = "int")
ANSWER_CLASS_SCHEMA <- c(condition = "chr", answer_class = "chr", outcome = "chr", state = "chr",
                         n = "int", denominator = "int")
UNAVAILABLE_SCHEMA <- c(analysis = "chr", status = "chr", reason = "chr", required_data = "chr")
POPULATIONS_SCHEMA <- c(table = "chr", population = "chr", unit = "chr", denominator = "chr",
                        evaluability = "chr", missingness = "chr", n_rows = "int")

SECONDARY_SCHEMAS <- list(
  "secondary_stage_a.csv" = STAGE_A_SCHEMA,
  "secondary_stage_b_failure.csv" = STAGE_B_FAILURE_SCHEMA,
  "secondary_robustness.csv" = ROBUSTNESS_SCHEMA,
  "secondary_scope.csv" = SCOPE_SCHEMA,
  "secondary_repeat_outcomes.csv" = REPEAT_OUTCOMES_SCHEMA,
  "secondary_repeat_variability.csv" = REPEAT_VARIABILITY_SCHEMA,
  "secondary_discordance.csv" = DISCORDANCE_SCHEMA,
  "secondary_correct_without_evidence.csv" = CORRECT_WITHOUT_EVIDENCE_SCHEMA,
  "secondary_completion.csv" = COMPLETION_SCHEMA,
  "secondary_answer_class.csv" = ANSWER_CLASS_SCHEMA,
  "secondary_unavailable.csv" = UNAVAILABLE_SCHEMA,
  "secondary_populations.csv" = POPULATIONS_SCHEMA
)

secondary_error <- function(message) input_error("secondary_input", message)

# ---- helpers ----------------------------------------------------------------------------------

REQUIRED_ROW_COLUMNS <- c("item_id", "paper_id", "answer_class", "matrix_arm", "condition", "repeat_index",
                          "execution_status", "completion_kind", "stage_a_label", "stage_b_status",
                          "context_composition", "evidence_delivered", "answer_correct", "earliest_failure",
                          "correct_without_evidence")

check_secondary_rows <- function(rows, rules) {
  missing <- setdiff(REQUIRED_ROW_COLUMNS, names(rows))
  if (length(missing)) secondary_error(sprintf("rows lack column(s): %s", paste(missing, collapse = ", ")))
  if (!setequal(names(rules), EXPECTED_FAMILY)) secondary_error("outcome rules must be the two co-primary outcomes")
  if (anyDuplicated(rows[c("matrix_arm", "item_id", "condition", "repeat_index")])) {
    secondary_error("duplicate (arm, item, condition, repeat) rows")
  }
  invisible(TRUE)
}

fill_missing <- function(x, token) {
  x <- as.character(x)
  x[is.na(x)] <- token
  x
}

# Count rows per distinct combination of `cols`, in radix order. A zero-row input gives a typed
# zero-row result.
tally <- function(df, cols) {
  if (nrow(df) == 0) {
    out <- df[0, cols, drop = FALSE]
    out$n <- integer(0)
    return(out)
  }
  key <- do.call(paste, c(unname(as.list(df[cols])), sep = "\x1f"))
  first <- !duplicated(key)
  out <- df[first, cols, drop = FALSE]
  out$n <- as.integer(tabulate(match(key, key[first])))
  out <- out[do.call(order, c(unname(as.list(out[cols])), list(method = "radix"))), , drop = FALSE]
  rownames(out) <- NULL
  out
}

# Denominator of each row: all counted rows in its `by` group (every row when `by` is empty), every
# state included. Each source row is counted once, so the denominator is the size of the group, not
# of its evaluable part. Also used by the supplement tables.
with_denominator <- function(counts, by) {
  if (!nrow(counts)) {
    counts$denominator <- integer(0)
    return(counts)
  }
  key <- if (length(by)) do.call(paste, c(lapply(unname(as.list(counts[by])), as.character), sep = "\x1f")) else rep("all", nrow(counts))
  counts$denominator <- as.integer(tapply(counts$n, key, sum)[key])
  counts
}

empty_table <- function(schema) {
  cols <- lapply(schema, function(type) switch(type, chr = character(0), int = integer(0), dbl = numeric(0)))
  as.data.frame(cols, stringsAsFactors = FALSE)
}

primary_repeat <- function(rows) rows[rows$repeat_index == 1L, , drop = FALSE]
representations <- function(rows) rows[rows$condition %in% REPRESENTATIONS, , drop = FALSE]

# The populations the tables use. Each is a function of the rows only.
population_primary <- function(rows) {
  representations(primary_repeat(rows[rows$matrix_arm == "confirmatory", , drop = FALSE]))
}
population_retrieval_repeat1 <- function(rows) {
  representations(primary_repeat(rows[rows$matrix_arm != "closed_book", , drop = FALSE]))
}

# ---- tables -----------------------------------------------------------------------------------

# Stage A is judged once per item and representation artifact. Only the confirmatory arm's first
# repeat is used so that neither generation repeats nor other arms count an item twice.
secondary_stage_a <- function(rows) {
  pop <- population_primary(rows)
  pop$stage_a_label <- fill_missing(pop$stage_a_label, NO_STAGE_A)
  with_denominator(tally(pop, c("condition", "stage_a_label")), "condition")[, names(STAGE_A_SCHEMA), drop = FALSE]
}

# Stage B status and earliest failure stage, per arm and representation, first repeat only. Loss
# at Stage A (not reachable), no evidence in the context, an unresolved judgment and a missing
# record are separate values.
secondary_stage_b_failure <- function(rows) {
  pop <- population_retrieval_repeat1(rows)
  if (nrow(pop) == 0) return(empty_table(STAGE_B_FAILURE_SCHEMA))
  one <- function(column, token) {
    d <- pop
    d$measure <- column
    d$value <- fill_missing(d[[column]], token)
    with_denominator(tally(d, c("matrix_arm", "condition", "measure", "value")), c("matrix_arm", "condition", "measure"))
  }
  out <- rbind(one("stage_b_status", NO_STAGE_B), one("earliest_failure", "no_record"))
  out <- out[do.call(order, c(unname(as.list(out[c("matrix_arm", "condition", "measure", "value")])), list(method = "radix"))), ]
  rownames(out) <- NULL
  out[, names(STAGE_B_FAILURE_SCHEMA), drop = FALSE]
}

# Paired JATS - PDF_DOCLING difference per arm and outcome, first repeat, same complete-case rule
# as the primary analysis. Item pairing is by item id inside the arm. Descriptive: no interval
# and no p-value. An absent arm is a row with status arm_absent, not a missing row.
secondary_robustness <- function(rows, rules) {
  out <- list()
  for (arm in c("confirmatory", ROBUSTNESS_ARMS)) {
    arm_rows <- representations(primary_repeat(rows[rows$matrix_arm == arm, , drop = FALSE]))
    role <- if (arm == "confirmatory") "reference_primary_arm" else "robustness"
    for (outcome in EXPECTED_FAMILY) {
      rule <- rules[[outcome]]
      items <- sort(unique(arm_rows$item_id), method = "radix")
      j <- arm_rows[arm_rows$condition == "JATS", , drop = FALSE]
      p <- arm_rows[arm_rows$condition == "PDF_DOCLING", , drop = FALSE]
      both <- intersect(j$item_id, p$item_id)
      jj <- j[match(both, j$item_id), , drop = FALSE]
      pp <- p[match(both, p$item_id), , drop = FALSE]
      evaluable <- c(rule$success, rule$failure)
      ok <- jj[[rule$column]] %in% evaluable & pp[[rule$column]] %in% evaluable
      code <- function(x) as.numeric(x == rule$success)
      s <- summarize_pairs(code(jj[[rule$column]][ok]), code(pp[[rule$column]][ok]), jj$paper_id[ok])
      status <- if (nrow(arm_rows) == 0) "arm_absent" else s$status
      out[[length(out) + 1]] <- data.frame(
        matrix_arm = arm, outcome = outcome, role = role, status = status, n_items = length(items),
        n_paired = length(both), n_complete = as.integer(sum(ok)), n_papers = s$n_papers,
        rate_jats = s$rate_jats, rate_pdf = s$rate_pdf, rd = s$rd, rd_pp = s$rd_pp,
        equal_paper_rd = s$equal_paper_rd, stringsAsFactors = FALSE)
    }
  }
  out <- do.call(rbind, out)
  out$n_items <- as.integer(out$n_items); out$n_paired <- as.integer(out$n_paired)
  out$n_papers <- as.integer(out$n_papers)
  rownames(out) <- NULL
  out[, names(ROBUSTNESS_SCHEMA), drop = FALSE]
}

# Full-corpus scope cell: what the supplied context contained, against whether the evidence
# arrived. Context composition comes from Python; R does not classify a miss.
secondary_scope <- function(rows) {
  pop <- representations(primary_repeat(rows[rows$matrix_arm == "scope_robustness", , drop = FALSE]))
  pop$context_composition <- fill_missing(pop$context_composition, NO_CONTEXT)
  with_denominator(tally(pop, c("condition", "context_composition", "evidence_delivered")), "condition")[, names(SCOPE_SCHEMA), drop = FALSE]
}

repeat_groups <- function(rows) {
  rows[(rows$matrix_arm == "confirmatory" & rows$condition %in% REPRESENTATIONS) |
         (rows$matrix_arm == "closed_book" & rows$condition == "CLOSED_BOOK"), , drop = FALSE]
}

# Answer correctness at each prespecified repeat. Closed book is its own arm and condition, never
# a third representation. No repeat is chosen, pooled or ranked.
secondary_repeat_outcomes <- function(rows) {
  pop <- repeat_groups(rows)
  pop$repeat_index <- as.integer(pop$repeat_index)
  with_denominator(tally(pop, c("matrix_arm", "condition", "repeat_index", "answer_correct")),
                   c("matrix_arm", "condition", "repeat_index"))[, names(REPEAT_OUTCOMES_SCHEMA), drop = FALSE]
}

# Per item and condition over repeats 1-3: how many repeats were correct. Only an item with all
# three repeat rows and a yes/no correctness in each is counted by its correct number; the rest
# are counted separately (missing repeat row first, then an unevaluable repeat). This is
# correctness agreement. It says nothing about whether two incorrect answers were the same.
secondary_repeat_variability <- function(rows, rules) {
  pop <- repeat_groups(rows)
  evaluable <- c(rules$answer_correct$success, rules$answer_correct$failure)
  out <- list()
  for (key in sort(unique(paste(pop$matrix_arm, pop$condition, sep = "\x1f")), method = "radix")) {
    parts <- strsplit(key, "\x1f", fixed = TRUE)[[1]]
    g <- pop[pop$matrix_arm == parts[1] & pop$condition == parts[2], , drop = FALSE]
    items <- sort(unique(g$item_id), method = "radix")
    k <- integer(4); missing_rows <- 0L; unevaluable <- 0L
    for (it in items) {
      r <- g[g$item_id == it, , drop = FALSE]
      if (!setequal(r$repeat_index, REPEATS)) { missing_rows <- missing_rows + 1L; next }
      if (!all(r$answer_correct %in% evaluable)) { unevaluable <- unevaluable + 1L; next }
      c_yes <- sum(r$answer_correct == rules$answer_correct$success)
      k[c_yes + 1] <- k[c_yes + 1] + 1L
    }
    out[[length(out) + 1]] <- data.frame(
      matrix_arm = parts[1], condition = parts[2], n_items = length(items), n_all_repeats_evaluable = sum(k),
      n_correct_0 = k[1], n_correct_1 = k[2], n_correct_2 = k[3], n_correct_3 = k[4],
      n_missing_repeat_rows = missing_rows, n_with_unevaluable_repeat = unevaluable, stringsAsFactors = FALSE)
  }
  if (!length(out)) return(empty_table(REPEAT_VARIABILITY_SCHEMA))
  out <- do.call(rbind, out)
  out$n_items <- as.integer(out$n_items)
  rownames(out) <- NULL
  out[, names(REPEAT_VARIABILITY_SCHEMA), drop = FALSE]
}

pair_class <- function(j, p, rule) {
  evaluable <- c(rule$success, rule$failure)
  ifelse(!(j %in% evaluable & p %in% evaluable), "not_complete",
         ifelse(j == rule$success & p == rule$success, "both_success",
                ifelse(j == rule$failure & p == rule$failure, "both_failure",
                       ifelse(j == rule$success, "jats_only_success", "pdf_only_success"))))
}

# Paired states of the primary population, item by item: the full cross table of the two
# representations' states, so unevaluable pairs stay visible beside the complete ones. This is
# disagreement in the binary outcome. Answer-content disagreement is in secondary_unavailable().
secondary_discordance <- function(rows, rules) {
  pop <- population_primary(rows)
  j <- pop[pop$condition == "JATS", , drop = FALSE]
  p <- pop[pop$condition == "PDF_DOCLING", , drop = FALSE]
  both <- sort(intersect(j$item_id, p$item_id), method = "radix")
  jj <- j[match(both, j$item_id), , drop = FALSE]
  pp <- p[match(both, p$item_id), , drop = FALSE]
  out <- list()
  for (outcome in EXPECTED_FAMILY) {
    col <- rules[[outcome]]$column
    d <- data.frame(outcome = rep(outcome, length(both)), jats_state = jj[[col]], pdf_state = pp[[col]],
                    stringsAsFactors = FALSE)
    d$pair_class <- pair_class(d$jats_state, d$pdf_state, rules[[outcome]])
    out[[outcome]] <- with_denominator(tally(d, c("outcome", "jats_state", "pdf_state", "pair_class")), "outcome")
  }
  out <- do.call(rbind, out)
  rownames(out) <- NULL
  if (is.null(out)) return(empty_table(DISCORDANCE_SCHEMA))
  out[, names(DISCORDANCE_SCHEMA), drop = FALSE]
}

secondary_correct_without_evidence <- function(rows) {
  pop <- population_primary(rows)
  pop$correct_without_evidence <- ifelse(is.na(pop$correct_without_evidence) | pop$correct_without_evidence == MISSING_TOKEN,
                                         NOT_DETERMINED, pop$correct_without_evidence)
  with_denominator(tally(pop, c("condition", "correct_without_evidence")), "condition")[, names(CORRECT_WITHOUT_EVIDENCE_SCHEMA), drop = FALSE]
}

# How each primary cell ended: operational failure, abstention, unparseable output and a scored
# answer are different execution/completion states, shown against the correctness label they got.
secondary_completion <- function(rows) {
  pop <- population_primary(rows)
  pop$completion_kind <- fill_missing(pop$completion_kind, NO_COMPLETION)
  with_denominator(tally(pop, c("condition", "execution_status", "completion_kind", "answer_correct")),
                   "condition")[, names(COMPLETION_SCHEMA), drop = FALSE]
}

# Both co-primary outcomes' states by answer class in the primary population. The evaluated
# population of each class is the sum of its yes and no rows; the others are shown beside them.
secondary_answer_class <- function(rows, rules) {
  pop <- population_primary(rows)
  if (nrow(pop) == 0) return(empty_table(ANSWER_CLASS_SCHEMA))
  out <- lapply(EXPECTED_FAMILY, function(outcome) {
    d <- pop
    d$outcome <- outcome
    d$state <- d[[rules[[outcome]]$column]]
    with_denominator(tally(d, c("condition", "answer_class", "outcome", "state")), c("condition", "answer_class", "outcome"))
  })
  out <- do.call(rbind, out)
  out <- out[do.call(order, c(unname(as.list(out[c("condition", "answer_class", "outcome", "state")])), list(method = "radix"))), ]
  rownames(out) <- NULL
  out[, names(ANSWER_CLASS_SCHEMA), drop = FALSE]
}

# Prespecified analyses the validated analysis table cannot support. They are listed so that their
# absence is part of the result, not an omission.
secondary_unavailable <- function() {
  data.frame(
    analysis = c("answer_identity_disagreement", "identical_incorrect_answer_stability", "table_vs_prose",
                 "evidence_location"),
    status = "deferred_data_not_in_analysis_table",
    reason = c(
      "the table holds a correctness label per observation, not the generated answer or an identity of it, so correctness disagreement cannot be read as answer disagreement",
      "three incorrect repeats do not show the same incorrect answer; without response identities only correctness agreement across repeats is reported",
      "no table-versus-prose attribute of benchmark evidence is a column of the frozen analysis table",
      "no evidence-location attribute of benchmark evidence is a column of the frozen analysis table"),
    required_data = c("per-observation response identity (for example a hash of the normalized parsed answer)",
                      "per-observation response identity across repeats 1-3",
                      "per-item table/prose attribute fixed before analysis",
                      "per-item evidence location fixed before analysis"),
    stringsAsFactors = FALSE)
}

# Population, unit, denominator and missingness rules of each table, with the number of source
# rows that fed it (zero means the arm or condition is absent from the input).
secondary_populations <- function(rows) {
  retrieval <- population_retrieval_repeat1(rows)
  desc <- function(table, population, unit, denominator, evaluability, missingness, n_rows) {
    data.frame(table = table, population = population, unit = unit, denominator = denominator,
               evaluability = evaluability, missingness = missingness, n_rows = as.integer(n_rows),
               stringsAsFactors = FALSE)
  }
  prim <- nrow(population_primary(rows))
  out <- rbind(
    desc("secondary_stage_a.csv", "confirmatory arm, repeat 1, JATS and PDF_DOCLING rows", "item x representation",
         "rows of the representation", "none: every Stage A label is counted", "a row without a Stage A record is no_stage_a_record", prim),
    desc("secondary_stage_b_failure.csv", "retrieval arms, repeat 1, JATS and PDF_DOCLING rows", "item x representation per arm",
         "rows of the arm, representation and measure", "none: every status and failure stage is counted",
         "not_reachable_stage_a, non_evaluable, unresolved and no record stay separate values", nrow(retrieval)),
    desc("secondary_robustness.csv", "confirmatory reference and the three robustness arms, repeat 1", "item paired across representations",
         "n_complete: items evaluable under both representations for the outcome",
         "success or failure state of the outcome in both representations", "incomplete pairs are excluded from the rate and counted in n_paired - n_complete; an absent arm is arm_absent",
         nrow(representations(primary_repeat(rows[rows$matrix_arm %in% c("confirmatory", ROBUSTNESS_ARMS), , drop = FALSE])))),
    desc("secondary_scope.csv", "scope_robustness arm, repeat 1, JATS and PDF_DOCLING rows", "item x representation",
         "rows of the representation", "none: composition and delivery are counted as recorded", "a row without a context record is no_context_record",
         nrow(representations(primary_repeat(rows[rows$matrix_arm == "scope_robustness", , drop = FALSE])))),
    desc("secondary_repeat_outcomes.csv", "confirmatory JATS and PDF_DOCLING, and closed_book CLOSED_BOOK, all repeats", "observation per prespecified repeat",
         "rows of the arm, condition and repeat", "none: every correctness state is counted", "missing, non_evaluable and not_executed stay separate from no",
         nrow(repeat_groups(rows))),
    desc("secondary_repeat_variability.csv", "same as secondary_repeat_outcomes.csv", "item within arm and condition",
         "n_items: distinct items", "an item counts by its number of correct repeats only with three repeat rows each yes or no",
         "missing repeat rows and unevaluable repeats are counted separately; correctness agreement only", nrow(repeat_groups(rows))),
    desc("secondary_discordance.csv", "confirmatory arm, repeat 1, items present under both representations", "item",
         "paired items", "pair_class not_complete when either state is not success or failure", "all state pairs listed", prim),
    desc("secondary_correct_without_evidence.csv", "confirmatory arm, repeat 1, JATS and PDF_DOCLING rows", "item x representation",
         "rows of the representation", "none", "the tri-state value NA is not_determined", prim),
    desc("secondary_completion.csv", "confirmatory arm, repeat 1, JATS and PDF_DOCLING rows", "item x representation",
         "rows of the representation", "none", "operational failure, abstention and unparseable output are separate from a scored answer", prim),
    desc("secondary_answer_class.csv", "confirmatory arm, repeat 1, JATS and PDF_DOCLING rows", "item x representation per answer class and outcome",
         "rows of the class and representation", "yes and no rows are the evaluated population", "other states are listed beside them", prim),
    desc("secondary_unavailable.csv", "prespecified analyses without supporting data", "analysis", "not applicable", "not applicable",
         "listed as deferred", 4L),
    desc("secondary_populations.csv", "this table", "output table", "not applicable", "not applicable",
         "n_rows is the number of source rows that fed the table", length(SECONDARY_SCHEMAS) - 1L)
  )
  rownames(out) <- NULL
  out
}

compute_secondary_tables <- function(rows, rules) {
  check_secondary_rows(rows, rules)
  populations <- secondary_populations(rows)
  tables <- list(
    "secondary_stage_a.csv" = secondary_stage_a(rows),
    "secondary_stage_b_failure.csv" = secondary_stage_b_failure(rows),
    "secondary_robustness.csv" = secondary_robustness(rows, rules),
    "secondary_scope.csv" = secondary_scope(rows),
    "secondary_repeat_outcomes.csv" = secondary_repeat_outcomes(rows),
    "secondary_repeat_variability.csv" = secondary_repeat_variability(rows, rules),
    "secondary_discordance.csv" = secondary_discordance(rows, rules),
    "secondary_correct_without_evidence.csv" = secondary_correct_without_evidence(rows),
    "secondary_completion.csv" = secondary_completion(rows),
    "secondary_answer_class.csv" = secondary_answer_class(rows, rules),
    "secondary_unavailable.csv" = secondary_unavailable(),
    "secondary_populations.csv" = populations
  )
  tables[names(SECONDARY_SCHEMAS)]
}
