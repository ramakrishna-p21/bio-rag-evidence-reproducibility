# Supplement summary: counts of the labels Python decided for the deferred secondary analyses.
#
# Python (evaluation/supplement.py) decides every label: answer-content identity, pair and
# stability classes, evidence form and location. This file re-establishes the hash links to the
# analysis table and handoff, re-reads the Python supplement strictly, and counts. It never
# compares answers, re-classifies a pair, or infers an evidence form. The result is a separate
# hashed directory; the primary and secondary results are not changed.
# While the input is a synthetic rehearsal every file says so and nothing is a finding.

SUPPLEMENT_LABEL <- "supplement_exploratory"
SUPPLEMENT_PY_RECORD <- "supplement_record.json"
SUPPLEMENT_SUMMARY_RECORD <- "supplement_summary_record.json"
SUPPLEMENT_SUMMARY_KIND <- "r_supplement_summary_record"
SUPPLEMENT_PY_KIND <- "analysis_supplement"

PY_SUPPLEMENT_SCHEMAS <- list(
  "supplement_response.csv" = c(observation_key = "chr", item_id = "chr", item_sha256 = "chr", matrix_arm = "chr",
                                condition = "chr", repeat_index = "int", attempt_id = "chr", stage_c_artifact = "chr",
                                answer_state = "chr", stage_c_status = "chr", answer_shape = "chr",
                                byte_group = "int", norm_group = "int"),
  "supplement_pairs.csv" = c(item_id = "chr", matrix_arm = "chr", repeat_index = "int", jats_state = "chr",
                             pdf_state = "chr", jats_status = "chr", pdf_status = "chr", pair_class = "chr"),
  "supplement_stability.csv" = c(item_id = "chr", matrix_arm = "chr", condition = "chr", n_rows = "int",
                                 n_answers = "int", stability_class = "chr", byte_identical = "chr"),
  "supplement_evidence_items.csv" = c(item_id = "chr", item_sha256 = "chr", n_jats_refs = "int", n_pdf_refs = "int",
                                      n_evidence_sets = "int", n_sets_with_table_roles = "int", jats_form = "chr",
                                      jats_location = "chr", pdf_form = "chr", form_relation = "chr", item_form = "chr"),
  "supplement_evidence_observations.csv" = c(observation_key = "chr", item_id = "chr", matrix_arm = "chr",
                                             condition = "chr", repeat_index = "int", stage_a_label = "chr",
                                             stage_a_form = "chr", stage_a_sets_table = "int",
                                             stage_a_sets_prose = "int", stage_a_sets_mixed = "int",
                                             stage_a_sets_other = "int", stage_b_status = "chr", stage_b_form = "chr")
)

PAIRS_SUMMARY_SCHEMA <- c(matrix_arm = "chr", repeat_index = "int", pair_class = "chr", jats_state = "chr",
                          pdf_state = "chr", n = "int", denominator = "int")
STABILITY_SUMMARY_SCHEMA <- c(matrix_arm = "chr", condition = "chr", stability_class = "chr",
                              byte_identical = "chr", n = "int", denominator = "int")
EVIDENCE_SUMMARY_SCHEMA <- c(item_form = "chr", jats_location = "chr", form_relation = "chr", n = "int",
                             denominator = "int")
EVIDENCE_OUTCOMES_SCHEMA <- c(condition = "chr", item_form = "chr", answer_correct = "chr", n = "int",
                              denominator = "int")
STAGE_FORMS_SCHEMA <- c(condition = "chr", stage_a_form = "chr", stage_b_form = "chr", n = "int",
                        denominator = "int")
SUPPLEMENT_POPULATIONS_SCHEMA <- c(table = "chr", population = "chr", unit = "chr", denominator = "chr",
                                   evaluability = "chr", missingness = "chr", n_rows = "int")

SUPPLEMENT_SCHEMAS <- list(
  "supplement_pairs_summary.csv" = PAIRS_SUMMARY_SCHEMA,
  "supplement_stability_summary.csv" = STABILITY_SUMMARY_SCHEMA,
  "supplement_evidence_summary.csv" = EVIDENCE_SUMMARY_SCHEMA,
  "supplement_evidence_outcomes.csv" = EVIDENCE_OUTCOMES_SCHEMA,
  "supplement_stage_forms.csv" = STAGE_FORMS_SCHEMA,
  "supplement_populations.csv" = SUPPLEMENT_POPULATIONS_SCHEMA
)
SUPPLEMENT_DATA_FILES <- names(SUPPLEMENT_SCHEMAS)
SUPPLEMENT_FILES <- c(SUPPLEMENT_DATA_FILES, SUPPLEMENT_SUMMARY_RECORD)

# Group columns whose rows must sum to the denominator.
SUPPLEMENT_DENOMINATOR_GROUPS <- list(
  "supplement_pairs_summary.csv" = c("matrix_arm", "repeat_index"),
  "supplement_stability_summary.csv" = c("matrix_arm", "condition"),
  "supplement_evidence_summary.csv" = character(0),
  "supplement_evidence_outcomes.csv" = c("condition", "item_form"),
  "supplement_stage_forms.csv" = "condition"
)

supplement_error <- function(message) input_error("supplement_input", message)

# ---- identity ---------------------------------------------------------------------------------

# Python's canonical JSON of {analysis_table_sha256, definitions_version, files, handoff_sha256, kind}.
python_supplement_id <- function(handoff_sha256, csv_sha256, file_sha256) {
  files <- files_json(file_sha256)
  sha256_raw(charToRaw(sprintf(
    "{\"analysis_table_sha256\":\"%s\",\"definitions_version\":\"1\",\"files\":[%s],\"handoff_sha256\":\"%s\",\"kind\":\"%s\"}",
    csv_sha256, files, handoff_sha256, SUPPLEMENT_PY_KIND)))
}

supplement_summary_id <- function(handoff_sha256, csv_sha256, supplement_sha256, file_sha256) {
  if (!all(vapply(list(handoff_sha256, csv_sha256, supplement_sha256), is_sha256_hex, logical(1))) ||
      !setequal(names(file_sha256), SUPPLEMENT_DATA_FILES) || !all(vapply(file_sha256, is_sha256_hex, logical(1)))) {
    run_error("identity", "supplement summary identity inputs must be SHA-256 hex strings for exactly its tables")
  }
  files <- files_json(file_sha256)
  sha256_raw(charToRaw(sprintf(
    "{\"csv_sha256\":\"%s\",\"files\":[%s],\"handoff_sha256\":\"%s\",\"kind\":\"supplement_summary\",\"supplement_sha256\":\"%s\"}",
    csv_sha256, files, handoff_sha256, supplement_sha256)))
}

# ---- reading the Python supplement ------------------------------------------------------------

# Strictly read the Python supplement directory. Hashes and links are re-established here from
# bytes; Python's own re-derivation of the dependent labels is not repeated by R.
read_supplement <- function(dir, expected_record_sha256, inputs) {
  if (!dir.exists(dir)) supplement_error("supplement directory not found")
  expected <- c(names(PY_SUPPLEMENT_SCHEMAS), SUPPLEMENT_PY_RECORD)
  present <- list.files(dir, all.files = TRUE, no.. = TRUE)
  if (!setequal(present, expected)) {
    supplement_error(sprintf("the supplement files are not the expected set (missing: %s; unexpected: %s)",
                             paste(setdiff(expected, present), collapse = ", "), paste(setdiff(present, expected), collapse = ", ")))
  }
  paths <- stats::setNames(file.path(dir, expected), expected)
  if (any(nzchar(Sys.readlink(paths))) || any(dir.exists(paths))) supplement_error("every supplement file must be a regular file")
  bytes <- lapply(paths, read_bytes)
  record_sha256 <- sha256_raw(bytes[[SUPPLEMENT_PY_RECORD]])
  if (!is_sha256_hex(expected_record_sha256) || !identical(record_sha256, expected_record_sha256)) {
    supplement_error("the supplement record differs from the trusted SHA-256")
  }
  record <- tryCatch(jsonlite::fromJSON(rawToChar(bytes[[SUPPLEMENT_PY_RECORD]]), simplifyVector = FALSE),
                     error = function(e) supplement_error("the supplement record is not valid JSON"))
  claims <- list(kind = SUPPLEMENT_PY_KIND, schema_version = "1", definitions_version = "1",
                 purpose = "synthetic_rehearsal", scientific_authority = "none", final = FALSE,
                 banner = SYNTHETIC_BANNER)
  for (n in names(claims)) {
    if (!identical(record[[n]], claims[[n]])) supplement_error(sprintf("supplement record field %s must be %s", n, format(claims[[n]])))
  }
  if (!identical(record$analysis_table_sha256, inputs$csv_sha256) ||
      !identical(record$handoff_sha256, inputs$handoff_sha256)) {
    supplement_error("the supplement does not cite this handoff and analysis table")
  }
  if (!is.list(record$files) || !setequal(names(record$files), names(PY_SUPPLEMENT_SCHEMAS))) {
    supplement_error("the supplement record must list exactly the supplement tables")
  }
  for (n in names(PY_SUPPLEMENT_SCHEMAS)) {
    e <- record$files[[n]]
    if (!identical(e$sha256, sha256_raw(bytes[[n]])) || !identical(as.numeric(e$bytes), as.numeric(length(bytes[[n]])))) {
      supplement_error(sprintf("%s differs from the hash and size recorded for it", n))
    }
  }
  id <- python_supplement_id(record$handoff_sha256, record$analysis_table_sha256,
                             vapply(record$files, function(e) e$sha256, ""))
  if (!identical(id, record$supplement_sha256)) supplement_error("supplement_sha256 does not match the files and inputs")
  tables <- lapply(stats::setNames(names(PY_SUPPLEMENT_SCHEMAS), names(PY_SUPPLEMENT_SCHEMAS)), function(n) {
    tryCatch(parse_canonical_csv(bytes[[n]], PY_SUPPLEMENT_SCHEMAS[[n]], n),
             run_error = function(e) supplement_error(conditionMessage(e)))
  })
  check_supplement_links(tables, inputs$rows)
  list(tables = tables, record = record, record_sha256 = record_sha256, supplement_sha256 = id)
}

# The label rows must be exactly the analysis table's rows; R compares identities, not content.
check_supplement_links <- function(tables, rows) {
  resp <- tables[["supplement_response.csv"]]
  if (anyDuplicated(resp$observation_key) || !setequal(resp$observation_key, rows$observation_key) ||
      nrow(resp) != nrow(rows)) {
    supplement_error("the response rows are not exactly the analysis-table rows")
  }
  at <- match(resp$observation_key, rows$observation_key)
  same <- resp$item_id == rows$item_id[at] & resp$matrix_arm == rows$matrix_arm[at] &
    resp$condition == rows$condition[at] & resp$repeat_index == rows$repeat_index[at]
  if (!all(same)) supplement_error("a response row disagrees with its analysis-table row")
  obs <- tables[["supplement_evidence_observations.csv"]]
  if (anyDuplicated(obs$observation_key) || !setequal(obs$observation_key, rows$observation_key)) {
    supplement_error("the evidence rows are not exactly the analysis-table rows")
  }
  if (!setequal(tables[["supplement_evidence_items.csv"]]$item_id, rows$item_id)) {
    supplement_error("the evidence items are not exactly the table's items")
  }
  invisible(TRUE)
}

# ---- tables -----------------------------------------------------------------------------------

# Count rows by `cols`; groups sorted in radix order by those columns.
tabulate_groups <- function(df, cols) {
  if (!nrow(df)) {
    out <- df[0, cols, drop = FALSE]
    out$n <- integer(0)
    return(out)
  }
  key <- do.call(paste, c(lapply(unname(as.list(df[cols])), function(x) ifelse(is.na(x), "\001NA", as.character(x))), sep = "\x1f"))
  first <- !duplicated(key)
  out <- df[first, cols, drop = FALSE]
  out$n <- as.integer(table(factor(key, levels = key[first])))
  out <- out[do.call(order, c(unname(as.list(out[cols])), list(method = "radix"))), , drop = FALSE]
  rownames(out) <- NULL
  out
}

# Confirmatory repeat 1 of one condition per item: the primary-row population, one row per item.
supplement_primary_rows <- function(rows) {
  rows[rows$matrix_arm == "confirmatory" & rows$repeat_index == 1L, , drop = FALSE]
}

compute_supplement_tables <- function(py, rows) {
  pairs <- py[["supplement_pairs.csv"]]
  stab <- py[["supplement_stability.csv"]]
  items <- py[["supplement_evidence_items.csv"]]
  obs <- py[["supplement_evidence_observations.csv"]]
  resp <- py[["supplement_response.csv"]]

  pairs_summary <- with_denominator(
    tabulate_groups(pairs, c("matrix_arm", "repeat_index", "pair_class", "jats_state", "pdf_state")),
    c("matrix_arm", "repeat_index"))
  stability_summary <- with_denominator(
    tabulate_groups(stab, c("matrix_arm", "condition", "stability_class", "byte_identical")),
    c("matrix_arm", "condition"))
  evidence_summary <- with_denominator(tabulate_groups(items, c("item_form", "jats_location", "form_relation")),
                                       character(0))

  primary <- supplement_primary_rows(rows)
  joined <- merge(primary[c("item_id", "condition", "answer_correct")], items[c("item_id", "item_form")],
                  by = "item_id", all.x = TRUE, sort = FALSE)
  evidence_outcomes <- with_denominator(tabulate_groups(joined, c("condition", "item_form", "answer_correct")),
                                        c("condition", "item_form"))
  forms <- obs[obs$matrix_arm == "confirmatory" & obs$repeat_index == 1L, , drop = FALSE]
  stage_forms <- with_denominator(tabulate_groups(forms, c("condition", "stage_a_form", "stage_b_form")), "condition")

  desc <- function(table, population, unit, denominator, evaluability, missingness, n_rows) {
    data.frame(table = table, population = population, unit = unit, denominator = denominator,
               evaluability = evaluability, missingness = missingness, n_rows = as.integer(n_rows),
               stringsAsFactors = FALSE)
  }
  populations <- rbind(
    desc("supplement_pairs_summary.csv", "JATS and PDF_DOCLING responses of one item, retrieval arm and repeat",
         "item x arm x repeat pair", "pairs of the arm and repeat",
         "Python pair_class; both_correct is not scientific equivalence; differ_both_incorrect_unresolved is an upper bound",
         "absent side or an abstained, unparseable, failed or unexecuted response is not_comparable with its states", nrow(pairs)),
    desc("supplement_stability_summary.csv", "three repeats of one item in one condition, confirmatory and closed-book arms",
         "item x arm x condition", "groups of the arm and condition",
         "Python stability_class; stable incorrect needs three completed scored incorrect repeats with one normalized answer",
         "an absent, failed or unexecuted repeat is incomplete; abstention or unparseable output is never an identical incorrect answer",
         nrow(stab)),
    desc("supplement_evidence_summary.csv", "all benchmark items of the table", "item", "all items",
         "Python item_form and jats_location from the original JATS path structure; PDF location is not classified",
         "unresolved and other are kept; a missing table id is not prose; JATS and PDF disagreement is form_relation", nrow(items)),
    desc("supplement_evidence_outcomes.csv", "confirmatory repeat 1 rows by item evidence form", "observation",
         "rows of the condition and item form", "answer_correct exactly as in the analysis table; no re-scoring",
         "not evaluable and missing answer states keep their own values", nrow(primary)),
    desc("supplement_stage_forms.csv", "confirmatory repeat 1 rows by preserved and delivered evidence form",
         "observation", "rows of the condition",
         "Python stage_a_form and stage_b_form from the canonical document of that representation",
         "no_stage_a_record, no_stage_b_record, no_located_evidence and none_delivered are distinct values", nrow(forms))
  )
  populations <- populations[order(populations$table, method = "radix"), ]
  rownames(populations) <- NULL
  list("supplement_pairs_summary.csv" = pairs_summary, "supplement_stability_summary.csv" = stability_summary,
       "supplement_evidence_summary.csv" = evidence_summary, "supplement_evidence_outcomes.csv" = evidence_outcomes,
       "supplement_stage_forms.csv" = stage_forms, "supplement_populations.csv" = populations,
       .response_rows = nrow(resp))
}

check_supplement_tables <- function(tables) {
  bad <- function(why) verify_error(why)
  if (!setequal(names(tables), SUPPLEMENT_DATA_FILES)) bad("the supplement summary tables are not the expected set")
  for (name in names(SUPPLEMENT_DENOMINATOR_GROUPS)) {
    t <- tables[[name]]
    if (!nrow(t)) next
    if (any(t$n < 1L) || any(t$n > t$denominator)) bad(sprintf("%s: a count is outside 1..denominator", name))
    g <- SUPPLEMENT_DENOMINATOR_GROUPS[[name]]
    group <- if (length(g)) do.call(paste, c(lapply(unname(as.list(t[g])), as.character), sep = "\x1f")) else rep("all", nrow(t))
    if (!identical(as.integer(tapply(t$n, group, sum)[group]), t$denominator)) {
      bad(sprintf("%s: a denominator is not the sum of its group", name))
    }
  }
  p <- tables[["supplement_populations.csv"]]
  if (!setequal(p$table, setdiff(SUPPLEMENT_DATA_FILES, "supplement_populations.csv")) || anyNA(p$n_rows)) {
    bad("supplement_populations.csv does not describe exactly the supplement tables")
  }
  invisible(TRUE)
}

# ---- computation, record, publication ---------------------------------------------------------

compute_supplement <- function(handoff_path, handoff_sha256, csv_path, supplement_dir, supplement_record_sha256) {
  if (identical(handoff_origin(handoff_path, handoff_sha256), "verified_official_run")) {
    run_error("origin", "the Python-supplement analyses are deferred for official inputs: official-origin support is not implemented")
  }
  inputs <- read_inputs(handoff_path, handoff_sha256, csv_path)
  if (!identical(inputs$input_origin, "synthetic_rehearsal")) {
    run_error("origin", sprintf("input_origin '%s' is not accepted; only a synthetic rehearsal is", inputs$input_origin))
  }
  py <- read_supplement(supplement_dir, supplement_record_sha256, inputs)
  computed <- compute_supplement_tables(py$tables, inputs$rows)
  tables <- computed[SUPPLEMENT_DATA_FILES]
  files <- Map(csv_bytes, tables, SUPPLEMENT_SCHEMAS)
  file_sha256 <- vapply(files, sha256_raw, "")
  list(inputs = inputs, py = py, tables = tables, files = files, file_sha256 = file_sha256,
       summary_sha256 = supplement_summary_id(handoff_sha256, inputs$csv_sha256, py$supplement_sha256, file_sha256))
}

build_supplement_record <- function(derived, primary, code, execution) {
  entry <- function(b) list(bytes = length(b), sha256 = sha256_raw(b))
  list(
    kind = SUPPLEMENT_SUMMARY_KIND, schema_version = "1",
    purpose = "synthetic_rehearsal", scientific_authority = "none", banner = SYNTHETIC_BANNER,
    analysis_label = SUPPLEMENT_LABEL,
    inference = "none: counts of Python-decided labels; no interval, test or adjustment",
    inputs = list(handoff_sha256 = derived$inputs$handoff_sha256, csv_sha256 = derived$inputs$csv_sha256,
                  supplement_sha256 = derived$py$supplement_sha256, supplement_record_sha256 = derived$py$record_sha256,
                  input_origin = derived$inputs$input_origin),
    primary_result = if (is.null(primary)) list(citation = "not_cited") else
      list(run_record_sha256 = primary$record_sha256, scientific_result_sha256 = primary$scientific_result_sha256,
           verification = "verified_by_this_run_before_use"),
    scientific_files = lapply(derived$files, entry),
    supplement_summary_sha256 = derived$summary_sha256,
    code = list(git_commit = code$commit, dirty = code$dirty, script_sha256 = as.list(code$script_sha256)),
    execution = execution
  )
}

run_supplement <- function(handoff_path, handoff_sha256, csv_path, supplement_dir, supplement_record_sha256, out_root,
                           project_dir, primary_dir = NULL, expected_primary_record_sha256 = NULL, now = Sys.time,
                           code_fn = code_state) {
  started <- now()
  if (!dir.exists(out_root)) run_error("publish", "the output root must be an existing directory")
  derived <- compute_supplement(handoff_path, handoff_sha256, csv_path, supplement_dir, supplement_record_sha256)
  primary <- NULL
  if (!is.null(primary_dir)) {
    primary <- verify_result(primary_dir, expected_record_sha256 = expected_primary_record_sha256)
    if (!identical(primary$record$inputs$handoff_sha256, handoff_sha256) ||
        !identical(primary$record$inputs$csv_sha256, derived$inputs$csv_sha256)) {
      run_error("primary", "the primary result was computed from different inputs")
    }
  }
  code <- code_fn(project_dir)
  if (!identical(code$status, "ok")) run_error("code", "the git state of the analysis code cannot be read")
  run_id <- sprintf("%s-%s", format(started, "%Y%m%dT%H%M%OS3Z", tz = "UTC"), substr(derived$summary_sha256, 1, 8))
  execution <- list(run_id = run_id, started_utc = iso_utc(started), finished_utc = iso_utc(now()), hostname = Sys.info()[["nodename"]])
  record_bytes <- serialize_record(build_supplement_record(derived, primary, code, execution))
  contents <- c(derived$files, stats::setNames(list(record_bytes), SUPPLEMENT_SUMMARY_RECORD))
  staging <- file.path(out_root, paste0(".staging-supplement-", run_id))
  if (!dir.create(staging, showWarnings = FALSE, mode = "0755")) run_error("publish", "the staging directory already exists")
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  for (f in SUPPLEMENT_FILES) writeBin(contents[[f]], file.path(staging, f))
  target <- file.path(out_root, paste0("supplement_", run_id))
  publish_result(staging, target, files = SUPPLEMENT_FILES, verifier = verify_supplement_summary)
  list(result_dir = target, record_sha256 = sha256_raw(record_bytes), supplement_summary_sha256 = derived$summary_sha256,
       run_id = run_id)
}

# ---- verification -----------------------------------------------------------------------------

verify_supplement_summary <- function(dir, expected_record_sha256 = NULL, require_read_only = FALSE, primary_dir = NULL) {
  opened <- open_published(dir, SUPPLEMENT_FILES, SUPPLEMENT_SUMMARY_RECORD, "supplement summary", expected_record_sha256, require_read_only)
  paths <- opened$paths; sealed <- opened$sealed; record <- opened$record; record_sha256 <- opened$record_sha256
  claims <- list(
    list("kind", SUPPLEMENT_SUMMARY_KIND), list("schema_version", "1"), list("purpose", "synthetic_rehearsal"),
    list("scientific_authority", "none"), list("banner", SYNTHETIC_BANNER), list("analysis_label", SUPPLEMENT_LABEL),
    list(c("inputs", "input_origin"), "synthetic_rehearsal"))
  check_claims(record, claims)
  scientific <- get_path(record, "scientific_files")
  if (!is.list(scientific) || !setequal(names(scientific), SUPPLEMENT_DATA_FILES)) verify_error("the record must list exactly the supplement summary tables")
  bytes <- lapply(stats::setNames(SUPPLEMENT_DATA_FILES, SUPPLEMENT_DATA_FILES), function(n) read_bytes(paths[[n]]))
  check_listed(scientific, bytes)
  id <- tryCatch(supplement_summary_id(record$inputs$handoff_sha256, record$inputs$csv_sha256, record$inputs$supplement_sha256,
                                       vapply(scientific, function(e) e$sha256, "")),
                 error = function(e) verify_error("the identity inputs of the record are malformed"))
  if (!identical(id, record$supplement_summary_sha256)) verify_error("supplement_summary_sha256 does not match the files and inputs")
  tables <- lapply(stats::setNames(SUPPLEMENT_DATA_FILES, SUPPLEMENT_DATA_FILES),
                   function(n) parse_canonical_csv(bytes[[n]], SUPPLEMENT_SCHEMAS[[n]], n))
  check_supplement_tables(tables)
  if (!is_sha256_hex(get_path(record, c("inputs", "supplement_record_sha256")))) verify_error("the cited supplement record hash is malformed")
  cited <- get_path(record, c("primary_result", "run_record_sha256"))
  if (!is.null(primary_dir)) {
    if (!is_sha256_hex(cited)) verify_error("the record cites no primary result to check against")
    check_cited_primary(record, primary_dir, cited)
  }
  list(ok = TRUE, sealed = sealed, record_sha256 = record_sha256, record_authenticated = !is.null(expected_record_sha256),
       supplement_summary_sha256 = id, record = record, tables = tables)
}

# Recompute from the original inputs and the Python supplement and require byte equality.
verify_supplement_rerun <- function(dir, handoff_path, handoff_sha256, csv_path, supplement_dir, primary_dir = NULL) {
  v <- verify_supplement_summary(dir, primary_dir = primary_dir)
  if (!identical(v$record$inputs$handoff_sha256, handoff_sha256)) verify_error("the supplied handoff hash differs from the record")
  fresh <- compute_supplement(handoff_path, handoff_sha256, csv_path, supplement_dir, v$record$inputs$supplement_record_sha256)
  if (!identical(fresh$summary_sha256, v$supplement_summary_sha256)) verify_error("re-execution gives a different supplement summary")
  for (n in SUPPLEMENT_DATA_FILES) {
    if (!identical(read_bytes(file.path(dir, n)), fresh$files[[n]])) verify_error(sprintf("re-execution differs in %s", n))
  }
  list(comparison = "byte_identical_tables", supplement_summary_sha256 = v$supplement_summary_sha256)
}
