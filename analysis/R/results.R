# Statistical result tables and their deterministic serialization.
#
# This file only arranges what the inference functions already computed. Nothing here estimates,
# resamples or tests anything. Values stay full-precision doubles until the moment they are
# written; the writer is the only place that decides how a number becomes bytes.

run_error <- function(step, message) {
  stop(structure(
    class = c("run_error", "error", "condition"),
    list(message = sprintf("[%s] %s", step, message), call = NULL)
  ))
}

sha256_raw <- function(bytes) digest::digest(bytes, algo = "sha256", serialize = FALSE)

# ---- column schemas ---------------------------------------------------------------------------
# A cell is "chr" (text; NA is the missing token), "int" or "dbl". The order is the file order.

COPRIMARY_SCHEMA <- c(
  outcome = "chr", n_eligible = "int", n_complete = "int", n_papers = "int", summary_status = "chr",
  rate_jats = "dbl", rate_pdf = "dbl", rd = "dbl", rd_pp = "dbl", equal_paper_rd = "dbl",
  boot_status = "chr", boot_n_resamples = "int", boot_n_empty = "int",
  boot_ci_lower = "dbl", boot_ci_upper = "dbl",
  test_status = "chr", test_reason = "chr", test_se = "dbl", test_df = "dbl", test_t = "dbl", test_p = "dbl",
  holm_status = "chr", holm_p = "dbl",
  bound_lower = "dbl", bound_upper = "dbl", bound_n_unknown_jats = "int", bound_n_unknown_pdf = "int",
  bound_n_operational_excluded = "int"
)
PAPER_COUNTS_SCHEMA <- c(outcome = "chr", paper_id = "chr", eligible = "int", complete = "int")
STATE_COUNTS_SCHEMA <- c(outcome = "chr", condition = "chr", state = "chr", n = "int")
INVENTORY_SCHEMA <- c(package = "chr", version = "chr", priority = "chr", library_role = "chr", in_lock = "chr")

# The three scientific files (a function of the inputs and the seed alone) and their schemas.
SCIENTIFIC_SCHEMAS <- list(
  "coprimary_results.csv" = COPRIMARY_SCHEMA,
  "paper_counts.csv" = PAPER_COUNTS_SCHEMA,
  "state_counts.csv" = STATE_COUNTS_SCHEMA
)

# ---- serialization ----------------------------------------------------------------------------

# 17 significant digits reproduce every finite double exactly. Negative zero is written as 0.
# NA is the project missing token; NaN and infinities are never published.
format_double <- function(x) {
  if (length(x) != 1 || !is.double(x)) run_error("serialize", "a double cell must be one double")
  if (is.nan(x)) run_error("serialize", "NaN cannot be published; an undefined value must be NA with a status")
  if (is.na(x)) return("NA")
  if (!is.finite(x)) run_error("serialize", "an infinite value cannot be published")
  if (x == 0) return("0")
  sprintf("%.17g", x)
}

format_cell <- function(x, type, column) {
  if (length(x) != 1) run_error("serialize", sprintf("column %s: one value per cell", column))
  if (type == "chr") {
    if (!is.character(x)) run_error("serialize", sprintf("column %s must be character", column))
    if (is.na(x)) return("NA")
    # A literal "NA" would read back as missing; an empty string or a line break would break the file.
    if (x == "NA" || x == "" || grepl("[\r\n]", x) || !validUTF8(x)) {
      run_error("serialize", sprintf("column %s holds a value that cannot be written unambiguously", column))
    }
    if (grepl("[,\"]", x)) return(paste0("\"", gsub("\"", "\"\"", x, fixed = TRUE), "\""))
    return(x)
  }
  if (type == "int") {
    if (!(is.integer(x) || is.double(x)) || (!is.na(x) && (!is.finite(x) || x != round(x) || abs(x) > .Machine$integer.max))) {
      run_error("serialize", sprintf("column %s must hold whole numbers", column))
    }
    return(if (is.na(x)) "NA" else sprintf("%d", as.integer(x)))
  }
  if (type == "dbl") {
    if (!is.double(x)) run_error("serialize", sprintf("column %s must be double", column))
    return(format_double(x))
  }
  run_error("serialize", sprintf("unknown column type %s", type))
}

# UTF-8, no BOM, LF line endings, header, comma separator, "NA" for missing.
csv_bytes <- function(df, schema) {
  if (!is.data.frame(df) || !identical(names(df), names(schema))) {
    run_error("serialize", "table columns differ from the schema")
  }
  lines <- paste(names(schema), collapse = ",")
  for (i in seq_len(nrow(df))) {
    cells <- vapply(seq_along(schema), function(j) format_cell(df[[j]][i], schema[[j]], names(schema)[j]), "")
    lines <- c(lines, paste(cells, collapse = ","))
  }
  charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
}

# The `files` array of an identity: {"name", "sha256"} objects in radix order of the name, as in
# Python's canonical JSON.
files_json <- function(file_sha256) {
  ordered <- file_sha256[order(names(file_sha256), method = "radix")]
  paste(sprintf("{\"name\":\"%s\",\"sha256\":\"%s\"}", names(ordered), unlist(ordered, use.names = FALSE)),
        collapse = ",")
}

# ---- identity of the scientific result --------------------------------------------------------

# SHA-256 of canonical JSON (keys sorted, no whitespace): the inputs, the seed and the hash of each
# scientific file. Execution facts (time, host, run id) are deliberately not part of it.
scientific_result_id <- function(handoff_sha256, csv_sha256, seed, file_sha256) {
  names_ok <- !is.null(names(file_sha256)) && all(grepl("^[A-Za-z0-9_.]+$", names(file_sha256)))
  if (!is_sha256_hex(handoff_sha256) || !is_sha256_hex(csv_sha256) || !names_ok || !all(vapply(file_sha256, is_sha256_hex, logical(1)))) {
    run_error("identity", "identity inputs must be SHA-256 hex strings and plain file names")
  }
  if (length(seed) != 1 || !is.numeric(seed) || !is.finite(seed) || seed != round(seed)) {
    run_error("identity", "seed must be one integer")
  }
  files <- files_json(file_sha256)
  json <- sprintf("{\"csv_sha256\":\"%s\",\"files\":[%s],\"handoff_sha256\":\"%s\",\"seed\":%s}",
                  csv_sha256, files, handoff_sha256, sprintf("%d", as.integer(seed)))
  sha256_raw(charToRaw(json))
}

# ---- tables -----------------------------------------------------------------------------------

coprimary_table <- function(accounting, inference, handoff) {
  rules <- outcome_rules(handoff)
  rows <- lapply(EXPECTED_FAMILY, function(outcome) {
    items <- accounting[[outcome]]$items
    o <- inference$outcomes[[outcome]]
    bounds <- missingness_bounds(items$jats_state, items$pdf_state, rules[[outcome]])
    s <- o$summary; b <- o$bootstrap; t <- o$test
    data.frame(
      outcome = outcome, n_eligible = nrow(items), n_complete = s$n_items, n_papers = s$n_papers,
      summary_status = s$status, rate_jats = s$rate_jats, rate_pdf = s$rate_pdf, rd = s$rd, rd_pp = s$rd_pp,
      equal_paper_rd = s$equal_paper_rd,
      boot_status = b$status, boot_n_resamples = b$n_resamples, boot_n_empty = b$n_empty,
      boot_ci_lower = b$ci_lower, boot_ci_upper = b$ci_upper,
      test_status = t$status, test_reason = t$reason, test_se = t$se, test_df = t$df, test_t = t$t, test_p = t$p,
      holm_status = inference$holm$status, holm_p = unname(inference$holm$adjusted[[outcome]]),
      bound_lower = bounds$lower, bound_upper = bounds$upper,
      bound_n_unknown_jats = bounds$n_unknown_jats, bound_n_unknown_pdf = bounds$n_unknown_pdf,
      bound_n_operational_excluded = bounds$n_operational_excluded,
      stringsAsFactors = FALSE
    )
  })
  out <- do.call(rbind, rows)
  rownames(out) <- NULL
  out
}

paper_counts_table <- function(accounting) {
  out <- do.call(rbind, lapply(EXPECTED_FAMILY, function(outcome) {
    cbind(outcome = rep(outcome, nrow(accounting[[outcome]]$papers)), accounting[[outcome]]$papers)
  }))
  out$outcome <- as.character(out$outcome)
  rownames(out) <- NULL
  out[, names(PAPER_COUNTS_SCHEMA), drop = FALSE]
}

# The accounting keeps "CONDITION|state" keys in radix order; they are split at the first bar.
state_counts_table <- function(accounting) {
  out <- do.call(rbind, lapply(EXPECTED_FAMILY, function(outcome) {
    counts <- accounting[[outcome]]$state_counts
    key <- names(counts)
    bar <- regexpr("|", key, fixed = TRUE)
    if (any(bar < 2)) run_error("serialize", "state count key without a condition")
    data.frame(outcome = outcome, condition = substr(key, 1, bar - 1), state = substr(key, bar + 1, nchar(key)),
               n = unname(as.integer(counts)), stringsAsFactors = FALSE)
  }))
  rownames(out) <- NULL
  out
}

# ---- the computation, from pinned inputs to hashed bytes ---------------------------------------

# The reader verifies input bytes and asks Python to verify official source lineage.
# Repeat the origin guard here before any statistical computation.
compute_results <- function(handoff_path, handoff_sha256, csv_path, seed, official_request = NULL, python = NULL, project_dir = NULL) {
  inputs <- read_inputs(handoff_path, handoff_sha256, csv_path, official_request, python, project_dir)
  if (!inputs$input_origin %in% c("synthetic_rehearsal", "verified_official_run")) {
    run_error("origin", sprintf("input_origin '%s' is not accepted; only a synthetic rehearsal is", inputs$input_origin))
  }
  if (identical(inputs$input_origin, "verified_official_run") &&
      (is.null(inputs$official_proof) || is.null(official_request) ||
       !identical(inputs$handoff$input_origin, "verified_official_run"))) {
    run_error("origin", "official inputs lack fresh Python verification")
  }
  if (!is.null(inputs$handoff$scoring_reference$primary_bootstrap_registry_sha256)) {
    seed_record(seed, origin_labels(TRUE, identical(inputs$handoff$phase, "final_test"))$seed_label,
                inputs$handoff, inputs$official_proof$scientific_bootstrap_seed)
  }
  accounting <- derive_accounting(inputs$rows, inputs$handoff)
  inference <- coprimary_inference(accounting, inputs$handoff, seed)
  tables <- list(
    "coprimary_results.csv" = coprimary_table(accounting, inference, inputs$handoff),
    "paper_counts.csv" = paper_counts_table(accounting),
    "state_counts.csv" = state_counts_table(accounting)
  )
  files <- Map(csv_bytes, tables, SCIENTIFIC_SCHEMAS)
  file_sha256 <- vapply(files, sha256_raw, "")
  list(
    inputs = inputs, accounting = accounting, inference = inference, files = files, file_sha256 = file_sha256,
    scientific_result_sha256 = scientific_result_id(handoff_sha256, inputs$csv_sha256, seed, file_sha256)
  )
}
