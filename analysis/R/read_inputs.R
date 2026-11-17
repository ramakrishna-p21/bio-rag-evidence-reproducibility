# Read and validate the Python-authored analysis table and handoff.
#
# What this establishes: the handoff file and CSV are the exact bytes pinned by a hash the
# caller supplies, the CSV obeys the reading contract the handoff states, and every row is a
# development row consistent with the handoff. What it does not establish: lineage, official
# origin, or that the contract matches Python's row model. Those need the Python preflight
# (`biorag-analysis verify-handoff` for a rehearsal) run before R; R only records that it did
# not verify them. Nothing here rescores any Stage A/B/C judgment.

input_error <- function(step, message) {
  stop(structure(
    class = c("input_error", "error", "condition"),
    list(message = sprintf("[%s] %s", step, message), call = NULL)
  ))
}

sha256_file <- function(path) {
  digest::digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
}

# ---- frozen expectations (EXPERIMENT_PLAN 8-9, REPRODUCIBILITY_SPEC 8; mirrored from the
# Python ANALYSIS_SPEC so a changed handoff is refused rather than silently followed) ---------

EXPECTED_SPEC <- list(
  spec_version = "analysis_spec_v1",
  contrast = "JATS - PDF_DOCLING",
  direction = "JATS_minus_PDF_DOCLING",
  estimand = "item_weighted_paired_risk_difference",
  pair_key = "item_id",
  cluster_key = "paper_id",
  bootstrap_method = "whole_paper_percentile",
  bootstrap_resamples = 10000L,
  interval_percent = 95L,
  bootstrap_seed_name = "primary_bootstrap",
  model = "intercept_only_paired_differences",
  variance = "CR2",
  test = "satterthwaite_wald_two_sided",
  package = "clubSandwich",
  multiplicity = "holm"
)
EXPECTED_FAMILY <- c("evidence_delivered", "answer_correct")
EXPECTED_SENSITIVITY <- c("equal_paper_weighted", "best_worst_case_bounds")

EXPECTED_CONTRACT <- list(
  encoding = "utf-8", byte_order_mark = FALSE, line_ending = "\n", delimiter = ",",
  quote = "\"", quote_escape = "doubled", header_row = TRUE, missing_token = "NA",
  integer_pattern = "^(0|[1-9][0-9]*)$", sort_collation = "unicode_code_point"
)
EXPECTED_SORT_KEY <- c("matrix_arm", "item_id", "condition", "repeat_index", "observation_key")

MISSING_TOKEN <- "NA"
# Vocabularies of the columns the analysis relies on. The two outcome vocabularies come from
# the handoff rules; these are the rest.
TRI_STATE <- c("TRUE", "FALSE", "NA")
CONDITIONS <- c("JATS", "PDF_DOCLING", "CLOSED_BOOK")
MATRIX_ARMS <- c("confirmatory", "dense_robustness", "chunk_robustness", "scope_robustness", "closed_book")
EXECUTION_STATUSES <- c("planned_not_executed", "operational_failure", "completed")

# ---- column-list hash -----------------------------------------------------------------------

# Canonical JSON (RFC 8785 for this shape): objects with keys sorted, no whitespace. Column
# names are restricted to ASCII identifiers, so no string escaping is needed.
columns_canonical_json <- function(columns) {
  cells <- vapply(columns, function(column) {
    name <- column$name
    type <- column$type
    if (!is.character(name) || length(name) != 1 || !grepl("^[a-z][a-z0-9_]*$", name)) {
      input_error("columns_hash", "column name is not a plain ASCII identifier")
    }
    if (!type %in% c("integer", "logical", "string")) {
      input_error("columns_hash", sprintf("column %s has unknown type %s", name, type))
    }
    if (!is.logical(column$nullable) || length(column$nullable) != 1 || is.na(column$nullable)) {
      input_error("columns_hash", sprintf("column %s has no nullable flag", name))
    }
    sprintf('{"name":"%s","nullable":%s,"type":"%s"}', name, tolower(column$nullable), type)
  }, character(1))
  paste0("[", paste(cells, collapse = ","), "]")
}

columns_sha256 <- function(columns) {
  digest::digest(columns_canonical_json(columns), algo = "sha256", serialize = FALSE)
}

# ---- handoff --------------------------------------------------------------------------------

same_values <- function(a, b) identical(unname(unlist(a)), unname(b))

# Execute the fixed repository verifier; a citation or status flag is insufficient.
read_official_proof <- function(request, python, project_dir) {
  if (is.null(request) || is.null(python) || is.null(project_dir)) {
    input_error("origin", "an official origin requires a verification request, Python and project directory")
  }
  script <- file.path(project_dir, "..", "scripts", "evaluate_development.py")
  if (!file.exists(script) || !file.exists(request) || !file.exists(python)) {
    input_error("origin", "official verification inputs are unavailable")
  }
  errors <- tempfile("official-preflight-", fileext = ".log")
  output <- suppressWarnings(system2(normalizePath(python),
    c(shQuote(normalizePath(script)), "--verify-request", shQuote(normalizePath(request))),
    stdout = TRUE, stderr = errors))
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    input_error("origin", sprintf("Python refused official provenance; failure evidence: %s", errors))
  }
  proof <- tryCatch(jsonlite::fromJSON(paste(output, collapse = "\n"), simplifyVector = FALSE),
                    error = function(e) input_error("origin", "Python returned no valid verification result"))
  if (!identical(proof$status, "verified_official_development_handoff") ||
      !identical(proof$source_verification, "fresh_python_verification") ||
      proof$observations != 1290L) {
    input_error("origin", "Python did not establish the complete official development origin")
  }
  proof$request_sha256 <- sha256_file(request)
  proof
}

check_handoff <- function(handoff, official_proof = NULL) {
  if (!identical(handoff$kind, "analysis_handoff") || !identical(handoff$schema_version, "1") ||
      !identical(handoff$phase, "development")) {
    input_error("handoff_schema", "not a version 1 development analysis_handoff")
  }
  if (!identical(handoff$input_origin, "synthetic_rehearsal") &&
      !(identical(handoff$input_origin, "verified_official_run") && !is.null(official_proof))) {
    input_error("origin", sprintf("input_origin '%s' lacks fresh verification", handoff$input_origin))
  }
  spec <- handoff$analysis_spec
  for (name in names(EXPECTED_SPEC)) {
    if (!identical(spec[[name]], EXPECTED_SPEC[[name]])) {
      input_error("analysis_spec", sprintf("%s differs from the frozen method", name))
    }
  }
  if (!same_values(spec$multiplicity_family, EXPECTED_FAMILY)) {
    input_error("analysis_spec", "multiplicity family is not exactly the two co-primary outcomes")
  }
  if (!same_values(spec$sensitivity, EXPECTED_SENSITIVITY)) {
    input_error("analysis_spec", "sensitivity analyses differ from the frozen method")
  }
  outcomes <- spec$outcomes
  if (!same_values(lapply(outcomes, `[[`, "name"), EXPECTED_FAMILY) ||
      !all(vapply(outcomes, function(o) identical(o$success, "yes") && identical(o$failure, "no") &&
                    identical(o$name, o$column), logical(1)))) {
    input_error("analysis_spec", "outcome rules differ from the frozen yes/no definition")
  }
  contract <- handoff$csv_contract
  for (name in names(EXPECTED_CONTRACT)) {
    if (!identical(contract[[name]], EXPECTED_CONTRACT[[name]])) {
      input_error("csv_contract", sprintf("%s differs from the reading contract", name))
    }
  }
  if (!same_values(contract$logical_tokens, c("TRUE", "FALSE")) ||
      !same_values(contract$sort_key, EXPECTED_SORT_KEY)) {
    input_error("csv_contract", "logical tokens or sort key differ from the reading contract")
  }
  if (!identical(columns_sha256(contract$columns), contract$columns_sha256)) {
    input_error("columns_hash", "the handoff's column list does not reproduce its recorded hash")
  }
  names <- vapply(contract$columns, `[[`, "", "name")
  if (anyDuplicated(names)) input_error("csv_contract", "duplicate column names")
  if (!all(unlist(contract$na_value_columns) %in% names)) {
    input_error("csv_contract", "na_value_columns names an unknown column")
  }
  invisible(handoff)
}

read_handoff_file <- function(path, expected_sha256, official_request = NULL, python = NULL, project_dir = NULL) {
  if (!is.character(expected_sha256) || length(expected_sha256) != 1 ||
      !grepl("^[0-9a-f]{64}$", expected_sha256)) {
    input_error("handoff_hash", "the pinned handoff hash is not a SHA-256")
  }
  if (!identical(sha256_file(path), expected_sha256)) {
    input_error("handoff_hash", "handoff bytes differ from the pinned SHA-256")
  }
  handoff <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  proof <- NULL
  if (identical(handoff$input_origin, "verified_official_run")) {
    proof <- read_official_proof(official_request, python, project_dir)
    if (!identical(proof$handoff$sha256, expected_sha256) ||
        !identical(proof$csv_sha256, handoff$csv$sha256)) {
      input_error("origin", "Python verified different handoff or table bytes")
    }
  } else if (!is.null(official_request)) {
    input_error("origin", "a synthetic rehearsal cannot claim an official verification request")
  }
  check_handoff(handoff, proof)
  if (!is.null(proof)) attr(handoff, "official_proof") <- proof
  invisible(handoff)
}

# ---- table ----------------------------------------------------------------------------------

check_bytes <- function(path) {
  raw <- readBin(path, "raw", file.size(path))
  if (length(raw) == 0) input_error("bytes", "the table is empty")
  if (length(raw) >= 3 && identical(raw[1:3], as.raw(c(0xef, 0xbb, 0xbf)))) {
    input_error("bytes", "byte order mark")
  }
  if (any(raw == as.raw(0x0d))) input_error("bytes", "carriage return (LF line endings only)")
  if (any(raw == as.raw(0x00)) || !validUTF8(rawToChar(raw[raw != as.raw(0x00)]))) {
    input_error("bytes", "not valid UTF-8")
  }
  if (raw[length(raw)] != as.raw(0x0a)) input_error("bytes", "no final line feed")
  invisible(TRUE)
}

parse_cells <- function(values, column, na_data) {
  name <- column$name
  if (any(values == "")) input_error("empty_cell", sprintf("%s has an empty cell", name))
  is_token <- values == MISSING_TOKEN
  if (any(is_token) && !column$nullable && !name %in% na_data) {
    input_error("missing_token", sprintf("%s is not nullable but has the missing token", name))
  }
  if (column$type == "logical") {
    if (!all(values[!is_token] %in% c("TRUE", "FALSE"))) {
      input_error("type", sprintf("%s: logical values are TRUE or FALSE", name))
    }
    parsed <- values == "TRUE"
  } else if (column$type == "integer") {
    ok <- grepl(EXPECTED_CONTRACT$integer_pattern, values[!is_token])
    parsed <- rep(NA_integer_, length(values))
    if (!all(ok)) input_error("type", sprintf("%s: not a canonical integer", name))
    converted <- suppressWarnings(as.integer(values[!is_token]))
    if (anyNA(converted)) input_error("type", sprintf("%s: integer out of range", name))
    parsed[!is_token] <- converted
  } else {
    parsed <- values
  }
  if (column$nullable) parsed[is_token] <- NA
  parsed
}

check_vocabulary <- function(rows, handoff) {
  allowed <- list(
    correct_without_evidence = TRI_STATE, condition = CONDITIONS, matrix_arm = MATRIX_ARMS,
    execution_status = EXECUTION_STATUSES
  )
  for (outcome in handoff$analysis_spec$outcomes) {
    allowed[[outcome$column]] <- c(outcome$success, outcome$failure, unlist(outcome$non_evaluable))
  }
  for (name in names(allowed)) {
    if (!all(rows[[name]] %in% allowed[[name]])) {
      input_error("vocabulary", sprintf("%s has a value outside its vocabulary", name))
    }
  }
}

check_purpose <- function(rows, handoff) {
  # Development rehearsal data only; qualification, final-test or confirmatory-role rows are
  # never mixed in.
  if (!all(rows$phase == "development" & rows$split == "development" &
             rows$analysis_role == "development")) {
    input_error("purpose", "rows are not all development-phase, development-split, development-role")
  }
  if (length(unique(rows$benchmark_version)) != 1 || rows$benchmark_version[1] != handoff$benchmark_version ||
      length(unique(rows$corpus_version)) != 1 || rows$corpus_version[1] != handoff$corpus_version) {
    input_error("version", "rows differ from the handoff's benchmark or corpus version")
  }
  arms <- handoff$arms
  for (arm in unique(rows$matrix_arm)) {
    pinned <- Filter(function(a) identical(a$matrix_arm, arm), arms)
    arm_rows <- rows[rows$matrix_arm == arm, ]
    configs <- unique(paste(arm_rows$config_sha256, ifelse(is.na(arm_rows$retrieval_config_sha256), "NA", arm_rows$retrieval_config_sha256)))
    expected <- if (length(pinned) == 1) paste(pinned[[1]]$resolved_config_sha256,
                                               if (is.null(pinned[[1]]$retrieval_config_sha256)) "NA" else pinned[[1]]$retrieval_config_sha256)
    if (length(pinned) != 1 || !identical(configs, expected)) {
      input_error("arm_config", sprintf("matrix arm %s differs from the handoff's configuration", arm))
    }
  }
}

read_table <- function(csv_path, handoff) {
  contract <- handoff$csv_contract
  if (!identical(sha256_file(csv_path), handoff$csv$sha256)) {
    input_error("csv_hash", "table bytes differ from the SHA-256 pinned in the handoff")
  }
  check_bytes(csv_path)
  columns <- contract$columns
  expected_names <- vapply(columns, `[[`, "", "name")
  header <- strsplit(readLines(csv_path, n = 1, encoding = "UTF-8", warn = FALSE), ",", fixed = TRUE)[[1]]
  if (!identical(header, expected_names)) {
    input_error("header", "table header differs from the contract's column names and order")
  }
  counts <- utils::count.fields(csv_path, sep = ",", quote = "\"", comment.char = "", blank.lines.skip = FALSE)
  if (anyNA(counts) || any(counts != length(expected_names))) {
    input_error("cell_count", "a line does not have exactly one cell per column")
  }
  raw <- utils::read.csv(
    csv_path, header = TRUE, colClasses = "character", na.strings = character(0),
    check.names = FALSE, comment.char = "", quote = "\"", stringsAsFactors = FALSE,
    fileEncoding = "UTF-8", strip.white = FALSE, blank.lines.skip = FALSE
  )
  if (nrow(raw) != contract$row_count) {
    input_error("row_count", sprintf("table has %d rows, the handoff states %d", nrow(raw), contract$row_count))
  }
  na_data <- unlist(contract$na_value_columns)
  rows <- raw
  for (column in columns) {
    rows[[column$name]] <- parse_cells(raw[[column$name]], column, na_data)
  }
  check_vocabulary(rows, handoff)
  if (anyDuplicated(rows$observation_key) ||
      anyDuplicated(rows[c("matrix_arm", "item_id", "condition", "repeat_index")])) {
    input_error("duplicate_key", "duplicate observation key or duplicate (arm, item, condition, repeat)")
  }
  check_purpose(rows, handoff)
  ordering <- order(rows$matrix_arm, rows$item_id, rows$condition, rows$repeat_index,
                    rows$observation_key, method = "radix")
  if (!identical(ordering, seq_len(nrow(rows)))) {
    input_error("row_order", "rows are not in the contract's sort order")
  }
  rownames(rows) <- NULL
  rows
}

# The entry point of the reader. `lineage_verified_by_r` is always FALSE: lineage and official
# origin are Python's to verify before R runs.
read_inputs <- function(handoff_path, handoff_sha256, csv_path, official_request = NULL, python = NULL, project_dir = NULL) {
  handoff <- read_handoff_file(handoff_path, handoff_sha256, official_request, python, project_dir)
  proof <- attr(handoff, "official_proof")
  attr(handoff, "official_proof") <- NULL
  rows <- read_table(csv_path, handoff)
  out <- list(
    handoff = handoff, rows = rows, input_origin = handoff$input_origin,
    handoff_sha256 = handoff_sha256, csv_sha256 = handoff$csv$sha256,
    lineage_verified_by_r = FALSE
  )
  if (!is.null(proof)) {
    out$official_proof <- proof
    out$official_request_sha256 <- proof$request_sha256
  }
  out
}
