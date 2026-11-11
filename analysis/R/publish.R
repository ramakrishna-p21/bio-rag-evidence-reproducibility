# Run record, immutable publication, and the integrity verifier.
#
# A published result is a directory of five files. Four are data (three scientific CSVs and the
# package inventory); the fifth, run_record.json, lists their hashes and is written last. The
# record cannot contain its own hash: its SHA-256 is its external identity. Checking a directory
# against itself shows internal consistency only; the record is authenticated only when a trusted
# SHA-256 of it is supplied from outside.

RECORD_FILE <- "run_record.json"
INVENTORY_FILE <- "package_inventory.csv"
DATA_FILES <- c(names(SCIENTIFIC_SCHEMAS), INVENTORY_FILE)
RESULT_FILES <- c(DATA_FILES, RECORD_FILE)
RECORD_KIND <- "r_run_record"
RECORD_SCHEMA_VERSION <- "1"

# What a run record says when a Python preflight was cited. R hashed the file's bytes and recorded
# the hash; it did not read the contents as evidence and did not repeat any Python verification.
PREFLIGHT_NOT_RECORDED <- "not_recorded"
PREFLIGHT_CITED <- "cited_bytes_hashed_not_interpreted"
PREFLIGHT_SCOPE <- "bytes_hashed_by_r_only_contents_not_read_python_verification_not_repeated"

# The nine facts the handoff requires from the R run, and where the record states each.
REQUIRED_PROVENANCE <- list(
  analysis_script_git_sha = c("code", "git_commit"),
  analysis_script_sha256 = c("code", "script_sha256"),
  applied_bootstrap_seed = c("seed", "value"),
  handoff_sha256 = c("inputs", "handoff_sha256"),
  installed_package_inventory_sha256 = c("environment_actual", "installed_package_inventory_sha256"),
  output_hashes = c("outputs", "files"),
  package_versions_used = c("environment_actual", "package_versions_used"),
  r_dependency_lock_sha256 = c("environment_intended", "renv_lock_sha256"),
  r_version = c("environment_actual", "r_version")
)

verify_error <- function(message) run_error("verify", message)

read_bytes <- function(path) readBin(path, "raw", file.size(path))

iso_utc <- function(t) format(t, "%Y-%m-%dT%H:%M:%OS3Z", tz = "UTC")

is_sha256_hex <- function(x) is.character(x) && length(x) == 1 && !is.na(x) && grepl("^[0-9a-f]{64}$", x)

# Hash the bytes of a cited Python preflight. Both the path and the SHA-256 are required, the file
# must be a non-empty regular file, and its bytes must match the hash. The contents are not read
# as evidence: this proves only that these bytes existed and were the ones named.
read_preflight_citation <- function(path, sha256) {
  if (is.null(path) && is.null(sha256)) return(NULL)
  if (is.null(path) || is.null(sha256)) run_error("preflight", "a preflight needs both its path and its SHA-256")
  if (!is_sha256_hex(sha256)) run_error("preflight", "the preflight SHA-256 must be 64 lowercase hexadecimal characters")
  if (!is.character(path) || length(path) != 1 || is.na(path) || !nzchar(path)) run_error("preflight", "the preflight path must be one file path")
  link <- Sys.readlink(path)                                       # NA when the path does not exist
  is_link <- !is.na(link) && nzchar(link)
  if (!file.exists(path) && !is_link) run_error("preflight", "preflight file not found")
  if (is_link || dir.exists(path)) run_error("preflight", "the preflight must be a regular file")
  bytes <- read_bytes(path)
  if (!length(bytes)) run_error("preflight", "the preflight file is empty")
  actual <- sha256_raw(bytes)
  if (!identical(actual, sha256)) {
    run_error("preflight", sprintf("the preflight bytes (sha256 %s) differ from the SHA-256 supplied (%s)", actual, sha256))
  }
  list(sha256 = actual, bytes = length(bytes))
}

# ---- the record -------------------------------------------------------------------------------

build_run_record <- function(results, seed, lock, environment, inventory_sha256, used_versions, code, execution, preflight = NULL) {
  handoff <- results$inputs$handoff
  files <- lapply(c(results$files, list("package_inventory.csv" = results$inventory_bytes)),
                  function(b) list(bytes = length(b), sha256 = sha256_raw(b)))
  record <- list(
    kind = RECORD_KIND, schema_version = RECORD_SCHEMA_VERSION,
    purpose = "synthetic_rehearsal", scientific_authority = "none",
    verification = list(
      input_origin = results$inputs$input_origin,
      r_byte_verification = "handoff_and_csv_bytes_matched_supplied_sha256_and_reading_contract",
      python_preflight = if (is.null(preflight)) PREFLIGHT_NOT_RECORDED else PREFLIGHT_CITED,
      lineage_verified_by_r = FALSE,
      official_source_runs = "not_applicable_synthetic_rehearsal",
      statistical_execution = "completed",
      output_integrity = "hashes_recorded_in_this_record_only",
      record_authentication = "none_without_external_record_sha256"
    ),
    inputs = list(
      handoff_sha256 = results$inputs$handoff_sha256, csv_sha256 = results$inputs$csv_sha256,
      input_origin = results$inputs$input_origin, handoff_kind = handoff$kind,
      handoff_schema_version = handoff$schema_version, analysis_spec_version = handoff$analysis_spec$spec_version
    ),
    seed = seed,
    environment_intended = list(r_version_required = REQUIRED_R_VERSION, renv_lock_sha256 = lock$sha256,
                                lock_packages = as.list(lock$packages)),
    environment_actual = c(environment, list(
      renv_synchronized = TRUE, installed_package_inventory_sha256 = inventory_sha256,
      package_versions_used = as.list(used_versions)
    )),
    code = list(
      status = code$status, git_commit = code$commit, dirty = code$dirty,
      tracked_modified = as.list(code$tracked_modified),
      tracked_diff_sha256 = if (is.null(code$tracked_diff_sha256)) NA_character_ else code$tracked_diff_sha256,
      untracked_source = code$untracked_source, script_sha256 = as.list(code$script_sha256)
    ),
    execution = execution,
    outputs = list(files = files, scientific_result_sha256 = results$scientific_result_sha256),
    provenance = list(required = as.list(unlist(handoff$execution_provenance_required)))
  )
  if (!is.null(preflight)) {
    at <- match("python_preflight", names(record$verification))
    record$verification <- append(record$verification, list(python_preflight_scope = PREFLIGHT_SCOPE), after = at)
    record$inputs <- c(record$inputs, list(python_preflight_sha256 = preflight$sha256,
                                           python_preflight_bytes = as.integer(preflight$bytes)))
  }
  record
}

serialize_record <- function(record) {
  json <- jsonlite::toJSON(record, auto_unbox = TRUE, pretty = TRUE, null = "null", na = "null", digits = NA)
  charToRaw(paste0(as.character(json), "\n"))
}

# ---- strict reading of the published CSV files -------------------------------------------------

# Parse a file written by csv_bytes and require that writing the parsed table again reproduces the
# same bytes. That rejects any other number format, quoting, blank line or line ending.
parse_canonical_csv <- function(bytes, schema, name) {
  bad <- function(why) verify_error(sprintf("%s: %s", name, why))
  text <- tryCatch(rawToChar(bytes), error = function(e) bad("not valid text"))
  if (!validUTF8(text) || !endsWith(text, "\n") || grepl("\r", text, fixed = TRUE) ||
      identical(bytes[1:min(3, length(bytes))], as.raw(c(0xef, 0xbb, 0xbf)))) bad("not UTF-8 text with LF endings")
  raw_df <- tryCatch(
    utils::read.csv(text = text, colClasses = "character", na.strings = character(0), stringsAsFactors = FALSE,
                    check.names = FALSE, comment.char = "", quote = "\""),
    error = function(e) bad("cannot be parsed as CSV"))
  if (!identical(names(raw_df), names(schema))) bad("columns differ from the schema")
  typed <- raw_df
  for (j in seq_along(schema)) {
    col <- raw_df[[j]]
    typed[[j]] <- switch(schema[[j]],
      chr = ifelse(col == "NA", NA_character_, col),
      int = {
        if (!all(col == "NA" | grepl("^-?[0-9]+$", col))) bad(sprintf("column %s is not integer", names(schema)[j]))
        suppressWarnings(as.integer(ifelse(col == "NA", NA_character_, col)))
      },
      dbl = {
        v <- suppressWarnings(as.numeric(ifelse(col == "NA", NA_character_, col)))
        if (any(is.na(v) & col != "NA") || any(is.infinite(v), na.rm = TRUE)) bad(sprintf("column %s is not finite numeric", names(schema)[j]))
        v
      })
  }
  again <- tryCatch(csv_bytes(typed, schema), run_error = function(e) bad("cannot be written back unambiguously"))
  if (!identical(again, bytes)) bad("is not in the canonical form")
  typed
}

# ---- consistency of the tables with one another ------------------------------------------------

check_tables <- function(co, pc, sc) {
  bad <- function(why) verify_error(why)
  if (!identical(co$outcome, EXPECTED_FAMILY)) bad("coprimary_results.csv must hold the two co-primary outcomes in order")
  in_set <- function(x, set) all(x %in% set)
  if (!in_set(co$summary_status, c("ok", "no_complete_pairs")) ||
      !in_set(co$boot_status, c("ok", "single_cluster_degenerate", "no_complete_pairs")) ||
      !in_set(co$test_status, c("ok", "inference_undefined")) || !in_set(co$holm_status, c("ok", "holm_undefined"))) {
    bad("a status is outside the known vocabulary")
  }
  if (any(co$n_complete > co$n_eligible) || any(co$n_papers > co$n_complete)) bad("complete cases exceed eligible items or papers exceed items")
  within <- function(x, lo, hi) all(is.na(x) | (x >= lo & x <= hi))
  if (!within(co$rate_jats, 0, 1) || !within(co$rate_pdf, 0, 1) || !within(co$test_p, 0, 1) || !within(co$holm_p, 0, 1) ||
      !within(co$rd, -1, 1) || !within(co$bound_lower, -1, 1) || !within(co$bound_upper, -1, 1)) bad("a value lies outside its range")
  if (any(co$bound_lower > co$bound_upper) || any(co$boot_ci_lower > co$boot_ci_upper, na.rm = TRUE)) bad("an interval is reversed")
  if (any(co$holm_p < co$test_p, na.rm = TRUE)) bad("a Holm-adjusted p-value is below its raw p-value")
  undefined <- co$test_status == "inference_undefined"
  if (any(undefined & (!is.na(co$test_p) | is.na(co$test_reason))) || any(!undefined & !is.na(co$test_reason))) bad("test status and reason disagree")
  if (any(co$holm_status == "holm_undefined" & !is.na(co$holm_p))) bad("an undefined Holm family has a p-value")
  has_boot <- co$boot_status != "no_complete_pairs"
  if (any(has_boot & co$boot_n_resamples != PRIMARY_RESAMPLES)) bad("a bootstrap used a resample count other than 10000")
  for (i in seq_len(nrow(co))) {
    p <- pc[pc$outcome == co$outcome[i], ]; s <- sc[sc$outcome == co$outcome[i], ]
    if (sum(p$eligible) != co$n_eligible[i] || sum(p$complete) != co$n_complete[i] || any(p$complete > p$eligible) ||
        sum(p$complete > 0) != co$n_papers[i]) bad(sprintf("paper_counts.csv disagrees with the table for %s", co$outcome[i]))
    if (sum(s$n) != 2L * co$n_eligible[i]) bad(sprintf("state_counts.csv disagrees with the table for %s", co$outcome[i]))
  }
  invisible(TRUE)
}

get_path <- function(x, path) {
  for (p in path) {
    if (!is.list(x) || is.null(x[[p]])) return(NULL)
    x <- x[[p]]
  }
  x
}
populated <- function(x) !is.null(x) && length(x) > 0 && !(length(x) == 1 && is.atomic(x) && (is.na(x) || identical(x, "")))

# ---- the verifier -----------------------------------------------------------------------------

# The opening shared by the verifiers of the three published directories (result, secondary,
# supplement summary): the directory holds exactly `files`, each a regular file; the record is read
# from the bytes that are there, compared with the trusted hash when one is given, and parsed.
# `noun` names the directory in messages. The read-only seal is a local file-permission measure: it
# is reported, and demanded only on request; content integrity does not depend on it and it proves
# neither authenticity nor scientific provenance.
open_published <- function(dir, files, record_file, noun, expected_record_sha256, require_read_only) {
  if (!dir.exists(dir)) verify_error(sprintf("%s directory not found", noun))
  present <- list.files(dir, all.files = TRUE, no.. = TRUE)
  if (length(setdiff(files, present))) verify_error(sprintf("missing file(s): %s", paste(setdiff(files, present), collapse = ", ")))
  if (length(setdiff(present, files))) verify_error(sprintf("unexpected file(s): %s", paste(setdiff(present, files), collapse = ", ")))
  paths <- stats::setNames(file.path(dir, files), files)
  if (any(nzchar(Sys.readlink(paths))) || any(dir.exists(paths))) verify_error(sprintf("every %s file must be a regular file", noun))
  sealed <- all(bitwAnd(as.integer(file.mode(c(paths, dir))), 146L) == 0L)
  if (require_read_only && !sealed) verify_error("published files and directory must be read-only")
  record_bytes <- read_bytes(paths[[record_file]])
  record_sha256 <- sha256_raw(record_bytes)
  if (!is.null(expected_record_sha256) && !identical(record_sha256, expected_record_sha256)) {
    verify_error(sprintf("%s sha256 %s differs from the trusted %s", record_file, record_sha256, expected_record_sha256))
  }
  record <- tryCatch(jsonlite::fromJSON(rawToChar(record_bytes), simplifyVector = FALSE),
                     error = function(e) verify_error(sprintf("%s is not valid JSON", record_file)))
  if (!is.list(record)) verify_error(sprintf("%s is not a JSON object", record_file))
  list(paths = paths, sealed = sealed, record = record, record_sha256 = record_sha256)
}

# Each claim is list(path, required value); a differing field is reported with `suffix`.
check_claims <- function(record, claims, suffix = "") {
  for (cl in claims) {
    if (!identical(get_path(record, cl[[1]]), cl[[2]])) {
      verify_error(sprintf("record field %s must be %s%s", paste(cl[[1]], collapse = "."), format(cl[[2]]), suffix))
    }
  }
}

# The record lists the SHA-256 and size of each file it names; the bytes read must agree.
check_listed <- function(listed, bytes) {
  for (n in names(bytes)) {
    if (!identical(listed[[n]]$sha256, sha256_raw(bytes[[n]])) || !identical(as.numeric(listed[[n]]$bytes), as.numeric(length(bytes[[n]])))) {
      verify_error(sprintf("%s differs from the hash and size recorded for it", n))
    }
  }
}

# A derived record cites its primary result; that result is verified and must carry the same inputs.
check_cited_primary <- function(record, primary_dir, cited) {
  p <- verify_result(primary_dir, expected_record_sha256 = cited)
  if (!identical(p$scientific_result_sha256, record$primary_result$scientific_result_sha256) ||
      !identical(p$record$inputs$handoff_sha256, record$inputs$handoff_sha256) ||
      !identical(p$record$inputs$csv_sha256, record$inputs$csv_sha256)) {
    verify_error("the cited primary result does not match the record")
  }
}

# Integrity check of a published result directory, reading the bytes that are actually there.
# `record_authenticated` is TRUE only when the caller supplied an external SHA-256 of the record
# and it matched. Without one the result is internally consistent, nothing more.
verify_result <- function(dir, expected_record_sha256 = NULL, expected_lock_sha256 = NULL, require_read_only = FALSE) {
  opened <- open_published(dir, RESULT_FILES, RECORD_FILE, "result", expected_record_sha256, require_read_only)
  paths <- opened$paths; sealed <- opened$sealed; record <- opened$record; record_sha256 <- opened$record_sha256

  claims <- list(
    list(c("kind"), RECORD_KIND), list(c("schema_version"), RECORD_SCHEMA_VERSION),
    list(c("purpose"), "synthetic_rehearsal"), list(c("scientific_authority"), "none"),
    list(c("verification", "input_origin"), "synthetic_rehearsal"),
    list(c("verification", "lineage_verified_by_r"), FALSE),
    list(c("verification", "official_source_runs"), "not_applicable_synthetic_rehearsal"),
    list(c("verification", "statistical_execution"), "completed"),
    list(c("verification", "record_authentication"), "none_without_external_record_sha256"),
    list(c("inputs", "input_origin"), "synthetic_rehearsal"),
    list(c("seed", "label"), REHEARSAL_SEED_LABEL), list(c("seed", "registry"), "not_registered"),
    list(c("seed", "status"), "synthetic_rehearsal_not_scientific"),
    list(c("seed", "production_seed"), "awaiting_approval"), list(c("seed", "spec_name"), "primary_bootstrap"),
    list(c("environment_intended", "r_version_required"), REQUIRED_R_VERSION)
  )
  check_claims(record, claims, " in this pass")

  # Either nothing about a preflight, or a citation of its bytes and hash; never a claim that
  # R verified anything Python did.
  pre_state <- get_path(record, c("verification", "python_preflight"))
  pre_sha <- get_path(record, c("inputs", "python_preflight_sha256"))
  pre_bytes <- get_path(record, c("inputs", "python_preflight_bytes"))
  pre_scope <- get_path(record, c("verification", "python_preflight_scope"))
  if (identical(pre_state, PREFLIGHT_CITED)) {
    if (!is_sha256_hex(pre_sha) || !is.numeric(pre_bytes) || length(pre_bytes) != 1 || is.na(pre_bytes) ||
        pre_bytes < 1 || pre_bytes != round(pre_bytes) || !identical(pre_scope, PREFLIGHT_SCOPE)) {
      verify_error("a cited preflight needs its SHA-256, its size and the stated scope")
    }
  } else if (identical(pre_state, PREFLIGHT_NOT_RECORDED)) {
    if (!is.null(pre_sha) || !is.null(pre_bytes) || !is.null(pre_scope)) verify_error("a record without a preflight must not carry a preflight citation")
  } else {
    verify_error(sprintf("record field verification.python_preflight must be %s or %s", PREFLIGHT_NOT_RECORDED, PREFLIGHT_CITED))
  }

  listed <- get_path(record, c("outputs", "files"))
  if (!is.list(listed) || !setequal(names(listed), DATA_FILES)) verify_error("the record must list exactly the four data files")
  bytes <- lapply(stats::setNames(DATA_FILES, DATA_FILES), function(n) read_bytes(paths[[n]]))
  check_listed(listed, bytes)
  sci <- vapply(names(SCIENTIFIC_SCHEMAS), function(n) listed[[n]]$sha256, "")
  id <- tryCatch(
    scientific_result_id(record$inputs$handoff_sha256, record$inputs$csv_sha256, record$seed$value, sci),
    error = function(e) verify_error("the identity inputs of the record are malformed"))
  if (!identical(id, get_path(record, c("outputs", "scientific_result_sha256")))) verify_error("scientific_result_sha256 does not match the files, inputs and seed")
  if (!identical(get_path(record, c("environment_actual", "installed_package_inventory_sha256")), listed[[INVENTORY_FILE]]$sha256)) {
    verify_error("the recorded inventory hash differs from package_inventory.csv")
  }

  tables <- lapply(stats::setNames(names(SCIENTIFIC_SCHEMAS), names(SCIENTIFIC_SCHEMAS)),
                   function(n) parse_canonical_csv(bytes[[n]], SCIENTIFIC_SCHEMAS[[n]], n))
  inventory <- parse_canonical_csv(bytes[[INVENTORY_FILE]], INVENTORY_SCHEMA, INVENTORY_FILE)
  check_tables(tables[["coprimary_results.csv"]], tables[["paper_counts.csv"]], tables[["state_counts.csv"]])

  installed <- stats::setNames(inventory$version, inventory$package)
  for (pkg in names(record$environment_intended$lock_packages)) {
    if (!identical(unname(installed[pkg]), record$environment_intended$lock_packages[[pkg]])) {
      verify_error(sprintf("the inventory has %s %s, the lock has %s", pkg, installed[pkg], record$environment_intended$lock_packages[[pkg]]))
    }
  }
  used <- get_path(record, c("environment_actual", "package_versions_used"))
  for (pkg in names(used)) {
    if (!identical(unname(installed[pkg]), used[[pkg]])) verify_error(sprintf("package %s was used at %s but the inventory has %s", pkg, used[[pkg]], installed[pkg]))
  }
  if (!is.null(expected_lock_sha256) && !identical(record$environment_intended$renv_lock_sha256, expected_lock_sha256)) {
    verify_error("the record's lock hash differs from the trusted lock hash")
  }

  required <- unlist(record$provenance$required)
  if (!setequal(required, names(REQUIRED_PROVENANCE)) || anyDuplicated(required)) verify_error("the provenance checklist is not the required set")
  for (name in names(REQUIRED_PROVENANCE)) {
    if (!populated(get_path(record, REQUIRED_PROVENANCE[[name]]))) verify_error(sprintf("provenance field %s is empty", name))
  }
  list(ok = TRUE, sealed = sealed, record_sha256 = record_sha256, record_authenticated = !is.null(expected_record_sha256),
       scientific_result_sha256 = id, record = record, tables = tables)
}

# Re-run the computation from the original inputs and compare with the published scientific files.
# Equal bytes are reported as such; last-digit differences from another numerical library are
# reported separately; anything larger is an error.
verify_rerun <- function(dir, handoff_path, handoff_sha256, csv_path, tolerance = 1e-10) {
  v <- verify_result(dir)
  rec <- v$record
  if (!identical(rec$inputs$handoff_sha256, handoff_sha256)) verify_error("the supplied handoff hash differs from the record")
  fresh <- compute_results(handoff_path, handoff_sha256, csv_path, rec$seed$value)
  if (!identical(rec$inputs$csv_sha256, fresh$inputs$csv_sha256)) verify_error("the supplied CSV differs from the record")
  published <- lapply(stats::setNames(names(SCIENTIFIC_SCHEMAS), names(SCIENTIFIC_SCHEMAS)),
                      function(n) read_bytes(file.path(dir, n)))
  if (identical(published, fresh$files[names(published)])) {
    return(list(comparison = "byte_identical", scientific_result_sha256 = fresh$scientific_result_sha256))
  }
  for (n in names(published)) {
    a <- parse_canonical_csv(published[[n]], SCIENTIFIC_SCHEMAS[[n]], n)
    b <- parse_canonical_csv(fresh$files[[n]], SCIENTIFIC_SCHEMAS[[n]], n)
    for (col in names(a)) {
      if (SCIENTIFIC_SCHEMAS[[n]][[col]] == "dbl") {
        same_na <- identical(is.na(a[[col]]), is.na(b[[col]]))
        close <- same_na && all(abs(a[[col]] - b[[col]]) <= tolerance * pmax(1, abs(b[[col]])), na.rm = TRUE)
        if (!close) verify_error(sprintf("re-execution differs in %s column %s beyond %g", n, col, tolerance))
      } else if (!identical(a[[col]], b[[col]])) {
        verify_error(sprintf("re-execution differs in %s column %s", n, col))
      }
    }
  }
  list(comparison = "numerically_equal_not_byte_identical", scientific_result_sha256 = fresh$scientific_result_sha256)
}

# ---- publication ------------------------------------------------------------------------------

# Publish a staging directory as `target`. What this does and does not guarantee:
#  * No artifact is ever replaced: `dir.create` fails if the path exists and `file.link` fails if its
#    destination exists, each decided by the file system in one step (no check-then-act window).
#  * Invalid staging content is rejected before the destination is created.
#  * The complete result does NOT appear atomically. The directory is visible from `dir.create` on,
#    with 0 to 4 files, until the record is linked. The record is linked last and `verify_result`
#    requires the exact file set, so every such state, and any interruption, is rejected by the
#    verifier. Callers must verify before using a result directory.
#  * Read-only permissions are applied at the end and checked. They are a local protection only; a
#    complete, hash-consistent directory without them still passes content verification.
# An ordinary failure removes only what this call created; an interrupt leaves an inert partial
# directory whose name cannot be reused. Hard links need staging and target on one file system.
publish_result <- function(staging, target, link = file.link, chmod = Sys.chmod, files = RESULT_FILES,
                           verifier = verify_result) {
  # Nothing may become visible at the destination unless the staged content is already valid.
  verifier(staging)
  if (!dir.create(target, showWarnings = FALSE, mode = "0755")) {
    run_error("publish", "the result directory already exists or cannot be created; nothing was replaced")
  }
  made <- character(0)
  undo <- function() {
    suppressWarnings(file.remove(made))
    suppressWarnings(file.remove(target))
  }
  for (f in files) {                                         # the file list ends with the record
    to <- file.path(target, f)
    if (!isTRUE(link(file.path(staging, f), to))) {
      undo()
      run_error("publish", sprintf("could not link %s into the result directory; nothing was replaced", f))
    }
    made <- c(made, to)
  }
  tryCatch(verifier(target), error = function(e) { undo(); stop(e) })
  chmod(made, "0444")
  chmod(target, "0555")
  # Read-back of the published bytes once more, now that the directory is read-only.
  verifier(target, require_read_only = TRUE)
  invisible(target)
}

active_renv_project <- function() if (requireNamespace("renv", quietly = TRUE)) renv::project() else NULL

# The whole run: gates, computation, record, staging, verification, publication.
run_analysis <- function(handoff_path, handoff_sha256, csv_path, lock_sha256, seed, seed_label, out_root, project_dir,
                         now = Sys.time, synchronized = NULL, code_fn = code_state,
                         preflight_path = NULL, preflight_sha256 = NULL) {
  started <- now()
  seed <- check_seed(seed, seed_label)
  if (!dir.exists(out_root)) run_error("publish", "the output root must be an existing directory")
  preflight <- read_preflight_citation(preflight_path, preflight_sha256)

  lock <- read_lock(file.path(project_dir, "renv.lock"), lock_sha256)
  inventory <- installed_inventory(.libPaths(), file.path(project_dir, "renv", "library"), lock$packages)
  check_environment(lock, inventory, project_dir, active_project = active_renv_project(), synchronized = synchronized)

  results <- compute_results(handoff_path, handoff_sha256, csv_path, seed)
  seed_rec <- seed_record(seed, seed_label, results$inputs$handoff)
  code <- code_fn(project_dir)
  if (!identical(code$status, "ok")) run_error("code", "the git state of the analysis code cannot be read")
  results$inventory_bytes <- csv_bytes(inventory, INVENTORY_SCHEMA)
  finished <- now()

  run_id <- sprintf("%s-%s", format(started, "%Y%m%dT%H%M%OS3Z", tz = "UTC"),
                    substr(sha256_raw(charToRaw(paste(handoff_sha256, results$inputs$csv_sha256, seed))), 1, 8))
  execution <- list(run_id = run_id, started_utc = iso_utc(started), finished_utc = iso_utc(finished),
                    hostname = Sys.info()[["nodename"]])
  record <- build_run_record(results, seed_rec, lock, execution_environment(), sha256_raw(results$inventory_bytes),
                             loaded_package_versions(), code, execution, preflight)
  record_bytes <- serialize_record(record)

  staging <- file.path(out_root, paste0(".staging-", run_id))
  if (!dir.create(staging, showWarnings = FALSE, mode = "0755")) run_error("publish", "the staging directory already exists")
  # No force = TRUE: it would chmod the staged files, which share their inode with the published ones.
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  contents <- c(results$files, stats::setNames(list(results$inventory_bytes), INVENTORY_FILE),
                stats::setNames(list(record_bytes), RECORD_FILE))
  for (f in RESULT_FILES) writeBin(contents[[f]], file.path(staging, f))

  target <- file.path(out_root, paste0("rehearsal_", run_id))
  publish_result(staging, target)
  list(result_dir = target, record_sha256 = sha256_raw(record_bytes),
       scientific_result_sha256 = results$scientific_result_sha256, run_id = run_id)
}
