# Secondary result directory: hashed descriptive tables, a report and one figure, and a record.
#
# This is a separate published directory from the primary result. The primary result, its files
# and its scientific-result hash are read and verified here, never changed. Three kinds of content
# are kept apart:
#   * scientific secondary tables: CSV files, a function of the handoff and table bytes only; their
#     hashes give secondary_result_sha256;
#   * presentation: report.md (a function of the verified primary and secondary tables) and the
#     figure (a rendering whose bytes depend on the graphics stack; its hash is recorded but is not
#     part of any scientific identity);
#   * provenance: the record with the cited primary result, code and environment.
# While the input is a synthetic rehearsal every file says so and nothing is a finding.

SECONDARY_RECORD_FILE <- "secondary_record.json"
REPORT_FILE <- "report.md"
FIGURE_FILE <- "coprimary_effects.svg"
SECONDARY_DATA_FILES <- names(SECONDARY_SCHEMAS)
PRESENTATION_FILES <- c(REPORT_FILE, FIGURE_FILE)
DERIVED_FILES <- c(SECONDARY_DATA_FILES, PRESENTATION_FILES, SECONDARY_RECORD_FILE)
SECONDARY_RECORD_KIND <- "r_secondary_record"
SECONDARY_SCHEMA_VERSION <- "1"
SYNTHETIC_BANNER <- "SYNTHETIC REHEARSAL — NOT SCIENTIFIC FINDINGS"

# ---- identity of the secondary tables ---------------------------------------------------------

secondary_result_id <- function(handoff_sha256, csv_sha256, file_sha256) {
  hex <- function(x) is.character(x) && length(x) == 1 && grepl("^[0-9a-f]{64}$", x)
  if (!hex(handoff_sha256) || !hex(csv_sha256) || !setequal(names(file_sha256), SECONDARY_DATA_FILES) ||
      !all(vapply(file_sha256, hex, logical(1)))) {
    run_error("identity", "secondary identity inputs must be SHA-256 hex strings for exactly the secondary tables")
  }
  ordered <- file_sha256[order(names(file_sha256), method = "radix")]
  files <- paste(sprintf("{\"name\":\"%s\",\"sha256\":\"%s\"}", names(ordered), unlist(ordered, use.names = FALSE)),
                 collapse = ",")
  json <- sprintf("{\"csv_sha256\":\"%s\",\"files\":[%s],\"handoff_sha256\":\"%s\",\"kind\":\"secondary_tables\"}",
                  csv_sha256, files, handoff_sha256)
  sha256_raw(charToRaw(json))
}

# ---- consistency of the secondary tables ------------------------------------------------------

# The columns that define a denominator group in each count table.
DENOMINATOR_GROUPS <- list(
  "secondary_stage_a.csv" = "condition",
  "secondary_stage_b_failure.csv" = c("matrix_arm", "condition", "measure"),
  "secondary_scope.csv" = "condition",
  "secondary_repeat_outcomes.csv" = c("matrix_arm", "condition", "repeat_index"),
  "secondary_discordance.csv" = "outcome",
  "secondary_correct_without_evidence.csv" = "condition",
  "secondary_completion.csv" = "condition",
  "secondary_answer_class.csv" = c("condition", "answer_class", "outcome")
)
ROBUSTNESS_STATUSES <- c("ok", "no_complete_pairs", "arm_absent")

check_secondary_tables <- function(tables) {
  bad <- function(why) verify_error(why)
  if (!setequal(names(tables), SECONDARY_DATA_FILES)) bad("the secondary tables are not the expected set")
  for (name in names(DENOMINATOR_GROUPS)) {
    t <- tables[[name]]
    if (!nrow(t)) next
    if (any(t$n < 1L) || any(t$n > t$denominator)) bad(sprintf("%s: a count is outside 1..denominator", name))
    group <- do.call(paste, c(unname(as.list(t[DENOMINATOR_GROUPS[[name]]])), sep = "\x1f"))
    if (!identical(as.integer(tapply(t$n, group, sum)[group]), t$denominator)) {
      bad(sprintf("%s: a denominator is not the sum of its group", name))
    }
  }
  r <- tables[["secondary_robustness.csv"]]
  if (!all(r$status %in% ROBUSTNESS_STATUSES)) bad("secondary_robustness.csv: unknown status")
  if (any(r$n_complete > r$n_paired) || any(r$n_paired > r$n_items) || any(r$n_papers > r$n_complete)) {
    bad("secondary_robustness.csv: counts are not nested")
  }
  inside <- function(x, lo, hi) all(is.na(x) | (x >= lo & x <= hi))
  if (!inside(r$rate_jats, 0, 1) || !inside(r$rate_pdf, 0, 1) || !inside(r$rd, -1, 1) || !inside(r$equal_paper_rd, -1, 1)) {
    bad("secondary_robustness.csv: a value lies outside its range")
  }
  if (any(!is.na(r$rd) & abs(r$rd_pp - 100 * r$rd) > 1e-9) || any(xor(is.na(r$rd), is.na(r$rd_pp)))) {
    bad("secondary_robustness.csv: rd_pp is not 100 * rd")
  }
  if (any((r$status != "ok") != is.na(r$rd))) bad("secondary_robustness.csv: an estimate and its status disagree")
  v <- tables[["secondary_repeat_variability.csv"]]
  if (any(v$n_all_repeats_evaluable + v$n_missing_repeat_rows + v$n_with_unevaluable_repeat != v$n_items) ||
      any(v$n_correct_0 + v$n_correct_1 + v$n_correct_2 + v$n_correct_3 != v$n_all_repeats_evaluable)) {
    bad("secondary_repeat_variability.csv: item counts do not add up")
  }
  u <- tables[["secondary_unavailable.csv"]]
  if (!setequal(u$analysis, secondary_unavailable()$analysis) || !all(u$status == "deferred_data_not_in_analysis_table")) {
    bad("secondary_unavailable.csv is not the expected statement of deferred analyses")
  }
  p <- tables[["secondary_populations.csv"]]
  if (!setequal(p$table, SECONDARY_DATA_FILES) || anyNA(p$n_rows) || any(p$n_rows < 0L)) {
    bad("secondary_populations.csv does not describe exactly the secondary tables")
  }
  invisible(TRUE)
}

# ---- report -----------------------------------------------------------------------------------

md_cell <- function(x) {
  if (is.na(x)) return("NA")
  if (is.double(x)) return(sprintf("%.4f", x))
  gsub("|", "\\|", as.character(x), fixed = TRUE)
}

md_table <- function(df) {
  if (!ncol(df)) return(character(0))
  head <- sprintf("| %s |", paste(names(df), collapse = " | "))
  rule <- sprintf("|%s|", paste(rep("---", ncol(df)), collapse = "|"))
  body <- vapply(seq_len(nrow(df)), function(i) {
    sprintf("| %s |", paste(vapply(df, function(col) md_cell(col[i]), ""), collapse = " | "))
  }, "")
  c(head, rule, body)
}

build_report <- function(primary, tables, hashes) {
  co <- primary$tables[["coprimary_results.csv"]]
  key <- c("outcome", "n_eligible", "n_complete", "n_papers", "rate_jats", "rate_pdf", "rd_pp", "boot_status",
           "boot_ci_lower", "boot_ci_upper", "test_status", "test_p", "holm_status", "holm_p", "equal_paper_rd",
           "bound_lower", "bound_upper")
  lines <- c(
    paste0("# ", SYNTHETIC_BANNER), "",
    "Derived presentation of a synthetic rehearsal. No value below is a scientific finding. The primary analysis is unchanged; the secondary tables are descriptive counts and proportions with no inference.", "",
    "## Provenance", "",
    sprintf("- handoff sha256: %s", hashes$handoff_sha256),
    sprintf("- analysis table sha256: %s", hashes$csv_sha256),
    sprintf("- primary run record sha256: %s", primary$record_sha256),
    sprintf("- primary scientific result sha256: %s", primary$scientific_result_sha256),
    sprintf("- secondary result sha256: %s", hashes$secondary_result_sha256), "",
    "## Co-primary results (item-weighted paired risk difference, JATS minus PDF_DOCLING)", "",
    "Rates are proportions; rd_pp is in percentage points; boot_ci_* is the 95% whole-paper percentile interval of the risk difference (proportion scale); holm_p is adjusted across the two co-primary p-values only.", "",
    md_table(co[key]), "",
    "## Secondary and exploratory tables (descriptive; not confirmatory)", ""
  )
  pops <- tables[["secondary_populations.csv"]]
  for (name in setdiff(SECONDARY_DATA_FILES, c("secondary_populations.csv", "secondary_unavailable.csv"))) {
    pp <- pops[pops$table == name, ]
    lines <- c(lines, sprintf("### %s", name), "",
               sprintf("Population: %s. Unit: %s. Denominator: %s. Evaluability: %s. Missingness: %s. Source rows: %d.",
                       pp$population, pp$unit, pp$denominator, pp$evaluability, pp$missingness, pp$n_rows), "",
               md_table(tables[[name]]), "")
  }
  lines <- c(lines, "## Prespecified analyses not supported by the analysis table", "",
             md_table(tables[["secondary_unavailable.csv"]]), "")
  charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n")))
}

# ---- figure -----------------------------------------------------------------------------------

# Point estimate and percentile interval (percentage points) of each co-primary outcome. An
# undefined interval or estimate is written as text, never drawn.
draw_coprimary_figure <- function(co) {
  n <- nrow(co)
  lower <- 100 * co$boot_ci_lower
  upper <- 100 * co$boot_ci_upper
  finite <- c(co$rd_pp, lower, upper, 0)
  finite <- finite[is.finite(finite)]
  pad <- max(5, 0.1 * diff(range(finite)))
  xlim <- c(min(finite) - pad, max(finite) + pad)
  graphics::plot(NA, xlim = xlim, ylim = c(0.5, n + 0.5), yaxt = "n", xlab = "Risk difference, JATS minus PDF_DOCLING (percentage points)",
                 ylab = "", main = SYNTHETIC_BANNER, sub = "Item-weighted paired risk difference; 95% whole-paper percentile bootstrap interval")
  graphics::abline(v = 0, lty = 2)
  graphics::axis(2, at = seq_len(n), labels = rev(co$outcome), las = 1, tick = FALSE)
  for (i in seq_len(n)) {
    y <- n + 1 - i
    if (is.finite(lower[i]) && is.finite(upper[i]) && is.finite(co$rd_pp[i])) {
      graphics::segments(lower[i], y, upper[i], y, lwd = 2)
    }
    if (is.finite(co$rd_pp[i])) {
      graphics::points(co$rd_pp[i], y, pch = 19)
      note <- if (is.finite(lower[i])) sprintf("n=%d items, %d papers", co$n_complete[i], co$n_papers[i])
              else sprintf("interval undefined (%s); n=%d items", co$boot_status[i], co$n_complete[i])
      graphics::text(co$rd_pp[i], y + 0.25, note, cex = 0.8)
    } else {
      graphics::text(mean(xlim), y, sprintf("no estimate (%s)", co$summary_status[i]), cex = 0.9)
    }
  }
  invisible(NULL)
}

# The device draws text as glyph outlines, so the banner is also written as a comment in the file.
render_figure <- function(co, path) {
  grDevices::svg(path, width = 7, height = 3.2)
  drawn <- FALSE
  on.exit(if (!drawn) grDevices::dev.off(), add = TRUE)
  draw_coprimary_figure(co)
  grDevices::dev.off()
  drawn <- TRUE
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  at <- which(startsWith(lines, "<svg"))[1]
  lines <- append(lines, sprintf("<!-- %s -->", SYNTHETIC_BANNER), after = at - 1L)
  writeBin(charToRaw(enc2utf8(paste0(paste(lines, collapse = "\n"), "\n"))), path)
  invisible(path)
}

# ---- computation ------------------------------------------------------------------------------

compute_derived <- function(handoff_path, handoff_sha256, csv_path, primary_dir, expected_primary_record_sha256 = NULL) {
  inputs <- read_inputs(handoff_path, handoff_sha256, csv_path)
  if (!identical(inputs$input_origin, "synthetic_rehearsal")) {
    run_error("origin", sprintf("input_origin '%s' is not accepted; only a synthetic rehearsal is", inputs$input_origin))
  }
  primary <- verify_result(primary_dir, expected_record_sha256 = expected_primary_record_sha256)
  if (!identical(primary$record$inputs$handoff_sha256, handoff_sha256) ||
      !identical(primary$record$inputs$csv_sha256, inputs$csv_sha256)) {
    run_error("primary", "the primary result was computed from different inputs")
  }
  tables <- compute_secondary_tables(inputs$rows, outcome_rules(inputs$handoff))
  files <- Map(csv_bytes, tables, SECONDARY_SCHEMAS)
  file_sha256 <- vapply(files, sha256_raw, "")
  hashes <- list(handoff_sha256 = handoff_sha256, csv_sha256 = inputs$csv_sha256,
                 secondary_result_sha256 = secondary_result_id(handoff_sha256, inputs$csv_sha256, file_sha256))
  list(inputs = inputs, primary = primary, tables = tables, files = files, file_sha256 = file_sha256,
       hashes = hashes, report = build_report(primary, tables, hashes))
}

# Render twice and keep the bytes of the first. Whether the two renders agree is recorded, not assumed.
render_figure_bytes <- function(co) {
  one <- function() {
    path <- tempfile(fileext = ".svg")
    on.exit(unlink(path))
    render_figure(co, path)
    readBin(path, "raw", file.size(path))
  }
  first <- one()
  list(bytes = first, repeatable = identical(first, one()))
}

build_secondary_record <- function(derived, figure, code, execution) {
  entry <- function(b) list(bytes = length(b), sha256 = sha256_raw(b))
  primary <- derived$primary
  list(
    kind = SECONDARY_RECORD_KIND, schema_version = SECONDARY_SCHEMA_VERSION,
    purpose = "synthetic_rehearsal", scientific_authority = "none", banner = SYNTHETIC_BANNER,
    analysis_label = SECONDARY_LABEL,
    inference = "none: descriptive counts and proportions; no interval, test or adjustment",
    inputs = list(handoff_sha256 = derived$hashes$handoff_sha256, csv_sha256 = derived$hashes$csv_sha256,
                  input_origin = derived$inputs$input_origin),
    primary_result = list(run_record_sha256 = primary$record_sha256,
                          scientific_result_sha256 = primary$scientific_result_sha256,
                          verification = "verified_by_this_run_before_use"),
    scientific_files = lapply(derived$files, entry),
    secondary_result_sha256 = derived$hashes$secondary_result_sha256,
    presentation = list(
      "report.md" = c(entry(derived$report), list(role = "derived_from_verified_tables", byte_deterministic = TRUE)),
      "coprimary_effects.svg" = c(entry(figure$bytes), list(
        role = "rendering_dependent_not_part_of_any_scientific_identity",
        rendered_twice_byte_identical = figure$repeatable, device = "grDevices::svg (cairo)",
        r_version = R.version.string))
    ),
    code = list(git_commit = code$commit, dirty = code$dirty, script_sha256 = as.list(code$script_sha256)),
    execution = execution
  )
}

run_derived <- function(handoff_path, handoff_sha256, csv_path, primary_dir, out_root, project_dir,
                        expected_primary_record_sha256 = NULL, now = Sys.time, code_fn = code_state) {
  started <- now()
  if (!dir.exists(out_root)) run_error("publish", "the output root must be an existing directory")
  derived <- compute_derived(handoff_path, handoff_sha256, csv_path, primary_dir, expected_primary_record_sha256)
  code <- code_fn(project_dir)
  if (!identical(code$status, "ok")) run_error("code", "the git state of the analysis code cannot be read")
  figure <- render_figure_bytes(derived$primary$tables[["coprimary_results.csv"]])
  run_id <- sprintf("%s-%s", format(started, "%Y%m%dT%H%M%OS3Z", tz = "UTC"), substr(derived$hashes$secondary_result_sha256, 1, 8))
  execution <- list(run_id = run_id, started_utc = iso_utc(started), finished_utc = iso_utc(now()), hostname = Sys.info()[["nodename"]])
  record_bytes <- serialize_record(build_secondary_record(derived, figure, code, execution))
  contents <- c(derived$files, stats::setNames(list(derived$report, figure$bytes, record_bytes), c(REPORT_FILE, FIGURE_FILE, SECONDARY_RECORD_FILE)))
  staging <- file.path(out_root, paste0(".staging-secondary-", run_id))
  if (!dir.create(staging, showWarnings = FALSE, mode = "0755")) run_error("publish", "the staging directory already exists")
  on.exit(unlink(staging, recursive = TRUE), add = TRUE)
  for (f in DERIVED_FILES) writeBin(contents[[f]], file.path(staging, f))
  target <- file.path(out_root, paste0("secondary_", run_id))
  publish_result(staging, target, files = DERIVED_FILES, verifier = verify_secondary)
  list(result_dir = target, record_sha256 = sha256_raw(record_bytes),
       secondary_result_sha256 = derived$hashes$secondary_result_sha256, run_id = run_id)
}

# ---- verification -----------------------------------------------------------------------------

verify_secondary <- function(dir, expected_record_sha256 = NULL, require_read_only = FALSE, primary_dir = NULL) {
  if (!dir.exists(dir)) verify_error("secondary directory not found")
  present <- list.files(dir, all.files = TRUE, no.. = TRUE)
  if (length(setdiff(DERIVED_FILES, present))) verify_error(sprintf("missing file(s): %s", paste(setdiff(DERIVED_FILES, present), collapse = ", ")))
  if (length(setdiff(present, DERIVED_FILES))) verify_error(sprintf("unexpected file(s): %s", paste(setdiff(present, DERIVED_FILES), collapse = ", ")))
  paths <- stats::setNames(file.path(dir, DERIVED_FILES), DERIVED_FILES)
  if (any(nzchar(Sys.readlink(paths))) || any(dir.exists(paths))) verify_error("every secondary file must be a regular file")
  sealed <- all(bitwAnd(as.integer(file.mode(c(paths, dir))), 146L) == 0L)
  if (require_read_only && !sealed) verify_error("published files and directory must be read-only")
  record_bytes <- read_bytes(paths[[SECONDARY_RECORD_FILE]])
  record_sha256 <- sha256_raw(record_bytes)
  if (!is.null(expected_record_sha256) && !identical(record_sha256, expected_record_sha256)) {
    verify_error(sprintf("%s sha256 %s differs from the trusted %s", SECONDARY_RECORD_FILE, record_sha256, expected_record_sha256))
  }
  record <- tryCatch(jsonlite::fromJSON(rawToChar(record_bytes), simplifyVector = FALSE),
                     error = function(e) verify_error("secondary_record.json is not valid JSON"))
  if (!is.list(record)) verify_error("secondary_record.json is not a JSON object")
  claims <- list(
    list(c("kind"), SECONDARY_RECORD_KIND), list(c("schema_version"), SECONDARY_SCHEMA_VERSION),
    list(c("purpose"), "synthetic_rehearsal"), list(c("scientific_authority"), "none"),
    list(c("banner"), SYNTHETIC_BANNER), list(c("analysis_label"), SECONDARY_LABEL),
    list(c("inputs", "input_origin"), "synthetic_rehearsal")
  )
  for (cl in claims) {
    if (!identical(get_path(record, cl[[1]]), cl[[2]])) {
      verify_error(sprintf("record field %s must be %s", paste(cl[[1]], collapse = "."), format(cl[[2]])))
    }
  }
  scientific <- get_path(record, "scientific_files")
  presentation <- get_path(record, "presentation")
  if (!is.list(scientific) || !setequal(names(scientific), SECONDARY_DATA_FILES)) verify_error("the record must list exactly the secondary tables")
  if (!is.list(presentation) || !setequal(names(presentation), PRESENTATION_FILES)) verify_error("the record must list exactly the report and the figure")
  bytes <- lapply(stats::setNames(c(SECONDARY_DATA_FILES, PRESENTATION_FILES), c(SECONDARY_DATA_FILES, PRESENTATION_FILES)),
                  function(n) read_bytes(paths[[n]]))
  listed <- c(scientific, presentation)
  for (n in names(bytes)) {
    if (!identical(listed[[n]]$sha256, sha256_raw(bytes[[n]])) || !identical(as.numeric(listed[[n]]$bytes), as.numeric(length(bytes[[n]])))) {
      verify_error(sprintf("%s differs from the hash and size recorded for it", n))
    }
  }
  id <- tryCatch(secondary_result_id(record$inputs$handoff_sha256, record$inputs$csv_sha256,
                                     vapply(scientific, function(e) e$sha256, "")),
                 error = function(e) verify_error("the identity inputs of the record are malformed"))
  if (!identical(id, record$secondary_result_sha256)) verify_error("secondary_result_sha256 does not match the files and inputs")
  tables <- lapply(stats::setNames(SECONDARY_DATA_FILES, SECONDARY_DATA_FILES),
                   function(n) parse_canonical_csv(bytes[[n]], SECONDARY_SCHEMAS[[n]], n))
  check_secondary_tables(tables)
  banner <- charToRaw(SYNTHETIC_BANNER)
  contains <- function(b) length(b) >= length(banner) && any(vapply(seq_len(length(b) - length(banner) + 1L), function(i) identical(b[i:(i + length(banner) - 1L)], banner), NA))
  if (!contains(bytes[[REPORT_FILE]]) || !contains(bytes[[FIGURE_FILE]])) verify_error("the report and the figure must state that they are synthetic")
  cited <- get_path(record, c("primary_result", "run_record_sha256"))
  if (!is_sha256_hex(cited) || !is_sha256_hex(get_path(record, c("primary_result", "scientific_result_sha256")))) {
    verify_error("the cited primary result hashes are malformed")
  }
  if (!is.null(primary_dir)) {
    p <- verify_result(primary_dir, expected_record_sha256 = cited)
    if (!identical(p$scientific_result_sha256, record$primary_result$scientific_result_sha256) ||
        !identical(p$record$inputs$handoff_sha256, record$inputs$handoff_sha256) ||
        !identical(p$record$inputs$csv_sha256, record$inputs$csv_sha256)) {
      verify_error("the cited primary result does not match the record")
    }
  }
  list(ok = TRUE, sealed = sealed, record_sha256 = record_sha256, record_authenticated = !is.null(expected_record_sha256),
       secondary_result_sha256 = id, record = record, tables = tables, report = bytes[[REPORT_FILE]])
}

# Recompute the tables and the report from the original inputs and the primary result, and require
# byte equality. The figure is not compared: its bytes depend on the graphics stack.
verify_secondary_rerun <- function(dir, handoff_path, handoff_sha256, csv_path, primary_dir) {
  v <- verify_secondary(dir, primary_dir = primary_dir)
  if (!identical(v$record$inputs$handoff_sha256, handoff_sha256)) verify_error("the supplied handoff hash differs from the record")
  fresh <- compute_derived(handoff_path, handoff_sha256, csv_path, primary_dir, v$record$primary_result$run_record_sha256)
  if (!identical(fresh$hashes$secondary_result_sha256, v$secondary_result_sha256)) verify_error("re-execution gives a different secondary result")
  for (n in SECONDARY_DATA_FILES) {
    if (!identical(read_bytes(file.path(dir, n)), fresh$files[[n]])) verify_error(sprintf("re-execution differs in %s", n))
  }
  if (!identical(v$report, fresh$report)) verify_error("re-execution differs in report.md")
  list(comparison = "byte_identical_tables_and_report", secondary_result_sha256 = v$secondary_result_sha256)
}
