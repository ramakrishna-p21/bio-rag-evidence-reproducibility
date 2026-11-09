# Secondary result directory: publication, determinism, verification and tamper detection.

dclock <- function(iso) function() as.POSIXct(iso, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS")

d_root <- cleanup_dir(tempfile("derived"), teardown_env())
dir.create(d_root)
d_primary_out <- file.path(d_root, "primary"); dir.create(d_primary_out)
d_args <- fixture_args()
d_primary <- run_analysis(d_args$handoff_path, d_args$handoff_sha256, d_args$csv_path, real_lock_sha(), TEST_SEED,
                          "synthetic_rehearsal", d_primary_out, analysis_dir(),
                          now = dclock("2026-11-07T10:00:00.000"), synchronized = TRUE)
primary_hashes <- function() vapply(list.files(d_primary$result_dir, full.names = TRUE), sha256_file, "")
primary_before <- primary_hashes()

derive <- function(out, clock = "2026-11-07T11:00:00.000", primary = d_primary$result_dir, args = d_args, rec_sha = NULL) {
  run_derived(args$handoff_path, args$handoff_sha256, args$csv_path, primary, out, analysis_dir(),
              expected_primary_record_sha256 = rec_sha, now = dclock(clock))
}
d_out1 <- file.path(d_root, "one"); dir.create(d_out1)
d_out2 <- file.path(d_root, "two"); dir.create(d_out2)
d1 <- derive(d_out1, rec_sha = d_primary$record_sha256)
d2 <- derive(d_out2, clock = "2026-11-07T12:30:00.000")

copy_derived <- function(dir = d1$result_dir) {
  dest <- cleanup_dir(tempfile("dcopy"), parent.frame())
  dir.create(dest)
  file.copy(list.files(dir, full.names = TRUE), dest)
  Sys.chmod(list.files(dest, full.names = TRUE), "0644")
  dest
}
drecord <- function(dir) jsonlite::fromJSON(file.path(dir, SECONDARY_RECORD_FILE), simplifyVector = FALSE)
rewrite <- function(dir, name, f) {
  path <- file.path(dir, name)
  writeBin(f(readBin(path, "raw", file.size(path))), path)
}
# Keep the record consistent with edited files, as a forger would.
reseal <- function(dir) {
  rec <- drecord(dir)
  for (n in c(SECONDARY_DATA_FILES, PRESENTATION_FILES)) {
    b <- readBin(file.path(dir, n), "raw", file.size(file.path(dir, n)))
    section <- if (n %in% SECONDARY_DATA_FILES) "scientific_files" else "presentation"
    rec[[section]][[n]]$bytes <- length(b); rec[[section]][[n]]$sha256 <- sha256_raw(b)
  }
  rec$secondary_result_sha256 <- secondary_result_id(rec$inputs$handoff_sha256, rec$inputs$csv_sha256,
                                                     vapply(rec$scientific_files, function(e) e$sha256, ""))
  writeBin(serialize_record(rec), file.path(dir, SECONDARY_RECORD_FILE))
  dir
}
expect_dverify_fails <- function(dir, ..., pattern = NULL) {
  err <- tryCatch({ verify_secondary(dir, ...); NULL }, run_error = function(e) e)
  expect_false(is.null(err), info = "verify_secondary did not fail")
  if (!is.null(err) && !is.null(pattern)) expect_match(conditionMessage(err), pattern)
}

test_that("the secondary directory holds exactly the declared files, read-only", {
  expect_match(basename(d1$result_dir), "^secondary_")
  expect_setequal(list.files(d1$result_dir, all.files = TRUE, no.. = TRUE), DERIVED_FILES)
  for (f in list.files(d1$result_dir, full.names = TRUE)) expect_equal(unname(file.mode(f) & as.octmode("0222")), as.octmode("0000"))
  expect_identical(list.files(d_out1, all.files = TRUE, no.. = TRUE), basename(d1$result_dir))
  expect_identical(d1$record_sha256, sha256_file(file.path(d1$result_dir, SECONDARY_RECORD_FILE)))
  v <- verify_secondary(d1$result_dir, expected_record_sha256 = d1$record_sha256, require_read_only = TRUE,
                        primary_dir = d_primary$result_dir)
  expect_true(v$ok && v$sealed && v$record_authenticated)
})

test_that("the primary result and its golden scientific hash are untouched", {
  expect_identical(primary_hashes(), primary_before)
  expect_identical(d_primary$scientific_result_sha256, "17c44709e57fd9ac259f3d239498693d31e34258f8577d9e73d0e6cb86b01e36")
  expect_identical(drecord(d1$result_dir)$primary_result$scientific_result_sha256, d_primary$scientific_result_sha256)
  expect_true(verify_result(d_primary$result_dir)$ok)
})

test_that("tables, report and secondary identity are deterministic; execution facts differ", {
  expect_identical(d1$secondary_result_sha256, d2$secondary_result_sha256)
  for (n in c(SECONDARY_DATA_FILES, REPORT_FILE)) {
    expect_identical(readBin(file.path(d1$result_dir, n), "raw", 1e6), readBin(file.path(d2$result_dir, n), "raw", 1e6))
  }
  expect_false(identical(d1$record_sha256, d2$record_sha256))
  expect_false(identical(d1$secondary_result_sha256, d_primary$scientific_result_sha256))
})

test_that("the figure is recorded as rendering dependent and its repeatability is reported, not assumed", {
  fig <- drecord(d1$result_dir)$presentation[[FIGURE_FILE]]
  expect_match(fig$role, "rendering_dependent")
  expect_type(fig$rendered_twice_byte_identical, "logical")
  expect_identical(fig$sha256, sha256_file(file.path(d1$result_dir, FIGURE_FILE)))
  expect_false(FIGURE_FILE %in% names(drecord(d1$result_dir)$scientific_files))
})

test_that("every output states that it is synthetic", {
  rec <- drecord(d1$result_dir)
  expect_identical(rec$banner, SYNTHETIC_BANNER)
  expect_identical(rec$scientific_authority, "none")
  report <- read_bytes_as_text(file.path(d1$result_dir, REPORT_FILE))
  expect_true(startsWith(report, paste0("# ", SYNTHETIC_BANNER)))
  svg <- read_bytes_as_text(file.path(d1$result_dir, FIGURE_FILE))
  expect_match(svg, SYNTHETIC_BANNER, fixed = TRUE)
})

test_that("the confirmatory reference row equals the primary result", {
  v <- verify_secondary(d1$result_dir)
  co <- verify_result(d_primary$result_dir)$tables[["coprimary_results.csv"]]
  r <- v$tables[["secondary_robustness.csv"]]
  r <- r[r$matrix_arm == "confirmatory", ]
  expect_identical(r$outcome, co$outcome)
  expect_identical(r$n_complete, co$n_complete)
  expect_identical(r$n_papers, co$n_papers)
  expect_equal(r$rd, co$rd, tolerance = 1e-15)
  expect_equal(r$equal_paper_rd, co$equal_paper_rd, tolerance = 1e-15)
})

test_that("absent arms in the fixture are visible as absent, not as zero effects", {
  v <- verify_secondary(d1$result_dir)
  r <- v$tables[["secondary_robustness.csv"]]
  expect_true(all(r$status[r$matrix_arm %in% ROBUSTNESS_ARMS] == "arm_absent"))
  p <- v$tables[["secondary_populations.csv"]]
  expect_identical(p$n_rows[p$table == "secondary_scope.csv"], 0L)
})

test_that("a rerun reproduces tables and report byte for byte", {
  a <- d_args
  r <- verify_secondary_rerun(d1$result_dir, a$handoff_path, a$handoff_sha256, a$csv_path, d_primary$result_dir)
  expect_identical(r$comparison, "byte_identical_tables_and_report")
})

test_that("a changed table, forged or not, is detected", {
  d <- copy_derived()
  rewrite(d, "secondary_stage_a.csv", function(b) { b[length(b) - 2] <- as.raw(bitwXor(as.integer(b[length(b) - 2]), 1L)); b })
  expect_dverify_fails(d, pattern = "differs from the hash")
  forged <- copy_derived()
  rewrite(forged, "secondary_stage_a.csv", function(b) charToRaw(sub("preserved,[0-9]+,", "preserved,9,", rawToChar(b))))
  reseal(forged)
  expect_dverify_fails(forged, pattern = "denominator|count|canonical")
  expect_dverify_fails(forged, expected_record_sha256 = d1$record_sha256, pattern = "differs from the trusted")
})

test_that("a changed report passes content checks but fails the byte rerun", {
  d <- copy_derived()
  rewrite(d, REPORT_FILE, function(b) charToRaw(sub("rd_pp", "rd_PP", rawToChar(b), fixed = TRUE)))
  reseal(d)
  expect_true(verify_secondary(d)$ok)
  a <- d_args
  err <- tryCatch(verify_secondary_rerun(d, a$handoff_path, a$handoff_sha256, a$csv_path, d_primary$result_dir), run_error = function(e) e)
  expect_s3_class(err, "run_error")
  expect_match(conditionMessage(err), "report.md")
})

test_that("a report or figure without the synthetic statement is refused", {
  d <- copy_derived()
  rewrite(d, REPORT_FILE, function(b) charToRaw(sub(SYNTHETIC_BANNER, "RESULTS", rawToChar(b), fixed = TRUE)))
  reseal(d)
  expect_dverify_fails(d, pattern = "synthetic")
  f <- copy_derived()
  rewrite(f, FIGURE_FILE, function(b) charToRaw(sub(SYNTHETIC_BANNER, "RESULTS", rawToChar(b), fixed = TRUE)))
  reseal(f)
  expect_dverify_fails(f, pattern = "synthetic")
})

test_that("record claims, file set and links are checked", {
  d <- copy_derived()
  rec <- drecord(d); rec$scientific_authority <- "primary"
  writeBin(serialize_record(rec), file.path(d, SECONDARY_RECORD_FILE))
  expect_dverify_fails(d, pattern = "scientific_authority")
  e <- copy_derived(); file.remove(file.path(e, "secondary_scope.csv"))
  expect_dverify_fails(e, pattern = "missing")
  x <- copy_derived(); writeLines("x", file.path(x, "extra.txt"))
  expect_dverify_fails(x, pattern = "unexpected")
  expect_dverify_fails(d1$result_dir, expected_record_sha256 = strrep("0", 64), pattern = "differs from the trusted")
  expect_dverify_fails(copy_derived(), require_read_only = TRUE, pattern = "read-only")
})

test_that("a primary result from other inputs is refused", {
  other <- make_inputs(csv_edit = edit_cell(2, "attempt_id", "attempt_other"))
  mismatch_out <- file.path(d_root, "mismatch"); dir.create(mismatch_out)
  err <- tryCatch(derive(mismatch_out, primary = d_primary$result_dir,
                         args = list(handoff_path = other$handoff, handoff_sha256 = other$handoff_sha, csv_path = other$csv)),
                  error = function(e) e)
  expect_s3_class(err, "run_error")
  expect_match(conditionMessage(err), "different inputs")
  expect_length(list.files(mismatch_out, all.files = TRUE, no.. = TRUE), 0L)
  wrong_primary <- copy_derived()  # a secondary directory is not a primary result
  expect_error(compute_derived(d_args$handoff_path, d_args$handoff_sha256, d_args$csv_path, wrong_primary))
  expect_run_error(compute_derived(d_args$handoff_path, d_args$handoff_sha256, d_args$csv_path, d_primary$result_dir,
                                   expected_primary_record_sha256 = strrep("0", 64)), "verify")
})

test_that("nothing is replaced, and a failed run leaves no directory behind", {
  expect_run_error(derive(d_out1), "publish")
  expect_identical(verify_secondary(d1$result_dir)$secondary_result_sha256, d1$secondary_result_sha256)
  missing_root <- file.path(d_root, "absent_root")
  expect_run_error(derive(missing_root), "publish")
  unreadable <- function(dir) list(status = "unavailable", script_sha256 = character(0))
  out <- file.path(d_root, "code"); dir.create(out)
  expect_run_error(run_derived(d_args$handoff_path, d_args$handoff_sha256, d_args$csv_path, d_primary$result_dir, out,
                               analysis_dir(), code_fn = unreadable), "code")
  expect_length(list.files(out, all.files = TRUE, no.. = TRUE), 0L)
})

test_that("the figure handles undefined intervals and estimates honestly", {
  co <- verify_result(d_primary$result_dir)$tables[["coprimary_results.csv"]]
  undefined <- co
  undefined$boot_ci_lower <- NA_real_; undefined$boot_ci_upper <- NA_real_; undefined$boot_status <- "single_cluster_degenerate"
  none <- co
  none$rd_pp <- NA_real_; none$boot_ci_lower <- NA_real_; none$boot_ci_upper <- NA_real_
  none$summary_status <- "no_complete_pairs"; none$boot_status <- "no_complete_pairs"
  for (x in list(co, undefined, none)) {
    p <- tempfile(fileext = ".svg")
    render_figure(x, p)
    expect_gt(file.size(p), 500)
    expect_match(readLines(p, n = 3, warn = FALSE)[2], SYNTHETIC_BANNER, fixed = TRUE)
  }
  all_na <- none; all_na$rd_pp <- NA_real_
  expect_no_error(render_figure(all_na, tempfile(fileext = ".svg")))
})

test_that("secondary table consistency checks catch broken counts and statuses", {
  t <- verify_secondary(d1$result_dir)$tables
  expect_true(check_secondary_tables(t))
  bad <- t; bad[["secondary_stage_a.csv"]]$denominator[1] <- bad[["secondary_stage_a.csv"]]$denominator[1] + 1L
  expect_run_error(check_secondary_tables(bad), "verify")
  bad <- t; bad[["secondary_robustness.csv"]]$status[1] <- "ok"; bad[["secondary_robustness.csv"]]$rd[1] <- NA_real_
  expect_run_error(check_secondary_tables(bad), "verify")
  bad <- t; bad[["secondary_robustness.csv"]]$status[1] <- "no_complete_pairs"
  expect_run_error(check_secondary_tables(bad), "verify")
  bad <- t; a <- bad[["secondary_stage_a.csv"]]
  bad[["secondary_stage_a.csv"]] <- rbind(a, transform(a[1, ], n = 0L))
  expect_run_error(check_secondary_tables(bad), "verify")
  bad <- t; bad[["secondary_repeat_variability.csv"]]$n_items[1] <- bad[["secondary_repeat_variability.csv"]]$n_items[1] + 1L
  expect_run_error(check_secondary_tables(bad), "verify")
  bad <- t; bad[["secondary_unavailable.csv"]] <- bad[["secondary_unavailable.csv"]][-1, ]
  expect_run_error(check_secondary_tables(bad), "verify")
})

test_that("multi-arm tables survive the full write and strict read-back", {
  tabs <- compute_secondary_tables(syn_rows(), SYN_RULES)
  for (nm in names(tabs)) {
    back <- parse_canonical_csv(csv_bytes(tabs[[nm]], SECONDARY_SCHEMAS[[nm]]), SECONDARY_SCHEMAS[[nm]], nm)
    expect_equal(back, tabs[[nm]], ignore_attr = TRUE)
  }
  expect_true(check_secondary_tables(tabs))
})

cli_d <- function(...) {
  args <- c(file.path(analysis_dir(), "run_analysis.R"), ...)
  out <- withr::with_dir(analysis_dir(), suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE)))
  list(output = out, status = if (is.null(attr(out, "status"))) 0L else attr(out, "status"))
}

test_that("the command line derives, verifies and re-executes the secondary result", {
  skip_if(Sys.which("Rscript") == "")
  out <- cleanup_dir(tempfile("clid"), environment()); dir.create(out)
  r <- cli_d("derive", "--handoff", d_args$handoff_path, "--handoff-sha256", d_args$handoff_sha256, "--csv", d_args$csv_path,
             "--primary", d_primary$result_dir, "--primary-record-sha256", d_primary$record_sha256, "--out-root", out)
  expect_identical(r$status, 0L)
  expect_true(any(grepl("NOT SCIENTIFIC FINDINGS", r$output, fixed = TRUE)))
  dirs <- list.files(out, full.names = TRUE)
  expect_length(dirs, 1)
  expect_true(any(grepl(sha256_file(file.path(dirs, SECONDARY_RECORD_FILE)), r$output, fixed = TRUE)))
  expect_true(any(grepl(d1$secondary_result_sha256, r$output, fixed = TRUE)))
  v <- cli_d("verify-derived", "--result", dirs, "--primary", d_primary$result_dir, "--rerun", "--handoff", d_args$handoff_path,
             "--handoff-sha256", d_args$handoff_sha256, "--csv", d_args$csv_path)
  expect_identical(v$status, 0L)
  expect_true(any(grepl("byte_identical_tables_and_report", v$output, fixed = TRUE)))
  expect_true(any(grepl("NOT authenticated", v$output, fixed = TRUE)))
})

test_that("the derive command refuses missing or unknown arguments", {
  skip_if(Sys.which("Rscript") == "")
  expect_gt(cli_d("derive", "--csv", d_args$csv_path)$status, 0L)
  expect_gt(cli_d("verify-derived", "--result", d1$result_dir, "--rerun")$status, 0L)
  expect_gt(cli_d("derive", "--bogus", "x")$status, 0L)
})
