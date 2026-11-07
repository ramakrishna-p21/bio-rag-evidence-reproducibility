# Stage 3c: immutable publication, the integrity verifier, re-execution, gates and the CLI.

fixed_clock <- function(iso) function() as.POSIXct(iso, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS")

run_once <- function(out_root, seed = TEST_SEED, clock = fixed_clock("2026-11-06T10:00:00.000"),
                     label = "synthetic_rehearsal", lock_sha = real_lock_sha(), args = fixture_args()) {
  run_analysis(args$handoff_path, args$handoff_sha256, args$csv_path, lock_sha, seed, label, out_root,
               analysis_dir(), now = clock, synchronized = TRUE)
}

base_out <- cleanup_dir(tempfile("results"), teardown_env())
dir.create(base_out)
base <- run_once(base_out)

copy_result <- function(dir = base$result_dir) {
  dest <- cleanup_dir(tempfile("copy"), parent.frame())
  dir.create(dest)
  file.copy(list.files(dir, full.names = TRUE), dest)
  Sys.chmod(list.files(dest, full.names = TRUE), "0644")
  dest
}
read_record <- function(dir) jsonlite::fromJSON(file.path(dir, "run_record.json"), simplifyVector = FALSE)
write_record <- function(dir, record) writeBin(serialize_record(record), file.path(dir, "run_record.json"))
flip_byte <- function(path, from_end = 2) {
  b <- readBin(path, "raw", file.size(path))
  i <- length(b) - from_end
  b[i] <- as.raw(bitwXor(as.integer(b[i]), 1L))
  writeBin(b, path)
}

# An edit that leaves the record internally consistent: new file bytes, new file hashes and a new
# scientific identity. Internal consistency is not authentication; these are what an external
# record hash and re-execution must catch.
forge <- function(dir, csv_edits = list(), record_edit = identity, keep_inventory_field = FALSE) {
  rec <- read_record(dir)
  for (name in names(csv_edits)) {
    path <- file.path(dir, name)
    writeBin(charToRaw(csv_edits[[name]](rawToChar(readBin(path, "raw", file.size(path))))), path)
  }
  for (name in setdiff(RESULT_FILES, "run_record.json")) {
    bytes <- readBin(file.path(dir, name), "raw", file.size(file.path(dir, name)))
    rec$outputs$files[[name]] <- list(bytes = length(bytes), sha256 = sha256_raw(bytes))
  }
  if (!keep_inventory_field) {
    rec$environment_actual$installed_package_inventory_sha256 <- rec$outputs$files[["package_inventory.csv"]]$sha256
  }
  sci <- vapply(names(SCIENTIFIC_SCHEMAS), function(n) rec$outputs$files[[n]]$sha256, "")
  rec$outputs$scientific_result_sha256 <- scientific_result_id(rec$inputs$handoff_sha256, rec$inputs$csv_sha256,
                                                              rec$seed$value, sci)
  write_record(dir, record_edit(rec))
  dir
}
expect_verify_fails <- function(dir, ..., pattern = NULL) {
  err <- tryCatch({ verify_result(dir, ...); NULL }, run_error = function(e) e)
  expect_false(is.null(err), info = "verify_result did not fail")
  if (!is.null(err)) {
    expect_match(conditionMessage(err), "[verify]", fixed = TRUE)
    if (!is.null(pattern)) expect_match(conditionMessage(err), pattern)
  }
}

test_that("a run publishes exactly five read-only files in a rehearsal directory", {
  expect_match(basename(base$result_dir), "^rehearsal_")
  expect_identical(sort(list.files(base$result_dir, all.files = TRUE, no.. = TRUE), method = "radix"),
                   sort(RESULT_FILES, method = "radix"))
  expect_identical(list.files(base_out, all.files = TRUE, no.. = TRUE), basename(base$result_dir))   # no staging left
  for (f in list.files(base$result_dir, full.names = TRUE)) expect_equal(unname(file.mode(f) & as.octmode("0222")), as.octmode("0000"))
  expect_equal(unname(file.mode(base$result_dir) & as.octmode("0222")), as.octmode("0000"))
  expect_identical(base$record_sha256, sha256_file(file.path(base$result_dir, "run_record.json")))
  expect_identical(base$scientific_result_sha256, "17c44709e57fd9ac259f3d239498693d31e34258f8577d9e73d0e6cb86b01e36")
})

test_that("a published result verifies, and only an external record hash authenticates the record", {
  v <- verify_result(base$result_dir, require_read_only = TRUE)
  expect_false(v$record_authenticated)
  expect_identical(v$record_sha256, base$record_sha256)
  a <- verify_result(base$result_dir, expected_record_sha256 = base$record_sha256, expected_lock_sha256 = real_lock_sha())
  expect_true(a$record_authenticated)
  expect_verify_fails(base$result_dir, expected_record_sha256 = strrep("0", 64))
  expect_verify_fails(base$result_dir, expected_lock_sha256 = strrep("0", 64))
})

test_that("the run record separates the verification levels and states what it does not know", {
  r <- read_record(base$result_dir)
  expect_identical(r$kind, "r_run_record"); expect_identical(r$purpose, "synthetic_rehearsal")
  expect_identical(r$scientific_authority, "none")
  v <- r$verification
  expect_identical(v$input_origin, "synthetic_rehearsal")
  expect_identical(v$python_preflight, "not_recorded")
  expect_false(v$lineage_verified_by_r)
  expect_identical(v$official_source_runs, "not_applicable_synthetic_rehearsal")
  expect_identical(v$statistical_execution, "completed")
  expect_identical(v$record_authentication, "none_without_external_record_sha256")
  expect_identical(r$seed$value, TEST_SEED); expect_identical(r$seed$registry, "not_registered")
  expect_identical(r$seed$production_seed, "awaiting_approval")
  expect_identical(r$environment_intended$renv_lock_sha256, real_lock_sha())
  expect_identical(r$environment_actual$package_versions_used$clubSandwich, "0.7.0")
  expect_true(r$environment_actual$renv_synchronized)
  expect_identical(r$inputs$handoff_sha256, fixture_args()$handoff_sha256)
  expect_identical(r$inputs$csv_sha256, jsonlite::fromJSON(file.path(fixture_dir(), "fixture.json"))$csv_sha256)
  expect_identical(r$code$git_commit, trimws(system2("git", c("-C", analysis_dir(), "rev-parse", "HEAD"), stdout = TRUE)))
  expect_true(all(c("R/results.R", "R/publish.R", "R/provenance.R", "run_analysis.R") %in% names(r$code$script_sha256)))
  # The nine names the handoff requires from the R run are all present and populated.
  h <- jsonlite::fromJSON(fixture_args()$handoff_path, simplifyVector = FALSE)
  expect_setequal(unlist(r$provenance$required), unlist(h$execution_provenance_required))
  expect_setequal(names(REQUIRED_PROVENANCE), unlist(h$execution_provenance_required))
  expect_identical(r$environment_actual$installed_package_inventory_sha256, r$outputs$files[["package_inventory.csv"]]$sha256)
})

test_that("the inventory file lists actual versions for every locked package", {
  inv <- utils::read.csv(file.path(base$result_dir, "package_inventory.csv"), colClasses = "character", na.strings = character(0))
  l <- real_lock()
  expect_identical(setNames(inv$version, inv$package)[names(l$packages)], l$packages)
  expect_true(all(c("base", "stats", "MASS") %in% inv$package))
})

test_that("a second run is a separate directory with the same scientific result", {
  second <- run_once(base_out, clock = fixed_clock("2026-11-06T10:00:01.000"))
  expect_false(identical(second$result_dir, base$result_dir))
  expect_identical(second$scientific_result_sha256, base$scientific_result_sha256)
  for (f in names(SCIENTIFIC_SCHEMAS)) {
    expect_identical(sha256_file(file.path(second$result_dir, f)), sha256_file(file.path(base$result_dir, f)))
  }
  a <- read_record(base$result_dir); b <- read_record(second$result_dir)
  expect_false(identical(a$execution$run_id, b$execution$run_id))
  expect_false(identical(second$record_sha256, base$record_sha256))
  expect_identical(a[c("inputs", "seed", "environment_intended", "verification")], b[c("inputs", "seed", "environment_intended", "verification")])
  expect_identical(a$outputs$scientific_result_sha256, b$outputs$scientific_result_sha256)
  expect_true(verify_result(second$result_dir)$ok)
})

test_that("a repeated run id never replaces the published result", {
  before <- sha256_file(file.path(base$result_dir, "run_record.json"))
  expect_run_error(run_once(base_out), "publish")
  expect_identical(sha256_file(file.path(base$result_dir, "run_record.json")), before)
  expect_false(any(startsWith(list.files(base_out, all.files = TRUE), ".staging")))
  expect_true(verify_result(base$result_dir)$ok)
})

stage_copy <- function() {
  s <- cleanup_dir(tempfile("stage"), parent.frame()); dir.create(s)
  file.copy(list.files(base$result_dir, full.names = TRUE), s)
  Sys.chmod(list.files(s, full.names = TRUE), "0644")
  s
}

test_that("publish_result refuses an existing directory and leaves it untouched", {
  stage <- stage_copy()
  target <- cleanup_dir(tempfile("target"), environment()); dir.create(target)
  writeLines("precious", file.path(target, "keep.txt"))
  expect_run_error(publish_result(stage, target), "publish")
  expect_identical(list.files(target), "keep.txt")
  expect_identical(readLines(file.path(target, "keep.txt")), "precious")
  empty <- cleanup_dir(tempfile("empty"), environment()); dir.create(empty)
  expect_run_error(publish_result(stage, empty), "publish")                       # even an empty directory is not replaced
  expect_identical(list.files(empty, all.files = TRUE, no.. = TRUE), character(0))
})

test_that("a failure while publishing leaves no published directory", {
  stage <- stage_copy()
  target <- file.path(cleanup_dir(tempfile("o"), environment()), "rehearsal_x"); dir.create(dirname(target))
  count <- 0
  flaky <- function(from, to) { count <<- count + 1; if (count == 3) FALSE else file.link(from, to) }
  expect_run_error(publish_result(stage, target, link = flaky), "publish")
  expect_false(file.exists(target))
  expect_true(file.exists(file.path(stage, "run_record.json")))                    # the staging copy is still intact
})

test_that("read-back verification uses the published bytes and removes a corrupt publication", {
  stage <- stage_copy()
  target <- file.path(cleanup_dir(tempfile("o"), environment()), "rehearsal_y"); dir.create(dirname(target))
  corrupting <- function(from, to) {
    file.copy(from, to)
    if (basename(to) == "paper_counts.csv") flip_byte(to)
    TRUE
  }
  expect_run_error(publish_result(stage, target, link = corrupting), "verify")
  expect_false(file.exists(target))
})

test_that("a publication that cannot be made read-only is reported, not accepted", {
  stage <- stage_copy()
  target <- file.path(cleanup_dir(tempfile("o"), environment()), "rehearsal_z"); dir.create(dirname(target))
  expect_run_error(publish_result(stage, target, chmod = function(...) invisible(FALSE)), "verify")
})

test_that("a failed staging verification publishes nothing and cleans up", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  broken <- function(...) stop("simulated failure")
  expect_error(run_analysis(fixture_args()$handoff_path, fixture_args()$handoff_sha256, fixture_args()$csv_path,
                            real_lock_sha(), TEST_SEED, "synthetic_rehearsal", out, analysis_dir(),
                            now = broken, synchronized = TRUE), "simulated failure")
  expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))
})

test_that("the verifier rejects a changed byte in every data file", {
  for (f in setdiff(RESULT_FILES, "run_record.json")) {
    d <- copy_result(); flip_byte(file.path(d, f))
    expect_verify_fails(d)
  }
})

test_that("the verifier rejects a record that was changed", {
  d <- copy_result(); flip_byte(file.path(d, "run_record.json"), from_end = 6)
  expect_verify_fails(d, expected_record_sha256 = base$record_sha256)           # internally consistent changes need the trusted hash
  d2 <- copy_result()
  txt <- rawToChar(readBin(file.path(d2, "run_record.json"), "raw", file.size(file.path(d2, "run_record.json"))))
  writeBin(charToRaw(replace_once(txt, '"seed": {', '"seed": {\n    "extra": 1,')), file.path(d2, "run_record.json"))
  expect_verify_fails(d2, expected_record_sha256 = base$record_sha256)
})

test_that("the verifier rejects missing, unexpected and non-regular files", {
  for (f in RESULT_FILES) {
    d <- copy_result(); file.remove(file.path(d, f))
    expect_verify_fails(d, pattern = paste0("missing file.*", f))
  }
  d <- copy_result(); writeLines("x", file.path(d, "notes.txt")); expect_verify_fails(d, pattern = "unexpected file.*notes.txt")
  d <- copy_result(); dir.create(file.path(d, "sub")); expect_verify_fails(d, pattern = "unexpected file.*sub")
  d <- copy_result(); file.remove(file.path(d, "paper_counts.csv"))
  file.symlink(file.path(base$result_dir, "paper_counts.csv"), file.path(d, "paper_counts.csv")); expect_verify_fails(d, pattern = "regular file")
  expect_verify_fails(file.path(tempdir(), "does-not-exist"))
})

test_that("require_read_only fails on a writable copy", {
  expect_verify_fails(copy_result(), require_read_only = TRUE)
})

cell_text <- function(dir, file, column, row = 1) {
  utils::read.csv(file.path(dir, file), colClasses = "character", na.strings = character(0))[[column]][row]
}

test_that("an internally consistent forgery verifies without an external hash and fails with it", {
  se <- cell_text(base$result_dir, "coprimary_results.csv", "test_se")
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub(se, "0.25", t, fixed = TRUE)))
  expect_true(verify_result(d)$ok)                                                    # consistency alone proves nothing
  expect_false(verify_result(d)$record_authenticated)
  expect_verify_fails(d, expected_record_sha256 = base$record_sha256)                 # the trusted hash catches it
  args <- fixture_args()
  expect_run_error(verify_rerun(d, args$handoff_path, args$handoff_sha256, args$csv_path), "verify")   # so does re-execution
})

test_that("the verifier checks formats and sanity even when hashes are consistent", {
  d <- forge(copy_result(), csv_edits = list("paper_counts.csv" = function(t) paste0(t, "\n")))        # blank line
  expect_verify_fails(d)
  p17 <- cell_text(base$result_dir, "coprimary_results.csv", "test_p")
  p15 <- sprintf("%.15g", as.numeric(p17)); expect_false(identical(p15, p17))
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub(p17, p15, t, fixed = TRUE)))
  expect_verify_fails(d)                                                              # not the 17-digit canonical form
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub(",ok,1,0,0.2222", ",ok,0.5,0,0.2222", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "Holm")                                           # Holm below the raw p-value
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub("evidence_delivered,9,7", "evidence_delivered,6,7", t, fixed = TRUE)))
  expect_verify_fails(d)                                                              # more complete than eligible
  d <- forge(copy_result(), csv_edits = list("paper_counts.csv" = function(t) sub("P001,1,1", "P001,1,0", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "paper_counts")                                    # per-paper counts disagree with the table
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub("evidence_delivered,9,7,5", "evidence_delivered,9,7,4", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "paper_counts")                                    # only the paper count differs
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub("-0.25,0.80000000000000004", "0.80000000000000004,-0.25", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "interval")                                        # reversed bootstrap interval
  d <- forge(copy_result(), csv_edits = list("coprimary_results.csv" = function(t) sub(",0,0.22222222222222221,1,1", ",0.22222222222222221,0,1,1", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "interval")                                        # reversed missingness bounds
  d <- forge(copy_result(), csv_edits = list("state_counts.csv" = function(t) sub("JATS,no,1", "JATS,no,2", t, fixed = TRUE)))
  expect_verify_fails(d)
})

test_that("the verifier cross-checks the inventory, the lock and the versions used", {
  d <- forge(copy_result(), csv_edits = list("package_inventory.csv" = function(t) sub("clubSandwich,0.7.0", "clubSandwich,0.6.9", t, fixed = TRUE)))
  expect_verify_fails(d)                                                              # inventory contradicts the lock
  d <- forge(copy_result(), keep_inventory_field = TRUE, csv_edits = list("package_inventory.csv" = function(t) sub("r_base", "r_other", t, fixed = TRUE)))
  expect_verify_fails(d, pattern = "inventory hash")                                  # inventory hash field not updated
  d <- forge(copy_result(), record_edit = function(r) { r$environment_actual$package_versions_used$clubSandwich <- "0.6.9"; r })
  expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$environment_intended$lock_packages$clubSandwich <- "0.6.9"; r })
  expect_verify_fails(d)
})

test_that("the verifier refuses records that claim more than this pass can establish", {
  claims <- list(
    function(r) { r$verification$python_preflight <- "passed"; r },
    function(r) { r$verification$lineage_verified_by_r <- TRUE; r },
    function(r) { r$scientific_authority <- "final"; r },
    function(r) { r$purpose <- "official"; r },
    function(r) { r$verification$input_origin <- "verified_official_run"; r },
    function(r) { r$inputs$input_origin <- "verified_official_run"; r },
    function(r) { r$seed$registry <- "primary_bootstrap"; r },
    function(r) { r$seed$production_seed <- "approved"; r },
    function(r) { r$kind <- "analysis_result_record"; r },
    function(r) { r$schema_version <- "2"; r }
  )
  for (edit in claims) expect_verify_fails(forge(copy_result(), record_edit = edit))
})

test_that("the record must describe its files and identity exactly", {
  d <- forge(copy_result(), record_edit = function(r) { r$outputs$files$extra.csv <- list(bytes = 1L, sha256 = strrep("a", 64)); r })
  expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$outputs$files[["paper_counts.csv"]]$bytes <- 1L; r })
  expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$seed$value <- 5L; r })              # identity no longer matches the seed
  expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$outputs$scientific_result_sha256 <- strrep("a", 64); r })
  expect_verify_fails(d)
})

test_that("the provenance checklist must be complete and populated", {
  d <- forge(copy_result(), record_edit = function(r) { r$provenance$required[[1]] <- NULL; r }); expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$code$git_commit <- NA_character_; r }); expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$code$script_sha256 <- list(); r }); expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$provenance$required[[1]] <- "something_else"; r }); expect_verify_fails(d)
  d <- forge(copy_result(), record_edit = function(r) { r$environment_actual$package_versions_used <- list(); r }); expect_verify_fails(d)
})

test_that("re-execution reproduces the published scientific bytes", {
  args <- fixture_args()
  r <- verify_rerun(base$result_dir, args$handoff_path, args$handoff_sha256, args$csv_path)
  expect_identical(r$comparison, "byte_identical")
  expect_identical(r$scientific_result_sha256, base$scientific_result_sha256)
  other <- make_inputs(csv_edit = function(t) sub("q_001", "q_00x", t, fixed = TRUE))
  expect_error(verify_rerun(base$result_dir, other$handoff, other$handoff_sha, other$csv))
  expect_run_error(verify_rerun(base$result_dir, args$handoff_path, strrep("0", 64), args$csv_path), "verify")
})

test_that("re-execution refuses a record whose CSV hash is not the supplied table's", {
  args <- fixture_args()
  d <- forge(copy_result(), record_edit = function(r) {
    r$inputs$csv_sha256 <- strrep("a", 64)
    sci <- vapply(names(SCIENTIFIC_SCHEMAS), function(n) r$outputs$files[[n]]$sha256, "")
    r$outputs$scientific_result_sha256 <- scientific_result_id(r$inputs$handoff_sha256, r$inputs$csv_sha256, r$seed$value, sci)
    r
  })
  expect_true(verify_result(d)$ok)
  expect_run_error(verify_rerun(d, args$handoff_path, args$handoff_sha256, args$csv_path), "verify")
})

test_that("re-execution tolerates last-digit differences but reports them separately", {
  d <- copy_result(); args <- fixture_args()
  se <- cell_text(base$result_dir, "coprimary_results.csv", "test_se")
  bumped <- format_double(as.numeric(se) * (1 + 1e-14))
  expect_false(identical(bumped, se))
  forge(d, csv_edits = list("coprimary_results.csv" = function(t) sub(se, bumped, t, fixed = TRUE)))
  r <- verify_rerun(d, args$handoff_path, args$handoff_sha256, args$csv_path)
  expect_identical(r$comparison, "numerically_equal_not_byte_identical")
  expect_false(identical(r$scientific_result_sha256, read_record(d)$outputs$scientific_result_sha256))
})

test_that("the gates stop a run before anything is published", {
  args <- fixture_args()
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  none <- function() expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))
  expect_run_error(run_once(out, lock_sha = strrep("0", 64)), "environment"); none()
  expect_run_error(run_once(out, label = "primary_bootstrap"), "seed"); none()
  expect_run_error(run_once(out, seed = NULL), "seed"); none()
  expect_run_error(run_once(out, seed = 1.5), "seed"); none()
  official <- make_inputs(handoff_edit = function(t) replace_once(t, '"input_origin": "synthetic_rehearsal"', '"input_origin": "verified_official_run"'))
  expect_error(run_analysis(official$handoff, official$handoff_sha, official$csv, real_lock_sha(), TEST_SEED,
                            "synthetic_rehearsal", out, analysis_dir(), synchronized = TRUE)); none()
  tampered <- make_inputs(csv_edit = function(t) sub("q_001", "q_00x", t, fixed = TRUE), repin_csv = FALSE)
  expect_error(run_analysis(tampered$handoff, tampered$handoff_sha, tampered$csv, real_lock_sha(), TEST_SEED,
                            "synthetic_rehearsal", out, analysis_dir(), synchronized = TRUE), class = "input_error"); none()
  wrong_sha <- make_inputs(handoff_sha = strrep("0", 64))
  expect_input_error(run_analysis(wrong_sha$handoff, wrong_sha$handoff_sha, wrong_sha$csv, real_lock_sha(), TEST_SEED,
                                  "synthetic_rehearsal", out, analysis_dir(), synchronized = TRUE), "handoff_hash"); none()
  expect_run_error(run_analysis(args$handoff_path, args$handoff_sha256, args$csv_path, real_lock_sha(), TEST_SEED,
                                "synthetic_rehearsal", file.path(out, "missing"), analysis_dir(), synchronized = TRUE), "publish")
})

test_that("a run whose code state cannot be read is refused before publishing", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  a <- fixture_args()
  unreadable <- function(dir) { s <- code_state(dir); s$status <- "not_a_repository"; s }
  expect_run_error(run_analysis(a$handoff_path, a$handoff_sha256, a$csv_path, real_lock_sha(), TEST_SEED,
                                "synthetic_rehearsal", out, analysis_dir(), synchronized = TRUE, code_fn = unreadable), "code")
  expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))
})

ANALYSIS_DIR <- analysis_dir()
cli <- function(..., dir = ANALYSIS_DIR) {
  args <- c(file.path(ANALYSIS_DIR, "run_analysis.R"), ...)                          # evaluate before the directory changes
  out <- withr::with_dir(dir, suppressWarnings(system2("Rscript", args, stdout = TRUE, stderr = TRUE)))
  list(output = out, status = if (is.null(attr(out, "status"))) 0L else attr(out, "status"))
}

test_that("the command line runs, verifies and re-executes a rehearsal", {
  skip_if(Sys.which("Rscript") == "")
  a <- fixture_args()
  out <- cleanup_dir(tempfile("cli"), environment()); dir.create(out)
  r <- cli("run", "--handoff", a$handoff_path, "--handoff-sha256", a$handoff_sha256, "--csv", a$csv_path,
           "--lock-sha256", real_lock_sha(), "--seed", "20260601", "--seed-label", "synthetic_rehearsal", "--out-root", out)
  expect_identical(r$status, 0L)
  dirs <- list.files(out, full.names = TRUE)
  expect_length(dirs, 1)
  rec_sha <- sha256_file(file.path(dirs, "run_record.json"))
  expect_true(any(grepl(rec_sha, r$output, fixed = TRUE)))
  expect_true(any(grepl("synthetic rehearsal", r$output, ignore.case = TRUE)))
  v <- cli("verify", "--result", dirs)
  expect_identical(v$status, 0L)
  expect_true(any(grepl("NOT authenticated", v$output, fixed = TRUE)))
  v2 <- cli("verify", "--result", dirs, "--record-sha256", rec_sha, "--lock-sha256", real_lock_sha())
  expect_identical(v2$status, 0L)
  expect_true(any(grepl("authenticated", v2$output)) && !any(grepl("NOT authenticated", v2$output, fixed = TRUE)))
  v3 <- cli("verify", "--result", dirs, "--rerun", "--handoff", a$handoff_path, "--handoff-sha256", a$handoff_sha256, "--csv", a$csv_path)
  expect_identical(v3$status, 0L)
  expect_true(any(grepl("byte_identical", v3$output)))
  bad <- cli("verify", "--result", dirs, "--record-sha256", strrep("0", 64))
  expect_false(identical(bad$status, 0L))
  unlock_tree(dirs)
})

test_that("the command line refuses missing options, unknown options and a wrong working directory", {
  expect_false(identical(cli("run")$status, 0L))
  expect_false(identical(cli("frobnicate")$status, 0L))
  expect_true(any(grepl("[usage]", cli("run")$output, fixed = TRUE)))
  a <- fixture_args(); out <- cleanup_dir(tempfile("cli"), environment()); dir.create(out)
  args <- c("run", "--handoff", a$handoff_path, "--handoff-sha256", a$handoff_sha256, "--csv", a$csv_path,
            "--lock-sha256", real_lock_sha(), "--seed", "1", "--seed-label", "synthetic_rehearsal", "--out-root", out)
  usage <- function(r) expect_true(any(grepl("[usage]", r$output, fixed = TRUE)) && !identical(r$status, 0L))
  usage(cli(args, "--bogus", "1"))
  usage(cli(args, "--seed", "2"))                                                       # repeated option
  usage(cli("verify", "--result", out, "--rerun"))                                      # --rerun needs its inputs
  usage(cli("verify", "--result", out, "--csv", a$csv_path))                            # inputs without --rerun
  r <- cli(args, dir = tempdir())                                                     # renv would not be active here
  expect_false(identical(r$status, 0L))
  expect_identical(list.files(out), character(0))
})
