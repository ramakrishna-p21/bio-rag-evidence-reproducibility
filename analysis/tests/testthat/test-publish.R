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
  expect_identical(base$scientific_result_sha256, "075ec3af857b5eb994080078260b56154be4078ceec4c5855ff128242d6cf63d")
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

fresh_target <- function(envir = parent.frame()) {
  root <- cleanup_dir(tempfile("o"), envir); dir.create(root)
  file.path(root, "rehearsal_t")
}
recording_link <- function() {
  calls <- character(0); target_seen <- logical(0)
  list(link = function(from, to) {
         calls <<- c(calls, basename(to)); target_seen <<- c(target_seen, dir.exists(dirname(to)))
         file.link(from, to)
       },
       calls = function() calls, target_seen = function() target_seen)
}

test_that("the four data files are linked before the record, which is linked last", {
  stage <- stage_copy(); target <- fresh_target(); rec <- recording_link()
  publish_result(stage, target, link = rec$link)
  expect_identical(rec$calls(), c(DATA_FILES, "run_record.json"))
  expect_identical(utils::tail(rec$calls(), 1), RECORD_FILE)
  expect_setequal(utils::head(rec$calls(), -1), DATA_FILES)
  expect_true(verify_result(target, require_read_only = TRUE)$ok)
})

test_that("invalid staging content is rejected before any destination is created or linked", {
  breakers <- list(
    corrupt_data = function(s) flip_byte(file.path(s, "paper_counts.csv")),
    stale_record = function(s) { r <- read_record(s); r$outputs$files[["state_counts.csv"]]$sha256 <- strrep("0", 64); write_record(s, r) },
    missing_data = function(s) file.remove(file.path(s, "state_counts.csv")),
    missing_record = function(s) file.remove(file.path(s, "run_record.json")),
    extra_file = function(s) writeLines("x", file.path(s, "notes.txt"))
  )
  for (name in names(breakers)) {
    stage <- stage_copy(); breakers[[name]](stage)
    target <- fresh_target(); rec <- recording_link()
    err <- tryCatch(publish_result(stage, target, link = rec$link), error = function(e) e)
    expect_s3_class(err, "run_error"); expect_match(conditionMessage(err), "[verify]", fixed = TRUE)
    expect_identical(rec$calls(), character(0), info = name)          # nothing was linked
    expect_false(file.exists(target), info = name)                    # the destination never appeared
  }
})

test_that("an existing destination file is never replaced and the call stops", {
  stage <- stage_copy(); target <- fresh_target(); n <- 0
  planted <- function(from, to) {
    n <<- n + 1
    if (n == 2) writeLines("foreign", to)                            # another writer wins the race for this name
    suppressWarnings(file.link(from, to))                            # base R warns "File exists"; the FALSE result is what matters
  }
  expect_run_error(publish_result(stage, target, link = planted), "publish")
  expect_identical(readLines(file.path(target, "paper_counts.csv")), "foreign")
  expect_false("run_record.json" %in% list.files(target))
  expect_verify_fails(target, pattern = "missing file")             # the leftover is not a result
  expect_run_error(publish_result(stage, target), "publish")        # and the name cannot be used again
})

test_that("a failed link at any position leaves no destination", {
  for (k in seq_along(RESULT_FILES)) {
    stage <- stage_copy(); target <- fresh_target(); n <- 0
    failing <- function(from, to) { n <<- n + 1; if (n == k) FALSE else file.link(from, to) }
    expect_run_error(publish_result(stage, target, link = failing), "publish")
    expect_false(file.exists(target), info = paste("failed link", k))
  }
})

test_that("an interruption after any number of links leaves a directory the verifier rejects", {
  for (k in 0:4) {
    stage <- stage_copy(); target <- fresh_target(); n <- 0
    interrupted <- function(from, to) { if (n == k) stop("interrupted"); n <<- n + 1; file.link(from, to) }
    err <- tryCatch(publish_result(stage, target, link = interrupted), error = function(e) e)
    expect_false(inherits(err, "run_error"))                          # an interrupt is not a handled failure: nothing cleans up
    expect_identical(conditionMessage(err), "interrupted")
    expect_identical(list.files(target), sort(DATA_FILES[seq_len(k)], method = "radix"), info = paste("links", k))
    expect_false(file.exists(file.path(target, "run_record.json")))   # the record is the completion marker and is absent
    expect_verify_fails(target, pattern = "missing file.*run_record.json")
    expect_run_error(publish_result(stage, target), "publish")        # the leftover name is not reusable
    expect_identical(list.files(target), sort(DATA_FILES[seq_len(k)], method = "radix"))   # and was not touched
  }
})

test_that("a complete, hash-consistent directory without the read-only seal passes content verification only", {
  stage <- stage_copy(); target <- fresh_target()
  err <- tryCatch(publish_result(stage, target, chmod = function(...) invisible(FALSE)), error = function(e) e)
  expect_s3_class(err, "run_error"); expect_match(conditionMessage(err), "read-only")
  expect_identical(sort(list.files(target), method = "radix"), sort(RESULT_FILES, method = "radix"))
  v <- verify_result(target)                                        # content integrity, as documented
  expect_true(v$ok); expect_false(v$sealed); expect_false(v$record_authenticated)
  expect_verify_fails(target, require_read_only = TRUE, pattern = "read-only")
  sealed <- verify_result(base$result_dir)
  expect_true(sealed$sealed)                                        # a sealed result reports it
  expect_false(verify_result(copy_result())$sealed)
  # Read-only files in a writable directory are not sealed: files could still be added or removed.
  d <- copy_result(); Sys.chmod(list.files(d, full.names = TRUE), "0444")
  expect_false(verify_result(d)$sealed)
  expect_verify_fails(d, require_read_only = TRUE, pattern = "read-only")
  Sys.chmod(d, "0555"); expect_true(verify_result(d)$sealed)
  Sys.chmod(d, "0755")
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
  expect_true(any(grepl("content verified", v$output, fixed = TRUE)))
  expect_true(any(grepl("read-only seal: present", v$output, fixed = TRUE)))
  expect_true(any(grepl("NOT authenticated", v$output, fixed = TRUE)))
  expect_true(any(grepl("no scientific authority", v$output, fixed = TRUE)))
  writable <- copy_result(dirs)
  u <- cli("verify", "--result", writable)
  expect_identical(u$status, 0L)                                                  # content integrity does not depend on the seal
  expect_true(any(grepl("content verified", u$output, fixed = TRUE)))
  expect_true(any(grepl("read-only seal: ABSENT", u$output, fixed = TRUE)))
  expect_false(any(grepl("read-only seal: present", u$output, fixed = TRUE)))
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

# ---- citing a Python preflight -------------------------------------------------------------------
# R hashes the cited bytes and records the hash. It does not read them as Python's verification and
# never claims to have repeated it. The scientific files, the seed and the RNG are not involved.

PRE_SCOPE <- "bytes_hashed_by_r_only_contents_not_read_python_verification_not_repeated"
preflight_file <- function(content = "preflight bytes", envir = parent.frame()) {
  path <- write_text(content, tempfile("preflight"))
  withr::defer(unlink(path), envir = envir)
  list(path = path, sha = sha256_file(path))
}
run_cited <- function(out_root, pre, ...) {
  a <- fixture_args()
  run_analysis(a$handoff_path, a$handoff_sha256, a$csv_path, real_lock_sha(), TEST_SEED, "synthetic_rehearsal", out_root,
               analysis_dir(), now = fixed_clock("2026-11-06T10:00:00.000"), synchronized = TRUE,
               preflight_path = pre$path, preflight_sha256 = pre$sha, ...)
}
expect_preflight_refused <- function(out, ..., pattern = NULL) {
  err <- tryCatch({ run_cited(out, ...); NULL }, run_error = function(e) e)
  expect_false(is.null(err), info = "the run was not refused")
  if (!is.null(err)) {
    expect_match(conditionMessage(err), "[preflight]", fixed = TRUE)
    if (!is.null(pattern)) expect_match(conditionMessage(err), pattern)
  }
  expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))   # nothing staged or published
}

test_that("a run without a preflight keeps the previous record structure and behaviour", {
  r <- read_record(base$result_dir)
  expect_identical(names(r$inputs), c("handoff_sha256", "csv_sha256", "input_origin", "handoff_kind",
                                      "handoff_schema_version", "analysis_spec_version"))
  expect_identical(names(r$verification), c("input_origin", "r_byte_verification", "python_preflight", "lineage_verified_by_r",
                                            "official_source_runs", "statistical_execution", "output_integrity",
                                            "record_authentication"))
  expect_identical(r$verification$python_preflight, "not_recorded")
})

test_that("a cited preflight is hashed and recorded without being interpreted", {
  out <- cleanup_dir(tempfile("cited"), environment()); dir.create(out)
  pre <- preflight_file()
  res <- run_cited(out, pre)
  r <- read_record(res$result_dir)
  expect_identical(r$verification$python_preflight, "cited_bytes_hashed_not_interpreted")
  expect_identical(r$verification$python_preflight_scope, PRE_SCOPE)
  expect_identical(r$inputs$python_preflight_sha256, pre$sha)
  expect_identical(r$inputs$python_preflight_bytes, as.integer(file.size(pre$path)))
  expect_false(r$verification$lineage_verified_by_r)                              # R still claims no lineage check
  expect_identical(r$verification$official_source_runs, "not_applicable_synthetic_rehearsal")
  expect_identical(r$scientific_authority, "none")
  v <- verify_result(res$result_dir, expected_record_sha256 = res$record_sha256)
  expect_true(v$record_authenticated)
  # Science is untouched: same identity, same scientific bytes, same seed record as the run without one.
  expect_identical(res$scientific_result_sha256, "075ec3af857b5eb994080078260b56154be4078ceec4c5855ff128242d6cf63d")
  expect_identical(res$scientific_result_sha256, base$scientific_result_sha256)
  for (f in names(SCIENTIFIC_SCHEMAS)) {
    expect_identical(read_bytes(file.path(res$result_dir, f)), read_bytes(file.path(base$result_dir, f)))
  }
  expect_identical(r$seed, read_record(base$result_dir)$seed)
  expect_identical(r$outputs$files, read_record(base$result_dir)$outputs$files)
})

test_that("a preflight needs both its path and its hash", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  pre <- preflight_file()
  expect_preflight_refused(out, list(path = pre$path, sha = NULL), pattern = "both its path and its SHA-256")
  expect_preflight_refused(out, list(path = NULL, sha = pre$sha), pattern = "both its path and its SHA-256")
})

test_that("a cited preflight must exist, be a regular file and match its hash exactly", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  pre <- preflight_file()
  for (bad in c("", "abc", strrep("A", 64), substr(pre$sha, 1, 63), paste0(pre$sha, "0"), toupper(pre$sha), NA_character_)) {
    expect_preflight_refused(out, list(path = pre$path, sha = bad), pattern = "hexadecimal")
  }
  expect_preflight_refused(out, list(path = pre$path, sha = strrep("0", 64)), pattern = "differ")
  expect_preflight_refused(out, list(path = file.path(tempdir(), "no-such-preflight"), sha = pre$sha), pattern = "not found")
  expect_preflight_refused(out, list(path = tempdir(), sha = pre$sha), pattern = "regular file")
  link <- file.path(tempfile("link")); file.symlink(pre$path, link); withr::defer(unlink(link))
  expect_preflight_refused(out, list(path = link, sha = pre$sha), pattern = "regular file")
  other <- preflight_file("different bytes")
  expect_preflight_refused(out, list(path = pre$path, sha = other$sha), pattern = "differ")
})

test_that("an empty preflight file is refused", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  empty <- preflight_file("")
  expect_preflight_refused(out, empty, pattern = "empty")
})

test_that("a changed preflight file is detected by its hash, not by reading it", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  pre <- preflight_file()
  write_text("tampered", pre$path)
  expect_preflight_refused(out, pre, pattern = "differ")
})

test_that("the verifier accepts exactly the two preflight forms and a consistent citation", {
  out <- cleanup_dir(tempfile("o"), environment()); dir.create(out)
  res <- run_cited(out, preflight_file())
  ok <- function(f) { d <- forge(copy_result(res$result_dir), record_edit = f); verify_result(d)$ok }
  expect_true(ok(identity))
  bad <- list(
    function(r) { r$verification$python_preflight <- "verified"; r },
    function(r) { r$verification$python_preflight <- "passed"; r },
    function(r) { r$inputs$python_preflight_sha256 <- NULL; r },
    function(r) { r$inputs$python_preflight_sha256 <- "abc"; r },
    function(r) { r$inputs$python_preflight_sha256 <- toupper(r$inputs$python_preflight_sha256); r },
    function(r) { r$inputs$python_preflight_bytes <- 0L; r },
    function(r) { r$inputs$python_preflight_bytes <- NULL; r },
    function(r) { r$verification$python_preflight_scope <- "python_verified_lineage"; r },
    function(r) { r$verification$python_preflight_scope <- NULL; r },
    function(r) { r$verification$lineage_verified_by_r <- TRUE; r }
  )
  for (edit in bad) expect_verify_fails(forge(copy_result(res$result_dir), record_edit = edit))
  # A record that says nothing was recorded must not carry a citation.
  stray <- list(
    function(r) { r$inputs$python_preflight_sha256 <- strrep("a", 64); r },
    function(r) { r$inputs$python_preflight_bytes <- 5L; r },
    function(r) { r$verification$python_preflight_scope <- PRE_SCOPE; r }
  )
  for (edit in stray) expect_verify_fails(forge(copy_result(), record_edit = edit))
  forged_to_uncited <- function(r) { r$verification$python_preflight <- "not_recorded"; r }
  expect_verify_fails(forge(copy_result(res$result_dir), record_edit = forged_to_uncited))
})

test_that("the command line cites a preflight and refuses incomplete or wrong pairs", {
  skip_if(Sys.which("Rscript") == "")
  a <- fixture_args(); pre <- preflight_file()
  out <- cleanup_dir(tempfile("cli"), environment()); dir.create(out)
  base_args <- c("run", "--handoff", a$handoff_path, "--handoff-sha256", a$handoff_sha256, "--csv", a$csv_path,
                 "--lock-sha256", real_lock_sha(), "--seed", "20260601", "--seed-label", "synthetic_rehearsal", "--out-root", out)
  expect_false(identical(cli(base_args, "--preflight", pre$path)$status, 0L))
  expect_false(identical(cli(base_args, "--preflight-sha256", pre$sha)$status, 0L))
  expect_true(any(grepl("[usage]", cli(base_args, "--preflight", pre$path)$output, fixed = TRUE)))
  wrong <- cli(base_args, "--preflight", pre$path, "--preflight-sha256", strrep("0", 64))
  expect_false(identical(wrong$status, 0L)); expect_true(any(grepl("[preflight]", wrong$output, fixed = TRUE)))
  expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))
  r <- cli(base_args, "--preflight", pre$path, "--preflight-sha256", pre$sha)
  expect_identical(r$status, 0L)
  expect_true(any(grepl(pre$sha, r$output, fixed = TRUE)))
  expect_true(any(grepl("contents were not read", r$output, fixed = TRUE)))
  dirs <- list.files(out, full.names = TRUE)
  v <- cli("verify", "--result", dirs)
  expect_identical(v$status, 0L)
  expect_true(any(grepl("preflight cited", v$output, fixed = TRUE)) && any(grepl("not repeated by R", v$output, fixed = TRUE)))
  unlock_tree(dirs)
})
