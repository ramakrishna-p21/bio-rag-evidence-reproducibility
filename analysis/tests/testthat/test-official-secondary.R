# Official confirmatory-only secondary analysis: the proof is inherited from the authenticated primary
# record, and nothing in these tests may call the Python verification.

osclock <- function(iso) function() as.POSIXct(iso, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS")

os_root <- cleanup_dir(tempfile("officialsec"), teardown_env())
dir.create(os_root)
os_in <- official_scope_inputs()
local({
  stub_python_proof(os_in$proof)
  dir.create(file.path(os_root, "primary"))
  os_primary <<- run_analysis(os_in$handoff, os_in$handoff_sha, os_in$csv, real_lock_sha(), os_in$seed, "final_test",
                              file.path(os_root, "primary"), analysis_dir(), synchronized = TRUE,
                              official_request = os_in$request, python = "unused", now = osclock("2026-11-25T10:00:00.000"))
})

# From here on the Python verification is a failure, not a stub.
no_python <- function(env = parent.frame()) {
  home <- environment(read_handoff_file)
  original <- get("read_official_proof", envir = home)
  assign("read_official_proof", function(...) stop("the Python verification must not run in a secondary analysis"), envir = home)
  withr::defer(assign("read_official_proof", original, envir = home), envir = env)
}

os_derive <- function(out = file.path(os_root, "sec"), rec_sha = os_primary$record_sha256, handoff = os_in$handoff,
                      handoff_sha = os_in$handoff_sha, csv = os_in$csv, primary = os_primary$result_dir,
                      clock = "2026-11-25T11:00:00.000") {
  if (!dir.exists(out)) dir.create(out)
  run_derived(handoff, handoff_sha, csv, primary, out, analysis_dir(), expected_primary_record_sha256 = rec_sha, now = osclock(clock))
}

test_that("an official secondary analysis inherits the primary proof, publishes read-only and verifies", {
  no_python()
  d <- os_derive()
  expect_identical(d$purpose, OFFICIAL_SECONDARY_PURPOSE)
  v <- verify_secondary(d$result_dir, expected_record_sha256 = d$record_sha256, require_read_only = TRUE, primary_dir = os_primary$result_dir)
  expect_true(v$sealed)
  r <- v$record
  expect_identical(r$scientific_authority, "secondary_exploratory_descriptive")
  expect_identical(r$inputs$input_origin, "verified_official_run")
  expect_identical(r$inputs$authorization_sha256, strrep("d", 64))
  expect_identical(jsonlite::fromJSON(file.path(os_primary$result_dir, "run_record.json"), simplifyVector = FALSE)$inputs$authorization_sha256, strrep("d", 64))
  expect_identical(r$primary_result$verification, "inherited_from_authenticated_primary_record")
  expect_false(r$verification$fresh_python_verification_in_this_run)
  expect_identical(r$primary_result$run_record_sha256, os_primary$record_sha256)
  report <- rawToChar(v$report)
  expect_match(report, OFFICIAL_SECONDARY_BANNER, fixed = TRUE)
  expect_false(grepl("SYNTHETIC", report, fixed = TRUE))
  expect_match(report, "inherited from the authenticated primary run record", fixed = TRUE)
  expect_match(report, "computational replays of the frozen greedy configuration", fixed = TRUE)
  expect_match(report, "Deferred: the analyses below need Python-supplement data", fixed = TRUE)
  # the deferred arms stay visible as absent, and the supplement-dependent analyses stay registered
  rob <- v$tables[["secondary_robustness.csv"]]
  expect_true(all(rob$status[rob$matrix_arm != "confirmatory"] == "arm_absent"))
  expect_setequal(v$tables[["secondary_unavailable.csv"]]$analysis,
                  c("answer_identity_disagreement", "identical_incorrect_answer_stability", "table_vs_prose", "evidence_location"))
  # the primary result is untouched and the secondary result is a separate directory with its own kind
  expect_identical(verify_result(os_primary$result_dir, expected_record_sha256 = os_primary$record_sha256, inherit = TRUE)$record$purpose, "final_test_confirmatory")
  expect_false(identical(dirname(d$result_dir), os_primary$result_dir))
  expect_match(basename(d$result_dir), "^secondary_")
})

test_that("a published secondary result is never replaced", {
  d <- os_derive(out = file.path(os_root, "once"), clock = "2026-11-25T12:00:00.000")
  expect_error(os_derive(out = file.path(os_root, "once"), clock = "2026-11-25T12:00:00.000"), "already exists")
  expect_true(verify_secondary(d$result_dir, require_read_only = TRUE)$sealed)
})

test_that("the secondary tables do not depend on the origin and do not count repeats twice", {
  rows <- read_inputs(os_in$handoff, os_in$handoff_sha, os_in$csv, given_proof = os_in$proof)$rows
  fixture <- fixture_args()
  synthetic <- read_inputs(fixture$handoff_path, fixture$handoff_sha256, fixture$csv_path)
  rules <- outcome_rules(synthetic$handoff)
  expect_identical(compute_secondary_tables(rows, rules)[setdiff(SECONDARY_DATA_FILES, "secondary_populations.csv")],
                   compute_secondary_tables(synthetic$rows, rules)[setdiff(SECONDARY_DATA_FILES, "secondary_populations.csv")])
  tables <- compute_secondary_tables(rows, rules)
  first <- rows[rows$matrix_arm == "confirmatory" & rows$repeat_index == 1L & rows$condition %in% REPRESENTATIONS, ]
  # repeat-1 tables count each item x representation once; the repeat table counts each repeat once
  stage_a <- tables[["secondary_stage_a.csv"]]
  expect_identical(sum(stage_a$n), nrow(first))
  by_condition <- tapply(stage_a$n, stage_a$condition, sum)
  expect_identical(as.integer(by_condition[REPRESENTATIONS]), as.integer(table(first$condition)[REPRESENTATIONS]))
  rep_out <- tables[["secondary_repeat_outcomes.csv"]]
  all_conf <- rows[rows$matrix_arm == "confirmatory" & rows$condition %in% REPRESENTATIONS, ]
  expect_identical(sum(rep_out$n[rep_out$matrix_arm == "confirmatory"]), nrow(all_conf))
  expect_true(all(rep_out$denominator == as.integer(table(all_conf$condition, all_conf$repeat_index)[cbind(rep_out$condition, as.character(rep_out$repeat_index))])))
})

test_that("an official secondary analysis refuses every mismatched, missing or altered input", {
  no_python()
  # the primary record's SHA-256 must be supplied from outside, and must be the right one
  expect_error(os_derive(out = file.path(os_root, "r1"), rec_sha = NULL), "needs the SHA-256")
  expect_error(os_derive(out = file.path(os_root, "r2"), rec_sha = paste(rep("a", 64), collapse = "")), "differs|SHA-256")
  # a different handoff or table than the primary record's
  other <- write_text(sub('"scope": "confirmatory_only"', '"scope": "complete_matrix"', read_bytes_as_text(os_in$handoff), fixed = TRUE), tempfile(fileext = ".json"))
  expect_error(os_derive(out = file.path(os_root, "r3"), handoff = other, handoff_sha = sha256_file(other)), "different handoff|differ")
  expect_error(os_derive(out = file.path(os_root, "r4"), handoff_sha = paste(rep("b", 64), collapse = "")), "pinned SHA-256")
  csv <- write_text(paste0(read_bytes_as_text(os_in$csv), "x\n"), tempfile(fileext = ".csv"))
  expect_error(os_derive(out = file.path(os_root, "r5"), csv = csv), "differ")
  # missing and altered primary results
  expect_error(os_derive(out = file.path(os_root, "r6"), primary = file.path(os_root, "no-such-primary")), "not found")
  altered <- cleanup_dir(tempfile("altered"), environment())
  dir.create(altered)
  file.copy(list.files(os_primary$result_dir, full.names = TRUE), altered)
  Sys.chmod(list.files(altered, full.names = TRUE), "0644")
  csv_file <- file.path(altered, "coprimary_results.csv")
  writeBin(c(readBin(csv_file, "raw", file.size(csv_file)), as.raw(0x20)), csv_file)
  expect_error(os_derive(out = file.path(os_root, "r7"), primary = altered), "differs|hash")
})

test_that("an official handoff cannot cite a synthetic primary result, and a synthetic one cannot cite an official primary", {
  no_python()
  synthetic_out <- file.path(os_root, "synthetic-primary"); dir.create(synthetic_out)
  fx <- fixture_args()
  sp <- run_analysis(fx$handoff_path, fx$handoff_sha256, fx$csv_path, real_lock_sha(), TEST_SEED, "synthetic_rehearsal", synthetic_out,
                     analysis_dir(), now = osclock("2026-11-25T09:00:00.000"), synchronized = TRUE)
  expect_error(os_derive(out = file.path(os_root, "s1"), primary = sp$result_dir, rec_sha = sp$record_sha256), "not an official final-test confirmatory")
  expect_error(run_derived(fx$handoff_path, fx$handoff_sha256, fx$csv_path, os_primary$result_dir, file.path(os_root, "s2"), analysis_dir(),
                           expected_primary_record_sha256 = os_primary$record_sha256), "")
  # the synthetic route is unchanged
  dir.create(file.path(os_root, "s3"))
  d <- run_derived(fx$handoff_path, fx$handoff_sha256, fx$csv_path, sp$result_dir, file.path(os_root, "s3"), analysis_dir(), now = osclock("2026-11-25T13:00:00.000"))
  expect_identical(d$purpose, "synthetic_rehearsal")
  v <- verify_secondary(d$result_dir)
  expect_identical(v$record$banner, SYNTHETIC_BANNER)
  expect_identical(v$record$primary_result$verification, "verified_by_this_run_before_use")
})

test_that("an official secondary run must use the primary run's analysis code", {
  no_python()
  other_code <- function(project_dir) { code <- code_state(project_dir); code$commit <- paste(rep("0", 40), collapse = ""); code }
  dir.create(file.path(os_root, "c1"))
  expect_error(run_derived(os_in$handoff, os_in$handoff_sha, os_in$csv, os_primary$result_dir, file.path(os_root, "c1"), analysis_dir(),
                           expected_primary_record_sha256 = os_primary$record_sha256, code_fn = other_code), "same analysis code")
  expect_identical(list.files(file.path(os_root, "c1"), all.files = TRUE, no.. = TRUE), character(0))
})

test_that("the Python-supplement analyses stay deferred for an official handoff", {
  no_python()
  expect_error(compute_supplement(os_in$handoff, os_in$handoff_sha, os_in$csv, tempdir(), paste(rep("c", 64), collapse = "")),
               "deferred for official inputs")
})

test_that("the official primary record carries the proof facts that a secondary run inherits", {
  f <- jsonlite::fromJSON(file.path(os_primary$result_dir, "run_record.json"), simplifyVector = FALSE)$verification$official_proof
  expect_identical(f$scope, "confirmatory_only")
  expect_identical(f$observations, 1830L)
  expect_identical(f$source_verification, "fresh_python_verification")
  expect_identical(f$handoff_sha256, os_in$handoff_sha)
  # a record whose facts are removed cannot be inherited
  expect_error(inherited_proof(list(verification = list())), "no official proof facts")
})
