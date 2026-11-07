# Stage 3b: environment gate, installed inventory, code state, seed and origin gates.

test_that("read_lock pins the lock bytes and exposes R and package versions", {
  l <- real_lock()
  expect_identical(l$sha256, real_lock_sha())
  expect_identical(l$r_version, "4.6.1")
  expect_length(l$packages, 30)
  expect_identical(l$packages[["clubSandwich"]], "0.7.0")
  expect_identical(l$packages[["sandwich"]], "3.1-3")
  expect_identical(l$packages[["zoo"]], "1.9-1")
  expect_run_error(read_lock(lock_path(), strrep("0", 64)), "environment")
  expect_run_error(read_lock(file.path(tempdir(), "no-such.lock"), real_lock_sha()), "environment")
})

test_that("the inventory records installed versions, not lock versions", {
  proj <- file.path(analysis_dir(), "renv", "library")
  inst <- fake_installed(file.path(proj, "x"), c("clubSandwich", "base"), c("9.9.9", "4.6.1"), c(NA, "base"))
  inst[2, "LibPath"] <- "/usr/lib/R/library"
  inv <- installed_inventory(c(file.path(proj, "x"), "/usr/lib/R/library"), proj, real_lock()$packages, installed = inst)
  expect_identical(names(inv), names(INVENTORY_SCHEMA))
  expect_identical(inv$version[inv$package == "clubSandwich"], "9.9.9")
  expect_identical(inv$library_role, c("r_base", "project"))
  expect_identical(inv$package, c("base", "clubSandwich"))                       # radix order
  expect_identical(inv$in_lock, c("no", "yes"))
})

test_that("the first library on the path wins when a package is installed twice", {
  proj <- file.path(analysis_dir(), "renv", "library")
  inst <- rbind(fake_installed(proj, "pkgA", "1.0", NA), fake_installed("/other", "pkgA", "2.0", NA))
  inv <- installed_inventory(c(proj, "/other"), proj, c(pkgA = "1.0"), installed = inst)
  expect_identical(inv$version, "1.0"); expect_identical(inv$library_role, "project")
})

test_that("the real environment passes the gate and its inventory equals the lock", {
  inv <- real_inventory(); l <- real_lock()
  # Every locked package is installed at exactly its locked version. lattice is locked but comes
  # from the R installation; the other recommended packages are installed and not locked.
  expect_identical(setNames(inv$version, inv$package)[names(l$packages)], l$packages)
  expect_identical(inv$library_role[inv$package == "lattice"], "r_other")
  expect_true(all(inv$in_lock[inv$library_role == "project"] == "yes"))
  expect_true(all(c("base", "stats", "MASS") %in% inv$package))
  expect_silent(check_environment(l, inv, analysis_dir()))
})

test_that("the environment gate refuses each kind of drift", {
  l <- real_lock(); inv <- real_inventory(); d <- analysis_dir()
  expect_run_error(check_environment(l, inv, d, r_version = "4.5.2"), "environment")
  expect_run_error(check_environment(l, inv, d, active_project = tempdir()), "environment")
  expect_run_error(check_environment(l, inv, d, active_project = NULL), "environment")
  expect_run_error(check_environment(l, inv, d, lib_paths = c("/usr/lib/R/library", .libPaths())), "environment")
  expect_run_error(check_environment(l, inv, d, synchronized = FALSE), "environment")
  drift <- inv; drift$version[drift$package == "clubSandwich"] <- "0.6.9"
  expect_run_error(check_environment(l, drift, d), "environment")
  gone <- inv[inv$package != "sandwich", ]
  expect_run_error(check_environment(l, gone, d), "environment")
  extra <- rbind(inv, data.frame(package = "stowaway", version = "1.0", priority = NA_character_,
                                 library_role = "project", in_lock = "no", stringsAsFactors = FALSE))
  expect_run_error(check_environment(l, extra, d), "environment")
  lock_r <- l; lock_r$r_version <- "4.5.2"
  expect_run_error(check_environment(lock_r, inv, d), "environment")
  lock_c <- l; lock_c$packages[["clubSandwich"]] <- "0.6.9"; drift2 <- inv; drift2$version[drift2$package == "clubSandwich"] <- "0.6.9"
  expect_run_error(check_environment(lock_c, drift2, d), "environment")
})

git_in <- function(dir, ...) {
  out <- suppressWarnings(system2("git", c("-C", dir, "-c", "user.name=t", "-c", "user.email=t@example.invalid", ...),
                                  stdout = TRUE, stderr = TRUE))
  invisible(out)
}
make_repo <- function() {
  root <- tempfile("repo"); dir.create(file.path(root, "analysis", "R"), recursive = TRUE)
  writeLines("a <- 1", file.path(root, "analysis", "R", "a.R"))
  writeLines("main()", file.path(root, "analysis", "run_analysis.R"))
  writeLines("lock", file.path(root, "analysis", "renv.lock"))
  writeLines("ignored/", file.path(root, ".gitignore"))
  git_in(root, "init", "-q"); git_in(root, "add", "-A"); git_in(root, "commit", "-q", "-m", "x")
  root
}

test_that("code state of a clean repository", {
  root <- make_repo()
  s <- code_state(file.path(root, "analysis"))
  expect_identical(s$status, "ok")
  expect_match(s$commit, "^[0-9a-f]{40}$")
  expect_false(s$dirty)
  expect_identical(s$tracked_modified, character(0)); expect_identical(s$untracked_source, list())
  expect_null(s$tracked_diff_sha256)
  expect_identical(names(s$script_sha256), c("R/a.R", "run_analysis.R"))
  expect_identical(s$script_sha256[["R/a.R"]], sha256_file(file.path(root, "analysis", "R", "a.R")))
})

test_that("a tracked modification makes the state dirty and is hashed", {
  root <- make_repo()
  writeLines("a <- 2", file.path(root, "analysis", "R", "a.R"))
  s <- code_state(file.path(root, "analysis"))
  expect_true(s$dirty)
  expect_identical(s$tracked_modified, "analysis/R/a.R")
  expect_match(s$tracked_diff_sha256, "^[0-9a-f]{64}$")
  expect_identical(s$script_sha256[["R/a.R"]], sha256_file(file.path(root, "analysis", "R", "a.R")))
  writeLines("a <- 3", file.path(root, "analysis", "R", "a.R"))
  expect_false(identical(code_state(file.path(root, "analysis"))$tracked_diff_sha256, s$tracked_diff_sha256))
})

test_that("an untracked source file is recorded, not hidden", {
  root <- make_repo()
  writeLines("b <- 1", file.path(root, "analysis", "R", "b.R"))
  s <- code_state(file.path(root, "analysis"))
  expect_true(s$dirty)
  expect_identical(names(s$untracked_source), "analysis/R/b.R")
  expect_identical(s$untracked_source[["analysis/R/b.R"]], sha256_file(file.path(root, "analysis", "R", "b.R")))
  expect_true("R/b.R" %in% names(s$script_sha256))
})

test_that("ignored outputs and untracked files outside the source set do not make the state dirty", {
  root <- make_repo()
  dir.create(file.path(root, "ignored")); writeLines("x", file.path(root, "ignored", "out.csv"))
  dir.create(file.path(root, "analysis", "results")); writeLines("x", file.path(root, "analysis", "results", "r.csv"))
  writeLines("notes", file.path(root, "NOTES.txt"))
  dir.create(file.path(root, "analysis", "R", "ignored")); writeLines("x <- 1", file.path(root, "analysis", "R", "ignored", "x.R"))
  s <- code_state(file.path(root, "analysis"))
  expect_false(s$dirty)
  expect_identical(s$untracked_source, list())
})

test_that("code state outside a repository is explicit, not clean", {
  d <- tempfile("norepo"); dir.create(file.path(d, "R"), recursive = TRUE); writeLines("a", file.path(d, "R", "a.R"))
  s <- code_state(d)
  expect_identical(s$status, "not_a_repository")
  expect_true(is.na(s$commit)); expect_true(is.na(s$dirty))
})

test_that("the seed gate accepts only an explicit integer with the rehearsal label", {
  expect_identical(check_seed(20260601L, "synthetic_rehearsal"), 20260601L)
  expect_identical(check_seed(7, "synthetic_rehearsal"), 7L)
  expect_run_error(check_seed(7L, "primary_bootstrap"), "seed")
  expect_run_error(check_seed(7L, NULL), "seed")
  expect_run_error(check_seed(7L, ""), "seed")
  expect_run_error(check_seed(NULL, "synthetic_rehearsal"), "seed")
  expect_run_error(check_seed(NA_integer_, "synthetic_rehearsal"), "seed")
  expect_run_error(check_seed(1.5, "synthetic_rehearsal"), "seed")
  expect_run_error(check_seed(c(1L, 2L), "synthetic_rehearsal"), "seed")
  expect_run_error(check_seed(2^31, "synthetic_rehearsal"), "seed")
  expect_run_error(check_seed("7", "synthetic_rehearsal"), "seed")
})

test_that("parse_seed reads a plain integer only", {
  expect_identical(parse_seed("20260601"), 20260601L)
  expect_identical(parse_seed("-5"), -5L)
  for (bad in c("", "1.5", "1e3", "0x10", " 5", "5 ", "abc", "99999999999", "3000000000", NA_character_)) {
    expect_run_error(parse_seed(bad), "seed")
  }
})

test_that("the seed record keeps the absence of an approved production seed explicit", {
  h <- fixture()$handoff
  s <- seed_record(20260601L, "synthetic_rehearsal", h)
  expect_identical(s$value, 20260601L)
  expect_identical(s$label, "synthetic_rehearsal")
  expect_identical(s$spec_name, "primary_bootstrap")
  expect_identical(s$registry, "not_registered")
  expect_identical(s$status, "synthetic_rehearsal_not_scientific")
  expect_identical(s$production_seed, "awaiting_approval")
  expect_identical(s$rng$kind, "Mersenne-Twister")
  expect_identical(s$quantile_type, 7L)
  expect_identical(s$n_resamples, 10000L)
  h$awaiting_approval <- list()
  expect_run_error(seed_record(1L, "synthetic_rehearsal", h), "seed")
})

test_that("script hashes cover run_analysis.R and every R source file and change with content", {
  h1 <- script_hashes(analysis_dir())
  expect_true(all(c("R/results.R", "R/provenance.R", "R/read_inputs.R") %in% names(h1)))
  expect_identical(h1[["R/results.R"]], sha256_file(file.path(analysis_dir(), "R", "results.R")))
  expect_identical(names(h1), sort(names(h1), method = "radix"))
})

test_that("execution environment and versions used are actual measurements", {
  e <- execution_environment()
  expect_identical(e$r_version, R.version.string)
  expect_identical(e$platform, R.version$platform)
  expect_true(nzchar(e$lapack))
  compute_results(fixture_args()$handoff_path, fixture_args()$handoff_sha256, fixture_args()$csv_path, TEST_SEED)
  loadNamespace("lattice")
  used <- loaded_package_versions()
  expect_identical(used[["clubSandwich"]], "0.7.0")
  expect_identical(used[["lattice"]], real_lock()$packages[["lattice"]])           # "0.22-5", not "0.22.5"
  expect_true(all(c("sandwich", "digest") %in% names(used)))
  expect_identical(names(used), sort(names(used), method = "radix"))
})
