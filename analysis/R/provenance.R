# Environment, code and seed provenance for one analysis run.
#
# The lock states the intended R dependencies. The inventory states what is actually installed.
# They are recorded separately and the run stops if the installed packages disagree with the lock.
# Nothing here installs, upgrades or removes a package.

REQUIRED_R_VERSION <- "4.6.1"
REQUIRED_CLUBSANDWICH_VERSION <- "0.7.0"
REHEARSAL_SEED_LABEL <- "synthetic_rehearsal"
# Files whose content is the analysis code. An untracked file here is part of the run's provenance.
SOURCE_PATHSPEC <- c("R", "run_analysis.R", "renv.lock", ".Rprofile", "renv/activate.R", "renv/settings.json")

current_r_version <- function() paste(R.version$major, R.version$minor, sep = ".")

# ---- lock and installed inventory -------------------------------------------------------------

read_lock <- function(path, expected_sha256) {
  if (!file.exists(path)) run_error("environment", "renv.lock not found")
  actual <- sha256_file(path)
  if (!identical(actual, expected_sha256)) {
    run_error("environment", sprintf("renv.lock sha256 %s differs from the expected %s", actual, expected_sha256))
  }
  lock <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  versions <- vapply(lock$Packages, function(p) p$Version, "")
  list(sha256 = actual, r_version = lock$R$Version, packages = versions[order(names(versions), method = "radix")])
}

# One row per package visible on the library path (first library wins). `library_role` is
# `project` for the renv library, `r_base` for base packages and `r_other` for the rest of what the
# R installation provides.
installed_inventory <- function(lib_paths, project_library, lock_packages, installed = NULL) {
  if (is.null(installed)) installed <- utils::installed.packages(lib.loc = lib_paths, noCache = TRUE)
  norm <- function(x) normalizePath(x, mustWork = FALSE)
  inst <- as.data.frame(installed[, c("Package", "LibPath", "Version", "Priority"), drop = FALSE],
                        stringsAsFactors = FALSE)
  rank <- match(norm(inst$LibPath), norm(lib_paths))
  inst <- inst[!is.na(rank), , drop = FALSE]
  inst <- inst[order(rank[!is.na(rank)], inst$Package, method = "radix"), , drop = FALSE]
  inst <- inst[!duplicated(inst$Package), , drop = FALSE]
  in_project <- startsWith(norm(inst$LibPath), norm(project_library))
  role <- ifelse(in_project, "project", ifelse(!is.na(inst$Priority) & inst$Priority == "base", "r_base", "r_other"))
  out <- data.frame(package = inst$Package, version = inst$Version, priority = inst$Priority, library_role = role,
                    in_lock = ifelse(inst$Package %in% names(lock_packages), "yes", "no"), stringsAsFactors = FALSE)
  out <- out[order(out$package, method = "radix"), , drop = FALSE]
  rownames(out) <- NULL
  out
}

renv_synchronized <- function(project_dir) isTRUE(renv::status(project = project_dir)$synchronized)

# Stop unless the running environment is the locked one. All problems are reported together.
check_environment <- function(lock, inventory, project_dir, r_version = current_r_version(),
                              lib_paths = .libPaths(), active_project = renv::project(),
                              synchronized = NULL) {
  norm <- function(x) normalizePath(x, mustWork = FALSE)
  problems <- character(0)
  if (!identical(r_version, REQUIRED_R_VERSION)) problems <- c(problems, sprintf("R %s is running, %s is required", r_version, REQUIRED_R_VERSION))
  if (!identical(lock$r_version, REQUIRED_R_VERSION)) problems <- c(problems, sprintf("the lock records R %s", lock$r_version))
  if (is.null(active_project) || !identical(norm(active_project), norm(project_dir))) {
    problems <- c(problems, "the analysis renv project is not active (start R inside analysis/)")
  }
  project_library <- file.path(project_dir, "renv", "library")
  if (!length(lib_paths) || !startsWith(norm(lib_paths[1]), norm(project_library))) {
    problems <- c(problems, "the first library is not the project library")
  }
  if (!identical(lock$packages[["clubSandwich"]], REQUIRED_CLUBSANDWICH_VERSION)) {
    problems <- c(problems, "the lock does not pin clubSandwich 0.7.0")
  }
  installed <- stats::setNames(inventory$version, inventory$package)
  for (pkg in names(lock$packages)) {
    if (!pkg %in% names(installed)) {
      problems <- c(problems, sprintf("locked package %s is not installed", pkg))
    } else if (!identical(unname(installed[[pkg]]), unname(lock$packages[[pkg]]))) {
      problems <- c(problems, sprintf("%s %s is installed, the lock has %s", pkg, installed[[pkg]], lock$packages[[pkg]]))
    }
  }
  unlocked <- inventory$package[inventory$library_role == "project" & !inventory$package %in% names(lock$packages)]
  if (length(unlocked)) problems <- c(problems, sprintf("project library holds unlocked package(s): %s", paste(unlocked, collapse = ", ")))
  if (!length(problems)) {
    if (is.null(synchronized)) synchronized <- renv_synchronized(project_dir)
    if (!isTRUE(synchronized)) problems <- "renv reports the project is not synchronized with its lock"
  }
  if (length(problems)) run_error("environment", paste(problems, collapse = "; "))
  invisible(TRUE)
}

# What the machine reports about itself. Recorded as measured; none of it enters the scientific hash.
execution_environment <- function() {
  info <- utils::sessionInfo()
  list(
    r_version = R.version.string, platform = R.version$platform,
    os = if (is.null(info$running)) NA_character_ else info$running,
    blas = if (is.null(info$BLAS)) NA_character_ else basename(info$BLAS),
    lapack = La_library(), lapack_version = La_version(),
    library_roles = c("project", "r_distribution")[seq_len(min(2L, length(.libPaths())))]
  )
}

# Versions of the packages actually loaded, read after the computation has run. The DESCRIPTION
# string is used as is: packageVersion() would turn "0.22-5" into "0.22.5".
loaded_package_versions <- function() {
  ns <- sort(loadedNamespaces(), method = "radix")
  stats::setNames(vapply(ns, function(n) as.character(utils::packageDescription(n, fields = "Version")), ""), ns)
}

# ---- code state -------------------------------------------------------------------------------

script_hashes <- function(analysis_dir) {
  files <- c(file.path("R", list.files(file.path(analysis_dir, "R"), pattern = "\\.R$")),
             if (file.exists(file.path(analysis_dir, "run_analysis.R"))) "run_analysis.R")
  files <- sort(files, method = "radix")
  stats::setNames(vapply(files, function(f) sha256_file(file.path(analysis_dir, f)), ""), files)
}

git_lines <- function(dir, args) {
  out <- suppressWarnings(system2("git", c("-C", dir, args), stdout = TRUE, stderr = FALSE))
  status <- attr(out, "status")
  if (!is.null(status) && status != 0) return(NULL)
  as.character(out)
}

# Git commit and what differs from it. Dirty means a tracked file differs from HEAD (staged or
# not), or a source file is untracked and not ignored. Ignored runtime outputs and untracked files
# outside the source set do not count. A repository that cannot be read is reported, never "clean".
code_state <- function(analysis_dir) {
  scripts <- script_hashes(analysis_dir)
  empty <- list(commit = NA_character_, dirty = NA, tracked_modified = character(0), tracked_diff_sha256 = NULL,
                untracked_source = list(), script_sha256 = scripts)
  top <- git_lines(analysis_dir, c("rev-parse", "--show-toplevel"))
  if (is.null(top)) return(c(list(status = "not_a_repository"), empty))
  head <- git_lines(analysis_dir, c("rev-parse", "HEAD"))
  modified <- git_lines(analysis_dir, c("diff", "HEAD", "--name-only"))
  others <- git_lines(analysis_dir, c("ls-files", "--others", "--exclude-standard", "--full-name", "--", SOURCE_PATHSPEC))
  if (is.null(head) || is.null(modified) || is.null(others)) return(c(list(status = "unavailable"), empty))
  diff_sha <- NULL
  if (length(modified)) {
    patch <- git_lines(analysis_dir, c("diff", "HEAD", "--binary"))
    if (is.null(patch)) return(c(list(status = "unavailable"), empty))
    diff_sha <- sha256_raw(charToRaw(paste(patch, collapse = "\n")))
  }
  untracked <- list()
  for (f in sort(others, method = "radix")) {
    untracked[[f]] <- sha256_file(file.path(top[1], f))
  }
  list(status = "ok", commit = head[1], dirty = length(modified) > 0 || length(untracked) > 0,
       tracked_modified = sort(modified, method = "radix"), tracked_diff_sha256 = diff_sha,
       untracked_source = untracked, script_sha256 = scripts)
}

# ---- seed -------------------------------------------------------------------------------------

# Only an explicitly labelled rehearsal seed is accepted. There is no default and no production label.
check_seed <- function(seed, label) {
  if (!(is.character(label) && length(label) == 1L && !is.na(label) &&
        label %in% c(REHEARSAL_SEED_LABEL, "development_rehearsal"))) {
    run_error("seed", sprintf("the seed label must be '%s'; production analysis is unavailable through this rehearsal interface", REHEARSAL_SEED_LABEL))
  }
  ok <- is.numeric(seed) && length(seed) == 1 && !is.na(seed) && is.finite(seed) && seed == round(seed) &&
    abs(seed) <= .Machine$integer.max
  if (!ok) run_error("seed", "the seed must be one integer given explicitly")
  as.integer(seed)
}

parse_seed <- function(text) {
  if (!is.character(text) || length(text) != 1 || is.na(text) || !grepl("^-?[0-9]{1,10}$", text)) {
    run_error("seed", "the seed must be a plain integer")
  }
  value <- as.numeric(text)
  if (abs(value) > .Machine$integer.max) run_error("seed", "the seed is outside the 32-bit integer range")
  as.integer(value)
}

# Historical rehearsals retain their unregistered seed semantics. Official development
# records bind the approved registry through fresh Python verification.
seed_record <- function(seed, label, handoff, registration = NULL) {
  seed <- check_seed(seed, label)
  expected_label <- if (identical(handoff$input_origin, "verified_official_run")) "development_rehearsal" else REHEARSAL_SEED_LABEL
  if (!identical(label, expected_label)) run_error("seed", "the rehearsal seed label differs from the verified input origin")
  registry_sha <- handoff$scoring_reference$primary_bootstrap_registry_sha256
  if (!is.null(registry_sha)) {
    if (!identical(handoff$input_origin, "verified_official_run") ||
        is.null(registration) || !identical(registration$registry_sha256, registry_sha) ||
        !identical(registration$record$name, "primary_bootstrap") ||
        !identical(as.integer(registration$record$value), seed) ||
        !identical(registration$record$resamples, primary_resamples(handoff)) ||
        !identical(registration$record$rng_algorithm, "Mersenne-Twister") ||
        !identical(registration$record$normal_kind, "Inversion") ||
        !identical(registration$record$sample_kind, "Rejection") ||
        "primary_bootstrap_seed" %in% unlist(handoff$awaiting_approval)) {
      run_error("seed", "registered seed differs from the freshly verified Python handoff")
    }
    return(list(value = seed, label = label, spec_name = "primary_bootstrap", registry = registry_sha,
                status = "registered_analysis_seed", production_seed = "approved_for_frozen_procedure",
                rng = BOOTSTRAP_RNG, quantile_type = BOOTSTRAP_QUANTILE_TYPE, n_resamples = primary_resamples(handoff)))
  }
  if (!"primary_bootstrap_seed" %in% unlist(handoff$awaiting_approval)) {
    run_error("seed", "the handoff no longer lists the production seed as awaiting approval; this pass has no approved seed")
  }
  list(value = seed, label = label, spec_name = handoff$analysis_spec$bootstrap_seed_name, registry = "not_registered",
       status = paste0(expected_label, "_not_scientific"), production_seed = "awaiting_approval",
       rng = BOOTSTRAP_RNG, quantile_type = BOOTSTRAP_QUANTILE_TYPE, n_resamples = primary_resamples(handoff))
}
