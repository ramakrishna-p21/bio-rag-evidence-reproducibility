# Command line for the R analysis rehearsal. Start it from the analysis/ directory so that the
# project's renv library is active; the run refuses to start otherwise.
#
#   Rscript run_analysis.R run --handoff F --handoff-sha256 H --csv F --lock-sha256 H \
#           --seed N --seed-label synthetic_rehearsal --out-root DIR \
#           [--preflight F --preflight-sha256 H]
#   Rscript run_analysis.R verify --result DIR [--record-sha256 H] [--lock-sha256 H] \
#           [--rerun --handoff F --handoff-sha256 H --csv F]
#   Rscript run_analysis.R derive --handoff F --handoff-sha256 H --csv F --primary DIR --out-root DIR \
#           [--primary-record-sha256 H]
#   An official handoff needs --primary-record-sha256 (the authenticated primary record). Its fresh Python verification
#   is inherited from that record and is not repeated; supplement stays refused for official inputs.
#   Rscript run_analysis.R verify-derived --result DIR [--record-sha256 H] [--primary DIR] \
#           [--rerun --handoff F --handoff-sha256 H --csv F]
#   Rscript run_analysis.R supplement --handoff F --handoff-sha256 H --csv F --supplement DIR \
#           --supplement-record-sha256 H --out-root DIR [--primary DIR --primary-record-sha256 H]
#   Rscript run_analysis.R verify-supplement --result DIR [--record-sha256 H] [--primary DIR] \
#           [--rerun --handoff F --handoff-sha256 H --csv F --supplement DIR]

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
for (f in c("read_inputs.R", "accounting.R", "estimates.R", "bootstrap.R", "inference.R",
            "results.R", "provenance.R", "publish.R", "secondary.R", "derived.R", "supplement.R")) {
  source(file.path(script_dir, "R", f))
}

OPTIONS <- list(
  run = c("handoff", "handoff-sha256", "csv", "lock-sha256", "seed", "seed-label", "out-root", "preflight", "preflight-sha256", "official-request", "python"),
  verify = c("result", "record-sha256", "lock-sha256", "handoff", "handoff-sha256", "csv", "official-request", "python"),
  derive = c("handoff", "handoff-sha256", "csv", "primary", "out-root", "primary-record-sha256"),
  "verify-derived" = c("result", "record-sha256", "primary", "handoff", "handoff-sha256", "csv"),
  supplement = c("handoff", "handoff-sha256", "csv", "supplement", "supplement-record-sha256", "out-root", "primary",
                 "primary-record-sha256"),
  "verify-supplement" = c("result", "record-sha256", "primary", "handoff", "handoff-sha256", "csv", "supplement")
)
REQUIRED <- list(run = setdiff(OPTIONS$run, c("preflight", "preflight-sha256", "official-request", "python")), verify = "result",
                 derive = setdiff(OPTIONS$derive, "primary-record-sha256"), "verify-derived" = "result",
                 supplement = setdiff(OPTIONS$supplement, c("primary", "primary-record-sha256")),
                 "verify-supplement" = "result")

parse_cli <- function(args) {
  command <- if (length(args)) args[1] else ""
  if (!command %in% names(OPTIONS)) run_error("usage", "the command must be 'run', 'verify', 'derive', 'verify-derived', 'supplement' or 'verify-supplement'")
  rest <- args[-1]
  flags <- if (command %in% c("verify", "verify-derived", "verify-supplement")) "rerun" else character(0)
  out <- list()
  i <- 1
  while (i <= length(rest)) {
    key <- sub("^--", "", rest[i])
    if (!startsWith(rest[i], "--") || key %in% names(out)) run_error("usage", sprintf("unexpected or repeated argument '%s'", rest[i]))
    if (key %in% flags) {
      out[[key]] <- TRUE; i <- i + 1
    } else if (key %in% OPTIONS[[command]]) {
      if (i == length(rest) || startsWith(rest[i + 1], "--")) run_error("usage", sprintf("option --%s needs a value", key))
      out[[key]] <- rest[i + 1]; i <- i + 2
    } else {
      run_error("usage", sprintf("unknown option '--%s'", key))
    }
  }
  missing <- setdiff(REQUIRED[[command]], names(out))
  if (length(missing)) run_error("usage", sprintf("missing option(s): %s", paste0("--", missing, collapse = ", ")))
  if (command == "run" && xor("preflight" %in% names(out), "preflight-sha256" %in% names(out))) {
    run_error("usage", "--preflight and --preflight-sha256 are used together")
  }
  if (command %in% c("run", "verify") && xor("official-request" %in% names(out), "python" %in% names(out))) {
    run_error("usage", "--official-request and --python are used together")
  }
  if (command %in% c("verify", "verify-derived", "verify-supplement")) {
    rerun_opts <- c("handoff", "handoff-sha256", "csv", if (command == "verify-supplement") "supplement")
    if (isTRUE(out$rerun) && length(setdiff(rerun_opts, names(out)))) run_error("usage", "--rerun needs --handoff, --handoff-sha256, --csv (and --supplement)")
    if (command == "verify-derived" && isTRUE(out$rerun) && !"primary" %in% names(out)) run_error("usage", "--rerun needs --primary")
    if (!isTRUE(out$rerun) && any(rerun_opts %in% names(out))) run_error("usage", "--handoff, --handoff-sha256, --csv and --supplement are only used with --rerun")
  }
  c(list(command = command), out)
}

# OpenBLAS picks its kernel from the CPU and the last bit of the CR2 standard error depends on
# it, so the byte-compared result hash is only defined for this kernel and one thread. The
# variables are read when R starts; setting them from here would be too late. The Python runner
# sets them; src/biorag_repro/evaluation/analysis_result.py repeats these values and a test
# compares them.
BLAS_PIN <- c(OPENBLAS_CORETYPE = "HASWELL", OPENBLAS_NUM_THREADS = "1")

require_blas_pin <- function() {
  have <- Sys.getenv(names(BLAS_PIN), unset = NA_character_)
  if (!identical(unname(have), unname(BLAS_PIN))) {
    run_error("environment", sprintf(
      "R must be started with OPENBLAS_CORETYPE=HASWELL and OPENBLAS_NUM_THREADS=1 in its environment (found OPENBLAS_CORETYPE=%s, OPENBLAS_NUM_THREADS=%s)",
      have[["OPENBLAS_CORETYPE"]], have[["OPENBLAS_NUM_THREADS"]]))
  }
}

main <- function(args) {
  a <- parse_cli(args)
  if (a$command == "run" || (a$command == "verify" && isTRUE(a$rerun))) require_blas_pin()
  if (a$command == "run") {
    r <- run_analysis(a$handoff, a[["handoff-sha256"]], a$csv, a[["lock-sha256"]], parse_seed(a$seed), a[["seed-label"]],
                      a[["out-root"]], script_dir, preflight_path = a$preflight, preflight_sha256 = a[["preflight-sha256"]], official_request = a[["official-request"]], python = a$python)
    cat(if (identical(r$purpose, "final_test_confirmatory")) "official final-test analysis (frozen confirmatory method; custody exception recorded in the handoff)\n" else if (is.null(a[["official-request"]])) "synthetic rehearsal: no scientific authority; the seed is not a registered or approved seed\n" else "development rehearsal: no scientific authority\n")
    cat("result directory: ", r$result_dir, "\n", sep = "")
    cat("run_record.json sha256: ", r$record_sha256, "\n", sep = "")
    cat("scientific_result_sha256: ", r$scientific_result_sha256, "\n", sep = "")
    if (!is.null(a$preflight)) {
      cat("python preflight cited: sha256 ", a[["preflight-sha256"]],
          " (contents were not read; the verification was not repeated by R)\n", sep = "")
    }
  } else if (a$command == "derive") {
    r <- run_derived(a$handoff, a[["handoff-sha256"]], a$csv, a$primary, a[["out-root"]], script_dir,
                     expected_primary_record_sha256 = a[["primary-record-sha256"]])
    cat(if (identical(r$purpose, OFFICIAL_SECONDARY_PURPOSE)) paste0(OFFICIAL_SECONDARY_BANNER, "\n") else "SYNTHETIC REHEARSAL \u2014 NOT SCIENTIFIC FINDINGS\n")
    cat("result directory: ", r$result_dir, "\n", sep = "")
    cat("secondary_record.json sha256: ", r$record_sha256, "\n", sep = "")
    cat("secondary_result_sha256: ", r$secondary_result_sha256, "\n", sep = "")
  } else if (a$command == "supplement") {
    r <- run_supplement(a$handoff, a[["handoff-sha256"]], a$csv, a$supplement, a[["supplement-record-sha256"]],
                        a[["out-root"]], script_dir, primary_dir = a$primary,
                        expected_primary_record_sha256 = a[["primary-record-sha256"]])
    cat("SYNTHETIC REHEARSAL \u2014 NOT SCIENTIFIC FINDINGS\n")
    cat("result directory: ", r$result_dir, "\n", sep = "")
    cat("supplement_summary_record.json sha256: ", r$record_sha256, "\n", sep = "")
    cat("supplement_summary_sha256: ", r$supplement_summary_sha256, "\n", sep = "")
  } else if (a$command == "verify-supplement") {
    v <- verify_supplement_summary(a$result, a[["record-sha256"]], primary_dir = a$primary)
    cat("content verified: supplement summary files, hashes and tables are internally consistent (synthetic rehearsal; no scientific authority)\n")
    cat("read-only seal: ", if (v$sealed) "present (local file permissions only)" else "ABSENT", "\n", sep = "")
    cat(if (v$record_authenticated) "record authenticated against the supplied sha256\n" else "record NOT authenticated: no --record-sha256 was supplied\n")
    if (isTRUE(a$rerun)) {
      r <- verify_supplement_rerun(a$result, a$handoff, a[["handoff-sha256"]], a$csv, a$supplement, a$primary)
      cat("re-execution: ", r$comparison, "\n", sep = "")
    }
  } else if (a$command == "verify-derived") {
    v <- verify_secondary(a$result, a[["record-sha256"]], primary_dir = a$primary)
    cat(sprintf("content verified: secondary files, hashes and tables are internally consistent (%s)\n",
                if (identical(v$record$purpose, OFFICIAL_SECONDARY_PURPOSE)) "official secondary descriptive analysis; exploratory, not confirmatory; primary verification inherited" else "synthetic rehearsal; no scientific authority"))
    cat("read-only seal: ", if (v$sealed) "present (local file permissions only)" else "ABSENT", "\n", sep = "")
    cat(if (v$record_authenticated) "record authenticated against the supplied sha256\n" else "record NOT authenticated: no --record-sha256 was supplied\n")
    if (!is.null(a$primary)) cat("cited primary result verified against ", a$primary, "\n", sep = "")
    if (isTRUE(a$rerun)) {
      r <- verify_secondary_rerun(a$result, a$handoff, a[["handoff-sha256"]], a$csv, a$primary)
      cat("re-execution: ", r$comparison, "\n", sep = "")
    }
  } else {
    v <- verify_result(a$result, a[["record-sha256"]], a[["lock-sha256"]],
                       official_request = a[["official-request"]], python = a$python, project_dir = script_dir)
    cat(sprintf("content verified: files, hashes and tables are internally consistent (%s)\n",
                if (identical(v$record$purpose, "final_test_confirmatory")) v$record$purpose
                else paste0(v$record$purpose, "; no scientific authority")))
    if (v$sealed) {
      cat("read-only seal: present (local file permissions only; not proof of immutability)\n")
    } else {
      cat("read-only seal: ABSENT (content is still verified; the files can be modified locally)\n")
    }
    if (identical(v$record$verification$python_preflight, PREFLIGHT_CITED)) {
      cat("python preflight cited: sha256 ", v$record$inputs$python_preflight_sha256,
          " (hash recorded by R; contents not read, verification not repeated by R)\n", sep = "")
    }
    if (v$record_authenticated) {
      cat("record authenticated against the supplied sha256 ", v$record_sha256, "\n", sep = "")
    } else {
      cat("record NOT authenticated: no --record-sha256 was supplied (internal consistency is not authentication)\n")
    }
    if (isTRUE(a$rerun)) {
      r <- verify_rerun(a$result, a$handoff, a[["handoff-sha256"]], a$csv,
                        official_request = a[["official-request"]], python = a$python, project_dir = script_dir)
      cat("re-execution: ", r$comparison, "\n", sep = "")
    }
  }
}

if (sys.nframe() == 0) {
  status <- tryCatch({ main(commandArgs(trailingOnly = TRUE)); 0L },
                     error = function(e) { message("error: ", conditionMessage(e)); 1L })
  quit(save = "no", status = status)
}
