# Command line for the R analysis rehearsal. Start it from the analysis/ directory so that the
# project's renv library is active; the run refuses to start otherwise.
#
#   Rscript run_analysis.R run --handoff F --handoff-sha256 H --csv F --lock-sha256 H \
#           --seed N --seed-label synthetic_rehearsal --out-root DIR \
#           [--preflight F --preflight-sha256 H]
#   Rscript run_analysis.R verify --result DIR [--record-sha256 H] [--lock-sha256 H] \
#           [--rerun --handoff F --handoff-sha256 H --csv F]

script_dir <- dirname(normalizePath(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])))
for (f in c("read_inputs.R", "accounting.R", "estimates.R", "bootstrap.R", "inference.R",
            "results.R", "provenance.R", "publish.R")) {
  source(file.path(script_dir, "R", f))
}

OPTIONS <- list(
  run = c("handoff", "handoff-sha256", "csv", "lock-sha256", "seed", "seed-label", "out-root", "preflight", "preflight-sha256"),
  verify = c("result", "record-sha256", "lock-sha256", "handoff", "handoff-sha256", "csv")
)
REQUIRED <- list(run = setdiff(OPTIONS$run, c("preflight", "preflight-sha256")), verify = "result")

parse_cli <- function(args) {
  command <- if (length(args)) args[1] else ""
  if (!command %in% names(OPTIONS)) run_error("usage", "the command must be 'run' or 'verify'")
  rest <- args[-1]
  flags <- if (command == "verify") "rerun" else character(0)
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
  if (command == "verify") {
    rerun_opts <- c("handoff", "handoff-sha256", "csv")
    if (isTRUE(out$rerun) && length(setdiff(rerun_opts, names(out)))) run_error("usage", "--rerun needs --handoff, --handoff-sha256 and --csv")
    if (!isTRUE(out$rerun) && any(rerun_opts %in% names(out))) run_error("usage", "--handoff, --handoff-sha256 and --csv are only used with --rerun")
  }
  c(list(command = command), out)
}

main <- function(args) {
  a <- parse_cli(args)
  if (a$command == "run") {
    r <- run_analysis(a$handoff, a[["handoff-sha256"]], a$csv, a[["lock-sha256"]], parse_seed(a$seed), a[["seed-label"]],
                      a[["out-root"]], script_dir, preflight_path = a$preflight, preflight_sha256 = a[["preflight-sha256"]])
    cat("synthetic rehearsal: no scientific authority; the seed is not a registered or approved seed\n")
    cat("result directory: ", r$result_dir, "\n", sep = "")
    cat("run_record.json sha256: ", r$record_sha256, "\n", sep = "")
    cat("scientific_result_sha256: ", r$scientific_result_sha256, "\n", sep = "")
    if (!is.null(a$preflight)) {
      cat("python preflight cited: sha256 ", a[["preflight-sha256"]],
          " (contents were not read; the verification was not repeated by R)\n", sep = "")
    }
  } else {
    v <- verify_result(a$result, a[["record-sha256"]], a[["lock-sha256"]])
    cat("content verified: files, hashes and tables are internally consistent (synthetic rehearsal; no scientific authority)\n")
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
      r <- verify_rerun(a$result, a$handoff, a[["handoff-sha256"]], a$csv)
      cat("re-execution: ", r$comparison, "\n", sep = "")
    }
  }
}

if (sys.nframe() == 0) {
  status <- tryCatch({ main(commandArgs(trailingOnly = TRUE)); 0L },
                     error = function(e) { message("error: ", conditionMessage(e)); 1L })
  quit(save = "no", status = status)
}
