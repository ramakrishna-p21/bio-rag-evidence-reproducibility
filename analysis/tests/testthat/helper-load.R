# Load the analysis sources into the test environment (the analysis is plain scripts, not a package).
r_dir <- normalizePath(test_path("..", "..", "R"))
for (f in c("read_inputs.R", "accounting.R", "estimates.R", "bootstrap.R", "inference.R")) {
  source(file.path(r_dir, f), local = TRUE)
}

fixture_dir <- function() normalizePath(test_path("fixtures", "rehearsal"))

read_bytes_as_text <- function(path) rawToChar(readBin(path, "raw", file.size(path)))

write_text <- function(text, path) {
  writeBin(charToRaw(text), path)
  path
}

edit_lines <- function(f) function(text) {
  lines <- strsplit(text, "\n", fixed = TRUE)[[1]]
  paste0(paste(f(lines), collapse = "\n"), "\n")
}

# Replace exactly one occurrence of `old` (fixed string); fail loudly if the fixture changed.
replace_once <- function(text, old, new) {
  hits <- gregexpr(old, text, fixed = TRUE)[[1]]
  stopifnot(length(hits) == 1, hits[1] > 0)
  sub(old, new, text, fixed = TRUE)
}

# Replace the n-th occurrence of `old`.
replace_nth <- function(text, old, new, n) {
  hits <- gregexpr(old, text, fixed = TRUE)[[1]]
  stopifnot(length(hits) >= n, hits[1] > 0)
  start <- hits[n]
  paste0(substr(text, 1, start - 1), new, substr(text, start + nchar(old), nchar(text)))
}

# Copy the frozen fixture to a temporary directory with optional edits, re-pinning the CSV hash
# in the handoff (unless told not to) so a test reaches the check it is aimed at.
make_inputs <- function(csv_edit = identity, handoff_edit = identity, repin_csv = TRUE,
                        handoff_sha = NULL) {
  dir <- tempfile("inputs")
  dir.create(dir)
  csv_text <- csv_edit(read_bytes_as_text(file.path(fixture_dir(), "analysis_table.csv")))
  old_sha <- sha256_file(file.path(fixture_dir(), "analysis_table.csv"))
  csv_path <- write_text(csv_text, file.path(dir, "analysis_table.csv"))
  handoff_text <- read_bytes_as_text(file.path(fixture_dir(), "handoff.json"))
  if (repin_csv) {
    handoff_text <- replace_once(
      handoff_text, paste0('"sha256": "', old_sha, '"'),
      paste0('"sha256": "', sha256_file(csv_path), '"')
    )
  }
  handoff_path <- write_text(handoff_edit(handoff_text), file.path(dir, "handoff.json"))
  list(
    handoff = handoff_path, csv = csv_path,
    handoff_sha = if (is.null(handoff_sha)) sha256_file(handoff_path) else handoff_sha
  )
}

load_fixture <- function(...) {
  inputs <- make_inputs(...)
  read_inputs(inputs$handoff, inputs$handoff_sha, inputs$csv)
}

fixture <- function() load_fixture()

# Split data lines on commas (the frozen fixture contains no quoted cells) and rebuild.
csv_columns <- function() strsplit(readLines(file.path(fixture_dir(), "analysis_table.csv"), n = 1), ",")[[1]]

edit_cell <- function(line_no, column, value) {
  index <- match(column, csv_columns())
  stopifnot(!is.na(index))
  edit_lines(function(lines) {
    cells <- strsplit(lines[line_no], ",", fixed = TRUE)[[1]]
    stopifnot(length(cells) == length(csv_columns()))
    cells[index] <- value
    lines[line_no] <- paste(cells, collapse = ",")
    lines
  })
}

expect_input_error <- function(expr, step) {
  err <- tryCatch({ force(expr); NULL }, input_error = function(e) e)
  expect_false(is.null(err), info = paste("no input_error for step", step))
  if (!is.null(err)) expect_match(conditionMessage(err), paste0("[", step, "]"), fixed = TRUE)
}
