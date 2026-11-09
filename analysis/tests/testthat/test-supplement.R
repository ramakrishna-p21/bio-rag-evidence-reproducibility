# Supplement summary: counting of Python-decided labels, links, publication and tamper detection.
# Expected counts are derived by hand from the frozen synthetic supplement fixture
# (tests/test_supplement_fixture.py): three items, JATS and PDF_DOCLING, three repeats.

sclock <- function(iso) function() as.POSIXct(iso, tz = "UTC", format = "%Y-%m-%dT%H:%M:%OS")
sup_fixture_dir <- function() normalizePath(test_path("fixtures", "supplement"))
sup_meta <- jsonlite::fromJSON(file.path(sup_fixture_dir(), "fixture.json"))
sup_args <- list(handoff = file.path(sup_fixture_dir(), "handoff.json"), handoff_sha = sup_meta$handoff_sha256,
                 csv = file.path(sup_fixture_dir(), "analysis_table.csv"))
sup_py <- file.path(sup_fixture_dir(), "supplement")
sup_inputs <- function() read_inputs(sup_args$handoff, sup_args$handoff_sha, sup_args$csv)

s_root <- cleanup_dir(tempfile("supp"), teardown_env())
dir.create(s_root)
s_out1 <- file.path(s_root, "one"); dir.create(s_out1)
s_out2 <- file.path(s_root, "two"); dir.create(s_out2)
run_s <- function(out, clock = "2026-11-08T10:00:00.000", py = sup_py, rec = sup_meta$supplement_record_sha256,
                  csv = sup_args$csv, handoff = sup_args$handoff, handoff_sha = sup_args$handoff_sha) {
  run_supplement(handoff, handoff_sha, csv, py, rec, out, analysis_dir(), now = sclock(clock))
}
s1 <- run_s(s_out1)
s2 <- run_s(s_out2, clock = "2026-11-08T12:30:00.000")
read_summary <- function(dir, name) parse_canonical_csv(readBin(file.path(dir, name), "raw", 1e6), SUPPLEMENT_SCHEMAS[[name]], name)
cell <- function(df, ...) {
  keep <- rep(TRUE, nrow(df))
  for (nm in names(list(...))) keep <- keep & (df[[nm]] %in% list(...)[[nm]])
  df[keep, , drop = FALSE]
}

test_that("pair classes are counted per arm and repeat with the denominator of that population", {
  t <- read_summary(s1$result_dir, "supplement_pairs_summary.csv")
  expect_true(all(t$denominator == 3L))
  one <- function(rep, cls, jats = "answer", pdf = "answer") sum(cell(t, repeat_index = rep, pair_class = cls, jats_state = jats, pdf_state = pdf)$n)
  expect_identical(one(1L, "differ_correctness_differs"), 1L)
  expect_identical(one(1L, "both_correct"), 1L)
  expect_identical(one(1L, "identical_bytes"), 1L)
  expect_identical(one(2L, "not_comparable", pdf = "abstained"), 1L)
  expect_identical(one(2L, "differ_correctness_differs"), 2L)
  expect_identical(one(2L, "identical_bytes"), 0L)
  expect_identical(one(3L, "identical_bytes"), 2L)
  expect_identical(sum(t$n), 9L)
})

test_that("stability classes keep the byte flag and never count abstention as an identical incorrect answer", {
  t <- read_summary(s1$result_dir, "supplement_stability_summary.csv")
  expect_true(all(t$denominator == 3L))
  n <- function(cond, cls, byte) sum(cell(t, condition = cond, stability_class = cls, byte_identical = byte)$n)
  expect_identical(n("JATS", "incorrect_identical_normalized", "no"), 1L)
  expect_identical(n("JATS", "mixed", "no"), 1L)
  expect_identical(n("JATS", "all_correct", "yes"), 0L)
  expect_identical(n("JATS", "incorrect_varying", "no"), 1L)
  expect_identical(n("PDF_DOCLING", "all_correct", "yes"), 1L)
  expect_identical(n("PDF_DOCLING", "all_correct", "no"), 1L)
  expect_identical(sum(is.na(t$byte_identical)), 1L)
  expect_identical(t$stability_class[is.na(t$byte_identical)], "contains_abstention_or_unparseable")
  expect_false(any(t$condition == "PDF_DOCLING" & t$stability_class == "incorrect_identical_normalized"))
})

test_that("evidence form and location are counted per item; outcomes use the table's own labels", {
  e <- read_summary(s1$result_dir, "supplement_evidence_summary.csv")
  expect_true(all(e$denominator == 3L))
  expect_identical(sum(cell(e, item_form = "prose", jats_location = "body")$n), 1L)
  expect_identical(sum(cell(e, item_form = "prose", jats_location = "abstract")$n), 1L)
  expect_identical(sum(cell(e, item_form = "table", jats_location = "body", form_relation = "agree")$n), 1L)
  o <- read_summary(s1$result_dir, "supplement_evidence_outcomes.csv")
  v <- function(cond, form, correct) sum(cell(o, condition = cond, item_form = form, answer_correct = correct)$n)
  expect_identical(c(v("JATS", "prose", "no"), v("JATS", "table", "yes"), v("JATS", "prose", "yes"), v("JATS", "table", "no")),
                   c(2L, 1L, 0L, 0L))   # repeat 1 only: item q_002's wrong JATS repeat 2 is not counted
  expect_identical(c(v("PDF_DOCLING", "prose", "no"), v("PDF_DOCLING", "prose", "yes"), v("PDF_DOCLING", "table", "yes")), c(1L, 1L, 1L))
  expect_identical(unique(cell(o, condition = "PDF_DOCLING", item_form = "prose")$denominator), 2L)
  f <- read_summary(s1$result_dir, "supplement_stage_forms.csv")
  expect_identical(cell(f, condition = "JATS")[c("stage_a_form", "stage_b_form", "n", "denominator")],
                   data.frame(stage_a_form = "prose", stage_b_form = "prose", n = 3L, denominator = 3L, stringsAsFactors = FALSE))
  expect_identical(cell(f, condition = "PDF_DOCLING")$stage_a_form, "table")
})

test_that("the directory is exactly the declared files, read-only, and verifies with the trusted record hash", {
  expect_match(basename(s1$result_dir), "^supplement_")
  expect_setequal(list.files(s1$result_dir, all.files = TRUE, no.. = TRUE), SUPPLEMENT_FILES)
  for (f in list.files(s1$result_dir, full.names = TRUE)) expect_equal(unname(file.mode(f) & as.octmode("0222")), as.octmode("0000"))
  v <- verify_supplement_summary(s1$result_dir, s1$record_sha256, require_read_only = TRUE)
  expect_true(v$ok && v$sealed && v$record_authenticated)
  rec <- jsonlite::fromJSON(file.path(s1$result_dir, SUPPLEMENT_SUMMARY_RECORD), simplifyVector = FALSE)
  expect_identical(rec$purpose, "synthetic_rehearsal")
  expect_identical(rec$scientific_authority, "none")
  expect_identical(rec$inputs$supplement_sha256, sup_meta$supplement_sha256)
  expect_identical(rec$inputs$csv_sha256, sup_meta$csv_sha256)
})

test_that("the summary identity and tables are deterministic; execution facts differ", {
  expect_identical(s1$supplement_summary_sha256, s2$supplement_summary_sha256)
  for (n in SUPPLEMENT_DATA_FILES) {
    expect_identical(readBin(file.path(s1$result_dir, n), "raw", 1e6), readBin(file.path(s2$result_dir, n), "raw", 1e6))
  }
  expect_false(identical(s1$record_sha256, s2$record_sha256))
  r <- verify_supplement_rerun(s1$result_dir, sup_args$handoff, sup_args$handoff_sha, sup_args$csv, sup_py)
  expect_identical(r$comparison, "byte_identical_tables")
})

# Replace one fixed string in a file's text; fail loudly if it is absent so a tamper test cannot pass vacuously.
edit_file <- function(path, old, new) {
  text <- rawToChar(readBin(path, "raw", file.size(path)))
  stopifnot(grepl(old, text, fixed = TRUE))
  writeBin(charToRaw(sub(old, new, text, fixed = TRUE)), path)
  invisible(path)
}

copy_supplement <- function(dir = sup_py) {
  dest <- cleanup_dir(tempfile("spy"), parent.frame())
  dir.create(dest)
  file.copy(list.files(dir, full.names = TRUE), dest)
  Sys.chmod(list.files(dest, full.names = TRUE), "0644")
  dest
}

test_that("a changed, missing, extra or foreign Python supplement is refused before counting", {
  rec <- sup_meta$supplement_record_sha256
  expect_input_error(read_supplement(sup_py, "0", sup_inputs()), "supplement_input")
  expect_input_error(read_supplement(sup_py, strrep("0", 64), sup_inputs()), "supplement_input")
  d <- copy_supplement()
  path <- file.path(d, "supplement_pairs.csv")
  edit_file(path, "identical_bytes", "both_correct")
  expect_input_error(read_supplement(d, rec, sup_inputs()), "supplement_input")
  d <- copy_supplement(); file.remove(file.path(d, "supplement_stability.csv"))
  expect_input_error(read_supplement(d, rec, sup_inputs()), "supplement_input")
  d <- copy_supplement(); writeLines("x", file.path(d, "extra.txt"))
  expect_input_error(read_supplement(d, rec, sup_inputs()), "supplement_input")
  other <- make_inputs()   # a different, valid rehearsal table and handoff
  expect_input_error(read_supplement(sup_py, rec, read_inputs(other$handoff, other$handoff_sha, other$csv)), "supplement_input")
})

test_that("a record edited to match its files still fails the Python identity and links", {
  rec <- sup_meta$supplement_record_sha256
  d <- copy_supplement()
  path <- file.path(d, "supplement_record.json")
  edit_file(path, '"final":false', '"final":true')
  expect_input_error(read_supplement(d, sha256_file(path), sup_inputs()), "supplement_input")
})

# Rewrite the Python record so that it is consistent with edited files, as a careful forger would:
# entries re-hashed and the supplement identity recomputed. Only R's own links can then catch it.
reseal_supplement <- function(dir, edit_record = identity) {
  path <- file.path(dir, "supplement_record.json")
  rec <- jsonlite::fromJSON(path, simplifyVector = FALSE)
  for (n in names(rec$files)) {
    b <- readBin(file.path(dir, n), "raw", file.size(file.path(dir, n)))
    rec$files[[n]]$bytes <- length(b); rec$files[[n]]$sha256 <- sha256_raw(b)
  }
  rec$supplement_sha256 <- python_supplement_id(rec$handoff_sha256, rec$analysis_table_sha256,
                                                vapply(rec$files, function(e) e$sha256, ""))
  rec <- edit_record(rec)
  writeBin(charToRaw(as.character(jsonlite::toJSON(rec, auto_unbox = TRUE))), path)
  sha256_file(path)
}

test_that("a sealed forgery is still caught by R's own links to the table and handoff", {
  d <- copy_supplement()
  sha <- reseal_supplement(d, function(r) { r$analysis_table_sha256 <- strrep("a", 64); r$supplement_sha256 <-
    python_supplement_id(r$handoff_sha256, r$analysis_table_sha256, vapply(r$files, function(e) e$sha256, "")); r })
  err <- tryCatch(read_supplement(d, sha, sup_inputs()), input_error = function(e) conditionMessage(e))
  expect_match(err, "does not cite this handoff and analysis table")
  d <- copy_supplement()
  sha <- reseal_supplement(d, function(r) { r$handoff_sha256 <- strrep("b", 64); r$supplement_sha256 <-
    python_supplement_id(r$handoff_sha256, r$analysis_table_sha256, vapply(r$files, function(e) e$sha256, "")); r })
  err <- tryCatch(read_supplement(d, sha, sup_inputs()), input_error = function(e) conditionMessage(e))
  expect_match(err, "does not cite this handoff and analysis table")
  d <- copy_supplement()
  sha <- reseal_supplement(d, function(r) { r$supplement_sha256 <- strrep("c", 64); r })
  err <- tryCatch(read_supplement(d, sha, sup_inputs()), input_error = function(e) conditionMessage(e))
  expect_match(err, "supplement_sha256 does not match")
  d <- copy_supplement()
  edit_file(file.path(d, "supplement_response.csv"), ",confirmatory,JATS,1,", ",confirmatory,JATS,2,")
  sha <- reseal_supplement(d)
  err <- tryCatch(read_supplement(d, sha, sup_inputs()), input_error = function(e) conditionMessage(e))
  expect_match(err, "disagrees with its analysis-table row")
  d <- copy_supplement()   # an unedited reseal is accepted, so the failures above are the edits
  sha <- reseal_supplement(d)
  expect_silent(read_supplement(d, sha, sup_inputs()))
})

test_that("summary files are verified against the record and each other", {
  copy_summary <- function() {
    dest <- cleanup_dir(tempfile("ssum"), parent.frame())
    dir.create(dest)
    file.copy(list.files(s1$result_dir, full.names = TRUE), dest)
    Sys.chmod(list.files(dest, full.names = TRUE), "0644")
    dest
  }
  expect_fails <- function(dir, ..., pattern = NULL) {
    err <- tryCatch({ verify_supplement_summary(dir, ...); NULL }, run_error = function(e) e)
    expect_false(is.null(err))
    if (!is.null(err) && !is.null(pattern)) expect_match(conditionMessage(err), pattern)
  }
  d <- copy_summary()
  p <- file.path(d, "supplement_pairs_summary.csv")
  edit_file(p, "both_correct", "both_correcx")
  expect_fails(d, pattern = "differs from the hash")
  d <- copy_summary(); writeLines("x", file.path(d, "extra.csv"))
  expect_fails(d, pattern = "unexpected file")
  expect_fails(s1$result_dir, expected_record_sha256 = strrep("0", 64), pattern = "differs from the trusted")
  d <- copy_summary()
  p <- file.path(d, SUPPLEMENT_SUMMARY_RECORD)
  edit_file(p, "synthetic_rehearsal", "official_origin")
  expect_fails(d, pattern = "must be")
  d <- copy_summary()   # a record whose identity no longer matches its tables
  edit_file(file.path(d, SUPPLEMENT_SUMMARY_RECORD), s1$supplement_summary_sha256, strrep("d", 64))
  expect_fails(d, pattern = "supplement_summary_sha256 does not match")
})

test_that("consistency checks reject counts outside their denominators", {
  tables <- lapply(stats::setNames(SUPPLEMENT_DATA_FILES, SUPPLEMENT_DATA_FILES), function(n) read_summary(s1$result_dir, n))
  expect_true(check_supplement_tables(tables))
  bad <- tables; bad[["supplement_pairs_summary.csv"]]$n[1] <- 9L
  expect_error(check_supplement_tables(bad), "denominator|outside")
  bad <- tables   # a zero count with the group sum kept intact: only the range check sees it
  bad[["supplement_pairs_summary.csv"]]$n[1:2] <- c(0L, bad[["supplement_pairs_summary.csv"]]$n[2] + bad[["supplement_pairs_summary.csv"]]$n[1])
  expect_error(check_supplement_tables(bad), "outside")
  bad <- tables; bad[["supplement_populations.csv"]] <- bad[["supplement_populations.csv"]][-1, ]
  expect_error(check_supplement_tables(bad), "populations")
})

test_that("the summary never modifies or depends on the primary or secondary scientific hashes", {
  expect_false("scientific_result_sha256" %in% names(jsonlite::fromJSON(file.path(s1$result_dir, SUPPLEMENT_SUMMARY_RECORD))$inputs))
  rec <- jsonlite::fromJSON(file.path(s1$result_dir, SUPPLEMENT_SUMMARY_RECORD), simplifyVector = FALSE)
  expect_identical(rec$primary_result$citation, "not_cited")
})

test_that("a handoff that claims an official origin never reaches the supplement", {
  i <- make_inputs(handoff_edit = function(t) replace_once(t, '"input_origin": "synthetic_rehearsal"', '"input_origin": "verified_official_run"'),
                   csv_edit = identity)
  expect_error(compute_supplement(i$handoff, i$handoff_sha, i$csv, sup_py, sup_meta$supplement_record_sha256))
})
