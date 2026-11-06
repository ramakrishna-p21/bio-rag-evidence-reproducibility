# Input reading and validation. Expected values below are written from the frozen fixture
# construction (tests/test_r_input_fixture.py), not read back from the code under test.

test_that("file hashing is SHA-256 of the raw bytes (known test vectors)", {
  vector_file <- function(bytes) {
    path <- tempfile()
    writeBin(bytes, path)
    path
  }
  expect_identical(
    sha256_file(vector_file(charToRaw("abc"))),
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
  )
  expect_identical(
    sha256_file(vector_file(raw(0))),
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
  )
  expect_identical(
    sha256_file(vector_file(charToRaw("The quick brown fox jumps over the lazy dog"))),
    "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592"
  )
  # Not the hash of the serialized R object, and a trailing newline matters.
  expect_false(identical(sha256_file(vector_file(charToRaw("abc\n"))), sha256_file(vector_file(charToRaw("abc")))))
})

test_that("the column-list hash uses the Python canonical JSON form", {
  columns <- list(
    list(name = "a", type = "string", nullable = FALSE),
    list(name = "b_1", type = "integer", nullable = TRUE)
  )
  # Literal canonical form (sorted keys, no whitespace) and its SHA-256, computed outside R.
  expect_identical(
    columns_canonical_json(columns),
    '[{"name":"a","nullable":false,"type":"string"},{"name":"b_1","nullable":true,"type":"integer"}]'
  )
  expect_identical(
    columns_sha256(columns),
    "5637e673ae854808d99f5dd7dff44bc617cf18c3f85e49318bed0e501b705fcc"
  )
  expect_input_error(columns_sha256(list(list(name = "café", type = "string", nullable = FALSE))), "columns_hash")
  expect_input_error(columns_sha256(list(list(name = "a", type = "float", nullable = FALSE))), "columns_hash")
})

test_that("the frozen fixture is read and the recorded column hash is reproduced in R", {
  inputs <- make_inputs()
  read <- read_inputs(inputs$handoff, inputs$handoff_sha, inputs$csv)
  h <- read$handoff
  expect_identical(h$csv_contract$columns_sha256, columns_sha256(h$csv_contract$columns))
  expect_identical(read$input_origin, "synthetic_rehearsal")
  expect_false(read$lineage_verified_by_r)  # R never verifies Python lineage
  rows <- read$rows
  expect_identical(dim(rows), c(18L, 37L))
  expect_identical(names(rows), vapply(h$csv_contract$columns, function(c) c$name, ""))
  expect_type(rows$repeat_index, "integer")
  expect_type(rows$primary_row, "logical")
  expect_true(all(rows$primary_row))
  expect_identical(unique(rows$run_id), "run1")
  # nullable NA -> missing: completion_kind is NA exactly for the three operational failures
  expect_identical(is.na(rows$completion_kind), rows$execution_status == "operational_failure")
  expect_identical(sum(is.na(rows$completion_kind)), 3L)
  expect_false(anyNA(rows$item_id))
  # q_004 PDF_DOCLING is an operational failure: correct_without_evidence is the string "NA".
  cell <- rows$correct_without_evidence[rows$item_id == "q_004" & rows$condition == "PDF_DOCLING"]
  expect_identical(cell, "NA")
  expect_false(anyNA(rows$correct_without_evidence))
  expect_identical(sort(unique(rows$correct_without_evidence)), c("FALSE", "NA", "TRUE"))
})

test_that("the handoff and table hashes are checked against the pinned values", {
  inputs <- make_inputs()
  expect_input_error(read_inputs(inputs$handoff, strrep("0", 64), inputs$csv), "handoff_hash")
  expect_input_error(read_inputs(inputs$handoff, "not-a-hash", inputs$csv), "handoff_hash")
  # CSV changed but the handoff still pins the original bytes.
  stale <- make_inputs(csv_edit = edit_cell(2, "benchmark_version", "benchmark_v2"), repin_csv = FALSE)
  expect_input_error(read_inputs(stale$handoff, stale$handoff_sha, stale$csv), "csv_hash")
})

test_that("byte-level format violations are refused", {
  expect_input_error(load_fixture(csv_edit = function(t) paste0("﻿", t)), "bytes")
  expect_input_error(load_fixture(csv_edit = function(t) gsub("\n", "\r\n", t, fixed = TRUE)), "bytes")
  expect_input_error(load_fixture(csv_edit = function(t) sub("\n$", "", t)), "bytes")
  expect_input_error(load_fixture(csv_edit = function(t) paste0(t, "\n")), "cell_count")  # blank line
})

test_that("header, cell-count, type and missing-token violations are refused", {
  expect_input_error(load_fixture(csv_edit = function(t) sub("item_id", "item_idx", t, fixed = TRUE)), "header")
  swapped <- edit_lines(function(l) {
    cells <- strsplit(l[1], ",", fixed = TRUE)[[1]]
    l[1] <- paste(cells[c(2, 1, 3:length(cells))], collapse = ",")
    l
  })
  expect_input_error(load_fixture(csv_edit = swapped), "header")
  expect_input_error(load_fixture(csv_edit = edit_lines(function(l) { l[3] <- paste0(l[3], ",x"); l })), "cell_count")
  expect_input_error(load_fixture(csv_edit = edit_lines(function(l) { l[3] <- sub(",[^,]*$", "", l[3]); l })), "cell_count")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "primary_row", "true")), "type")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "primary_row", "NA")), "missing_token")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "repeat_index", "01")), "type")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "repeat_index", "1.0")), "type")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "repeat_index", "99999999999999999999")), "type")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "item_id", "NA")), "missing_token")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "answer_correct", "NA")), "missing_token")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "item_id", "")), "empty_cell")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "run_id", "")), "empty_cell")
  # A literal NA in a nullable column is missing, in correct_without_evidence it is data.
  ok <- load_fixture(csv_edit = edit_cell(2, "run_id", "NA"))
  expect_true(is.na(ok$rows$run_id[1]))
})

test_that("a lower-case or numeric tri-state value is not accepted as correct_without_evidence data", {
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "correct_without_evidence", "na")), "vocabulary")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "correct_without_evidence", "yes")), "vocabulary")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "evidence_delivered", "maybe")), "vocabulary")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "answer_correct", "correct")), "vocabulary")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "condition", "BOTH")), "vocabulary")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "correct_without_evidence", "")), "empty_cell")
})

test_that("row count, order, uniqueness and unexpected rows are checked", {
  drop_last <- edit_lines(function(l) l[-length(l)])
  expect_input_error(load_fixture(csv_edit = drop_last), "row_count")
  reorder <- edit_lines(function(l) l[c(1, 3, 2, 4:length(l))])
  expect_input_error(load_fixture(csv_edit = reorder), "row_order")
  # Duplicate observation key (row 3 takes row 2's key).
  key_col <- match("observation_key", csv_columns())
  dup <- edit_lines(function(l) {
    first <- strsplit(l[2], ",", fixed = TRUE)[[1]][key_col]
    cells <- strsplit(l[3], ",", fixed = TRUE)[[1]]
    cells[key_col] <- first
    l[3] <- paste(cells, collapse = ",")
    l
  })
  expect_input_error(load_fixture(csv_edit = dup), "duplicate_key")
})

test_that("rows from another phase, split or version are refused", {
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "phase", "final_test")), "purpose")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "phase", "qualification")), "purpose")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "split", "final_test")), "purpose")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "analysis_role", "confirmatory")), "purpose")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "benchmark_version", "benchmark_v2")), "version")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "corpus_version", "corpus_v2")), "version")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "config_sha256", strrep("c", 64))), "arm_config")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "retrieval_config_sha256", strrep("c", 64))), "arm_config")
  expect_input_error(load_fixture(csv_edit = edit_cell(2, "matrix_arm", "dense_robustness")), "arm_config")
})

test_that("rows that are consistent among themselves but differ from the handoff's versions are refused", {
  expect_input_error(load_fixture(csv_edit = function(t) gsub("benchmark_v1", "benchmark_v2", t, fixed = TRUE)), "version")
  expect_input_error(load_fixture(csv_edit = function(t) gsub("corpus_v1", "corpus_v2", t, fixed = TRUE)), "version")
})

test_that("a handoff that is not the rehearsal development contract is refused", {
  h <- function(old, new) function(t) replace_once(t, old, new)
  expect_input_error(load_fixture(handoff_edit = h('"input_origin": "synthetic_rehearsal"', '"input_origin": "verified_official_run"')), "origin")
  expect_input_error(load_fixture(handoff_edit = h('"input_origin": "synthetic_rehearsal"', '"input_origin": "something_else"')), "origin")
  expect_input_error(load_fixture(handoff_edit = h('"kind": "analysis_handoff"', '"kind": "other"')), "handoff_schema")
  expect_input_error(load_fixture(handoff_edit = h('"schema_version": "1",\n  "scope"', '"schema_version": "2",\n  "scope"')), "handoff_schema")
  expect_input_error(load_fixture(handoff_edit = h('"phase": "development"', '"phase": "final_test"')), "handoff_schema")
  expect_input_error(load_fixture(handoff_edit = h('"bootstrap_resamples": 10000', '"bootstrap_resamples": 1000')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"multiplicity": "holm"', '"multiplicity": "bonferroni"')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"package": "clubSandwich"', '"package": "sandwich"')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"variance": "CR2"', '"variance": "HC3"')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"multiplicity_family": [\n      "evidence_delivered",\n      "answer_correct"\n    ]', '"multiplicity_family": [\n      "evidence_delivered"\n    ]')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"multiplicity_family": [\n      "evidence_delivered",\n      "answer_correct"\n    ]', '"multiplicity_family": [\n      "answer_correct",\n      "evidence_delivered"\n    ]')), "analysis_spec")
  expect_input_error(load_fixture(handoff_edit = h('"missing_token": "NA"', '"missing_token": "-"')), "csv_contract")
  expect_input_error(load_fixture(handoff_edit = h('"row_count": 18', '"row_count": 17')), "row_count")
  expect_input_error(load_fixture(handoff_edit = function(t) {
    old <- jsonlite::fromJSON(t, simplifyVector = FALSE)$csv_contract$columns_sha256
    replace_once(t, old, strrep("1", 64))
  }), "columns_hash")
})

test_that("the pinned seed is not part of the handoff and is never invented", {
  read <- load_fixture()
  expect_identical(unlist(read$handoff$awaiting_approval), "primary_bootstrap_seed")
})
