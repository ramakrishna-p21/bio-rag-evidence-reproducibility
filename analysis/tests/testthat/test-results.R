# Stage 3a: result tables, exact values and deterministic serialization.

parse_csv_bytes <- function(bytes) {
  utils::read.csv(text = rawToChar(bytes), colClasses = "character", na.strings = character(0),
                  stringsAsFactors = FALSE)
}

cell_num <- function(x) suppressWarnings(as.numeric(x))

run_fixture_results <- function(seed = TEST_SEED) {
  a <- fixture_args()
  compute_results(a$handoff_path, a$handoff_sha256, a$csv_path, seed)
}

test_that("format_double round-trips finite doubles exactly and normalises negative zero", {
  expect_identical(format_double(0.1), "0.10000000000000001")
  expect_identical(format_double(1 / 3), "0.33333333333333331")
  expect_identical(format_double(-0), "0")
  expect_identical(format_double(0), "0")
  expect_identical(format_double(1), "1")
  expect_identical(format_double(NA_real_), "NA")
  set.seed(1)
  x <- c(stats::rnorm(200), stats::runif(200) * 1e-300, stats::runif(50) * 1e22, 5e-324, .Machine$double.xmax)
  for (v in x) expect_identical(as.numeric(format_double(v)), v)
})

test_that("format_double refuses non-finite values", {
  for (v in c(NaN, Inf, -Inf)) expect_run_error(format_double(v), "serialize")
})

test_that("csv_bytes writes the project CSV contract", {
  df <- data.frame(name = c("a", "b,c", "d\"e"), n = c(1L, NA, 3L), x = c(0.5, NA, -0), stringsAsFactors = FALSE)
  out <- csv_bytes(df, c(name = "chr", n = "int", x = "dbl"))
  expect_type(out, "raw")
  expect_identical(rawToChar(out), "name,n,x\na,1,0.5\n\"b,c\",NA,NA\n\"d\"\"e\",3,0\n")
  expect_false(identical(out[1:3], as.raw(c(0xef, 0xbb, 0xbf))))
  expect_identical(csv_bytes(df, c(name = "chr", n = "int", x = "dbl")), out)
})

test_that("csv_bytes writes only a header for zero rows", {
  df <- data.frame(a = character(0), b = integer(0), stringsAsFactors = FALSE)
  expect_identical(rawToChar(csv_bytes(df, c(a = "chr", b = "int"))), "a,b\n")
})

test_that("csv_bytes refuses ambiguous or invalid cells", {
  s <- c(name = "chr", n = "int")
  expect_run_error(csv_bytes(data.frame(name = "NA", n = 1L, stringsAsFactors = FALSE), s), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "", n = 1L, stringsAsFactors = FALSE), s), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "a\nb", n = 1L, stringsAsFactors = FALSE), s), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "a", n = 1.5), s), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "a", n = 1L), c(name = "chr", n = "dbl")), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "a", n = 1L), c(n = "int", name = "chr")), "serialize")
  expect_run_error(csv_bytes(data.frame(a = "x", b = "y", stringsAsFactors = FALSE), c(c = "chr", d = "chr")), "serialize")
  expect_run_error(csv_bytes(data.frame(name = "a", x = Inf), c(name = "chr", x = "dbl")), "serialize")
})

test_that("coprimary table holds the hand-computed fixture values", {
  r <- run_fixture_results()
  t <- parse_csv_bytes(r$files[["coprimary_results.csv"]])
  expect_identical(names(t), names(COPRIMARY_SCHEMA))
  expect_identical(t$outcome, c("evidence_delivered", "answer_correct"))
  ev <- t[1, ]; ac <- t[2, ]
  # evidence_delivered: 9 eligible, 7 complete pairs, d sum = 1, JATS 6/7, PDF 5/7, paper means
  # (1, 0, -1/2, 1, 0) -> equal-paper 0.3; bounds [0, 2/9] with one unresolved cell per condition.
  expect_equal(c(ev$n_eligible, ev$n_complete, ev$n_papers), c("9", "7", "5"))
  expect_equal(cell_num(ev$rate_jats), 6 / 7, tolerance = 1e-15)
  expect_equal(cell_num(ev$rate_pdf), 5 / 7, tolerance = 1e-15)
  expect_equal(cell_num(ev$rd), 1 / 7, tolerance = 1e-15)
  expect_equal(cell_num(ev$rd_pp), 100 / 7, tolerance = 1e-14)
  expect_equal(cell_num(ev$equal_paper_rd), 0.3, tolerance = 1e-15)
  expect_equal(cell_num(ev$bound_lower), 0, tolerance = 1e-15)
  expect_equal(cell_num(ev$bound_upper), 2 / 9, tolerance = 1e-15)
  expect_equal(c(ev$bound_n_unknown_jats, ev$bound_n_unknown_pdf), c("1", "1"))
  expect_identical(ev$bound_n_operational_excluded, "0")
  # answer_correct: d sum = 0 (+1, -1, -1, +1 and zeros), 7 complete pairs. The two items with
  # operational failures are missing, not unresolved, so the bounds are [0, 0] over 7 items.
  expect_equal(c(ac$n_eligible, ac$n_complete, ac$n_papers), c("9", "7", "5"))
  expect_equal(cell_num(ac$rd), 0, tolerance = 1e-15)
  expect_equal(cell_num(ac$bound_lower), 0, tolerance = 1e-15)
  expect_equal(cell_num(ac$bound_upper), 0, tolerance = 1e-15)
  expect_equal(c(ac$bound_n_unknown_jats, ac$bound_n_unknown_pdf), c("0", "0"))
  expect_identical(ac$bound_n_operational_excluded, "2")
  expect_identical(c(ev$summary_status, ev$boot_status, ev$test_status, ev$holm_status), rep("ok", 4))
  expect_identical(ev$test_reason, "NA")
  expect_identical(c(ev$boot_n_resamples, ev$boot_n_empty), c("10000", "0"))
})

test_that("coprimary CR2 values agree with the independent matrix reference", {
  r <- run_fixture_results()
  t <- parse_csv_bytes(r$files[["coprimary_results.csv"]])
  d <- c(1, 0, 0, 0, -1, 1, 0)                       # evidence_delivered complete pairs, item order
  paper <- c("P001", "P002", "P002", "P003", "P003", "P004", "P005")
  ref <- ref_matrix(d, paper)
  expect_equal(cell_num(t$test_se[1]), unname(ref["se"]), tolerance = 1e-10)
  expect_equal(cell_num(t$test_df[1]), unname(ref["df"]), tolerance = 1e-10)
  expect_equal(cell_num(t$test_t[1]), unname(ref["t"]), tolerance = 1e-10)
  expect_equal(cell_num(t$test_p[1]), unname(ref["p"]), tolerance = 1e-10)
  # Holm over (p_evidence, 1): the adjusted values are 2 * min(p) capped at 1 and max(p).
  p <- cell_num(t$test_p)
  expect_equal(cell_num(t$holm_p), stats::p.adjust(p, "holm"), tolerance = 1e-15)
})

test_that("bootstrap interval in the table is the pinned Pass 2 computation", {
  r <- run_fixture_results()
  t <- parse_csv_bytes(r$files[["coprimary_results.csv"]])
  a <- fixture_args()
  inp <- read_inputs(a$handoff_path, a$handoff_sha256, a$csv_path)
  cc <- derive_accounting(inp$rows, inp$handoff)$evidence_delivered$items
  cc <- cc[cc$complete, ]
  direct <- percentile_bootstrap(cc$y_jats - cc$y_pdf, cc$paper_id, TEST_SEED, 10000L)
  expect_identical(as.numeric(t$boot_ci_lower[1]), direct$ci_lower)
  expect_identical(as.numeric(t$boot_ci_upper[1]), direct$ci_upper)
})

test_that("paper and state counts are long, radix-ordered and consistent with the coprimary table", {
  r <- run_fixture_results()
  pc <- parse_csv_bytes(r$files[["paper_counts.csv"]])
  sc <- parse_csv_bytes(r$files[["state_counts.csv"]])
  expect_identical(names(pc), c("outcome", "paper_id", "eligible", "complete"))
  expect_identical(names(sc), c("outcome", "condition", "state", "n"))
  ev <- pc[pc$outcome == "evidence_delivered", ]
  expect_identical(ev$paper_id, c("P001", "P002", "P003", "P004", "P005"))
  expect_identical(ev$eligible, c("1", "2", "3", "1", "2"))
  expect_identical(ev$complete, c("1", "2", "2", "1", "1"))
  ac_states <- sc[sc$outcome == "answer_correct", ]
  expect_identical(paste(ac_states$condition, ac_states$state, sep = "|"),
                   c("JATS|missing", "JATS|no", "JATS|yes", "PDF_DOCLING|missing", "PDF_DOCLING|no", "PDF_DOCLING|yes"))
  expect_identical(ac_states$n, c("1", "2", "6", "2", "2", "5"))
  expect_identical(unique(sc$outcome), c("evidence_delivered", "answer_correct"))
})

test_that("results are deterministic: same inputs and seed give identical bytes and identity", {
  a <- run_fixture_results(); b <- run_fixture_results()
  expect_identical(a$files, b$files)
  expect_identical(a$scientific_result_sha256, b$scientific_result_sha256)
  expect_identical(a$file_sha256, b$file_sha256)
  expect_match(a$scientific_result_sha256, "^[0-9a-f]{64}$")
})

test_that("another seed changes the identity and leaves everything but the bootstrap columns untouched", {
  a <- run_fixture_results(); b <- run_fixture_results(TEST_SEED + 1L)
  expect_false(identical(a$scientific_result_sha256, b$scientific_result_sha256))
  expect_identical(a$files[["paper_counts.csv"]], b$files[["paper_counts.csv"]])
  expect_identical(a$files[["state_counts.csv"]], b$files[["state_counts.csv"]])
  ta <- parse_csv_bytes(a$files[["coprimary_results.csv"]]); tb <- parse_csv_bytes(b$files[["coprimary_results.csv"]])
  boot <- c("boot_ci_lower", "boot_ci_upper")
  expect_identical(ta[setdiff(names(ta), boot)], tb[setdiff(names(tb), boot)])
  # The interval itself may coincide on five papers; that the seed is applied is tested in test-bootstrap.R.
})

test_that("scientific identity binds handoff, CSV, seed and file hashes", {
  h <- strrep("a", 64); c1 <- strrep("b", 64)
  f <- c("a.csv" = strrep("c", 64), "b.csv" = strrep("d", 64))
  base <- scientific_result_id(h, c1, 7L, f)
  expect_match(base, "^[0-9a-f]{64}$")
  expect_identical(scientific_result_id(h, c1, 7L, rev(f)), base)             # file order is not identity
  expect_false(identical(scientific_result_id(strrep("e", 64), c1, 7L, f), base))
  expect_false(identical(scientific_result_id(h, strrep("e", 64), 7L, f), base))
  expect_false(identical(scientific_result_id(h, c1, 8L, f), base))
  g <- f; g[["b.csv"]] <- strrep("e", 64)
  expect_false(identical(scientific_result_id(h, c1, 7L, g), base))
  # The canonical JSON is fixed: sorted keys, no whitespace.
  expect_identical(
    sha256_raw(charToRaw(paste0('{"csv_sha256":"', c1, '","files":[{"name":"a.csv","sha256":"', f[["a.csv"]],
                                '"},{"name":"b.csv","sha256":"', f[["b.csv"]], '"}],"handoff_sha256":"', h,
                                '","seed":7}'))), base)
})

test_that("compute_results refuses an input that is not a synthetic rehearsal", {
  a <- fixture_args()
  inputs <- make_inputs(handoff_edit = function(t) replace_once(t, '"input_origin": "synthetic_rehearsal"',
                                                                '"input_origin": "verified_official_run"'))
  expect_error(compute_results(inputs$handoff, inputs$handoff_sha, inputs$csv, TEST_SEED), "input_origin|origin")
})

test_that("the origin guard in compute_results stands on its own", {
  # The reader already refuses non-rehearsal handoffs; this reaches the second line of defence.
  env <- environment(compute_results)
  real_read <- get("read_inputs", envir = env)
  withr::defer(assign("read_inputs", real_read, envir = env))
  a <- fixture_args()
  for (origin in c("verified_official_run", "qualification", "final_test")) {
    assign("read_inputs", function(...) { x <- real_read(...); x$input_origin <- origin; x }, envir = env)
    expect_run_error(compute_results(a$handoff_path, a$handoff_sha256, a$csv_path, TEST_SEED), "origin")
  }
})

fake_accounting <- function(handoff, items) {
  one <- function(items) {
    items$complete <- !is.na(items$y_jats) & !is.na(items$y_pdf)
    list(items = items, complete_item_ids = items$item_id[items$complete],
         state_counts = c("JATS|yes" = 1L), papers = data.frame(paper_id = unique(items$paper_id), eligible = 1L, complete = 1L,
                                                                  stringsAsFactors = FALSE))
  }
  stats::setNames(list(one(items), one(items)), EXPECTED_FAMILY)
}
fake_items <- function(paper, y_jats, y_pdf) {
  n <- length(paper)
  data.frame(item_id = sprintf("q_%03d", seq_len(n)), paper_id = paper,
             jats_state = ifelse(is.na(y_jats), "non_evaluable", ifelse(y_jats == 1, "yes", "no")),
             pdf_state = ifelse(is.na(y_pdf), "non_evaluable", ifelse(y_pdf == 1, "yes", "no")),
             y_jats = y_jats, y_pdf = y_pdf, stringsAsFactors = FALSE)
}

test_that("each outcome gets its own Holm-adjusted p-value", {
  a <- fixture_args()
  inp <- read_inputs(a$handoff_path, a$handoff_sha256, a$csv_path)
  acc <- derive_accounting(inp$rows, inp$handoff)
  inf <- coprimary_inference(acc, inp$handoff, TEST_SEED)
  inf$holm$adjusted <- c(evidence_delivered = 0.25, answer_correct = 0.5)
  t <- coprimary_table(acc, inf, inp$handoff)
  expect_identical(t$holm_p, c(0.25, 0.5))
})

test_that("undefined inference keeps its status and reason with NA numerics in the table", {
  h <- read_inputs(fixture_args()$handoff_path, fixture_args()$handoff_sha256, fixture_args()$csv_path)$handoff
  acc <- fake_accounting(h, fake_items(c("P1", "P1"), c(1, 0), c(0, 0)))   # one paper
  inf <- coprimary_inference(acc, h, TEST_SEED)
  t <- parse_csv_bytes(csv_bytes(coprimary_table(acc, inf, h), COPRIMARY_SCHEMA))
  expect_identical(t$test_status, rep("inference_undefined", 2))
  expect_identical(t$test_reason, rep("single_cluster", 2))
  expect_identical(t$boot_status, rep("single_cluster_degenerate", 2))
  expect_identical(t$holm_status, rep("holm_undefined", 2))
  expect_identical(t$test_p, c("NA", "NA")); expect_identical(t$holm_p, c("NA", "NA"))
  expect_identical(t$test_se, c("NA", "NA"))
  expect_equal(cell_num(t$rd), c(0.5, 0.5))                                   # estimate is kept
})

test_that("no complete pair is its own status and the bounds still use every eligible item", {
  h <- read_inputs(fixture_args()$handoff_path, fixture_args()$handoff_sha256, fixture_args()$csv_path)$handoff
  acc <- fake_accounting(h, fake_items(c("P1", "P2"), c(NA_real_, NA_real_), c(NA_real_, NA_real_)))
  inf <- coprimary_inference(acc, h, TEST_SEED)
  t <- parse_csv_bytes(csv_bytes(coprimary_table(acc, inf, h), COPRIMARY_SCHEMA))
  expect_identical(t$summary_status, rep("no_complete_pairs", 2))
  expect_identical(t$n_complete, c("0", "0")); expect_identical(t$n_papers, c("0", "0"))
  expect_identical(t$rate_jats, c("NA", "NA")); expect_identical(t$boot_ci_lower, c("NA", "NA"))
  expect_equal(cell_num(t$bound_lower), c(-1, -1)); expect_equal(cell_num(t$bound_upper), c(1, 1))
})

test_that("golden hashes of the synthetic rehearsal outputs (seed 20260601)", {
  # The count tables contain only integers and strings, so their bytes are platform independent.
  r <- run_fixture_results()
  expect_identical(unname(r$file_sha256[["paper_counts.csv"]]),
                   "a10733bc370026725562ae2f7dd706eed314b7c10227115469e75b81a659cb33")
  expect_identical(unname(r$file_sha256[["state_counts.csv"]]),
                   "d182b7397d699bf1d9ed233fda506ca01567d1b278513dac45876dc220b0e95a")
  # The coprimary file carries CR2 values from BLAS/LAPACK at 17 digits, so its byte identity is
  # claimed only for this R / platform / lock. A change here means the computation or the format
  # changed and needs a recorded decision; on another machine compare the values with tolerance.
  expect_identical(unname(r$file_sha256[["coprimary_results.csv"]]),
                   "2f27d0f51c50279825ce29bf3749da12af051b4da0f8c104f214a14fb36e1239")
  expect_identical(r$scientific_result_sha256,
                   "075ec3af857b5eb994080078260b56154be4078ceec4c5855ff128242d6cf63d")
})
