# The confirmatory-only official path through every R gate after the Python verification.

test_that("a confirmatory-only official run passes every R gate, publishes and verifies", {
  i <- official_scope_inputs()
  stub_python_proof(i$proof)
  out <- tempfile("out"); dir.create(out)
  withr::defer(system2("chmod", c("-R", "u+w", shQuote(out)))); withr::defer(unlink(out, recursive = TRUE), priority = "last")
  r <- run_analysis(i$handoff, i$handoff_sha, i$csv, real_lock_sha(), i$seed, "final_test", out, analysis_dir(),
                    synchronized = TRUE, official_request = i$request, python = "unused")
  expect_identical(r$purpose, "final_test_confirmatory")
  v <- verify_result(r$result_dir, expected_record_sha256 = r$record_sha256, require_read_only = TRUE, official_proof = i$proof)
  expect_identical(v$record$inputs$input_origin, "verified_official_run")
  expect_identical(v$record$seed$value, i$seed)
})

test_that("the same run is refused when the proof carries the other scope's count", {
  i <- official_scope_inputs()
  i$proof$observations <- 4575L
  stub_python_proof(i$proof)
  out <- tempfile("out"); dir.create(out)
  withr::defer(unlink(out, recursive = TRUE))
  expect_error(run_analysis(i$handoff, i$handoff_sha, i$csv, real_lock_sha(), i$seed, "final_test", out, analysis_dir(),
                            synchronized = TRUE, official_request = i$request, python = "unused"), "complete official origin")
  expect_identical(list.files(out, all.files = TRUE, no.. = TRUE), character(0))
})
