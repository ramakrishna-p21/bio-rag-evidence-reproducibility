# Synthetic official-origin, confirmatory-only inputs and a stubbed Python proof, shared by the
# official-path tests. Only the Python subprocess is stubbed; every R gate after it is production code.

# Replace the Python subprocess by a proof built from the synthetic inputs; restore it afterwards.
stub_python_proof <- function(proof, env = parent.frame()) {
  home <- environment(read_handoff_file)
  original <- get("read_official_proof", envir = home)
  assign("read_official_proof", function(request, python, project_dir) { check_official_proof(proof, TRUE); proof }, envir = home)
  withr::defer(assign("read_official_proof", original, envir = home), envir = env)
}

official_scope_inputs <- function() {
  as_final_rows <- edit_lines(function(lines) {
    header <- strsplit(lines[1], ",")[[1]]
    c(lines[1], vapply(lines[-1], function(l) {
      cells <- strsplit(l, ",", fixed = TRUE)[[1]]
      cells[match(c("phase", "split", "analysis_role"), header)] <- c("final_test", "final_test", "confirmatory")
      paste(cells, collapse = ",")
    }, ""))
  })
  registry_path <- test_path("..", "..", "..", "configs", "analysis_seed.json")
  registry <- jsonlite::fromJSON(registry_path, simplifyVector = FALSE)
  registry_sha <- sha256_file(registry_path)
  as_official <- function(text) {
    h <- jsonlite::fromJSON(text, simplifyVector = FALSE)
    h$input_origin <- "verified_official_run"; h$phase <- "final_test"; h$scope <- "confirmatory_only"
    h$custody_exception <- list(status = "custody_gate_not_established_exception_v1",
                                authorization = list(artifact_id = paste0("evaluation_authorization_", strrep("d", 64)),
                                                     logical_ref = "approvals/final_test_evaluation.json", sha256 = strrep("d", 64)))
    h$awaiting_approval <- list()
    h$source_runs <- list(list(matrix_arm = "confirmatory", observations = 1830L))
    h$scoring_reference$primary_bootstrap_registry_sha256 <- registry_sha
    as.character(jsonlite::toJSON(h, auto_unbox = TRUE, null = "null", pretty = TRUE, digits = NA))
  }
  inputs <- make_inputs(csv_edit = as_final_rows, handoff_edit = as_official)
  request <- write_text("{\"authorization\": \"x\"}\n", tempfile(fileext = ".json"))
  inputs$request <- request
  inputs$proof <- list(
    status = "verified_official_final_test_handoff", source_verification = "fresh_python_verification",
    custody = "custody_gate_not_established_exception_v1", observations = 1830L, scope = "confirmatory_only",
    handoff = list(sha256 = inputs$handoff_sha), csv_sha256 = sha256_file(inputs$csv),
    scientific_bootstrap_seed = list(registry_sha256 = registry_sha, record = registry),
    request_sha256 = sha256_file(request))
  inputs$seed <- as.integer(registry$value)
  inputs
}
