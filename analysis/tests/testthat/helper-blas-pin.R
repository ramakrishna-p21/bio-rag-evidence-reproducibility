# The goldens here are byte-compared, and OpenBLAS picks its kernel from the CPU, so the suite has
# to run under the kernel the goldens were produced with. The variables are read when R starts;
# the child R processes started by the command-line tests inherit them.
local({
  pin <- c(OPENBLAS_CORETYPE = "HASWELL", OPENBLAS_NUM_THREADS = "1")
  if (!identical(unname(Sys.getenv(names(pin), unset = NA_character_)), unname(pin))) {
    stop("start R with OPENBLAS_CORETYPE=HASWELL OPENBLAS_NUM_THREADS=1 to run these tests (the BLAS kernel is chosen when R starts)",
         call. = FALSE)
  }
})
