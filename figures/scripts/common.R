# Shared theme, source loading, validation and export helpers.
# Reads published result CSVs only; no statistical inference is recomputed.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
})

# ---- locations -------------------------------------------------------------
# Inputs are the primary and secondary result directories of the official R analysis
# (the directories that contain run_record.json and secondary_record.json). They are
# not part of the repository; there is no default.
require_dir <- function(var) {
  path <- Sys.getenv(var, "")
  if (!nzchar(path) || !dir.exists(path))
    stop("Set ", var, " to an existing result directory (see figures/README.md).", call. = FALSE)
  normalizePath(path)
}
fig_root      <- normalizePath(Sys.getenv("FIG_ROOT", getwd()), mustWork = TRUE)
out_root      <- normalizePath(Sys.getenv("FIG_OUT", file.path(fig_root, "output")), mustWork = FALSE)
primary_dir   <- require_dir("FIG_PRIMARY")
secondary_dir <- require_dir("FIG_SECONDARY")

# ---- design tokens ---------------------------------------------------------
col <- c(
  JATS        = "#4D6478",
  PDF_DOCLING = "#007D76",
  unresolved  = "#C4A052",
  neutral     = "#E9EDF0",
  text        = "#20262C",
  bg          = "#FFFFFF"
)
grid_col <- "#D5DBE0"   # hairline gridlines, slightly darker than the neutral fill
rep_shape <- c(JATS = 16, PDF_DOCLING = 17)
font_family <- "Liberation Sans"   # metric-compatible with Arial

minus <- "−"
fmt_num <- function(x, digits = 1) {
  sub("^-", minus, formatC(x, format = "f", digits = digits))
}
fmt_p <- function(p) ifelse(p < 0.001, sprintf("%.4f", p), sprintf("%.3f", p))

theme_pub <- function(base_size = 8) {
  theme_minimal(base_size = base_size, base_family = font_family) %+replace%
    theme(
      text             = element_text(family = font_family, colour = col[["text"]]),
      plot.background  = element_rect(fill = col[["bg"]], colour = NA),
      panel.background = element_rect(fill = col[["bg"]], colour = NA),
      panel.grid.major = element_line(colour = grid_col, linewidth = 0.25),
      panel.grid.minor = element_blank(),
      axis.line.x      = element_line(colour = col[["text"]], linewidth = 0.3),
      axis.ticks.x     = element_line(colour = col[["text"]], linewidth = 0.3),
      axis.ticks.length = unit(2, "pt"),
      axis.text        = element_text(colour = col[["text"]], size = base_size),
      axis.title       = element_text(colour = col[["text"]], size = base_size),
      plot.title       = element_text(face = "bold", size = base_size + 0.5,
                                      hjust = 0, margin = margin(b = 4)),
      plot.title.position = "plot",
      plot.tag         = element_text(face = "bold", size = base_size + 3),
      plot.tag.position = c(0, 1),
      legend.position  = "top",
      legend.text      = element_text(size = base_size),
      legend.title     = element_blank(),
      legend.key.size  = unit(9, "pt"),
      strip.text       = element_text(face = "bold", hjust = 0),
      plot.margin      = margin(4, 6, 4, 4)
    )
}

update_geom_defaults("text", list(family = font_family, colour = col[["text"]]))

# ---- hash-verified source loading ------------------------------------------
registry <- local({
  py <- jsonlite::fromJSON(file.path(primary_dir, "run_record.json"))
  sr <- jsonlite::fromJSON(file.path(secondary_dir, "secondary_record.json"))
  list(
    primary   = vapply(py$outputs$files, function(f) f$sha256, ""),
    secondary = vapply(sr$scientific_files, function(f) f$sha256, ""),
    run_record_sha256_expected = sr$primary_result$run_record_sha256
  )
})

source_log <- data.frame(file = character(), published_sha256 = character(),
                         observed_sha256 = character(), match = logical(),
                         stringsAsFactors = FALSE)

record_source <- function(file, expected, path) {
  observed <- unname(tools::sha256sum(path))
  source_log[nrow(source_log) + 1L, ] <<- list(file, expected, observed, identical(expected, observed))
  if (!identical(expected, observed))
    stop("Source hash mismatch for ", file, ": expected ", expected, ", got ", observed)
}

# The primary run record is itself hash-linked from the secondary record.
record_source("primary run_record.json",
              registry$run_record_sha256_expected,
              file.path(primary_dir, "run_record.json"))

read_source <- function(name) {
  if (name %in% names(registry$primary)) {
    path <- file.path(primary_dir, name); expected <- registry$primary[[name]]
  } else if (name %in% names(registry$secondary)) {
    path <- file.path(secondary_dir, name); expected <- registry$secondary[[name]]
  } else stop("File not in a published hash registry: ", name)
  record_source(name, expected, path)
  read.csv(path, stringsAsFactors = FALSE, na.strings = "NA")
}

# ---- numeric validation log ------------------------------------------------
checks <- data.frame(figure = character(), check = character(), pass = logical(),
                     stringsAsFactors = FALSE)
check <- function(figure, description, condition) {
  ok <- isTRUE(condition)
  checks[nrow(checks) + 1L, ] <<- list(figure, description, ok)
  if (!ok) stop("Validation failed [", figure, "]: ", description)
  invisible(TRUE)
}
close_to <- function(a, b, tol = 1e-9) all(abs(a - b) < tol)

# ---- text helpers ----------------------------------------------------------
label_size <- function(pt) pt / .pt   # geom_text size is in mm

# ---- export ----------------------------------------------------------------
dir_out <- function(...) {
  d <- file.path(out_root, ...); dir.create(d, recursive = TRUE, showWarnings = FALSE); d
}

save_figure <- function(p, name, width, height) {
  pdf_f <- file.path(dir_out("figures", "pdf"), paste0(name, ".pdf"))
  svg_f <- file.path(dir_out("figures", "svg"), paste0(name, ".svg"))
  png_f <- file.path(dir_out("figures", "png"), paste0(name, ".png"))
  prv_f <- file.path(dir_out("preview"), paste0(name, "_preview.png"))

  ggsave(pdf_f, p, device = cairo_pdf, width = width, height = height, bg = "white")
  ggsave(svg_f, p, device = svglite::svglite, width = width, height = height, bg = "white")
  # Portable font stack for editors on machines without the exact family.
  s <- readLines(svg_f, warn = FALSE)
  s <- gsub(sprintf("font-family: %s;", font_family),
            sprintf("font-family: '%s', Arial, Helvetica, sans-serif;", font_family), s, fixed = TRUE)
  writeLines(s, svg_f)
  ggsave(png_f, p, device = ragg::agg_png, width = width, height = height,
         dpi = 600, bg = "white")
  ggsave(prv_f, p, device = ragg::agg_png, width = width, height = height,
         dpi = 200, bg = "white")
  invisible(c(pdf_f, svg_f, png_f))
}

# Plotted values come from the built ggplot object, not from the input frame.
layer_values <- function(p, layer = 1L) ggplot_build(p)$data[[layer]]

# Text-table column header used by the forest-style figures.
col_header <- function(x, y, label, hjust = 0.5)
  annotate("text", x = x, y = y, label = label, hjust = hjust, vjust = 0,
           size = label_size(7), fontface = "bold", colour = col[["text"]])

direction_labels <- function(y, zero = 0, gap = 0.4) {
  list(
    annotate("text", x = zero - gap, y = y, hjust = 1, vjust = 0.5,
             label = paste0("← Favors PDF_DOCLING"),
             size = label_size(7), colour = col[["text"]]),
    annotate("text", x = zero + gap, y = y, hjust = 0, vjust = 0.5,
             label = paste0("Favors JATS →"),
             size = label_size(7), colour = col[["text"]])
  )
}
