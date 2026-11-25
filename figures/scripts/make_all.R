# Generates every statistical figure, the validation logs and the previews.
# Run from the figures/ directory via ./make_figures.sh
source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))

figure_scripts <- c("fig_coprimary_effects.R", "fig_stage_diagnostics.R",
                    "fig_paired_discordance.R", "fig_answer_class_outcomes.R",
                    "fig_sensitivity_analysis.R")
built <- list()
for (s in figure_scripts) {
  message("== ", s)
  sys.source(file.path(fig_root, "scripts", s), envir = globalenv())
  save_figure(p, fig, fig_size[["width"]], fig_size[["height"]])
  built[[fig]] <- fig
}

# ---- contact sheets (colour and grayscale) ----------------------------------
read_preview <- function(name, gray = FALSE) {
  im <- png::readPNG(file.path(out_root, "preview", paste0(name, "_preview.png")))
  if (gray) {
    lum <- 0.2126 * im[, , 1] + 0.7152 * im[, , 2] + 0.0722 * im[, , 3]
    im <- array(rep(lum, 3), dim = c(dim(lum), 3))
  }
  patchwork::wrap_elements(full = grid::rasterGrob(im, interpolate = TRUE))
}
sheet <- function(out, gray) {
  items <- lapply(unlist(built), read_preview, gray = gray)
  s <- patchwork::wrap_plots(items, ncol = 2) &
    theme(plot.background = element_rect(fill = "white", colour = NA))
  ggsave(file.path(out_root, "preview", out), s, device = ragg::agg_png,
         width = 13.4, height = 14.5, dpi = 150, bg = "white")
}
sheet("contact_sheet.png", gray = FALSE)
sheet("contact_sheet_grayscale.png", gray = TRUE)

# ---- logs ---------------------------------------------------------------------
vd <- dir_out("validation")
write.csv(source_log, file.path(vd, "source_hash_check.csv"), row.names = FALSE)
write.csv(checks,     file.path(vd, "numeric_checks.csv"),    row.names = FALSE)
si <- as.data.frame(installed.packages(lib.loc = Sys.getenv("FIG_LIB", .libPaths()[1]))[, c("Package", "Version")])
write.csv(si, file.path(dir_out("environment"), "package_inventory.csv"), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(dir_out("environment"), "session_info.txt"))
message(sprintf("Done: %d figures, %d numeric checks passed, %d source hashes verified.",
                length(built), sum(checks$pass), sum(source_log$match)))
