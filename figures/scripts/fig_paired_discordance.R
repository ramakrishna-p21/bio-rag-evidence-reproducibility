# Paired outcome discordance: JATS (rows) x PDF_DOCLING (columns), repeat 1,
# complete pairs only. Cell fills encode agreement category, not counts.
if (!exists("read_source")) source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))
fig <- "paired_discordance"

dc <- read_source("secondary_discordance.csv")
cp <- read_source("coprimary_results.csv")
N <- 305L

tint <- function(hex, a) {   # blend with white; fixed per category, never count-scaled
  r <- col2rgb(hex)[, 1] / 255
  rgb(t(1 - a * (1 - r)))
}
fill_cat <- c(both_success = col[["neutral"]], both_failure = "#FFFFFF",
              jats_only_success = tint(col[["JATS"]], 0.50),
              pdf_only_success  = tint(col[["PDF_DOCLING"]], 0.22))
cell_name <- c(both_success = "Both successful", both_failure = "Both unsuccessful",
               jats_only_success = "JATS only\nsuccessful", pdf_only_success = "PDF_DOCLING only\nsuccessful")

make_panel <- function(outcome, pos, neg, tag, title) {
  d <- dc[dc$outcome == outcome, ]
  check(fig, paste(outcome, ": all state pairs account for 305 items"), sum(d$n) == N)
  comp <- d[d$pair_class != "not_complete", ]
  n_complete <- sum(comp$n)
  n_excl <- sum(d$n[d$pair_class == "not_complete"])
  cpo <- cp[cp$outcome == outcome, ]
  check(fig, paste(outcome, ": complete pairs equal co-primary n_complete"), n_complete == cpo$n_complete)
  check(fig, paste(outcome, ": excluded pairs equal n_eligible - n_complete"), n_excl == cpo$n_eligible - cpo$n_complete)
  # margins reconcile with the co-primary success counts / rates
  jats_succ <- sum(comp$n[comp$jats_state == "yes"]); pdf_succ <- sum(comp$n[comp$pdf_state == "yes"])
  check(fig, paste(outcome, ": JATS success margin reproduces published rate_jats"),
        close_to(jats_succ / n_complete, cpo$rate_jats, 1e-9))
  check(fig, paste(outcome, ": PDF_DOCLING success margin reproduces published rate_pdf"),
        close_to(pdf_succ / n_complete, cpo$rate_pdf, 1e-9))
  check(fig, paste(outcome, ": (JATS only - PDF only)/n reproduces published rd"),
        close_to((sum(comp$n[comp$pair_class == "jats_only_success"]) -
                  sum(comp$n[comp$pair_class == "pdf_only_success"])) / n_complete, cpo$rd, 1e-9))
  disc <- sum(comp$n[comp$pair_class %in% c("jats_only_success", "pdf_only_success")])

  comp$col <- ifelse(comp$pdf_state == "yes", 1, 2)   # PDF_DOCLING success on the left
  comp$row <- ifelse(comp$jats_state == "yes", 2, 1)  # JATS success on top (y = 2)
  comp$cat <- factor(comp$pair_class, levels = names(fill_cat))
  comp$pct <- 100 * comp$n / n_complete
  comp$name <- cell_name[comp$pair_class]
  comp$num <- format(comp$n)
  comp$pct_txt <- sprintf("%s%%", formatC(comp$pct, format = "f", digits = 1))

  rows_n <- c(jats_succ, n_complete - jats_succ); cols_n <- c(pdf_succ, n_complete - pdf_succ)
  xl <- sprintf("%s\n(n = %d)", c(pos, neg), cols_n)
  yl <- sprintf("%s\n(n = %d)", c(neg, pos), rev(rows_n))

  foot <- sprintf("Complete pairs: %d of %d   |   Discordant: %d (%s%%)\nUnresolved pairs excluded: %d",
                  n_complete, N, disc, formatC(100 * disc / n_complete, format = "f", digits = 1), n_excl)

  p <- ggplot(comp, aes(x = col, y = row)) +
    geom_tile(aes(fill = cat), width = 0.96, height = 0.96, colour = col[["text"]], linewidth = 0.4) +
    geom_text(aes(y = row + 0.27, label = name), vjust = 0.5, size = label_size(7), lineheight = 0.9, colour = col[["text"]]) +
    geom_text(aes(y = row - 0.02, label = num), vjust = 0.5, size = label_size(14), fontface = "bold", colour = col[["text"]]) +
    geom_text(aes(y = row - 0.27, label = pct_txt), vjust = 0.5, size = label_size(8), colour = col[["text"]]) +
    scale_fill_manual(values = fill_cat, guide = "none") +
    scale_x_continuous(breaks = 1:2, labels = xl, position = "top", expand = c(0, 0), limits = c(0.5, 2.5)) +
    scale_y_continuous(breaks = 1:2, labels = yl, expand = c(0, 0), limits = c(0.5, 2.5)) +
    labs(x = NULL, y = NULL, title = title, tag = tag, caption = foot) +
    theme_pub() +
    theme(panel.grid.major = element_blank(), axis.line = element_blank(),
          axis.ticks.x = element_blank(), axis.ticks.y = element_blank(),
          axis.text.x = element_text(lineheight = 0.95, face = "plain"),
          axis.text.y = element_text(lineheight = 0.95),
          plot.caption = element_text(hjust = 0, size = 7, lineheight = 1.1, margin = margin(t = 6)),
          plot.caption.position = "plot",
          plot.title = element_text(margin = margin(b = 14, l = 14)))
  list(plot = p, data = comp, n_complete = n_complete)
}

A <- make_panel("evidence_delivered", "Delivered", "Not delivered", "A", "Evidence-delivery agreement")
B <- make_panel("answer_correct", "Correct", "Incorrect", "B", "Answer-correctness agreement")

# axis titles: PDF_DOCLING outcome across the top, JATS outcome down the side
style_axes <- function(p) p +
  labs(x = "PDF_DOCLING outcome", y = "JATS outcome") +
  theme(axis.title.x = element_text(face = "bold", colour = col[["PDF_DOCLING"]], margin = margin(b = 8)),
        axis.title.y = element_text(face = "bold", colour = col[["JATS"]], angle = 90, margin = margin(r = 3)),
        axis.line.x = element_blank())
pa <- style_axes(A$plot); pb <- style_axes(B$plot)

p <- (pa | pb)

lv <- layer_values(A$plot, 2)
check(fig, "panel A plotted cell counts equal published counts",
      close_to(sort(as.numeric(trimws(A$data$num))), sort(A$data$n)))

fig_size <- c(width = 6.5, height = 3.3)
