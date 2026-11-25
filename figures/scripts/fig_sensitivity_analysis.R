# Sensitivity of the co-primary effects (JATS minus PDF_DOCLING, percentage points):
# registered item-weighted RD, equal-paper-weighted RD, and unresolved-outcome
# best/worst-case bounds. The bounds are not confidence intervals.
if (!exists("read_source")) source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))
fig <- "sensitivity_analysis"

cp  <- read_source("coprimary_results.csv")
rob <- read_source("secondary_robustness.csv")

# ---- validation ------------------------------------------------------------
ref <- rob[rob$matrix_arm == "confirmatory", ]
for (o in cp$outcome) {
  r <- ref[ref$outcome == o, ]; k <- cp[cp$outcome == o, ]
  check(fig, paste(o, ": secondary reference row equals co-primary rd and equal-paper rd"),
        close_to(c(r$rd, r$equal_paper_rd), c(k$rd, k$equal_paper_rd)))
}
check(fig, "bounds bracket the registered estimate and the equal-paper estimate",
      all(cp$bound_lower <= cp$rd & cp$rd <= cp$bound_upper &
          cp$bound_lower <= cp$equal_paper_rd & cp$equal_paper_rd <= cp$bound_upper))
check(fig, "bound_n_operational_excluded is 0 for both outcomes", all(cp$bound_n_operational_excluded == 0))
check(fig, "all plotted values are on the JATS minus PDF_DOCLING scale (negative = PDF_DOCLING higher)",
      all(cp$rd < 0, cp$equal_paper_rd < 0))
check(fig, "unresolved-outcome upper bound is below zero for both outcomes",
      all(cp$bound_upper < 0))

# ---- plotting data ---------------------------------------------------------
oc <- c(evidence_delivered = "Sufficient evidence delivery", answer_correct = "Scientific answer correctness")
rows <- do.call(rbind, lapply(seq_len(nrow(cp)), function(i) {
  k <- cp[i, ]; base <- if (k$outcome == "evidence_delivered") 6 else 2
  data.frame(outcome = k$outcome, base = base,
             y = base + c(2, 1, 0), 
             kind = c("item", "paper", "bound"),
             est = 100 * c(k$rd, k$equal_paper_rd, NA),
             lo = 100 * c(NA, NA, k$bound_lower), hi = 100 * c(NA, NA, k$bound_upper),
             stringsAsFactors = FALSE)
}))
rows$label <- c(item = "Registered item-weighted", paper = "Equal-paper-weighted",
                bound = "Unresolved-outcome bounds\n(best/worst case; not a CI)")[rows$kind]
rows$txt <- ifelse(rows$kind == "bound",
                   sprintf("%s to %s", fmt_num(rows$lo), fmt_num(rows$hi)), fmt_num(rows$est))
pts <- rows[rows$kind != "bound", ]; bnd <- rows[rows$kind == "bound", ]
hdr <- data.frame(y = c(9.0, 5.0), lab = unname(oc[c("evidence_delivered", "answer_correct")]))

xlim <- c(-20, 5)
ylim <- c(0.1, 9.5)
x_txt <- 6.5   # value column to the right of the axis, in data units
cap <- 0.22

p <- ggplot() +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = col[["text"]]) +
  # unresolved-outcome bounds: gold bracketed bar (not an interval estimate)
  geom_segment(data = bnd, aes(x = lo, xend = hi, y = y, yend = y), linewidth = 2.2,
               colour = col[["unresolved"]], lineend = "butt") +
  geom_segment(data = bnd, aes(x = lo, xend = lo, y = y - cap, yend = y + cap), linewidth = 0.7, colour = col[["text"]]) +
  geom_segment(data = bnd, aes(x = hi, xend = hi, y = y - cap, yend = y + cap), linewidth = 0.7, colour = col[["text"]]) +
  geom_point(data = pts[pts$kind == "item", ], aes(x = est, y = y), shape = 15, size = 3, colour = col[["PDF_DOCLING"]]) +
  geom_point(data = pts[pts$kind == "paper", ], aes(x = est, y = y), shape = 23, size = 2.8,
             fill = "white", colour = col[["PDF_DOCLING"]], stroke = 0.8) +
  geom_text(data = rows, aes(x = x_txt, y = y, label = txt), hjust = 0, size = label_size(8), colour = col[["text"]]) +
  geom_text(data = hdr, aes(x = xlim[1], y = y, label = lab), hjust = 0, size = label_size(8.5),
            fontface = "bold", colour = col[["text"]]) +
  annotate("text", x = x_txt, y = 9.0, label = "Estimate or\nbounds, pp", hjust = 0,
           size = label_size(7.5), fontface = "bold", colour = col[["text"]]) +
  direction_labels(y = 0.6, gap = 0.4) +
  scale_x_continuous(breaks = seq(-20, 5, 5), labels = function(x) fmt_num(x, 0)) +
  scale_y_continuous(breaks = rows$y, labels = rows$label) +
  labs(x = paste0("Risk difference, JATS ", minus, " PDF_DOCLING (percentage points)"), y = NULL) +
  coord_cartesian(xlim = xlim, ylim = ylim, expand = FALSE, clip = "off") +
  theme_pub() +
  theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(hjust = 1),
        plot.margin = margin(4, 85, 4, 4))

lv <- layer_values(p, 5)
check(fig, "plotted item-weighted estimates equal published rd_pp",
      close_to(sort(lv$x), sort(cp$rd_pp)))
lv <- layer_values(p, 6)
check(fig, "plotted equal-paper estimates equal 100 x published equal_paper_rd",
      close_to(sort(lv$x), sort(100 * cp$equal_paper_rd)))
lv <- layer_values(p, 2)
check(fig, "plotted bounds equal 100 x published bound_lower / bound_upper",
      close_to(sort(c(lv$x, lv$xend)), sort(100 * c(cp$bound_lower, cp$bound_upper))))

fig_size <- c(width = 6.5, height = 3.0)
