# Co-primary representation effects: item-weighted paired risk difference
# (JATS minus PDF_DOCLING) with the registered whole-paper percentile bootstrap CI.
if (!exists("read_source")) source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))
fig <- "coprimary_effects"

cp <- read_source("coprimary_results.csv")
sc <- read_source("state_counts.csv")

# ---- validation against the published table --------------------------------
check(fig, "two co-primary outcomes present",
      setequal(cp$outcome, c("evidence_delivered", "answer_correct")))
check(fig, "bootstrap, test and Holm statuses are all 'ok'",
      all(cp$boot_status == "ok", cp$test_status == "ok", cp$holm_status == "ok"))
check(fig, "bootstrap used 10000 resamples with none empty",
      all(cp$boot_n_resamples == 10000, cp$boot_n_empty == 0))
check(fig, "rd_pp equals 100 x rd", close_to(cp$rd_pp, 100 * cp$rd))
check(fig, "rd equals rate_jats - rate_pdf (JATS minus PDF_DOCLING)",
      close_to(cp$rd, cp$rate_jats - cp$rate_pdf))
check(fig, "all effects negative (PDF_DOCLING higher)", all(cp$rd < 0))
check(fig, "published CI brackets the point estimate",
      all(cp$boot_ci_lower < cp$rd & cp$rd < cp$boot_ci_upper))
check(fig, "Holm p is the Holm adjustment of the two published raw p-values",
      {
        o <- order(cp$test_p); adj <- pmin(1, cummax((2:1)[seq_along(o)] * cp$test_p[o]))
        close_to(cp$holm_p[o], adj, 1e-12)
      })
# n_complete must equal the sum of yes/no states (non_evaluable excluded per condition,
# so pair-level completeness is cross-checked in the discordance figure).
for (o in cp$outcome) {
  s <- sc[sc$outcome == o, ]
  check(fig, paste(o, ": each condition's states sum to n_eligible"),
        all(tapply(s$n, s$condition, sum) == cp$n_eligible[cp$outcome == o]))
}

# ---- plotting data ---------------------------------------------------------
cp$label <- c(evidence_delivered = "Sufficient evidence\ndelivery",
              answer_correct     = "Scientific answer\ncorrectness")[cp$outcome]
cp$y <- c(evidence_delivered = 2, answer_correct = 1)[cp$outcome]
cp$est <- cp$rd_pp
cp$lo  <- 100 * cp$boot_ci_lower
cp$hi  <- 100 * cp$boot_ci_upper
cp$est_txt <- sprintf("%s (%s, %s)", fmt_num(cp$est), fmt_num(cp$lo), fmt_num(cp$hi))
cp$n_txt   <- sprintf("%d / %d", cp$n_complete, cp$n_eligible)
cp$p_txt   <- fmt_p(cp$holm_p)

xlim <- c(-20, 10)
ylim <- c(0.4, 3.0)

p_forest <- ggplot(cp, aes(y = y)) +
  geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = col[["text"]]) +
  geom_errorbar(aes(xmin = lo, xmax = hi), orientation = "y", width = 0.12, linewidth = 0.5,
                 colour = col[["PDF_DOCLING"]]) +
  geom_point(aes(x = est), shape = 15, size = 3, colour = col[["PDF_DOCLING"]]) +
  direction_labels(y = 0.62, gap = 0.6) +
  scale_x_continuous(breaks = seq(-20, 10, 5), labels = function(x) fmt_num(x, 0),
                     limits = xlim, expand = c(0, 0)) +
  scale_y_continuous(breaks = cp$y, labels = cp$label, limits = ylim, expand = c(0, 0)) +
  labs(x = paste0("Paired risk difference, JATS ", minus, " PDF_DOCLING (percentage points)"), y = NULL) +
  theme_pub() +
  theme(panel.grid.major.y = element_blank(),
        axis.text.y = element_text(hjust = 0, size = 8.5, lineheight = 0.95))

# Text columns share the y scale and the axis height of the forest panel so rows align.
tab <- data.frame(y = cp$y, n = cp$n_txt, est = cp$est_txt, p = cp$p_txt)
hdr_y <- 2.55
p_table <- ggplot(tab, aes(y = y)) +
  geom_text(aes(x = 1.7, label = n),   hjust = 0.5, size = label_size(8), colour = col[["text"]]) +
  geom_text(aes(x = 6.2, label = est), hjust = 0.5, size = label_size(8), colour = col[["text"]]) +
  geom_text(aes(x = 10.3, label = p),   hjust = 0.5, size = label_size(8), colour = col[["text"]]) +
  col_header(1.7, hdr_y, "Complete pairs\n/ eligible") +
  col_header(6.2, hdr_y, "Risk difference, pp\n(95% bootstrap CI)") +
  col_header(10.3, hdr_y, "Holm-adjusted\np") +
  scale_x_continuous(limits = c(0, 12), expand = c(0, 0), breaks = 0) +
  scale_y_continuous(limits = ylim, expand = c(0, 0), breaks = NULL) +
  labs(x = "x", y = NULL) +
  theme_pub() +
  theme(panel.grid.major = element_blank(), axis.line.x = element_blank(),
        axis.ticks.x = element_blank(), axis.text = element_text(colour = NA),
        axis.title.x = element_text(colour = NA))

p <- p_forest + p_table + plot_layout(widths = c(3.3, 3.2))

lv <- layer_values(p_forest, 3)
check(fig, "plotted point estimates equal published rd_pp",
      close_to(sort(lv$x), sort(cp$rd_pp)))
lw <- layer_values(p_forest, 2)
check(fig, "plotted CI limits equal 100 x published boot_ci_*",
      close_to(sort(c(lw$xmin, lw$xmax)), sort(100 * c(cp$boot_ci_lower, cp$boot_ci_upper))))

fig_size <- c(width = 6.5, height = 2.2)
