# Outcomes by benchmark answer class: descriptive proportions over ALL eligible
# items in the class (unresolved retained in the denominator). No inference.
if (!exists("read_source")) source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))
fig <- "answer_class_outcomes"

ac <- read_source("secondary_answer_class.csv")
sc <- read_source("state_counts.csv")
conds <- c("JATS", "PDF_DOCLING")

# Registered display names: docs/frozen/BENCHMARK_SPEC.md answer-class table.
classes <- c(quantitative_value = "Quantitative value",
             categorical_conclusion = "Categorical conclusion",
             biological_entity = "Molecular or biological entity",
             bounded_entity_set = "Bounded entity set",
             dataset_cohort = "Dataset or cohort")
check(fig, "exactly the five registered answer classes are present",
      setequal(unique(ac$answer_class), names(classes)))

# ---- denominators and cross-source reconciliation --------------------------
tot <- aggregate(n ~ condition + answer_class + outcome, ac, sum)
den <- unique(ac[, c("condition", "answer_class", "outcome", "denominator")])
mm <- merge(tot, den)
check(fig, "state counts sum to the stated denominator in every class x condition x outcome",
      all(mm$n == mm$denominator))
dd <- unique(ac[, c("condition", "answer_class", "denominator")])
check(fig, "denominator is the same for both representations and both outcomes",
      all(tapply(dd$denominator, dd$answer_class, function(x) length(unique(x))) == 1))
check(fig, "class denominators sum to 305 per representation",
      all(tapply(dd$denominator, dd$condition, sum) == 305))
yes <- aggregate(n ~ condition + outcome, ac[ac$state == "yes", ], sum)
for (i in seq_len(nrow(yes)))
  check(fig, sprintf("%s %s: class 'yes' counts sum to state_counts yes", yes$condition[i], yes$outcome[i]),
        yes$n[i] == sc$n[sc$outcome == yes$outcome[i] & sc$condition == yes$condition[i] & sc$state == "yes"])

# ---- plotting data ---------------------------------------------------------
d <- ac[ac$state == "yes", c("condition", "answer_class", "outcome", "n", "denominator")]
# explicit zero rows are not expected here; every class x condition x outcome has a yes row
check(fig, "every class x condition x outcome has a 'yes' row", nrow(d) == 20)
d$pct <- 100 * d$n / d$denominator
d$class <- factor(d$answer_class, levels = rev(names(classes)),
                  labels = rev(sprintf("%s\n(n = %d)", classes, 
                                       dd$denominator[match(names(classes), dd$answer_class[dd$condition == "JATS"])] )))
den1 <- setNames(dd$denominator[dd$condition == "JATS"], dd$answer_class[dd$condition == "JATS"])
d$class <- factor(d$answer_class, levels = rev(names(classes)),
                  labels = rev(sprintf("%s\n(n = %d)", classes, den1[names(classes)])))
d$condition <- factor(d$condition, levels = conds)
d$lab <- sprintf("%d/%d (%d%%)", d$n, d$denominator, round(d$pct))
d$size <- d$denominator
dodge <- position_dodge(width = 0.55)

panel <- function(o, title, tag, show_y) {
  x <- d[d$outcome == o, ]
  x$ypos <- as.numeric(x$class) + ifelse(x$condition == "JATS", 0.16, -0.16)
  p <- ggplot(x, aes(x = pct, y = ypos, colour = condition, shape = condition, size = size)) +
    geom_hline(yintercept = seq_along(levels(x$class)), colour = grid_col, linewidth = 0.25) +
    geom_point() +
    # n / N (percent) labelled in aligned columns to the right of the 0-100% axis
    geom_text(aes(x = ifelse(condition == "JATS", 114, 138), y = as.numeric(class), label = lab),
              size = label_size(7), hjust = 0.5, show.legend = FALSE) +
    scale_colour_manual(values = col[conds], breaks = conds, name = NULL) +
    scale_shape_manual(values = rep_shape[conds], breaks = conds, name = NULL) +
    scale_size_continuous(range = c(1.9, 3.6), guide = "none") +
    scale_x_continuous(breaks = seq(0, 100, 25), labels = function(v) paste0(v, "%")) +
    scale_y_continuous(breaks = seq_along(levels(x$class)), labels = levels(x$class),
                       limits = c(0.5, length(levels(x$class)) + 0.95), expand = c(0, 0)) +
    labs(x = if (tag == "B") "Percent of eligible items in class (unresolved retained in denominator)" else NULL,
         y = NULL, title = title, tag = tag) +
    coord_cartesian(xlim = c(0, 100), expand = FALSE, clip = "off") +
    theme_pub() +
    theme(panel.grid.major.y = element_blank(), axis.text.y = element_text(hjust = 1, lineheight = 0.95),
          plot.title = element_text(margin = margin(b = 14, l = 14)),
          plot.margin = margin(4, 125, 4, 4))
  # column headers
  p + annotate("text", x = c(114, 138), y = length(levels(x$class)) + 0.62,
               label = c("JATS", "PDF_DOCLING"), size = label_size(7), fontface = "bold",
               colour = col[conds])
}
pA <- panel("evidence_delivered", "Sufficient evidence delivery", "A") + theme(legend.position = "none")
pB <- panel("answer_correct", "Scientific answer correctness", "B") + theme(legend.position = "bottom")
p <- pA / pB

lv <- layer_values(pA, 2)
xA <- d[d$outcome == "evidence_delivered", ]
check(fig, "panel A plotted dot positions equal 100 x yes / denominator",
      close_to(sort(lv$x), sort(100 * xA$n / xA$denominator)))
lv <- layer_values(pB, 2)
xB <- d[d$outcome == "answer_correct", ]
check(fig, "panel B plotted dot positions equal 100 x yes / denominator",
      close_to(sort(lv$x), sort(100 * xB$n / xB$denominator)))

fig_size <- c(width = 6.5, height = 5.4)
