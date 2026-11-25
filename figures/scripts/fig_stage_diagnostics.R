# Stage-specific diagnostic outcomes (descriptive, repeat 1, 305 eligible item-conditions
# per representation). Panel A: evidence-delivery status. Panel B: earliest observed failure.
if (!exists("read_source")) source(file.path(Sys.getenv("FIG_ROOT", getwd()), "scripts", "common.R"))
fig <- "stage_diagnostics"

sa <- read_source("secondary_stage_a.csv")
sb <- read_source("secondary_stage_b_failure.csv")
sc <- read_source("state_counts.csv")
conds <- c("JATS", "PDF_DOCLING")
N <- 305L

sb <- sb[sb$matrix_arm == "confirmatory", ]
check(fig, "stage_b_failure rows are the confirmatory arm only, denominator 305",
      all(sb$denominator == N) && nrow(sb) == 14)

status <- sb[sb$measure == "stage_b_status", ]
early  <- sb[sb$measure == "earliest_failure", ]

check(fig, "status categories sum to 305 in each representation",
      all(tapply(status$n, status$condition, sum) == N))
check(fig, "earliest-failure categories sum to 305 in each representation",
      all(tapply(early$n, early$condition, sum) == N))
check(fig, "Stage A labels sum to 305 in each representation",
      all(tapply(sa$n, sa$condition, sum) == N))
check(fig, "Stage A has only preserved / altered_but_sufficient labels",
      setequal(sa$stage_a_label, c("preserved", "altered_but_sufficient")))
check(fig, "no earliest-failure row exists for Stage A (count 0, derived from sum = 305)",
      !"stage_a" %in% early$value)

# Cross-source: Stage B status equals the co-primary evidence_delivered state counts.
ev <- sc[sc$outcome == "evidence_delivered", ]
m <- c(yes = "delivered", no = "not_delivered", non_evaluable = "unresolved")
for (cn in conds) for (k in names(m))
  check(fig, sprintf("%s %s: stage_b_status equals state_counts (%s)", cn, m[[k]], k),
        status$n[status$condition == cn & status$value == m[[k]]] ==
          ev$n[ev$condition == cn & ev$state == k])

# ---- plotting data ---------------------------------------------------------
cat_levels_a <- c("delivered", "not_delivered", "unresolved")
lab_a <- c(delivered = "Sufficient evidence\ndelivered",
           not_delivered = "Sufficient evidence\nnot delivered",
           unresolved = "Unresolved")
da <- status
da$cat <- factor(da$value, levels = rev(cat_levels_a), labels = rev(lab_a[cat_levels_a]))

cat_levels_b <- c("none", "stage_a", "stage_b", "stage_c", "unresolved")
lab_b <- c(none = "No observed failure", stage_a = "Stage A failure",
           stage_b = "Stage B failure", stage_c = "Stage C failure",
           unresolved = "Unresolved")
db <- early[, c("condition", "value", "n")]
db <- rbind(db, data.frame(condition = conds, value = "stage_a", n = 0L))  # derived zero
db$cat <- factor(db$value, levels = rev(cat_levels_b), labels = rev(lab_b[cat_levels_b]))

prep <- function(d) {
  d$condition <- factor(d$condition, levels = rev(conds))
  d$pct <- 100 * d$n / N
  d$lab <- sprintf("%d (%s%%)", d$n, formatC(d$pct, format = "f", digits = 1))
  d$is_unres <- grepl("^Unresolved", as.character(d$cat))
  d
}
da <- prep(da); db <- prep(db)

bar_panel <- function(d, title, tag) {
  ggplot(d, aes(y = cat, x = pct, fill = condition, group = condition)) +
    geom_col(aes(colour = condition), position = position_dodge(width = 0.78),
             width = 0.7, linewidth = 0.6) +
    # unresolved outcomes: gold fill, representation kept as outline colour
    geom_col(data = d[d$is_unres, ], aes(colour = condition),
             fill = col[["unresolved"]], position = position_dodge(width = 0.78),
             width = 0.7, linewidth = 0.7) +
    geom_text(aes(label = lab), position = position_dodge(width = 0.78),
              hjust = -0.08, size = label_size(7), colour = col[["text"]]) +
    scale_fill_manual(values = c(JATS = col[["JATS"]], PDF_DOCLING = "#8CC4C0"), breaks = conds, name = NULL) +
    scale_colour_manual(values = col[conds], guide = "none") +
    scale_x_continuous(limits = c(0, 100), breaks = seq(0, 100, 25),
                       labels = function(x) paste0(x, "%"), expand = c(0, 0)) +
    labs(x = NULL, y = NULL, title = title, tag = tag) +
    coord_cartesian(clip = "off") +
    theme_pub() +
    theme(panel.grid.major.y = element_blank(),
          axis.text.y = element_text(hjust = 1, lineheight = 0.95),
          plot.title = element_text(margin = margin(b = 3, l = 14)),
          plot.margin = margin(4, 34, 4, 4))
}

pA <- bar_panel(da, "Evidence-delivery status", "A") + theme(legend.position = "none")
pB <- bar_panel(db, "Earliest observed failure", "B") +
  labs(x = "Item-conditions per representation (% of 305 eligible, repeat 1)") +
  theme(legend.position = "bottom", legend.margin = margin(t = 2))

p <- pA / pB + plot_layout(heights = c(3, 5))

lA <- layer_values(pA, 1)
check(fig, "panel A plotted bar lengths equal 100 x n / 305",
      close_to(sort(lA$xmax), sort(100 * da$n / N)))
lB <- layer_values(pB, 1)
check(fig, "panel B plotted bar lengths equal 100 x n / 305 (includes the Stage A zero)",
      close_to(sort(lB$xmax), sort(100 * db$n / N)) && sum(lB$xmax == 0) == 2)

fig_size <- c(width = 6.5, height = 4.6)
