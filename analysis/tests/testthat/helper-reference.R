# Independent CR2/Satterthwaite references for the inference tests (they do not call clubSandwich).
#  * closed form, intercept-only: with n items, paper sizes n_g and paper sums S_g,
#      SE^2 = sum_g (S_g - n_g * mean)^2 / (n^2 * (1 - n_g / n))
#  * matrix form under the identity working model: Q = M C M with M = I - 11'/n and
#      C[i,j] = sqrt(w_i w_j) / n^2 for items in the same paper (w = 1 / (1 - n_g / n)),
#      SE^2 = d'Qd, df = (tr Q)^2 / tr(Q^2), p = 2 pt(-|t|, df).

ref_closed_se <- function(d, paper) {
  n <- length(d); m <- mean(d)
  S <- tapply(d, paper, sum); ng <- tapply(d, paper, length)
  sqrt(sum((S - ng * m)^2 / (n^2 * (1 - ng / n))))
}

ref_matrix <- function(d, paper) {
  n <- length(d)
  w <- 1 / (1 - as.numeric(table(paper)[paper]) / n)
  C <- outer(seq_len(n), seq_len(n), function(i, j) ifelse(paper[i] == paper[j], sqrt(w[i] * w[j]), 0)) / n^2
  M <- diag(n) - matrix(1 / n, n, n)
  Q <- M %*% C %*% M
  se <- sqrt(drop(crossprod(d, Q %*% d)))
  df <- sum(diag(Q))^2 / sum(diag(Q %*% Q))
  t <- mean(d) / se
  c(estimate = mean(d), se = se, df = df, t = t, p = 2 * stats::pt(-abs(t), df))
}

fields <- function(r) c(estimate = r$estimate, se = r$se, df = r$df, t = r$t, p = r$p)
