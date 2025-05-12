library(comets)
library(dHSIC)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ragg)
library(xgboost)

# ---- load one batch ----

batch_num <- 0
exper_id <- "five_latents"
z_true <- read.csv(paste0(
    "results/numerical/", exper_id, "/ztrue_batch",
    batch_num, ".csv"
), header = FALSE)
z0_est <- read.csv(paste0(
    "results/numerical/", exper_id, "/z0est_batch",
    batch_num, ".csv"
), header = FALSE)
# z0_est0 <- 0.5 * z_true[, 1]
# z0_est1 <- 0.5 * z_true[, 1]
# z0_est <- cbind(z0_est0, z0_est1)
z_all <- cbind(z_true, z0_est)
# z0_est0 is the recovered z0 from view 0
# z0_est1 is the recovered z0 from view 1
colnames(z_all) <- c("z0", "z1", "z2", "x", "y", "z0_est0", "z0_est1")
z_all <- data.frame(z_all)

# ---- try ATE on one batch ----

lm(y ~ z0 + x, data = z_all)
lm(y ~ z0_est0 + x, data = z_all)
lm(y ~ z0_est1 + x, data = z_all)

# ---- all ATE ----

batch_nums <- 0:49
dat_ls <- lapply(batch_nums, \(batch_num) {
    z_true <- read.csv(paste0(
        "results/numerical/", exper_id, "/ztrue_batch",
        batch_num, ".csv"
    ), header = FALSE)
    z0_est <- read.csv(paste0(
        "results/numerical/", exper_id, "/z0est_batch",
        batch_num, ".csv"
    ), header = FALSE)
    z_all <- cbind(z_true, z0_est)
    # z0_est0 is the recovered z0 from view 0
    # z0_est1 is the recovered z0 from view 1
    colnames(z_all) <- c("z0", "z1", "z2", "x", "y", "z0_est0", "z0_est1")
    z_all <- data.frame(z_all)
    z_all$batch_num <- batch_num
    z_all
})
dat <- do.call(rbind, dat_ls)
dat$batch_num <- as.integer(dat$batch_num)

beta_est <- lapply(batch_nums, \(batch_num) {
    beta_oracle <- coef(lm(y ~ z0 + x, data = dat[dat$batch_num == batch_num, ]))[3]
    beta_view0 <- coef(lm(y ~ z0_est0 + x, data = dat[dat$batch_num == batch_num, ]))[3]
    beta_view1 <- coef(lm(y ~ z0_est1 + x, data = dat[dat$batch_num == batch_num, ]))[3]
    data.frame(
        batch_num = batch_num,
        beta_oracle = beta_oracle,
        beta_view0 = beta_view0,
        beta_view1 = beta_view1
    )
})
beta_est_df <- do.call(rbind, beta_est)
head(beta_est_df)

# ---- plot ATE ----

beta_est_long <- beta_est_df %>%
    tidyr::pivot_longer(
        cols = c("beta_oracle", "beta_view0", "beta_view1"),
        names_to = "method",
        values_to = "beta_est"
    ) %>%
    mutate(
        method = factor(method,
            levels = c("beta_oracle", "beta_view0", "beta_view1"),
            labels = c("Oracle", "View 0", "View 1")
        )
    )

agg_jpeg(
    paste0(
        "results/numerical/", exper_id, "/beta_est.jpg"
    ),
    width = 10, height = 5,
    units = "in", res = 300
)

beta_est_long %>%
    ggplot2::ggplot(aes(
        x = method,
        y = beta_est
    )) +
    geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
    ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    theme_bw()

dev.off()
