library(comets)
library(dHSIC)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ragg)

# ---- load data ----

batch_num <- 0
z_true <- read.csv(paste0(
  "results/numerical/gumbel_softmax/ztrue_batch",
  batch_num, ".csv"
))
z0_est <- read.csv(paste0(
  "results/numerical/gumbel_softmax/z0est_batch",
  batch_num, ".csv"
))
z_all <- cbind(z_true, z0_est)
colnames(z_all) <- c("z0", "z1", "z2", "z0_est0", "z0_est1")
z_all <- data.frame(z_all)

# ---- marginal plots ----

p1 <- z_all %>%
  ggplot(aes(x = z0, y = z0_est0)) +
  geom_point() +
  theme_bw()

p2 <- z_all %>%
  ggplot(aes(x = z0, y = z0_est1)) +
  geom_point() +
  theme_bw()

p3 <- z_all %>%
  ggplot(aes(x = z1, y = z0_est0)) +
  geom_point() +
  theme_bw()

p4 <- z_all %>%
  ggplot(aes(x = z2, y = z0_est1)) +
  geom_point() +
  theme_bw()

p5 <- z_all %>%
  ggplot(aes(x = z2, y = z0_est0)) +
  geom_point() +
  theme_bw()

p6 <- z_all %>%
  ggplot(aes(x = z1, y = z0_est1)) +
  geom_point() +
  theme_bw()

# save the plots
agg_jpeg(
  paste0(
    "results/numerical/gumbel_softmax/scatter_batch",
    batch_num, ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)

ggpubr::ggarrange(p1, p2, p3, p4, p5, p6,
  ncol = 2, nrow = 3,
  common.legend = TRUE
)

dev.off()

p0 <- z_all %>%
  ggplot(aes(x = z1, y = z2)) +
  geom_point() +
  theme_bw()

agg_jpeg(
  paste0(
    "results/numerical/gumbel_softmax/scatter_z1z2_batch",
    batch_num, ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)

ggpubr::ggarrange(p0)

dev.off()

# ---- conditional indepedneces with comets ----

reg_mod <- "lrm" # "lrm" for linear models

gcm_z0_est0 <- gcm(
  X = z_all$z0_est0, Y = z_all$z1, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)
gcm_z0_est1 <- gcm(
  X = z_all$z0_est1, Y = z_all$z2, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z0_est0 <- pcm(
  X = z_all$z0_est0, Y = z_all$z1, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)
pcm_z0_est1 <- pcm(
  X = z_all$z0_est1, Y = z_all$z2, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

## ---- save results in a table ----

results <- data.frame(
  test = c("GCM z0_est0", "GCM z0_est1", "PCM z0_est0", "PCM z0_est1"),
  p_value = c(
    gcm_z0_est0$p.value, gcm_z0_est1$p.value,
    pcm_z0_est0$p.value, pcm_z0_est1$p.value
  ),
  statistic = c(
    gcm_z0_est0$statistic, gcm_z0_est1$statistic,
    pcm_z0_est0$statistic, pcm_z0_est1$statistic
  ),
  cond_cor = c(
    stats::cor(gcm_z0_est0$rX, gcm_z0_est0$rY),
    stats::cor(gcm_z0_est1$rX, gcm_z0_est1$rY),
    NA, NA
  )
)

write.csv(
  results,
  paste0(
    "results/numerical/gumbel_softmax/independence_tests_",
    reg_mod, "_batch", batch_num, ".csv"
  ),
  row.names = FALSE
)

# xtable::xtable(results,
#   digits = c(0, 3, 3, 3),
# )

## ---- plot the gcm residuals ----

p_res0 <- plot(gcm_z0_est0)
p_res1 <- plot(gcm_z0_est1)

agg_jpeg(
  paste0(
    "results/numerical/gumbel_softmax/scatter_gcm_",
    reg_mod,
    "_z0res_batch",
    batch_num, ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)

ggpubr::ggarrange(p_res0, p_res1,
  ncol = 2, nrow = 1,
  common.legend = TRUE
)

dev.off()

## ---- plot the pcm residuals ----

p_res0 <- plot(pcm_z0_est0)
p_res1 <- plot(pcm_z0_est1)

agg_jpeg(
  paste0(
    "results/numerical/gumbel_softmax/scatter_pcm_",
    reg_mod,
    "_z0res_batch",
    batch_num, ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)

ggpubr::ggarrange(p_res0, p_res1,
  ncol = 2, nrow = 1,
  common.legend = TRUE
)

dev.off()
