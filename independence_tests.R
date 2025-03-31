library(comets)
library(dHSIC)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ragg)

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

# conditional indepedneces with comets
gcm_z0_est0 <- gcm(
  X = z_all$z0_est0, Y = z_all$z1, Z = z_all$z0,
  reg_YonZ = "lrm", reg_XonZ = "lrm"
)
gcm_z0_est1 <- gcm(
  X = z_all$z0_est1, Y = z_all$z2, Z = z_all$z0,
  reg_YonZ = "lrm", reg_XonZ = "lrm"
)
pcm_z0_est0 <- pcm(
  X = z_all$z0_est0, Y = z_all$z1, Z = z_all$z0,
  reg_YonZ = "lrm", reg_XonZ = "lrm"
)
pcm_z0_est1 <- pcm(
  X = z_all$z0_est1, Y = z_all$z2, Z = z_all$z0,
  reg_YonZ = "lrm", reg_XonZ = "lrm"
)

# # marginal indepedences with dHSIC
# hsic_z0_est0 <- dhsic(
#   X = z_all$z0_est0, Y = z_all$z2,
# )
