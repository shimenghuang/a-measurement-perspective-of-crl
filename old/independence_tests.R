library(comets)
library(dHSIC)
library(ggplot2)
library(dplyr)
library(ggpubr)
library(ragg)
library(xgboost)

# ---- load data ----

batch_num <- 5
z_true <- read.csv(paste0(
  "results/numerical/gumbel_softmax/ztrue_batch",
  batch_num, ".csv"
), header = FALSE)
z0_est <- read.csv(paste0(
  "results/numerical/gumbel_softmax/z0est_batch",
  batch_num, ".csv"
), header = FALSE)
# z0_est0 <- 0.5 * z_true[, 1]
# z0_est1 <- 0.5 * z_true[, 1]
# z0_est <- cbind(z0_est0, z0_est1)
z_all <- cbind(z_true, z0_est)
# z0_est0 is the recovered z0 from view 0
# z0_est1 is the recovered z0 from view 1
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

# Null hypotheses for conditional indepence tests
# H0: z1 from view 1 is independent of z0_est0 given z0
# H0: z2 from view 2 is independent of z0_est1 given z0

# GCM H0: E[residuals X | Z \times residuals Y |Z] = 0
# PCM H0: E[residuals Y | Z \times f(X,Z)] = 0 for all f

reg_mod <- "lrm" # "lrm" for linear models

gcm_z0_est0 <- gcm(
  X = z_all$z0_est0, Y = cbind(z_all$z1, z_all$z2), Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

gcm_z0_est1 <- gcm(
  X = z_all$z0_est1, Y = cbind(z_all$z1, z_all$z2), Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

gcm_z0_est0_rev <- gcm(
  X = z_all$z0, Y = cbind(z_all$z1, z_all$z2), Z = z_all$z0_est0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

gcm_z0_est1_rev <- gcm(
  X = z_all$z0, Y = cbind(z_all$z1, z_all$z2), Z = z_all$z0_est1,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

gcm(
  X = z_all$z0, Y = cbind(z_all$z1, z_all$z2),
  Z = cbind(z_all$z0_est0, z_all$z0_est1),
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z0_est0 <- pcm(
  X = cbind(z_all$z1, z_all$z2), Y = z_all$z0_est0, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z0_est1 <- pcm(
  X = cbind(z_all$z1, z_all$z2), Y = z_all$z0_est1, Z = z_all$z0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z0_est0_rev <- pcm(
  X = cbind(z_all$z1, z_all$z2), Y = z_all$z0, Z = z_all$z0_est0,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z0_est1_rev <- pcm(
  X = cbind(z_all$z1, z_all$z2), Y = z_all$z0, Z = z_all$z0_est1,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

reg_mod <- "tuned_rf"
pcm_z1_est0 <- pcm(
  X = cbind(z_all$z0, z_all$z2), Y = z_all$z0_est1, Z = z_all$z1,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

pcm_z1_est0 <- pcm(
  X = z_all$z0_est0, Y = cbind(z_all$z1, z_all$z3), Z = z_all$z1,
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

p_res0 <- plot(gcm_z0_est0)
p_res1 <- plot(gcm_z0_est1)
p_res0_rev <- plot(gcm_z0_est0_rev)
p_res1_rev <- plot(gcm_z0_est1_rev)

ggpubr::ggarrange(p_res0, p_res1, p_res0_rev, p_res1_rev,
  ncol = 2, nrow = 2,
  common.legend = TRUE
)

dev.off()

## ---- plot the pcm residuals ----

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

p_res0 <- plot(pcm_z0_est0)
p_res1 <- plot(pcm_z0_est1)
p_res0_rev <- plot(pcm_z0_est0_rev)
p_res1_rev <- plot(pcm_z0_est1_rev)

ggpubr::ggarrange(p_res0, p_res1, p_res0_rev, p_res1_rev,
  ncol = 2, nrow = 2,
  common.legend = TRUE
)

dev.off()

# ---- small test ----

n <- 5000
Z1 <- rnorm(n)
Z2 <- rnorm(n)
Z3 <- rnorm(n)
Z_S <- cbind(Z1, Z2)
Z_Sh <- cbind(Z1 + Z2, Z1 - Z2)
reg_mod <- "lrm" # "lrm" for linear models
gcm(
  X = Z_S, Y = Z3, Z = Z_Sh,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)
gcm(
  X = Z_Sh, Y = Z3, Z = Z_S,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)
pcm(
  X = Z_S, Y = Z3, Z = Z_Sh,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)
pcm(
  X = Z_Sh, Y = Z3, Z = Z_S,
  reg_YonZ = reg_mod, reg_XonZ = reg_mod
)

# ---- small test for factors ----

val_label_text <- read.csv("results/multimodal3di/0/val_label_text.csv",
  sep = " ",
  header = FALSE
)
val_label_text <- val_label_text[, -4]
# "object_shape", "object_ypos", "object_xpos",
# "object_color_index", "text_phrasing"
colnames(val_label_text) <- c("X1", "X2", "X3", "X4", "X5")

V1 <- val_label_text$X1 + val_label_text$X2
V2 <- val_label_text$X1 + val_label_text$X3
V3 <- val_label_text$X2 + val_label_text$X3
dat <- cbind(val_label_text, V1, V2, V3)

reg_mod <- "rf"
fm1 <- cbind(V1, V2, V3) ~ X4 + X5 | X1 + X2 + X3
test1 <- comets(fm1, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)

fm2 <- cbind(X1, X2, X3) ~ X4 + X5 | V1 + V2 + V3
test2 <- comets(fm2, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)

fm3 <- cbind(V1, V2, V3) ~ X1 + X2 + X3 | X4 + X5
test3 <- comets(fm3, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)

# ---- multimodal data (text) ----

val_label_text <- read.csv("results/multimodal3di/0/val_label_text.csv",
  sep = " ",
  header = FALSE
)
val_label_text <- val_label_text[, -4]
# "object_shape", "object_ypos", "object_xpos",
# "object_color_index", "text_phrasing"
colnames(val_label_text) <- c("X1", "X2", "X3", "X4", "X5")

# content identified between view 0 and view 2
ss <- "(0, 2)"
val_hz_text <- read.csv(paste0(
  "results/multimodal3di/0/val_hz_text_",
  ss, ".csv"
), sep = " ", header = FALSE)
head(val_hz_text)

dat <- cbind(val_label_text, val_hz_text)
dat <- dat %>%
  mutate(
    X1 = as.factor(X1),
    X2 = as.factor(X2),
    X3 = as.factor(X3),
    # X4 = as.numeric(X4),
    X5 = as.factor(X5)
  )

idx_use <- sample(1:nrow(dat), 5000)
dat <- dat[idx_use, ]

reg_mod <- "rf" # "tuned_rf" # "tuned_xgb", "tuned_rf"

fm1 <- cbind(V1, V2, V3) ~ X4 + X5 | X1 + X2 + X3
test1 <- comets(fm1, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
test1
# cond indep accepted [:)]

fm2 <- cbind(X1, X2, X3) ~ X4 + X5 | V1 + V2 + V3
test2 <- comets(fm2, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
test2
# cond indep accepted [:)]

fm3 <- cbind(V1, V2, V3) ~ X1 + X2 + X3 | X4 + X5
test3 <- comets(fm3, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
test3
# cond indep accepted [:/]

fm4 <- cbind(V1, V2, V3) ~ X1 + X2 + X3 + X5 | X4
test4 <- comets(fm4, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
test4

fm5 <- cbind(V1, V2, V3) ~ X1 + X2 + X3 + X4 | X5
test5 <- comets(fm5, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
test5

args <- list(
  etas = c(0.0025, 0.005, 0.01),
  nrounds = c(25, 50, 100, 200),
  max_depths = 1:5,
  verbose = 0
)

reg_mod <- "tuned_xgb"

fm <- V1 ~ X1 + X2 + X3 | X4 + X5
pcm_test1 <- comets(fm,
  data = dat, reg_YonXZ = reg_mod, reg_YonZ = reg_mod, test = "pcm",
  args_YonXZ = args, args_YonZ = args,
  return_fitted_models = TRUE
) # cond indep accepted [:/]
pcm_test1$models$reg_YonXZ

fm <- V2 ~ X1 + X2 + X3 | X4 + X5
pcm_test2 <- comets(fm,
  data = dat, reg_YonXZ = reg_mod, reg_YonZ = reg_mod, test = "pcm",
  args_YonXZ = args, args_YonZ = args
) # cond indep accepted [:/]

fm <- V3 ~ X1 + X2 + X3 | X4 + X5
pcm_test3 <- comets(fm,
  data = dat, reg_YonXZ = reg_mod, reg_YonZ = reg_mod, test = "pcm",
  args_YonXZ = args, args_YonZ = args
) # cond indep accepted [:/]


agg_jpeg(
  paste0(
    "results/multimodal3di/0/scatter_pcm_",
    reg_mod,
    ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)
plot(pcm_test1)
dev.off()

agg_jpeg(
  paste0(
    "results/multimodal3di/0/tmp",
    ".jpg"
  ),
  width = 10, height = 10,
  units = "in", res = 300
)
plot(x = dat$V2, y = dat$X3)
dev.off()

# ---- multimodal data (image) ----

val_label_image <- read.csv("results/multimodal3di/0/val_label_image.csv",
  sep = " ",
  header = FALSE
)
val_label_image <- val_label_image[, -4]
# "object_shape", "object_ypos", "object_xpos",
# "object_color_index", "text_phrasing"
colnames(val_label_image) <- paste0("X", 1:10)

# content identified between view 0 and view 2
ss <- "(0, 2)"
val_hz_image <- read.csv(paste0(
  "results/multimodal3di/0/val_hz_image_",
  ss, ".csv"
), sep = " ", header = FALSE)
head(val_hz_image)

dat <- cbind(val_label_image, val_hz_image)
dat <- dat %>%
  mutate(
    X1 = as.factor(X1),
    X2 = as.factor(X2),
    X3 = as.factor(X3),
    # X4 = as.numeric(X4),
    X5 = as.factor(X5)
  )

reg_mod <- "tuned_rf"

fm1 <- cbind(V1, V2, V3) ~ X4 + X5 + X6 + X7 + X8 + X9 + X10 | X1 + X2 + X3
test1 <- comets(fm1, data = dat, reg_YonZ = reg_mod, reg_XonZ = reg_mod)
