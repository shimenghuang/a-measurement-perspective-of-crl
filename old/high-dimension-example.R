library(comets)
library(ranger)
options(ranger.num.threads = 8)

message("Running PCM X indep Y | Z")
nrep <- 30
start <- Sys.time()
res_all1 <- replicate(nrep,
    {
        n <- 3000
        dz <- 10
        dx <- 700
        reg_mod <- "rf"
        Z <- matrix(rnorm(n * dz), nrow = n)
        theta <- matrix(rnorm(dz), nrow = dz)
        Y <- Z %*% theta + rnorm(n)
        beta <- rnorm(dx)
        X <- matrix(rnorm(n * dx), nrow = n) + Y %*% beta
        pcm(
            X = X, Y = Y, Z = Z,
            eg_YonXZ = reg_mod,
            reg_YonZ = reg_mod,
            reg_YhatonZ = reg_mod,
            reg_VonXZ = reg_mod,
            reg_RonZ = reg_mod
        )
    },
    simplify = FALSE
)
end <- Sys.time()
time_used <- end - start
res_all1$time <- time_used
saveRDS(res_all1, file = "results/high-dimension-example-res1.rds")

message("Running GCM X indep Y | Z")
nrep <- 30
start <- Sys.time()
res_all2 <- replicate(nrep,
    {
        n <- 3000
        dz <- 10
        dx <- 700
        reg_mod <- "rf"
        Z <- matrix(rnorm(n * dz), nrow = n)
        theta <- matrix(rnorm(dz), nrow = dz)
        Y <- Z %*% theta + rnorm(n)
        beta <- rnorm(dx)
        X <- matrix(rnorm(n * dx), nrow = n) + Y %*% beta
        gcm(
            X = X, Y = Y, Z = Z,
            reg_YonZ = reg_mod,
            reg_XonZ = reg_mod,
        )
    },
    simplify = FALSE
)
end <- Sys.time()
time_used <- end - start
res_all2$time <- time_used
saveRDS(res_all2, file = "results/high-dimension-example-res2.rds")

message("Running GCM X indep Z | Y")
nrep <- 30
start <- Sys.time()
res_all3 <- replicate(nrep,
    {
        n <- 3000
        dz <- 10
        dx <- 700
        reg_mod <- "rf"
        Z <- matrix(rnorm(n * dz), nrow = n)
        theta <- matrix(rnorm(dz), nrow = dz)
        Y <- Z %*% theta + rnorm(n)
        beta <- rnorm(dx)
        X <- matrix(rnorm(n * dx), nrow = n) + Y %*% beta
        gcm(
            X = X, Y = Z, Z = Y,
            reg_YonZ = reg_mod,
            reg_XonZ = reg_mod,
        )
    },
    simplify = FALSE
)
end <- Sys.time()
time_used <- end - start
res_all3$time <- time_used
saveRDS(res_all3, file = "results/high-dimension-example-res3.rds")

# res_all1 <- readRDS("high-dimension-example-res1.rds")
# res_all1$time
# sum(sapply(res_all1[1:50], \(res) {
#     res$p.value
# }) < 0.05)
# res_all2 <- readRDS("high-dimension-example-res2.rds")
# res_all2$time
# sum(sapply(res_all2[1:50], \(res) {
#     res$p.value
# }) < 0.05)
