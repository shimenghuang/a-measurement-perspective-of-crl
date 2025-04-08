library(comets)
library(dHSIC)
library(dplyr)
library(ggplot2)
library(ggbeeswarm)
library(ggpubr)
library(ragg)
library(xgboost)
library(ranger)
options(ranger.num.threads = 8)

# ---- load data ----

batch_nums <- 0:29
dat_ls <- lapply(batch_nums, \(batch_num) {
    z_true <- read.csv(paste0(
        "results/numerical/gumbel_softmax/ztrue_batch",
        batch_num, ".csv"
    ), header = FALSE)
    z0_est <- read.csv(paste0(
        "results/numerical/gumbel_softmax/z0est_batch",
        batch_num, ".csv"
    ), header = FALSE)
    z_all <- cbind(z_true, z0_est)
    # z0_est0 is the recovered z0 from view 0
    # z0_est1 is the recovered z0 from view 1
    colnames(z_all) <- c("z0", "z1", "z2", "z0_est0", "z0_est1")
    z_all <- data.frame(z_all)
    z_all$batch_num <- batch_num
    z_all
})
dat <- do.call(rbind, dat_ls)
dat$batch_num <- as.integer(dat$batch_num)

# ---- compute cis scores ----

#' Perform tests for one batch.
#' @param other_latents The latent variables to test conditional
#'  independence with the learned block (each of them are tested separately).
#' @param cond_latents A string (or a vector of strings) which are column
#'   names in `dat` corresponding to the latent variables to be
#'   condition on (they are fixed).
#' @param learned_block A string (or a vector of strings)
#'   which are column names in `dat` corresponding to a learned block
#'   (they are fixed).
test_block_batch <- function(
    dat_batch, other_latents, cond_latents, learned_block,
    method, reg_mod, ...) {
    if (method == "pcm") {
        test_res <- lapply(other_latents, \(cc) {
            comets::pcm(
                X = dat_batch[cc],
                Y = dat_batch[learned_block],
                Z = dat_batch[cond_latents],
                reg_YonXZ = reg_mod,
                reg_YonZ = reg_mod,
                reg_YhatonZ = reg_mod,
                reg_VonXZ = reg_mod,
                reg_RonZ = reg_mod,
                ...
            )
        })
    } else if (method == "gcm") {
        test_res <- lapply(cond_latents, \(cc) {
            comets::gcm(
                X = dat_batch[cc],
                Y = dat_batch[learned_block],
                Z = dat_batch[cond_latents],
                reg_YonZ = reg_mod,
                reg_XonZ = reg_mod,
                ...
            )
        })
    } else {
        stop("Invalid method. Choose either 'pcm' or 'gcm'.")
    }
    list(
        checked_latents = other_latents,
        test_res = test_res
    )
}

#' Compute p-values for all batches for a given learned block.
#' @param batch_nums The batch numbers to test.
#' @param learned_block A string (or a vector of strings) which are
#'   column names in `dat` corresponding to a learned block.
comp_pvals_block <- function(
    dat, batch_nums, other_latents, cond_latents, learned_block,
    method, reg_mod, ...) {
    pvals_all <- lapply(batch_nums, \(bb) {
        dat_batch <- dat %>%
            dplyr::filter(batch_num == bb)
        res_lst <- test_block_batch(
            dat_batch,
            other_latents = other_latents,
            cond_latents = cond_latents,
            learned_block = learned_block,
            method = method,
            reg_mod = reg_mod,
            ...
        )
        pvals <- sapply(res_lst$test_res, \(res) {
            res$p.value
        })
        data.frame(
            batch_num = bb,
            checked_latent = res_lst$checked_latents,
            pval = pvals
        )
    })
    pvals_all <- do.call(rbind, pvals_all)
    pvals_all
}

#' Compute adjacency matrix in the bipartite graph
#' @examples
#' pvals_batch_block <- pvals_df %>%
#'     dplyr::filter(batch_num == 0, learned_block == "z0_est0")
#' comp_adj_mat(
#'     learned_block = "z0_est0",
#'     all_latents = c("z0", "z1", "z2"),
#'     pvals_batch = pvals_batch_block
#' )
comp_adj_mat <- function(
    learned_block, all_latents, pvals_batch_block,
    alpha = 0.05) {
    all_vars <- c(learned_block, all_latents)
    num_vars <- length(all_vars)
    mat <- matrix(0L, nrow = num_vars, ncol = num_vars)
    for (ii in seq.int(from = length(learned_block) + 1, to = num_vars)) {
        # message("ii", ii)
        for (jj in seq.int(from = 1, to = ii - 1)) {
            # message("jj", jj)
            pval <- pvals_batch_block %>%
                dplyr::filter(checked_latent == all_vars[ii]) %>%
                pull(pval)
            # message(pval)
            mat[ii, jj] <- as.integer(pval < alpha)
        }
    }
    colnames(mat) <- c(learned_block, all_latents)
    rownames(mat) <- c(learned_block, all_latents)
    mat[all_latents, learned_block]
}

#' Compute the conditional independence score for a given learned block
#' @examples
#' comp_cis_score(
#'     pvals_df = pvals_df,
#'     batch_nums = batch_nums,
#'     learned_block = "z0_est1",
#'     expected_adj_mat = matrix(c(1, 0, 0), nrow = 1)
#' )
comp_cis_score <- function(
    pvals_df, batch_nums, learned_block,
    expected_adj_mat, alpha = 0.05) {
    all_latents <- sort(unique(pvals_df$checked_latent))
    ham_dists <- sapply(batch_nums, \(bb) {
        pvals_batch_block <- pvals_df %>%
            dplyr::filter(batch_num == bb, learned_block == !!learned_block)
        mat <- comp_adj_mat(
            learned_block = learned_block,
            all_latents = all_latents,
            pvals_batch_block = pvals_batch_block,
            alpha = alpha
        )
        sum(mat != expected_adj_mat)
    })
    list(
        est = mean(ham_dists),
        sd = stats::sd(ham_dists)
    )
}

r2_view_batch <- function(
    dat_batch, all_latents, learned_block) {
    r2_all <- lapply(all_latents, \(cc) {
        cor(dat_batch[learned_block], dat_batch[cc])^2
    })
    r2_all <- do.call(rbind, r2_all)
    r2_all <- data.frame(r2_all)
    r2_all$checked_latent <- all_latents
    colnames(r2_all) <- c("r2", "checked_latent")
    r2_all
}

comp_r2 <- function(
    dat, batch_nums, all_latents, learned_block) {
    res_all <- lapply(batch_nums, \(bb) {
        dat_batch <- dat %>%
            dplyr::filter(batch_num == bb)
        res_lst <- r2_view_batch(
            dat_batch = dat_batch,
            all_latents = all_latents,
            learned_block = learned_block
        )
        data.frame(
            batch_num = bb,
            checked_latent = res_lst$checked_latent,
            r2 = res_lst$r2
        )
    })
    r2_all <- do.call(rbind, res_all)
    r2_all
}

helper_pval_block <- function(learned_block) {
    pvals_block_cond01 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z2"),
        cond_latents = c("z0", "z1"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond01$learned_block <- learned_block
    pvals_block_cond01$cond_latents <- paste("z0", "z1", sep = "_")

    pvals_block_cond02 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z1"),
        cond_latents = c("z0", "z2"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond02$learned_block <- learned_block
    pvals_block_cond02$cond_latents <- paste("z0", "z2", sep = "_")

    pvals_block_cond12 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z0"),
        cond_latents = c("z1", "z2"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond12$learned_block <- learned_block
    pvals_block_cond12$cond_latents <- paste("z1", "z2", sep = "_")

    rbind(pvals_block_cond01, pvals_block_cond02, pvals_block_cond12)
}

# ---- check results ----

all_latents <- c("z0", "z1", "z2")
bb <- 1L
learned_block <- "z0_est0"
dat_batch <- dat %>%
    dplyr::filter(batch_num == bb)

test_block_batch(
    dat_batch,
    other_latents = "z2",
    cond_latents = c("z0", "z1"),
    learned_block = learned_block,
    method = "pcm",
    reg_mod = "lrm",
    rep = 5
)

test_block_batch(
    dat_batch,
    other_latents = "z1",
    cond_latents = c("z0", "z2"),
    learned_block = learned_block,
    method = "pcm",
    reg_mod = "lrm",
    rep = 5
)

r2_view_batch(
    dat_batch = dat_batch,
    all_latents = all_latents,
    learned_block = "z0_est0"
)

pvals_block0 <- helper_pval_block(learned_block = "z0_est0")
pvals_block1 <- helper_pval_block(learned_block = "z0_est1")
pvals_df <- rbind(pvals_block0, pvals_block1)

comp_cis_score(
    pvals_df = pvals_df,
    batch_nums = batch_nums,
    learned_block = "z0_est0",
    expected_adj_mat = matrix(c(1, 0, 0), nrow = 1)
)
comp_cis_score(
    pvals_df = pvals_df,
    batch_nums = batch_nums,
    learned_block = "z0_est1",
    expected_adj_mat = matrix(c(0, 1, 0), nrow = 1)
)

agg_jpeg(
    paste0(
        "results/numerical/gumbel_softmax/simu_pvals.jpg"
    ),
    width = 10, height = 10,
    units = "in", res = 300
)

pvals_df %>%
    dplyr::mutate(view = as.factor(view)) %>%
    ggplot2::ggplot(aes(
        x = cond_latents,
        y = pval,
        color = view,
        group = interaction(view, cond_latents)
    )) +
    geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
    ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    # scale_y_log10() +
    theme_bw()

dev.off()

r2_df0 <- comp_r2(
    dat = dat,
    batch_nums = batch_nums,
    all_latents = all_latents,
    learned_block = "z0_est0"
)
r2_df0$view <- 0

r2_df1 <- comp_r2(
    dat = dat,
    batch_nums = batch_nums,
    all_latents = all_latents,
    learned_block = "z0_est1"
)
r2_df1$view <- 1

r2_df <- rbind(r2_df0, r2_df1)

r2_df %>%
    dplyr::group_by(view, checked_latent) %>%
    dplyr::summarise(
        avg_r2 = mean(r2),
        num_total = n()
    )

agg_jpeg(
    paste0(
        "results/numerical/gumbel_softmax/simu_r2s.jpg"
    ),
    width = 10, height = 10,
    units = "in", res = 300
)

r2_df %>%
    dplyr::mutate(view = as.factor(view)) %>%
    ggplot2::ggplot(aes(
        x = view, y = r2,
        group = interaction(view, checked_latent),
        color = checked_latent
    )) +
    geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
    ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    theme_bw()

dev.off()
