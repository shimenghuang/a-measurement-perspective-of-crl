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

# ---- load results from python ----

res_a <- read.csv(
    "results/numerical/res_five_latents_a.csv",
    header = TRUE
)
res_a %>%
    summarise(
        tmex_mean = mean(tmex_score),
        tmex_sd = sd(tmex_score),
    )

res_b <- read.csv(
    "results/numerical/res_five_latents_b.csv",
    header = TRUE
)
res_b %>%
    summarise(
        tmex_mean = mean(tmex_score),
        tmex_sd = sd(tmex_score),
    )

res_c <- read.csv(
    "results/numerical/res_five_latents_c.csv",
    header = TRUE
)
res_c %>%
    summarise(
        tmex_mean = mean(tmex_score),
        tmex_sd = sd(tmex_score),
    )

# ---- load data from one experiment ----

exper_id <- "five_latents_b"
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

# # create a mixed representation of Z0
# dat <- dat %>%
#     dplyr::mutate(
#         # z0_est0mix = z0_est0 + 0.1 * (z1 + 1)^2 - 0.1 * (z2 + 1)^2,
#         # z0_est1mix = z0_est1 + 0.1 * (z1 + 1)^2 - 0.1 * (z2 + 1)^2
#         z0_est0mix = z0_est0 + 0.01 * z1 - 0.01 * z2,
#         z0_est1mix = z0_est1 + 0.01 * z1 - 0.01 * z2
#     )

head(dat)

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

# pvals_batch_block <- pvals_df %>%
#     dplyr::filter(batch_num == 0, learned_block == "z0_est0")
# comp_adj_mat(
#     learned_block = "z0_est0",
#     all_latents = c("z0", "z1", "z2"),
#     pvals_batch = pvals_batch_block
# )

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
    expected_adj_mat, alpha = 0.05,
    return_mat = FALSE) {
    all_latents <- sort(unique(pvals_df$checked_latent))
    mat_lst <- lapply(batch_nums, \(bb) {
        pvals_batch_block <- pvals_df %>%
            dplyr::filter(batch_num == bb, learned_block == !!learned_block)
        comp_adj_mat(
            learned_block = learned_block,
            all_latents = all_latents,
            pvals_batch_block = pvals_batch_block,
            alpha = alpha
        )
    })
    ham_dists <- sapply(mat_lst, \(mat) {
        sum(mat != expected_adj_mat)
    })
    if (return_mat) {
        return(list(
            est = mean(ham_dists),
            sd = stats::sd(ham_dists),
            mat_lst = mat_lst
        ))
    } else {
        list(
            est = mean(ham_dists),
            sd = stats::sd(ham_dists)
        )
    }
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

bmcc_view_batch_block <- function(
    dat_batch, target_block, learned_block) {
    cor_all <- sapply(target_block, \(tv) {
        sapply(learned_block, \(lv) {
            cor(dat_batch[tv], dat_batch[lv], method = "spearman")
        })
    })
    mean(cor_all)
}

#' Compute B-MCC for one view in one batch.
#' @param target_blocks A list of M blocks of latent variables
#' @param learned_blocks A list of M blocks of representations
bmcc_view_batch <- function(
    dat_batch, target_blocks, learned_blocks) {
    cor_all <- mapply(
        \(tb, lb) {
            cor(dat_batch[tb], dat_batch[lb], method = "spearman")
        }, target_blocks, learned_blocks
    )
    mean(cor_all)
}

#' Compute B-MCC for one view for all batches.
comp_bmcc <- function(dat, batch_nums, target_blocks, learned_blocks) {
    res_all <- lapply(batch_nums, \(bb) {
        dat_batch <- dat %>%
            dplyr::filter(batch_num == bb)
        res <- bmcc_view_batch(
            dat_batch = dat_batch,
            target_blocks = target_blocks,
            learned_blocks = learned_blocks
        )
        data.frame(
            batch_num = bb,
            bmcc = res
        )
    })
    bmcc_all <- do.call(rbind, res_all)
    bmcc_all
}

helper_pval_block <- function(learned_block, exper_id = NULL) {
    if (!is.null(exper_id)) {
        dat <- dat %>%
            dplyr::filter(exper_id == !!exper_id)
    }

    pvals_block_cond01 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z2"),
        cond_latents = c("z0", "z1", "x", "y"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond01$learned_block <- learned_block
    pvals_block_cond01$cond_latents <- paste("z0", "z1", "x", "y", sep = "_")

    pvals_block_cond02 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z1"),
        cond_latents = c("z0", "z2", "x", "y"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond02$learned_block <- learned_block
    pvals_block_cond02$cond_latents <- paste("z0", "z2", "x", "y", sep = "_")

    pvals_block_cond12 <- comp_pvals_block(
        dat = dat,
        batch_nums = batch_nums,
        other_latents = c("z0"),
        cond_latents = c("z1", "z2", "x", "y"),
        learned_block = learned_block,
        method = "pcm",
        reg_mod = "lrm",
        rep = 9
    )
    pvals_block_cond12$learned_block <- learned_block
    pvals_block_cond12$cond_latents <- paste("z1", "z2", "x", "y", sep = "_")

    rbind(pvals_block_cond01, pvals_block_cond02, pvals_block_cond12)
}

# # ---- check results ----

# all_latents <- c("z0", "z1", "z2", "x", "y")
# bb <- 0L
# learned_block <- "z0_est0mix"
# dat_batch <- dat %>%
#     dplyr::filter(batch_num == bb)

# test_block_batch(
#     dat_batch,
#     other_latents = "z0",
#     cond_latents = c("z1", "z2", "x", "y"),
#     learned_block = learned_block,
#     method = "pcm",
#     reg_mod = "lrm",
#     rep = 5
# )

# test_block_batch(
#     dat_batch,
#     other_latents = "z1",
#     cond_latents = c("z0", "z2", "x", "y"),
#     learned_block = learned_block,
#     method = "pcm",
#     reg_mod = "lrm",
#     rep = 5
# )

# test_block_batch(
#     dat_batch,
#     other_latents = "z2",
#     cond_latents = c("z0", "z1", "x", "y"),
#     learned_block = learned_block,
#     method = "pcm",
#     reg_mod = "lrm",
#     rep = 5
# )

# r2_view_batch(
#     dat_batch = dat_batch,
#     all_latents = all_latents,
#     learned_block = "z0_est0"
# )

# bmcc_view_batch(
#     dat_batch = dat_batch,
#     target_blocks = "z0",
#     learned_blocks = "z0_est0mix"
# )

# bmcc_res0 <- comp_bmcc(
#     dat = dat,
#     batch_nums = batch_nums,
#     target_blocks = "z0",
#     learned_blocks = "z0_est1"
# )
# mean(bmcc_res0$bmcc)
# sd(bmcc_res0$bmcc)

# bmcc_res1 <- comp_bmcc(
#     dat = dat,
#     batch_nums = batch_nums,
#     target_blocks = "z0",
#     learned_blocks = "z0_est1"
# )
# mean(bmcc_res1$bmcc)
# sd(bmcc_res1$bmcc)

# pvals_block0 <- helper_pval_block(learned_block = "z0_est0")
# pvals_block1 <- helper_pval_block(learned_block = "z0_est1")
# pvals_df <- rbind(pvals_block0, pvals_block1)

# cis_res0 <- comp_cis_score(
#     pvals_df = pvals_df,
#     batch_nums = batch_nums,
#     learned_block = "z0_est0",
#     expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
#     return_mat = TRUE
# )
# cis_res0$est
# cis_res0$sd

# cis_res1 <- comp_cis_score(
#     pvals_df = pvals_df,
#     batch_nums = batch_nums,
#     learned_block = "z0_est1",
#     expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
#     return_mat = TRUE
# )
# cis_res1$est
# cis_res1$sd

# agg_jpeg(
#     paste0(
#         "results/numerical/", exper_id, "/simu_pvals.jpg"
#     ),
#     width = 10, height = 5,
#     units = "in", res = 300
# )

# pvals_df %>%
#     ggplot2::ggplot(aes(
#         x = cond_latents,
#         y = pval,
#         color = learned_block,
#         group = interaction(learned_block, cond_latents)
#     )) +
#     geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
#     ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
#     geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
#     # scale_y_log10() +
#     theme_bw()

# dev.off()

# # check the mixed representation
# pvals_block0mix <- helper_pval_block(learned_block = "z0_est0mix")
# pvals_block1mix <- helper_pval_block(learned_block = "z0_est1mix")
# pvals_df_mix <- rbind(pvals_block0mix, pvals_block1mix)

# cis_res0mix <- comp_cis_score(
#     pvals_df = pvals_df_mix,
#     batch_nums = batch_nums,
#     learned_block = "z0_est0mix",
#     expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
#     return_mat = TRUE
# )
# cis_res0mix$est
# cis_res0mix$sd

# cis_res1mix <- comp_cis_score(
#     pvals_df = pvals_df_mix,
#     batch_nums = batch_nums,
#     learned_block = "z0_est1mix",
#     expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
#     return_mat = TRUE
# )
# cis_res1mix$est
# cis_res1mix$sd

# agg_jpeg(
#     paste0(
#         "results/numerical/", exper_id, "/simu_pvals_mixed.jpg"
#     ),
#     width = 10, height = 5,
#     units = "in", res = 300
# )

# pvals_df_mix %>%
#     ggplot2::ggplot(aes(
#         x = cond_latents,
#         y = pval,
#         color = learned_block,
#         group = interaction(learned_block, cond_latents)
#     )) +
#     geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
#     ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
#     geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
#     # scale_y_log10() +
#     theme_bw()

# dev.off()

# r2_df0 <- comp_r2(
#     dat = dat,
#     batch_nums = batch_nums,
#     all_latents = all_latents,
#     learned_block = "z0_est0mix"
# )
# r2_df0$view <- 0

# r2_df1 <- comp_r2(
#     dat = dat,
#     batch_nums = batch_nums,
#     all_latents = all_latents,
#     learned_block = "z0_est1mix"
# )
# r2_df1$view <- 1

# r2_df <- rbind(r2_df0, r2_df1)

# r2_df %>%
#     dplyr::filter((checked_latent != "x") &
#         (checked_latent != "y")) %>%
#     dplyr::group_by(view, checked_latent) %>%
#     dplyr::summarise(
#         avg_r2 = mean(r2),
#         sd_r2 = sd(r2),
#         num_total = n()
#     )

# agg_jpeg(
#     paste0(
#         "results/numerical/", exper_id, "/simu_r2s.jpg"
#     ),
#     width = 10, height = 10,
#     units = "in", res = 300
# )

# r2_df %>%
#     dplyr::filter((checked_latent != "x") &
#         (checked_latent != "y")) %>%
#     dplyr::mutate(view = as.factor(view)) %>%
#     ggplot2::ggplot(aes(
#         x = view, y = r2,
#         group = interaction(view, checked_latent),
#         color = checked_latent
#     )) +
#     geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
#     ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
#     geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
#     theme_bw()

# dev.off()

# ---- compare three models ----

set.seed(123)
exper_id <- "five_latents_a"
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
dat_good <- do.call(rbind, dat_ls)
dat_good$batch_num <- as.integer(dat_good$batch_num)
dat_good$exper_id <- "five_latents"

exper_id <- "five_latents_b"
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
dat_bad <- do.call(rbind, dat_ls)
dat_bad$batch_num <- as.integer(dat_bad$batch_num)
dat_bad$exper_id <- "five_latents_bad"

dat_mix <- dat_good %>%
    dplyr::mutate(
        z0_est0 = z0_est0 + 0.2 * z1 - 0.1 * z2,
        z0_est1 = z0_est1 + 0.2 * z1 - 0.1 * z2
    )
dat_mix$exper_id <- "five_latents_mix"

# combine the two datasets
dat <- rbind(dat_good, dat_bad, dat_mix)

write.csv(
    dat,
    file = paste0(
        "results/numerical/", exper_id, "/simu_data.csv"
    ),
    row.names = FALSE
)

# compute p-values for both datasets
pvals_block_good <- helper_pval_block(learned_block = "z0_est0", exper_id = "five_latents")
pvals_block_bad <- helper_pval_block(learned_block = "z0_est0", exper_id = "five_latents_bad")
pvals_block_mix <- helper_pval_block(learned_block = "z0_est0", exper_id = "five_latents_mix")
pvals_block_good$exper_id <- "five_latents_good"
pvals_block_bad$exper_id <- "five_latents_bad"
pvals_block_mix$exper_id <- "five_latents_mix"
pvals_df <- rbind(pvals_block_good, pvals_block_bad, pvals_block_mix)
pvals_df$cond_latents <- factor(pvals_df$cond_latents,
    levels = c("z1_z2_x_y", "z0_z2_x_y", "z0_z1_x_y")
)
pvals_df$exper_id <- factor(pvals_df$exper_id,
    levels = c("five_latents_good", "five_latents_bad", "five_latents_mix")
)

write.csv(
    pvals_df,
    file = paste0(
        "results/numerical/", exper_id, "/simu_pvals.csv"
    ),
    row.names = FALSE
)

agg_jpeg(
    paste0(
        "results/numerical/", exper_id, "/compare_pvals.jpg"
    ),
    width = 5, height = 4,
    units = "in", res = 300
)

pvals_df %>%
    ggplot2::ggplot(aes(
        x = cond_latents,
        y = pval,
        color = exper_id,
        group = interaction(exper_id, cond_latents)
    )) +
    geom_boxplot(width = 0.3, alpha = 0, position = position_dodge(0.5)) +
    ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
    geom_hline(yintercept = 0.05, linetype = "dashed", color = "red") +
    # scale_color_discrete(labels = c(
    #     "Suff. trained",
    #     "Insuff. trained",
    #     "Suff. but mixed"
    # )) +
    scale_x_discrete(labels = c(
        expression(Z[1]),
        expression(Z[2]),
        expression(Z[3])
    )) +
    xlab(element_blank()) +
    ylab("p-value") +
    guides(color = guide_legend(title = "Training")) +
    theme_bw() +
    theme(
        legend.position = "top",
        text = element_text(size = 14, family = "Sans")
    )

dev.off()

## ---- T-MEX scores ----

cis_good <- comp_cis_score(
    pvals_df = pvals_df %>%
        dplyr::filter(exper_id == "five_latents_good"),
    batch_nums = batch_nums,
    learned_block = "z0_est0",
    expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
    return_mat = TRUE
)
cis_good$est
cis_good$sd

cis_bad <- comp_cis_score(
    pvals_df = pvals_df %>%
        dplyr::filter(exper_id == "five_latents_bad"),
    batch_nums = batch_nums,
    learned_block = "z0_est0",
    expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
    return_mat = TRUE
)
cis_bad$est
cis_bad$sd

cis_mix <- comp_cis_score(
    pvals_df = pvals_df %>%
        dplyr::filter(exper_id == "five_latents_mix"),
    batch_nums = batch_nums,
    learned_block = "z0_est0",
    expected_adj_mat = matrix(c(1, 0, 0), nrow = 1),
    return_mat = TRUE
)
cis_mix$est
cis_mix$sd

cis_tbl <- data.frame(
    training = c("sufficient", "insufficient", "suffmixed"),
    cis = c(cis_good$est, cis_bad$est, cis_mix$est),
    sd = c(cis_good$sd, cis_bad$sd, cis_mix$sd)
)

write.csv(
    cis_tbl,
    file = paste0(
        "results/numerical/", exper_id, "/simu_cis.csv"
    ),
    row.names = FALSE
)

## ---- BMCC scores ----
bmcc_res0_good <- comp_bmcc(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents"),
    batch_nums = batch_nums,
    target_blocks = "z0",
    learned_blocks = "z0_est0"
)
mean(bmcc_res0_good$bmcc)
sd(bmcc_res0_good$bmcc)

bmcc_res0_bad <- comp_bmcc(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents_bad"),
    batch_nums = batch_nums,
    target_blocks = "z0",
    learned_blocks = "z0_est0"
)
mean(bmcc_res0_bad$bmcc)
sd(bmcc_res0_bad$bmcc)

bmcc_res0_mix <- comp_bmcc(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents_mix"),
    batch_nums = batch_nums,
    target_blocks = "z0",
    learned_blocks = "z0_est0"
)
mean(bmcc_res0_mix$bmcc)
sd(bmcc_res0_mix$bmcc)

bmcc_tbl <- data.frame(
    training = c("sufficient", "insufficient", "suffmixed"),
    bmcc = c(
        mean(bmcc_res0_good$bmcc),
        mean(bmcc_res0_bad$bmcc),
        mean(bmcc_res0_mix$bmcc)
    ),
    sd = c(sd(bmcc_res0_good$bmcc), sd(bmcc_res0_bad$bmcc), sd(bmcc_res0_mix$bmcc))
)
write.csv(
    bmcc_tbl,
    file = paste0(
        "results/numerical/", exper_id, "/simu_bmcc.csv"
    ),
    row.names = FALSE
)

# ---- R2 ----
r2_df0_good <- comp_r2(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents"),
    batch_nums = batch_nums,
    all_latents = c("z0", "z1", "z2"),
    learned_block = "z0_est0"
)
r2_df0_good$view <- 0
r2_df0_good <- r2_df0_good %>%
    dplyr::group_by(checked_latent, view) %>%
    dplyr::summarise(mean_r2 = mean(r2), sd_r2 = sd(r2))

r2_df0_bad <- comp_r2(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents_bad"),
    batch_nums = batch_nums,
    all_latents = c("z0", "z1", "z2"),
    learned_block = "z0_est0"
)
r2_df0_bad$view <- 0
r2_df0_bad <- r2_df0_bad %>%
    dplyr::group_by(checked_latent, view) %>%
    dplyr::summarise(mean_r2 = mean(r2), sd_r2 = sd(r2))

r2_df0_mix <- comp_r2(
    dat = dat %>%
        dplyr::filter(exper_id == "five_latents_mix"),
    batch_nums = batch_nums,
    all_latents = c("z0", "z1", "z2"),
    learned_block = "z0_est0"
)
r2_df0_mix$view <- 0
r2_df0_mix <- r2_df0_mix %>%
    dplyr::group_by(checked_latent, view) %>%
    dplyr::summarise(mean_r2 = mean(r2), sd_r2 = sd(r2))

r2_tbl <- data.frame(
    training = c(rep("sufficient", 3), rep("insufficient", 3), rep("suffmixed", 3)),
    latent_var = c("z0", "z1", "z2"),
    r2 = c(
        r2_df0_good$mean_r2,
        r2_df0_bad$mean_r2,
        r2_df0_mix$mean_r2
    ),
    sd = c(
        r2_df0_good$sd_r2,
        r2_df0_bad$sd_r2,
        r2_df0_mix$sd_r2
    )
)

write.csv(
    r2_tbl,
    file = paste0(
        "results/numerical/", exper_id, "/simu_r2.csv"
    ),
    row.names = FALSE
)

# ---- compute bias of the ATE estimates ----
beta_est <- lapply(batch_nums, \(batch_num) {
    # beta_oracle <- coef(lm(y ~ z0 + x, data = dat_good[dat$batch_num == batch_num, ]))[3]
    beta_good <- coef(lm(y ~ z0_est0 + x, data = dat_good[dat_good$batch_num == batch_num, ]))[3]
    beta_bad <- coef(lm(y ~ z0_est0 + x, data = dat_bad[dat_bad$batch_num == batch_num, ]))[3]
    beta_mix <- coef(lm(y ~ z0_est0 + x, data = dat_mix[dat_mix$batch_num == batch_num, ]))[3]
    data.frame(
        batch_num = batch_num,
        # beta_oracle = beta_oracle,
        sufficient = beta_good,
        insufficient = beta_bad,
        suffmixed = beta_mix
    )
})
beta_est_df <- do.call(rbind, beta_est)
head(beta_est_df)

write.csv(
    beta_est_df,
    file = paste0(
        "results/numerical/", exper_id, "/simu_beta_est.csv"
    ),
    row.names = FALSE
)

agg_jpeg(
    paste0(
        "results/numerical/", exper_id, "/compare_beta.jpg"
    ),
    width = 3, height = 4,
    units = "in", res = 300
)

beta_est_df %>%
    tidyr::pivot_longer(
        cols = c("sufficient", "insufficient", "suffmixed"),
        names_to = "training",
        values_to = "beta_est"
    ) %>%
    dplyr::mutate(training = factor(training,
        levels = c("sufficient", "insufficient", "suffmixed"),
    )) %>%
    ggplot2::ggplot(aes(
        x = training,
        y = beta_est
    )) +
    geom_boxplot(width = 0.15, alpha = 0, position = position_dodge(0.5)) +
    ggbeeswarm::geom_quasirandom(width = 0.1, dodge.width = 0.5) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    # scale_x_discrete(labels = c(
    #     "Sufficient",
    #     "Insufficient"
    # )) +
    xlab("Training") +
    ylab("ATE estimates") +
    guides(color = guide_legend(title = "Training")) +
    theme_bw() +
    theme(
        legend.position = "top",
        text = element_text(size = 14, family = "Sans")
    )

dev.off()
