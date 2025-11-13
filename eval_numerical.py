# %%
import copy

import numpy as np
import pandas as pd
from causallearn.utils.cit import FastKCI_CInd, KCI_CInd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from pycomets.gcm import GCM
from pycomets.pcm import PCM
from pycomets.regression import KRR, LM, RF, XGB


# %%
# T-MEX score for this simulated experiment
def pcm_batch(
    df_batch, view_num=0, fun_reg=LM(), alpha=0.05, rng=np.random.default_rng()
):
    pcm = PCM()
    pcm.test(
        reg_yonxz=copy.deepcopy(fun_reg),
        reg_ronz=copy.deepcopy(fun_reg),
        reg_vonxz=copy.deepcopy(fun_reg),
        reg_yhatonz=copy.deepcopy(fun_reg),
        reg_yonz=copy.deepcopy(fun_reg),
        X=df_batch[["z0"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z1", "z2", "x", "y"]].to_numpy(),
        estimate_variance=False,
        rep=9,
        rng=rng,
        show_summary=False,
    )
    pval1 = pcm.pval
    pcm = PCM()
    pcm.test(
        reg_yonxz=copy.deepcopy(fun_reg),
        reg_ronz=copy.deepcopy(fun_reg),
        reg_vonxz=copy.deepcopy(fun_reg),
        reg_yhatonz=copy.deepcopy(fun_reg),
        reg_yonz=copy.deepcopy(fun_reg),
        X=df_batch[["z1"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z2", "x", "y"]].to_numpy(),
        estimate_variance=False,
        rep=9,
        rng=rng,
        show_summary=False,
    )
    pval2 = pcm.pval
    pcm = PCM()
    pcm.test(
        reg_yonxz=copy.deepcopy(fun_reg),
        reg_ronz=copy.deepcopy(fun_reg),
        reg_vonxz=copy.deepcopy(fun_reg),
        reg_yhatonz=copy.deepcopy(fun_reg),
        reg_yonz=copy.deepcopy(fun_reg),
        X=df_batch[["z2"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z1", "x", "y"]].to_numpy(),
        estimate_variance=False,
        rep=9,
        rng=rng,
        show_summary=False,
    )
    pval3 = pcm.pval
    score1 = abs(int(pval1 < alpha) - 1)
    score2 = int(pval2 < alpha)
    score3 = int(pval3 < alpha)
    tmex_score = score1 + score2 + score3
    return tmex_score, pval1, pval2, pval3


def gcm_batch(
    df_batch, view_num=0, fun_reg=LM(), alpha=0.05, rng=np.random.default_rng()
):
    gcm = GCM()
    gcm.test(
        reg_yz=copy.deepcopy(fun_reg),
        reg_xz=copy.deepcopy(fun_reg),
        X=df_batch[["z0"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z1", "z2", "x", "y"]].to_numpy(),
        test_type="max",
        B=499,
        show_summary=False,
    )
    pval1 = gcm.pval

    gcm = GCM()
    gcm.test(
        reg_yz=copy.deepcopy(fun_reg),
        reg_xz=copy.deepcopy(fun_reg),
        X=df_batch[["z1"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z2", "x", "y"]].to_numpy(),
        test_type="max",
        B=499,
        show_summary=False,
    )
    pval2 = gcm.pval

    gcm = GCM()
    gcm.test(
        reg_yz=copy.deepcopy(fun_reg),
        reg_xz=copy.deepcopy(fun_reg),
        X=df_batch[["z2"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z1", "x", "y"]].to_numpy(),
        test_type="max",
        B=499,
        show_summary=False,
    )
    pval3 = gcm.pval

    score1 = abs(int(pval1 < alpha) - 1)
    score2 = int(pval2 < alpha)
    score3 = int(pval3 < alpha)
    tmex_score = score1 + score2 + score3
    return tmex_score, pval1, pval2, pval3


def kci_batch(df_batch, fastkci=False, view_num=0, alpha=0.05):

    if fastkci:
        kci = FastKCI_CInd()
    else:
        kci = KCI_CInd(est_width="median")
    pval1, _ = kci.compute_pvalue(
        data_x=df_batch[["z0"]].to_numpy(),
        data_y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        data_z=df_batch[["z1", "z2", "x", "y"]].to_numpy(),
    )

    if fastkci:
        kci = FastKCI_CInd()
    else:
        kci = KCI_CInd(est_width="median")
    pval2, _ = kci.compute_pvalue(
        data_x=df_batch[["z1"]].to_numpy(),
        data_y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        data_z=df_batch[["z0", "z2", "x", "y"]].to_numpy(),
    )

    if fastkci:
        kci = FastKCI_CInd()
    else:
        kci = KCI_CInd(est_width="median")
    pval3, _ = kci.compute_pvalue(
        data_x=df_batch[["z2"]].to_numpy(),
        data_y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        data_z=df_batch[["z0", "z1", "x", "y"]].to_numpy(),
    )

    score1 = abs(int(pval1 < alpha) - 1)
    score2 = int(pval2 < alpha)
    score3 = int(pval3 < alpha)
    score = score1 + score2 + score3
    return score, pval1, pval2, pval3


def comp_r2(df_batch, view_num=0):
    r2_z0 = pearsonr(df_batch["z0"], df_batch[f"z0_est{view_num}"])[0] ** 2
    r2_z1 = pearsonr(df_batch["z1"], df_batch[f"z0_est{view_num}"])[0] ** 2
    r2_z2 = pearsonr(df_batch["z2"], df_batch[f"z0_est{view_num}"])[0] ** 2
    return r2_z0, r2_z1, r2_z2


# Compute the average treatment effect (ATE) using linear regression
def comp_ate(df_batch, view_num=0):
    model = LinearRegression()
    model.fit(df_batch[[f"z0_est{view_num}", "x"]], df_batch["y"])
    ate = model.coef_[1]
    return ate


def load_and_eval_helper(
    batch_nums,
    exper_id,
    view_num=0,
    ci_method="pcm",
    fun_reg=LM(),
    alpha=0.05,
    rng=np.random.default_rng(),
):
    res = pd.DataFrame(
        columns=[
            "tmex_score",
            "pval1",
            "pval2",
            "pval3",
            "r2_z0",
            "r2_z1",
            "r2_z2",
            "ate",
        ]
    )

    df_all = pd.DataFrame()
    # Model C also loads Model A but corrupt the data below
    if exper_id == "five_latents_c":
        exper_id_load = "five_latents_a"
    else:
        exper_id_load = exper_id
    for batch_num in batch_nums:
        df = pd.read_csv(
            f"~/multiview-crl-eval/results/numerical/{exper_id_load}/ztrue_batch{batch_num}.csv",
            header=None,
        )
        df_est = pd.read_csv(
            f"~/multiview-crl-eval/results/numerical/{exper_id_load}/z0est_batch{batch_num}.csv",
            header=None,
        )
        df_batch = pd.concat([df, df_est], axis=1)
        df_batch.columns = ["z0", "z1", "z2", "x", "y", "z0_est0", "z0_est1"]

        # Model C: corrupted data
        if exper_id == "five_latents_c":
            df_batch["z0_est0"] += 0.2 * df_batch["z1"] - 0.1 * df_batch["z2"]
            df_batch["z0_est1"] += 0.2 * df_batch["z1"] - 0.1 * df_batch["z2"]

        df_batch["batch_num"] = batch_num
        df_all = pd.concat([df_all, df_batch], axis=0)

        if ci_method == "pcm":
            # compute T-MEX with PCM
            score, pval1, pval2, pval3 = pcm_batch(
                df_batch,
                view_num=view_num,
                fun_reg=fun_reg,
                alpha=alpha,
                rng=rng,
            )
        elif ci_method == "gcm":
            # compute T-MEX with GCM
            score, pval1, pval2, pval3 = gcm_batch(
                df_batch,
                view_num=view_num,
                fun_reg=fun_reg,
                alpha=alpha,
                rng=rng,
            )
        elif ci_method == "kci":
            # compute T-MEX with KCI
            score, pval1, pval2, pval3 = kci_batch(
                df_batch, fastkci=False, view_num=view_num, alpha=alpha
            )
        elif ci_method == "fastkci":
            # compute T-MEX with KCI
            score, pval1, pval2, pval3 = kci_batch(
                df_batch, fastkci=True, view_num=view_num, alpha=alpha
            )

        # compute ATE
        ate = comp_ate(df_batch, view_num=view_num)
        # compute R^2
        r2_z0, r2_z1, r2_z2 = comp_r2(df_batch, view_num=view_num)
        res.loc[batch_num] = [
            score,
            pval1,
            pval2,
            pval3,
            r2_z0,
            r2_z1,
            r2_z2,
            ate,
        ]
        df_all.to_csv(
            f"~/multiview-crl-eval/results/numerical/{exper_id}_all_data.csv"
        )
    return res


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ci_method", type=str, default="pcm")
    parser.add_argument("--reg", type=str, default="LM")
    parser.add_argument("--view_num", type=int, default=0)
    parser.add_argument("--batch_num", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(123)

    batch_nums = range(args.batch_num, args.batch_num + 1)

    if args.reg == "LM":
        fun_reg = LM()
    elif args.reg == "KRR":
        fun_reg = KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]})
    elif args.reg == "RF":
        fun_reg = RF(
            # n_estimators=20, max_depth=5, min_samples_leaf=10,
            n_jobs=-1
        )
    elif args.reg == "XGB":
        fun_reg = XGB(
            param_grid={
                "n_estimators": [5, 20, 50],
                "max_depth": [2, 5, 10],
                "subsample": [0.8],
                "colsample_bytree": [0.8],
                "colsample_bylevel": [1.0],
                "colsample_bynode": [1.0],
            },
            n_jobs=-1,
        )
    else:
        fun_reg = None

    ## Model A, B, C
    rng = np.random.default_rng(123)
    exper_id = "five_latents_a"
    res_model_a = load_and_eval_helper(
        batch_nums=batch_nums,
        exper_id=exper_id,
        view_num=args.view_num,
        ci_method=args.ci_method,
        fun_reg=copy.deepcopy(fun_reg),
        alpha=0.05,
        rng=rng,
    )
    res_model_a.to_csv(
        f"~/multiview-crl-eval/results/numerical/res_five_latents_a_view{args.view_num}_{args.ci_method}{args.reg}_batch{args.batch_num}.csv",
        index=False,
    )
    print(f"Saved results for {exper_id} with {args.reg} regression.")

    exper_id = "five_latents_b"
    res_model_b = load_and_eval_helper(
        batch_nums=batch_nums,
        exper_id=exper_id,
        view_num=args.view_num,
        ci_method=args.ci_method,
        fun_reg=copy.deepcopy(fun_reg),
        alpha=0.05,
        rng=rng,
    )
    res_model_b.to_csv(
        f"~/multiview-crl-eval/results/numerical/res_five_latents_b_view{args.view_num}_{args.ci_method}{args.reg}_batch{args.batch_num}.csv",
        index=False,
    )
    print(f"Saved results for {exper_id} with {args.reg} regression.")

    exper_id = "five_latents_c"
    res_model_c = load_and_eval_helper(
        batch_nums=batch_nums,
        exper_id=exper_id,
        view_num=args.view_num,
        ci_method=args.ci_method,
        fun_reg=copy.deepcopy(fun_reg),
        alpha=0.05,
        rng=rng,
    )
    res_model_c.to_csv(
        f"~/multiview-crl-eval/results/numerical/res_five_latents_c_view{args.view_num}_{args.ci_method}{args.reg}_batch{args.batch_num}.csv",
        index=False,
    )
    print(f"Saved results for {exper_id} with {args.reg} regression.")
