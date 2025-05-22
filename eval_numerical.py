import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

from pycomets.pcm import PCM
from pycomets.regression import LM

np.random.seed(123)


# T-MEX score for this simulated experiment
def tmex_batch(df_batch, view_num=0, alpha=0.05, rng=np.random.default_rng()):
    pcm = PCM()
    pcm.test(
        reg_yonxz=LM(),
        reg_ronz=LM(),
        reg_vonxz=LM(),
        reg_yhatonz=LM(),
        reg_yonz=LM(),
        X=df_batch[["z0"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z1", "z2", "x", "y"]].to_numpy(),
        estimate_variance=True,
        rep=9,
        rng=rng,
    )
    pval1 = pcm.pval
    pcm = PCM()
    pcm.test(
        reg_yonxz=LM(),
        reg_ronz=LM(),
        reg_vonxz=LM(),
        reg_yhatonz=LM(),
        reg_yonz=LM(),
        X=df_batch[["z1"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z2", "x", "y"]].to_numpy(),
        estimate_variance=True,
        rep=9,
        rng=rng,
    )
    pval2 = pcm.pval
    pcm = PCM()
    pcm.test(
        reg_yonxz=LM(),
        reg_ronz=LM(),
        reg_vonxz=LM(),
        reg_yhatonz=LM(),
        reg_yonz=LM(),
        X=df_batch[["z2"]].to_numpy(),
        Y=df_batch[[f"z0_est{view_num}"]].to_numpy(),
        Z=df_batch[["z0", "z1", "x", "y"]].to_numpy(),
        estimate_variance=True,
        rep=9,
        rng=rng,
    )
    pval3 = pcm.pval
    score1 = abs(int(pval1 < alpha) - 1)
    score2 = int(pval2 < alpha)
    score3 = int(pval3 < alpha)
    tmex_score = score1 + score2 + score3
    return tmex_score, pval1, pval2, pval3


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
    batch_nums, exper_id, view_num=0, alpha=0.05, rng=np.random.default_rng()
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

        # compute T-MEX
        tmex_score, pval1, pval2, pval3 = tmex_batch(
            df_batch, view_num=view_num, alpha=0.05, rng=rng
        )
        # compute ATE
        ate = comp_ate(df_batch, view_num=view_num)
        # compute R^2
        r2_z0, r2_z1, r2_z2 = comp_r2(df_batch, view_num=view_num)
        res.loc[batch_num] = [
            tmex_score,
            pval1,
            pval2,
            pval3,
            r2_z0,
            r2_z1,
            r2_z2,
            ate,
        ]
    return res


## Model A, B, C
rng = np.random.default_rng(123)
exper_id = "five_latents_a"
view_num = 1
res_model_a = load_and_eval_helper(
    batch_nums=range(50), exper_id=exper_id, view_num=view_num, alpha=0.05
)
res_model_a.to_csv(
    "~/multiview-crl-eval/results/numerical/res_five_latents_a.csv", index=False
)

exper_id = "five_latents_b"
res_model_b = load_and_eval_helper(
    batch_nums=range(50), exper_id=exper_id, view_num=view_num, alpha=0.05
)
res_model_b.to_csv(
    "~/multiview-crl-eval/results/numerical/res_five_latents_b.csv", index=False
)

exper_id = "five_latents_c"
res_model_c = load_and_eval_helper(
    batch_nums=range(50), exper_id=exper_id, view_num=view_num, alpha=0.05
)
res_model_c.to_csv(
    "~/multiview-crl-eval/results/numerical/res_five_latents_c.csv", index=False
)
