import time

import numpy as np
import pandas as pd
import torch
from causallearn.utils.cit import CIT
from lightning import seed_everything
from noise_generator import GaussianNoise
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr
from scm import LocationScaleSCM
from sklearn.linear_model import LinearRegression
from utils import leaky_relu, leaky_sigmoid, leaky_tanh

from pycomets.gcm import GCM
from pycomets.regression import KRR, LM, RF, XGB


def artificial_measurement(Z, fun):
    Z = Z.clone()
    for i in range(Z.shape[1]):
        Z[:, i] = fun(Z[:, i])
    return Z


def tmex_blocks(W, Z, hZs, n_latent=3, reg=LM(), alpha=0.05):
    gcm = GCM()
    hW = np.zeros((len(hZs), n_latent))
    for ii, hZ in enumerate(hZs):
        for jj in range(n_latent):
            gcm.test(
                Y=hZ.detach().numpy(),
                X=Z[:, jj].detach().numpy(),
                Z=Z[:, np.setdiff1d(np.arange(n_latent), jj)].detach().numpy(),
                reg_yz=reg,
                reg_xz=reg,
                test_type="max",
                show_summary=False,
            )
            hW[ii, jj] = gcm.pval < alpha
    # print("hW:\n", hW)

    # ATE from Z1 to Z2 conditioning on hZ0
    model = LinearRegression()
    X = np.column_stack(
        (Z[:, 1].detach().numpy().reshape(-1, 1), hZs[0].detach().numpy())
    )
    # X = Z[:, :2]
    model.fit(X=X, y=Z[:, 2].detach().numpy())
    ate = model.coef_[0]

    return np.sum(hW != W), hW, ate


def gen_latent_data(n_sample):
    """Generate data from a partially-linear SCM."""
    Z1 = torch.randn(size=(n_sample,))
    Z2 = leaky_tanh(Z1) + torch.randn(size=(n_sample,)) * 0.5
    Z3 = Z2 + leaky_tanh(Z1) + torch.randn(size=(n_sample,)) * 0.5
    Z = torch.stack([Z1, Z2, Z3], dim=1)
    return Z


def gen_measurement_data_perfect_linear_nonoise(Z):
    hZs = []
    for ii in range(Z.shape[1]):
        hZs.append(artificial_measurement(Z[:, ii].reshape(-1, 1), lambda x: x))
    return hZs


def gen_measurement_data_perfect_linear_addnoise(Z, sd=0.2):
    hZs = []
    for ii in range(Z.shape[1]):
        hZs.append(
            artificial_measurement(
                Z[:, ii].reshape(-1, 1), lambda x: x + torch.randn_like(x) * sd
            )
        )
    return hZs


def gen_measurement_data_perfect_nonlinear_nonoise(Z):
    hZs = []
    for ii in range(Z.shape[1]):
        hZs.append(
            artificial_measurement(
                # Z[:, ii].reshape(-1, 1), lambda x: 0.5 * x**2 + x
                Z[:, ii].reshape(-1, 1),
                lambda x: leaky_sigmoid(x, beta=0),
            )
        )
    return hZs


def gen_measurement_data_perfect_nonlinear_addnoise(Z, sd=1):
    hZs = []
    for ii in range(Z.shape[1]):
        hZs.append(
            artificial_measurement(
                Z[:, ii].reshape(-1, 1),
                lambda x: 0.5 * x**2 + x + torch.randn_like(x) * sd,
            )
        )
    return hZs


def gen_measurement_data_fullymixed_linear_nonoise(Z):
    hZs = []
    for ii in range(Z.shape[1]):
        hZs.append(
            torch.tensor(
                0.8 * Z[:, ii].reshape(-1, 1)
                + 0.2 * Z.sum(axis=1).reshape(-1, 1)
            )
        )
        # hZs.append(Z @ np.repeat(0.1, Z.shape[1]))
    return hZs


def gen_measurement_data_mixed12_linear_nonoise(Z):
    hZs = []
    for ii in range(Z.shape[1]):
        if ii == 0:
            hZs.append(
                torch.tensor(
                    0.5 * Z[:, 0].reshape(-1, 1) + 0.5 * Z[:, 1].reshape(-1, 1)
                )
            )
        else:
            hZs.append(Z[:, ii].reshape(-1, 1))
    return hZs


def comp_mcc(z, hz, k):
    # z = z.detach().cpu().numpy()
    # hz = hz.detach().cpu().numpy()
    cor_abs = np.abs(np.corrcoef(z.T, hz.T))[:k, k:]

    assignments = linear_sum_assignment(-1 * cor_abs)
    maxcor = cor_abs[assignments].sum() / k
    return maxcor, cor_abs


def comp_r2(df_batch, n_latent=3, n_measure=3):
    r2_mat = np.zeros((n_measure, n_latent))
    for ii in range(n_measure):
        for jj in range(n_latent):
            r2_mat[ii, jj] = (
                pearsonr(df_batch[f"hat_z{ii+1}"], df_batch[f"z{jj+1}"])[0] ** 2
            )
    return r2_mat.max(axis=1).mean()
    # r2_z0 = pearsonr(df_batch["z1"], df_batch["hat_z1"])[0] ** 2
    # r2_z1 = pearsonr(df_batch["z2"], df_batch["hat_z1"])[0] ** 2
    # r2_z2 = pearsonr(df_batch["z3"], df_batch["hat_z1"])[0] ** 2
    # return r2_z0, r2_z1, r2_z2


def run_one_sim(
    n_sample,
    n_latent,
    n_rep,
    fun_gen_measurement_data,
    fun_reg={
        "LM": LM(),
        "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
        "XGB": XGB(param_grid={"n_estimators": [10, 50], "max_depth": [2, 5]}),
    },
):
    res_lst = []
    for ii in range(n_rep):
        Z = gen_latent_data(n_sample)
        W = np.eye(n_latent)  # Perfect correspondence
        hZs = fun_gen_measurement_data(Z)
        for reg_name, reg in fun_reg.items():
            start = time.time()
            score, hW, ate = tmex_blocks(
                W,
                Z,
                hZs,
                n_latent=n_latent,
                reg=reg,
                alpha=0.05,
            )
            end = time.time()
            df = pd.DataFrame(
                np.column_stack(hZs),
                columns=[f"hat_z{i+1}" for i in range(n_latent)],
            )
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        Z.numpy(), columns=[f"z{i+1}" for i in range(n_latent)]
                    ),
                ],
                axis=1,
            )
            r2 = comp_r2(df, n_measure=3, n_latent=3)
            mcc, _ = comp_mcc(Z, np.column_stack(hZs), n_latent)
            res_tmp = pd.DataFrame(
                [
                    {
                        "n_sample": n_sample,
                        "n_latent": n_latent,
                        "rep": ii,
                        "tmex_score": score,
                        "r2": r2,
                        # "r2z0": r2[0],
                        # "r2z1": r2[1],
                        # "r2z2": r2[2],
                        "mcc": mcc,
                        "ate": ate,
                        "time": end - start,
                        "reg": reg_name,
                    }
                ]
            )
            res_lst.append(res_tmp)
    res_df = pd.concat(res_lst, ignore_index=True)
    return res_df


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rep", type=int, default=50)
    parser.add_argument("--n_sample", type=int, default=500)
    args = parser.parse_args()
    n_latent = 3

    seed_everything(123)

    res_lst = []

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=n_latent,
        n_rep=args.n_rep,
        fun_gen_measurement_data=gen_measurement_data_perfect_linear_nonoise,
        fun_reg={
            "LM": LM(),
            # "RF": RF(
            #     n_estimators=30, max_depth=5, min_samples_leaf=10, n_jobs=-1
            # ),
            # "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
            # "XGB": XGB(
            #     param_grid={
            #         "n_estimators": [10, 20],
            #         "max_depth": [2, 5],
            #         "subsample": [0.5],
            #         "colsample_bytree": [0.5],
            #         "colsample_bylevel": [0.5],
            #         "colsample_bynode": [0.5],
            #     }
            # ),
        },
    )
    res_df["exp"] = "perfect_linear_nonoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=n_latent,
        n_rep=args.n_rep,
        fun_gen_measurement_data=gen_measurement_data_perfect_linear_addnoise,
        fun_reg={
            "LM": LM(),
            # "RF": RF(
            #     n_estimators=30, max_depth=5, min_samples_leaf=10, n_jobs=-1
            # ),
            # "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
            # "XGB": XGB(
            #     param_grid={
            #         "n_estimators": [10, 20],
            #         "max_depth": [2, 5],
            #         "subsample": [0.5],
            #         "colsample_bytree": [0.5],
            #         "colsample_bylevel": [0.5],
            #         "colsample_bynode": [0.5],
            #     }
            # ),
        },
    )
    res_df["exp"] = "perfect_linear_addnoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=n_latent,
        n_rep=args.n_rep,
        fun_gen_measurement_data=gen_measurement_data_perfect_nonlinear_nonoise,
        fun_reg={
            "LM": LM(),
            # "RF": RF(
            #     n_estimators=30, max_depth=5, min_samples_leaf=10, n_jobs=-1
            # ),
            # "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
            # "XGB": XGB(
            #     param_grid={
            #         "n_estimators": [10, 20],
            #         "max_depth": [2, 5],
            #         "subsample": [0.5],
            #         "colsample_bytree": [0.5],
            #         "colsample_bylevel": [0.5],
            #         "colsample_bynode": [0.5],
            #     }
            # ),
        },
    )
    res_df["exp"] = "perfect_nonlinear_nonoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=n_latent,
        n_rep=args.n_rep,
        fun_gen_measurement_data=gen_measurement_data_fullymixed_linear_nonoise,
        fun_reg={
            "LM": LM(),
            # "RF": RF(
            #     n_estimators=30, max_depth=5, min_samples_leaf=10, n_jobs=-1
            # ),
            # "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
            # "XGB": XGB(
            #     param_grid={
            #         "n_estimators": [10, 20],
            #         "max_depth": [2, 5],
            #         "subsample": [0.5],
            #         "colsample_bytree": [0.5],
            #         "colsample_bylevel": [0.5],
            #         "colsample_bynode": [0.5],
            #     }
            # ),
        },
    )
    res_df["exp"] = "fullymixed_linear_nonoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=n_latent,
        n_rep=args.n_rep,
        fun_gen_measurement_data=gen_measurement_data_mixed12_linear_nonoise,
        fun_reg={
            "LM": LM(),
            # "RF": RF(
            #     n_estimators=30, max_depth=5, min_samples_leaf=10, n_jobs=-1
            # ),
            # "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
            # "XGB": XGB(
            #     param_grid={
            #         "n_estimators": [10, 20],
            #         "max_depth": [2, 5],
            #         "subsample": [0.5],
            #         "colsample_bytree": [0.5],
            #         "colsample_bylevel": [0.5],
            #         "colsample_bynode": [0.5],
            #     }
            # ),
        },
    )
    res_df["exp"] = "mixed12_linear_nonoise"
    res_lst.append(res_df)

    res_df = pd.concat(res_lst, ignore_index=True)
    res_df.to_csv(
        f"tmex_demo_results/three_latent_res_{args.n_sample}sample.csv",
        index=False,
    )
