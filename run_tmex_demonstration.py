import time

import numpy as np
import pandas as pd
import torch
from causallearn.utils.cit import CIT
from lightning import seed_everything
from noise_generator import GaussianNoise
from scm import LocationScaleSCM
from sklearn.linear_model import LinearRegression
from utils import leaky_relu, leaky_sigmoid, leaky_tanh

from pycomets.gcm import GCM
from pycomets.regression import KRR, LM, RF, XGB


def generate_adjmat(n, rng=np.random.default_rng()):
    """
    Generate an n x n random strict lower triangular matrix with entries 0 or 1.

    Parameters:
        n (int): Size of the square matrix.
        seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: An n x n strict lower triangular matrix with 0/1 entries.
    """
    mat = np.zeros((n, n), dtype=int)
    lower_indices = np.tril_indices(n, k=-1)
    mat[lower_indices] = rng.integers(0, 2, size=len(lower_indices[0]))
    return mat


def artificial_measurement(Z, fun):
    Z = Z.clone()
    for i in range(Z.shape[1]):
        Z[:, i] = fun(Z[:, i])
    return Z


def tmex_blocks(W, Z, hZs, n_latent=20, reg=LM(), alpha=0.05):
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
    return np.sum(hW != W), hW


def gen_latent_data(n_sample, n_latent, B=None):
    """Generate latent data using a Location-Scale SCM."""
    if B is None:
        B = generate_adjmat(n_latent)
    noise_samples = torch.randn(n_sample, n_latent)
    lsscm = LocationScaleSCM(
        adjacency_matrix=B,
        latent_dim=n_latent,
        intervention_targets_per_env=np.eye(n_latent),
    )
    Z = lsscm.push_forward(noise_samples, env=0)
    return Z, B


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
                Z[:, ii].reshape(-1, 1), lambda x: 0.5 * x**2 + x
            )
        )
    return hZs


def gen_measurement_data_perfect_nonlinear_addnoise(Z, sd=0.2):
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
    B=None,
):
    res_lst = []
    for ii in range(n_rep):
        Z, _ = gen_latent_data(n_sample, n_latent, B=B)
        W = np.eye(n_latent)  # Perfect correspondence
        hZs = fun_gen_measurement_data(Z)
        for reg_name, reg in fun_reg.items():
            start = time.time()
            score, hW = tmex_blocks(
                W,
                Z,
                hZs,
                n_latent=n_latent,
                reg=reg,
                alpha=0.05,
            )
            end = time.time()
            res_tmp = pd.DataFrame(
                [
                    {
                        "n_sample": n_sample,
                        "n_latent": n_latent,
                        "rep": ii,
                        "score": score,
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
    parser.add_argument("--n_latent", type=int, required=True)
    parser.add_argument("--sim_id", type=int, required=True)
    parser.add_argument("--n_rep", type=int, default=2)
    parser.add_argument("--n_sample", type=int, default=250)
    args = parser.parse_args()

    seed_everything(123 + args.sim_id, workers=True)

    res_lst = []

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=args.n_latent,
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
    res_df["sim"] = args.sim_id
    res_df["exp"] = "perfect_linear_nonoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=args.n_latent,
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
    res_df["sim"] = args.sim_id
    res_df["exp"] = "perfect_linear_addnoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=args.n_latent,
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
    res_df["sim"] = args.sim_id
    res_df["exp"] = "perfect_nonlinear_nonoise"
    res_lst.append(res_df)

    res_df = run_one_sim(
        n_sample=args.n_sample,
        n_latent=args.n_latent,
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
    res_df["sim"] = args.sim_id
    res_df["exp"] = "fullymixed_linear_nonoise"
    res_lst.append(res_df)

    res_df = pd.concat(res_lst, ignore_index=True)
    res_df.to_csv(
        f"tmex_demo_results/lmonly_res_{args.n_sample}sample_{args.n_latent}latent_sim{args.sim_id}.csv",
        index=False,
    )
