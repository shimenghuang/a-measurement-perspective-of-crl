# %%
import time

import numpy as np
import pandas as pd
import torch
from causallearn.utils.cit import CIT
from noise_generator import GaussianNoise
from run_tmex_demonstration import (
    artificial_measurement,
    gen_latent_data,
    gen_measurement_data_fullymixed_linear_nonoise,
    gen_measurement_data_perfect_linear_addnoise,
    gen_measurement_data_perfect_linear_nonoise,
    gen_measurement_data_perfect_nonlinear_addnoise,
    gen_measurement_data_perfect_nonlinear_nonoise,
    generate_adjmat,
    run_one_sim,
    tmex_blocks,
)
from scm import LocationScaleSCM
from utils import leaky_relu, leaky_sigmoid, leaky_tanh

from pycomets.gcm import GCM
from pycomets.regression import KRR, LM, RF, XGB

# %%

# def generate_latent_lscm(
#     B, n_samples, n_latent=20, rng=np.random.default_rng()
# ):
#     """
#     Generate a latent space with a given number of samples and latent dimensions.
#     """
#     IB = np.eye(n_latent) - B
#     IB_inv = np.linalg.inv(IB)
#     Sigma_Z = IB_inv @ IB_inv.T
#     Z = rng.multivariate_normal(np.zeros(n_latent), Sigma_Z, size=n_samples)
#     return Z

# %%

n_latent = 20
B = generate_adjmat(n_latent)
# noise_generator = GaussianNoise(
#     latent_dim=20,
#     intervention_targets_per_env=np.eye(20),
#     mean=0.0,
#     std=1.0,
#     shift=False,
#     shift_type="mean",
# )

# %%
# Model 1: Perfect correspondence

# perfect correspondence implies a diagonal adjacency matrix
W = np.eye(n_latent)

# Generate latent variables using a Location-Scale SCM
n_sample = 1000
noise_samples = torch.randn(n_sample, n_latent)
lsscm = LocationScaleSCM(
    adjacency_matrix=B,
    latent_dim=n_latent,
    intervention_targets_per_env=np.eye(n_latent),
)
Z = lsscm.push_forward(noise_samples, env=0)

# Linear measurement functions (no noise)
hZs = []
for ii in range(n_latent):
    hZs.append(artificial_measurement(Z[:, ii].reshape(-1, 1), lambda x: x))

start = time.time()
score, hW = tmex_blocks(
    W,
    Z,
    hZs,
    n_latent=n_latent,
    reg=LM(),
    alpha=0.05,
)
end = time.time()
print(f"Done linear; score: {score}")
print(f"Elapsed time: {end - start:.4f} seconds")

# Non-linear measurement functions (no noise) with KRR
hZs = []
for ii in range(n_latent):
    hZs.append(
        artificial_measurement(
            Z[:, ii].reshape(-1, 1), lambda x: 0.5 * x**2 + x
        )
    )

start = time.time()
score, hW = tmex_blocks(
    W,
    Z,
    hZs,
    n_latent=n_latent,
    reg=KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
    alpha=0.05,
)
end = time.time()
print(f"Done KRR; score: {score}")
print(f"Elapsed time: {end - start:.4f} seconds")

# Non-linear measurement functions (no noise) with XGB
start = time.time()
score, hW = tmex_blocks(
    W,
    Z,
    hZs,
    n_latent=n_latent,
    reg=XGB(param_grid={"n_estimators": [10, 50], "max_depth": [2, 5]}),
    alpha=0.05,
)
end = time.time()
print(f"Done XGB; score: {score}")
print(f"Elapsed time: {end - start:.4f} seconds")

# %%

hZ1 = artificial_measurement(
    Z[:, 0].reshape(-1, 1), lambda x: leaky_relu(x, 0.1)
)
W = np.zeros((1, n_latent))
W[0, 0] = 1  # Only the first latent variable is measured
score, hW = tmex_blocks(
    W,
    Z,
    [hZ1],
    n_latent=n_latent,
    reg=XGB(param_grid={"n_estimators": [10, 50], "max_depth": [2, 5]}),
    alpha=0.05,
)

# %%

n_samples = [100, 250]
n_sim = 2
n_rep = 2
n_latent = 10
res_lst = []
for n_sample in n_samples:

    print(f"** n_sample: {n_sample} **")

    # Perfect linear nonoise
    for ss in range(n_sim):
        res_df = run_one_sim(
            n_sample=n_sample,
            n_latent=n_latent,
            n_rep=n_rep,
            fun_gen_measurement_data=gen_measurement_data_perfect_linear_nonoise,
            fun_reg={
                "LM": LM(),
                "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
                "XGB": XGB(
                    param_grid={
                        "n_estimators": [10, 20],
                        "max_depth": [2, 5],
                        "subsample": [0.5],
                        "colsample_bytree": [0.5],
                        "colsample_bylevel": [0.5],
                        "colsample_bynode": [0.5],
                    }
                ),
            },
        )
        res_df["sim"] = ss
        res_lst.append(res_df)

    # Perfect nonlinear nonoise
    for ss in range(n_sim):
        res_df = run_one_sim(
            n_sample=n_sample,
            n_latent=n_latent,
            n_rep=n_rep,
            fun_gen_measurement_data=gen_measurement_data_perfect_nonlinear_nonoise,
            fun_reg={
                "LM": LM(),
                "KRR": KRR(kernel="rbf", param_grid={"alpha": [0.1, 1, 10]}),
                "XGB": XGB(
                    param_grid={
                        "n_estimators": [10, 50],
                        "max_depth": [2, 5],
                    }
                ),
            },
        )
        res_df["sim"] = ss
        res_lst.append(res_df)

res_df = pd.concat(res_lst, ignore_index=True)

# %%
# read and summarize results

n_samples = [1000]
n_latents = [50]
df_all = []
for n_latent in n_latents:
    for n_sample in n_samples:
        for sim_id in range(20):
            df = pd.read_csv(
                f"tmex_demo_results/lmonly_res_{n_sample}sample_{n_latent}latent_sim{sim_id}.csv"
            )
            df_all.append(df)
df_all = pd.concat(df_all, ignore_index=True)
print(
    df_all.groupby(["reg", "exp", "n_latent", "n_sample"])["score"]
    .agg(["mean", "std"])
    .reset_index()
)
print(
    df_all.groupby(["reg", "exp", "n_latent", "n_sample"])["time"]
    .agg(["mean", "std"])
    .reset_index()
)

# %%

# table 1

print(
    df_all[
        # (df_all["reg"] == "LM") &
        (df_all["exp"] == "perfect_linear_nonoise")
    ]
    .groupby(["reg", "exp", "n_latent", "n_sample"])["score"]
    .agg(["mean", "std"])
    .reset_index()
    .drop(columns=["reg", "exp"])
    .to_markdown(index=False, floatfmt=".4f")
)

print("-----")

print(
    df_all[
        # (df_all["reg"] == "LM") &
        (df_all["exp"] == "perfect_linear_nonoise")
    ]
    .groupby(["reg", "exp", "n_latent", "n_sample"])["time"]
    .agg(["mean", "std"])
    .reset_index()
    .drop(columns=["reg", "exp"])
    .to_markdown(index=False, floatfmt=".4f")
)

# table 8

print(
    df_all[
        (df_all["exp"] == "perfect_linear_nonoise")
        | (df_all["exp"] == "perfect_linear_addnoise")
    ]
    .groupby(["reg", "exp", "n_latent", "n_sample"])["score"]
    .agg(["mean", "std"])
    .reset_index()
    .to_markdown(index=False, floatfmt=".4f")
)

# %%
# %%

df = pd.read_csv("tmex_demo_results/small_lm_res_500sample_3latent.csv")
print(
    df.groupby("exp")[["score", "ate"]]
    .agg(["mean", "std"])
    .reset_index()
    .to_markdown(index=False, floatfmt=".4f")
)
# df[df["exp"] == "fullymixed_linear_nonoise"]

# %%
df = pd.read_csv("tmex_demo_results/three_latent_res_1000sample.csv")
df["abs_ate_bias"] = np.abs(df["ate"] - 1.0)
print(
    df.groupby("exp")[["tmex_score", "abs_ate_bias", "r2", "mcc"]]
    .agg(["mean", "std"])
    .reset_index()
    .to_markdown(index=False, floatfmt=".4f")
)

# %%
