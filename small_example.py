# %%
import copy

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr

from pycomets.pcm import PCM
from pycomets.regression import KRR, LM, RF

# %%
n_tests = 100
N = 10_00
# alphas = np.random.uniform(0.01, 0.1, size=(n_tests, 3))
# betas = np.random.uniform(
#     0.005, 0.02, size=(n_tests, 2)
# )  # 0.01*np.ones((n_tests,2)) #
alphas = np.random.uniform(0.01, 0.1, 3)
betas = np.array([0.01, 0.01])


def generate_df(alphas, betas, N=10_000):
    Z1 = np.random.randn(N)
    Z2 = alphas[0] * Z1 + betas[0] * np.random.randn(N)  # Z1 -> Z2
    Z3 = (
        alphas[1] * Z1 + alphas[2] * Z2 + betas[1] * np.random.randn(N)
    )  # Z1 -> Z3; Z2 -> Z3
    # 2) Create a "mixed" representation
    #    Note how each learned dimension is a combination of the true latents
    # hatZ1 = alphas[3] * Z1 + alphas[4] * Z2
    # hatZ2 = alphas[5] * Z2
    # hatZ3 = alphas[6] * Z3
    hatZ1 = Z1 + Z2
    hatZ2 = Z2
    hatZ3 = Z3
    data = np.column_stack([Z1, Z2, Z3, hatZ1, hatZ2, hatZ3])
    df = pd.DataFrame(
        data, columns=["z1", "z2", "z3", "hat_z1", "hat_z2", "hat_z3"]
    )
    return df


def comp_tmex(df, fun_reg=LM(), alpha=0.05):
    W = np.eye(3)  # Perfect correspondence
    hW = np.zeros((3, 3))
    # go through each block
    for ii in range(3):
        # go through each latent
        for jj in range(3):
            pcm = PCM()
            pcm.test(
                reg_yonxz=copy.deepcopy(fun_reg),
                reg_ronz=copy.deepcopy(fun_reg),
                reg_vonxz=copy.deepcopy(fun_reg),
                reg_yhatonz=copy.deepcopy(fun_reg),
                reg_yonz=copy.deepcopy(fun_reg),
                X=df[[f"z{jj+1}"]].to_numpy(),
                Y=df[[f"hat_z{ii+1}"]].to_numpy(),
                Z=df.drop(
                    columns=["hat_z1", "hat_z2", "hat_z3"] + [f"z{jj+1}"]
                ).to_numpy(),
                estimate_variance=False,
                rep=9,
                rng=np.random.default_rng(),
                show_summary=False,
            )
            hW[ii, jj] = pcm.pval < alpha

    return np.sum(hW != W), hW


def comp_mcc(z, hz, k):
    # z = z.detach().cpu().numpy()
    # hz = hz.detach().cpu().numpy()
    cor_abs = np.abs(np.corrcoef(z.T, hz.T))[:k, k:]

    assignments = linear_sum_assignment(-1 * cor_abs)
    maxcor = cor_abs[assignments].sum() / k
    return maxcor, cor_abs


# def comp_r2(df_batch):
#     r2_z0 = pearsonr(df_batch["z1"], df_batch["hat_z1"])[0] ** 2
#     r2_z1 = pearsonr(df_batch["z2"], df_batch["hat_z1"])[0] ** 2
#     r2_z2 = pearsonr(df_batch["z3"], df_batch["hat_z1"])[0] ** 2
#     return r2_z0, r2_z1, r2_z2


def comp_r2(df_batch, n_latent=3, n_measure=3):
    r2_mat = np.zeros((n_measure, n_latent))
    for ii in range(n_measure):
        for jj in range(n_latent):
            r2_mat[ii, jj] = (
                pearsonr(df_batch[f"hat_z{ii+1}"], df_batch[f"z{jj+1}"])[0] ** 2
            )
    return r2_mat.max(axis=1).mean()


tmex_scores = []
r2_scores = []
mcc_scores = []
for i in range(n_tests):
    print(f"Test {i + 1}/{n_tests}")
    df = generate_df(alphas, betas, N)
    tmex_score, hW = comp_tmex(df, fun_reg=LM())
    print(f"T-MEX: {tmex_score}, hW:\n{hW}")
    tmex_scores.append(tmex_score)
    r2_score = comp_r2(df, n_measure=3, n_latent=3)
    print(f"R2: {r2_score}")
    r2_scores.append(r2_score)
    mcc_score, _ = comp_mcc(
        df[["z1", "z2", "z3"]].to_numpy(),
        df[["hat_z1", "hat_z2", "hat_z3"]].to_numpy(),
        3,
    )
    print(f"MCC: {mcc_score}")
    mcc_scores.append(mcc_score.item())

print(
    f"Average T-MEX score: {np.mean(tmex_scores):.4f} ± {np.std(tmex_scores):.4f}"
)


print(f"Average MR2 score: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# print(
#     pd.DataFrame(
#         {
#             "r2 mean": np.mean(r2_scores, axis=0),
#             "r2 std": np.std(r2_scores, axis=0),
#         }
#     ).to_markdown(index=False, floatfmt=".4f")
# )

print(
    f"Average MCC score: {np.mean(mcc_scores):.4f} ± {np.std(mcc_scores):.4f}"
)

# %%
