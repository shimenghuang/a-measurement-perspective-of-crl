# %%

import numpy as np
import pandas as pd

view_num = 1

# # %%
# ci_method = "pcm"
# reg = "LM"
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_a_view{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_b_view{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_cview{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")

# # %%
# ci_method = "gcm"
# reg = "LM"
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_a_view{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_b_view{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")
# df = pd.read_csv(
#     f"results/numerical/res_five_latents_c_view{view_num}_{ci_method}{reg}.csv"
# )
# print(f"{df['tmex_score'].mean()} \\pm {df['tmex_score'].std()}")

# %%
ci_method = "kci"
reg = ""
df_a = []
df_b = []
df_c = []
for batch_num in range(50):
    df = pd.read_csv(
        f"results/numerical/res_five_latents_a_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_a.append(df)

    df = pd.read_csv(
        f"results/numerical/res_five_latents_b_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_b.append(df)

    df = pd.read_csv(
        f"results/numerical/res_five_latents_c_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_c.append(df)

df_a = pd.concat(df_a, ignore_index=True)
df_b = pd.concat(df_b, ignore_index=True)
df_c = pd.concat(df_c, ignore_index=True)
print(f"{df_a['tmex_score'].mean()} \\pm {df_a['tmex_score'].std()}")
print(f"{df_b['tmex_score'].mean()} \\pm {df_b['tmex_score'].std()}")
print(f"{df_c['tmex_score'].mean()} \\pm {df_c['tmex_score'].std()}")


# %%
ci_method = "pcm"
reg = "RF"
df_a = []
df_b = []
df_c = []
for batch_num in range(50):
    df = pd.read_csv(
        f"results/numerical/res_five_latents_a_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_a.append(df)

    df = pd.read_csv(
        f"results/numerical/res_five_latents_b_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_b.append(df)

    df = pd.read_csv(
        f"results/numerical/res_five_latents_c_view{view_num}_{ci_method}{reg}_batch{batch_num}.csv"
    )
    df_c.append(df)

df_a = pd.concat(df_a, ignore_index=True)
df_b = pd.concat(df_b, ignore_index=True)
df_c = pd.concat(df_c, ignore_index=True)
print(f"{df_a['tmex_score'].mean()} \\pm {df_a['tmex_score'].std()}")
print(f"{df_b['tmex_score'].mean()} \\pm {df_b['tmex_score'].std()}")
print(f"{df_c['tmex_score'].mean()} \\pm {df_c['tmex_score'].std()}")

# %%
# Recompute scores using different alpha values


def comp_score(x, alpha=0.05):
    score1 = abs(int(x["pval1"] < alpha) - 1)
    score2 = int(x["pval2"] < alpha)
    score3 = int(x["pval3"] < alpha)
    return score1 + score2 + score3


alpha = 0.05
scores = df_a.apply(lambda x: comp_score(x, alpha=alpha), axis=1).to_numpy()
print(f"Score for A: {scores.mean()} \\pm {scores.std()}")
scores = df_b.apply(lambda x: comp_score(x, alpha=alpha), axis=1).to_numpy()
print(f"Score for B: {scores.mean()} \\pm {scores.std()}")
scores = df_c.apply(lambda x: comp_score(x, alpha=alpha), axis=1).to_numpy()
print(f"Score for C: {scores.mean()} \\pm {scores.std()}")
# %%
