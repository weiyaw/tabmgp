# %%
import numpy as np
import utils
import re
import os
import matplotlib.pyplot as plt
import pandas as pd

np.set_printoptions(formatter={"float_kind": "{:.2e}".format}, linewidth=200, threshold=np.inf)
pd.set_option("display.max_rows", None)

def compile_outer(outer_dir):
    # Compile and produce a single outer
    FnQn_paths = utils.get_matching_files(outer_dir, r"FnQn-\d+\.pickle")
    FnQn_paths.sort(key=lambda x: int(re.search(r"FnQn-(\d+)", x).group(1)))
    FnQn_raw = [utils.read_from(p) for p in FnQn_paths]
    F_n = np.asarray([d["F_n"] for d in FnQn_raw])
    Q_n = np.asarray([d["Q_n"] for d in FnQn_raw])
    n = np.asarray([d["n"] for d in FnQn_raw])
    return F_n, Q_n, n


def compile_all_outer(outer_dirs):
    # Compile all outer and average
    # outer0_delta0 = utils.read_from(
    #     utils.get_matching_files(outer_dirs[0], r"FnQn-\d+\.pickle")[0]
    # )
    # t = outer0_delta0["t"]
    # u = outer0_delta0["u"]
    # x_new = outer0_delta0["x_new"]
    outer0 = utils.read_from(f"{outer_dirs[0]}/Fn.pickle")
    t = outer0["t"]
    # u = outer0["u"]
    x_new = outer0["x_new"]

    F_n_all = []
    # Q_n_all = []
    n_all = []
    for d in outer_dirs:
        # F_n, Q_n, n = compile_outer(d)
        outer = utils.read_from(f"{d}/Fn.pickle")
        F_n_all.append(outer["F_n"])
        # Q_n_all.append(outer["Q_n"])
        n_all.append(outer["n"])
    F_n_all = np.stack(F_n_all) # (outer, n, t, x_new)
    # Q_n_all = np.stack(Q_n_all) # (outer, n, t, x_new)
    n_all = np.stack(n_all)  # (outer, n)
    assert np.all(n_all == n_all[0])
    n = n_all[0]
    return F_n_all, None, n, t, u, x_new


def plot_FF(ax, F_n, k):
    colors = plt.cm.viridis((k - k.min()) / (k.max() - k.min()))
    assert len(colors) == F_n.shape[0]
    for i, color in enumerate(colors):
        ax.plot(F_n[0], F_n[i], color=color, alpha=0.5)
    ax.plot([F_n.min(), F_n.max()], [F_n.min(), F_n.max()], color="black", linestyle="--")
    ax.grid()


def plot_QQ(ax, Q_n, k):
    colors = plt.cm.viridis((k - k.min()) / (k.max() - k.min()))
    for i, color in enumerate(colors):
        ax.plot(Q_n[0], Q_n[i], color=color, alpha=0.5)
    ax.plot([Q_n.min(), Q_n.max()], [Q_n.min(), Q_n.max()], color="black", linestyle="--")
    ax.grid()


# %%
id_dir = "./outputs/debug/"
image_dir = "../paper/images/"
n0 = 25
x_new_idx = 1

n_est_ls = [1]
n_reps = 1
# n_reps = 2
fig1, ax1 = plt.subplots(n_reps, 1, figsize=(12, 2.5 * n_reps), constrained_layout=True)
# fig2, ax2 = plt.subplots(n_reps, 4, figsize=(12, 2.5 * n_reps), constrained_layout=True)

# Handle single axis case if n_reps and n_est_ls are 1
if n_reps == 1 and len(n_est_ls) == 1:
    ax1 = np.array([[ax1]])

for col_idx, n_est in enumerate(n_est_ls):
    rep_dirs = utils.get_matching_dirs(id_dir, rf"cid.+n_est={n_est}")
    # assert len(rep_dirs) == n_reps
    rep_dirs.sort()

    for row_idx, d in enumerate(rep_dirs):
        if row_idx >= n_reps: break
        outer_dirs = utils.get_matching_dirs(d, r"outer-\d+")
        F_n_all, Q_n_all, n, t, u, x_new = compile_all_outer(outer_dirs)
        # F_n_all and Q_n_all are (outer, n, t, x_new)


        # I FORGOT TO RECORD N IN run-cid.py. HARDCODE VALUE AS A WORKAROUND
        # n = np.rint(np.linspace(25, 25 + 1000, 101)).astype(int)

        F_n = np.mean(F_n_all[..., x_new_idx], axis=0) # (n, t)
        # Q_n = np.mean(Q_n_all[..., x_new_idx], axis=0) # (n, t)
        k = n - n0

        a1 = ax1[row_idx, col_idx]
        # a2 = ax2[row_idx, col_idx]

        plot_FF(a1, F_n, k)
        # plot_QQ(a2, Q_n, k)

        if row_idx == 0:
            a1.set_title(f"n_estimators = {n_est}")
            # a2.set_title(f"n_estimators = {n_est}")
        
        if col_idx == 0:
            a1.set_ylabel(r"$\bar{F}_{n_0 + k}(t)$")
            # a2.set_ylabel(r"$\bar{F}^{-1}_{n_0 + k}(u)$")

        if row_idx == n_reps - 1:
            a1.set_xlabel(r"$F_{n_0}(t)$")
            # a2.set_xlabel(r"$F^{-1}_{n_0}(u)$")
            
fig1.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax1[:, -1],
)
# fig2.colorbar(
#     plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
#     label="k",
#     ax=ax2[:, -1],
# )

if not os.path.exists(image_dir):
    os.makedirs(image_dir)
# fig1.savefig(f"{image_dir}/cid-pp.pdf")
# fig2.savefig(f"{image_dir}/cid-qq.pdf")

# %%
