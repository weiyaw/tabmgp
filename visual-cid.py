# %%
import numpy as np
import utils
import re
import os
import matplotlib.pyplot as plt

np.set_printoptions(
    formatter={"float_kind": "{:.2e}".format}, linewidth=200, threshold=np.inf
)


def compile_all_Fn(sample_dirs):
    # Compile all samples of Fn
    sample0 = utils.read_from(f"{sample_dirs[0]}/Fn.pickle")
    t = sample0["t"]
    x_new = sample0["x_new"]

    F_n_all = []
    n_all = []
    for d in sample_dirs:
        sample = utils.read_from(f"{d}/Fn.pickle")
        F_n_all.append(sample["F_n"])
        n_all.append(sample["n"])
    F_n_all = np.stack(F_n_all)  # (samples, n, t, x_new)
    n_all = np.stack(n_all)  # (samples, n)
    assert np.all(n_all == n_all[0])
    n = n_all[0]
    return F_n_all, n, t, x_new


def compile_all_Qn(sample_dirs):
    # Compile all samples of Qn
    sample0 = utils.read_from(f"{sample_dirs[0]}/Qn.pickle")
    u = sample0["u"]
    x_new = sample0["x_new"]

    Q_n_all = []
    n_all = []
    for d in sample_dirs:
        sample = utils.read_from(f"{d}/Qn.pickle")
        Q_n_all.append(sample["Q_n"])
        n_all.append(sample["n"])
    Q_n_all = np.stack(Q_n_all)  # (samples, n, t, x_new)
    n_all = np.stack(n_all)  # (samples, n)
    assert np.all(n_all == n_all[0])
    n = n_all[0]
    return Q_n_all, n, u, x_new


def plot_FF(ax, F_n, n):
    # F_n is (n, t)
    assert F_n.shape[0] == len(n)
    colors = plt.cm.viridis((n - n.min()) / (n.max() - n.min()))
    for i, color in enumerate(colors):
        ax.plot(F_n[0], F_n[i], color=color, alpha=0.5)
    ax.plot([0, 1], [0, 1], color="black", linestyle="--")
    ax.grid()


def plot_QQ(ax, Q_n, n):
    # Q_n is (n, t)
    assert Q_n.shape[0] == len(n)
    colors = plt.cm.viridis((n - n.min()) / (n.max() - n.min()))
    for i, color in enumerate(colors):
        ax.plot(Q_n[0], Q_n[i], color=color, alpha=0.5)
    ax.plot(
        [Q_n.min(), Q_n.max()], [Q_n.min(), Q_n.max()], color="black", linestyle="--"
    )
    ax.grid()


# %%
id_dir = "../outputs/2025-08-01/"
image_dir = "../paper/images/"
# if not os.path.exists(image_dir):
#     os.makedirs(image_dir)
n0 = 25
x_new_idx_ls = [0, 1, 2, 3, 4]
n_est = 8
n_reps = 10

# %%
fig1, ax1 = plt.subplots(
    n_reps, len(x_new_idx_ls), figsize=(12, 1.5 * n_reps), constrained_layout=True
)
fig2, ax2 = plt.subplots(
    n_reps, len(x_new_idx_ls), figsize=(12, 1.5 * n_reps), constrained_layout=True
)
for col_idx, x_new_idx in enumerate(x_new_idx_ls):
    rep_dirs = utils.get_matching_dirs(
        id_dir, rf"cid.+dgp=regression-fixed.+n_est={n_est} "
    )
    assert len(rep_dirs) == n_reps
    rep_dirs.sort()

    for row_idx, d in enumerate(rep_dirs):
        sample_dirs = utils.get_matching_dirs(d, r"sample-\d+")

        # F_n_all is (samples, n, t, x_new)
        F_n_all, n, t, x_new = compile_all_Fn(sample_dirs)
        F_n = np.mean(F_n_all[..., x_new_idx], axis=0)  # (n, t)
        k = n - n0
        a1 = ax1[row_idx, col_idx]
        plot_FF(a1, F_n, k)
        if row_idx == 0:
            a1.set_title(rf"$x^\ast = {x_new[x_new_idx].item():.1f}$")
        if col_idx == 0:
            a1.set_ylabel(rf"$\bar{{F}}_{{n_0 + k}}(t)$, Rep {row_idx + 1}")
        if row_idx == n_reps - 1:
            a1.set_xlabel(r"$F_{n_0}(t)$")

        # Q_n_all is (samples, n, t, x_new)
        Q_n_all, n, u, x_new = compile_all_Qn(sample_dirs)
        Q_n = np.mean(Q_n_all[..., x_new_idx], axis=0)  # (n, t)
        k = n - n0
        a2 = ax2[row_idx, col_idx]
        plot_QQ(a2, Q_n, k)
        if row_idx == 0:
            a2.set_title(rf"$x^\ast = {x_new[x_new_idx].item():.1f}$")
        if col_idx == 0:
            a2.set_ylabel(rf"$\bar{{Q}}_{{n_0 + k}}(t)$, Rep {row_idx + 1}")
        if row_idx == n_reps - 1:
            a2.set_xlabel(r"$Q_{n_0}(t)$")
        # a2.set_xlim(-2, 2) # around 0.95 of N(0, 1)
        # a2.set_ylim(-2, 2)

fig1.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax1[:, -1],
)
fig2.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax2[:, -1],
)

fig1.savefig(f"{image_dir}/cid-regression-pp.pdf")
fig2.savefig(f"{image_dir}/cid-regression-qq.pdf")

# %%

fig1, ax1 = plt.subplots(
    n_reps, len(x_new_idx_ls), figsize=(12, 1.5 * n_reps), constrained_layout=True
)
fig2, ax2 = plt.subplots(
    n_reps, len(x_new_idx_ls), figsize=(12, 1.5 * n_reps), constrained_layout=True
)
for col_idx, x_new_idx in enumerate(x_new_idx_ls):
    rep_dirs = utils.get_matching_dirs(id_dir, rf"cid.+dgp=gamma.+n_est={n_est} ")
    assert len(rep_dirs) == n_reps
    rep_dirs.sort()

    for row_idx, d in enumerate(rep_dirs):
        sample_dirs = utils.get_matching_dirs(d, r"sample-\d+")

        # F_n_all is (samples, n, t, x_new)
        F_n_all, n, t, x_new = compile_all_Fn(sample_dirs)
        F_n = np.mean(F_n_all[..., x_new_idx], axis=0)  # (n, t)
        k = n - n0
        a1 = ax1[row_idx, col_idx]
        plot_FF(a1, F_n, k)
        if row_idx == 0:
            a1.set_title(rf"$x = {x_new[x_new_idx].item():.2f}$")
        if col_idx == 0:
            a1.set_ylabel(rf"$\bar{{F}}_{{n_0 + k}}(t)$, Rep {row_idx + 1}")
        if row_idx == n_reps - 1:
            a1.set_xlabel(r"$F_{n_0}(t)$")

        # Q_n_all is (samples, n, t, x_new)
        Q_n_all, n, u, x_new = compile_all_Qn(sample_dirs)
        Q_n = np.mean(Q_n_all[..., x_new_idx], axis=0)  # (n, t)
        k = n - n0
        a2 = ax2[row_idx, col_idx]
        plot_QQ(a2, Q_n, k)
        if row_idx == 0:
            a2.set_title(rf"$x = {x_new[x_new_idx].item():.2f}$")
        if col_idx == 0:
            a2.set_ylabel(rf"$\bar{{Q}}_{{n_0 + k}}(t)$, Rep {row_idx + 1}")
        if row_idx == n_reps - 1:
            a2.set_xlabel(r"$Q_{n_0}(t)$")
        # a2.set_xlim(0, 10) # around 0.95 of Gamma(2, 2)
        # a2.set_ylim(0, 10)

fig1.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax1[:, -1],
)
fig2.colorbar(
    plt.cm.ScalarMappable(norm=plt.Normalize(k.min(), k.max()), cmap="viridis"),
    label="k",
    ax=ax2[:, -1],
)

fig1.savefig(f"{image_dir}/cid-gamma-pp.pdf")
fig2.savefig(f"{image_dir}/cid-gamma-qq.pdf")
# %%

n_est = 4
fig1, ax1 = plt.subplots(
    n_reps, len(x_new_idx_ls), figsize=(12, 1.5 * n_reps), constrained_layout=True
)
for col_idx, x_new_idx in enumerate(x_new_idx_ls):
    rep_dirs = utils.get_matching_dirs(
        id_dir, rf"cid.+dgp=classification-fixed.+n_est={n_est} "
    )
    assert len(rep_dirs) == n_reps
    rep_dirs.sort()

    for row_idx, d in enumerate(rep_dirs):
        sample_dirs = utils.get_matching_dirs(d, r"sample-\d+")

        # F_n_all is (samples, n, t, x_new)
        F_n_all, n, t, x_new = compile_all_Fn(sample_dirs)
        F_n = np.mean(F_n_all[..., x_new_idx], axis=0)  # (n, t)
        k = n - n0
        a1 = ax1[row_idx, col_idx]
        a1.plot(k, F_n[:, 1], color="black", linestyle="-")
        a1.axhline(F_n[0, 1], color="black", linestyle="--")
        a1.set_ylim(0, 1)
        if row_idx == 0:
            a1.set_title(rf"$x = {x_new[x_new_idx].item():.2f}$")
        if col_idx == 0:
            a1.set_ylabel(rf"$\bar{{g}}_{{n_0 + k}}(t)$, Rep {row_idx + 1}")
        if row_idx == n_reps - 1:
            a1.set_xlabel(r"$k$")

fig1.savefig(f"{image_dir}/cid-classification-pp.pdf")
# %%
