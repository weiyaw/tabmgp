# %%
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

import utils

# %%
# PLOT ACID RESULTS FOR CLASSIFICATION

acid_dir = f"../outputs/2025-06-97/name=classification-fixed dim_x=2 resample_x=bb data=100 seed=1001/acid"
image_dir = "../paper/images/"

acid_eval_dir = [
    (m, f"{acid_dir}/{p}")
    for p in os.listdir(acid_dir)
    if (m := re.search(r"x-eval-(\d+)", p))
]
acid_eval_dir.sort(key=lambda mp: int(mp[0].group(1)))


def compile_cond_logpmf(acid_eval_dir):
    logpmf_matches = [
        (m, p)
        for p in os.listdir(acid_eval_dir)
        if (m := re.search(r"logpmf-(\d+)-(\d+)-(\d+)-(\d+)\.pickle", p))
    ]
    logpmf_matches.sort(key=lambda mp: int(mp[0].group(1)))
    seq_start = [int(m.group(2)) for m, _ in logpmf_matches]
    seq_end = [int(m.group(3)) for m, _ in logpmf_matches]
    seq_freq = [int(m.group(4)) for m, _ in logpmf_matches]

    if len(set(seq_start)) != 1 or len(set(seq_end)) != 1 or len(set(seq_freq)) != 1:
        print("Not all elements in seq_start, seq_end, and seq_freq are the same.")

    N_idx = np.arange(seq_start[0], seq_end[0] + 1, seq_freq[0])
    logpmf = [utils.read_from(f"{acid_eval_dir}/{p}") for _, p in logpmf_matches]
    logpmf = {k: np.stack([dic[k] for dic in logpmf]) for k in logpmf[0]}
    return logpmf, N_idx


cond_logpmf_y_x = [compile_cond_logpmf(p)[0] for _, p in acid_eval_dir]
cond_logpmf_y_x = {
    k: np.stack([dic[k] for dic in cond_logpmf_y_x]) for k in cond_logpmf_y_x[0]
}

_, N_idx = compile_cond_logpmf(acid_eval_dir[0][1])

logpmf_two_step_cond_y_x = cond_logpmf_y_x["two_step_cond_y_x"]  # (x, rollouts, N, y)
logpmf_one_step_cond_y_x = cond_logpmf_y_x["one_step_cond_y_x"]  # (x, rollouts, N, y)


# log |p2(Y | X) - p1(Y | X)|
log_delta_pmf_cond_y_x, sign = logsumexp(
    np.stack(
        [logpmf_two_step_cond_y_x, logpmf_one_step_cond_y_x],
        axis=-1,
    ),
    axis=-1,
    b=np.asarray([1, -1]),  # broadcast to the last axis
    return_sign=True,
)


# %%
def trapezoidal_cumsum(n, b):
    # n: (N, )
    # b: (N, )
    assert len(n) == len(b)
    assert np.all(n == np.sort(n))
    assert n.ndim == 1 and b.ndim == 1
    areas = (b[1:] + b[:-1]) / 2 * (n[1:] - n[:-1])
    areas = np.concatenate(([0], areas))
    return np.cumsum(areas), areas


# Partial sum plots
n_col = 5
n_row = 2
y_idx = 1
fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 3 * n_row))

delta_pmf_cond_y_x = np.exp(log_delta_pmf_cond_y_x)[..., y_idx]  # (x, rollouts, N)
for j, delta_pmf_cond_y_x_j in enumerate(delta_pmf_cond_y_x):
    ax = axes.flatten()[j]
    for delta_pmf_cond_y_x_j_r in delta_pmf_cond_y_x_j:
        # for each rollout, do a trapezoidal cumsum of |p2(y|x, z_1:N) - p1(y|x, z_1:N)|
        cumsum, _ = trapezoidal_cumsum(N_idx, delta_pmf_cond_y_x_j_r)
        ax.plot(N_idx, cumsum, color="grey", alpha=0.3)
    ax.set_title(rf"$x* = x[{j}]$")

for i in range(n_row):
    axes[i, 0].set_ylabel(r"$\sum_N |\Delta_i P(y = 1| x*)|$")
for i in range(n_col):
    axes[n_row - 1, i].set_xlabel("N")

plt.tight_layout()
plt.savefig(f"{image_dir}/classification-acid-sum-delta-y1.pdf")

# %%
