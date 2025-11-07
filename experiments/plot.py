# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax import vmap

import seaborn as sns
import utils
import pandas as pd
from utils import get_name, get_data_size, get_resample_x
from utils import tree_shape, OptDiagnostic
from posterior import load_experiment, Diagnostics, OptResult

from credible_set import marginal_credible_interval
from plot_settings import DATES, COLOR_PALETTE, POSTERIOR_NAMES

# from dgp import *
import os
import re


def get_plot_name(path):
    # get name and data size
    return f"{get_name(path)}-{get_data_size(path)}-{get_resample_x(path)}"


data_info = utils.read_from(f"./data_info.pickle")
savedir = "../paper/plots"
loss = "likelihood"


def read_trace(path, method, loss):
    """
    Read trace from the given path that matches method-*-*-*-trace.pickle.

    Return a dict of the form {T: array of shape (samples, theta.size)} where T
    is the number of forward recursion.
    """

    for root, _, files in os.walk(path):
        if not root.endswith(f"posterior-{loss}"):
            continue

        for f in sorted(files):
            # match bb/tabpfn/copula-x-post.pickle
            if match := re.search(rf"{method}-(\d+)-(\d+)-(\d+)-trace.pickle", f):
                trace, trace_opt = utils.read_from(f"{root}/{f}")
                info = {
                    "start": int(match.group(1)),
                    "end": int(match.group(2)),
                    "resolution": int(match.group(3)),
                }
                return trace, info
    return None, None


def read_post(path, method, loss):
    """
    Given a path, e.g. "... seed=1001", Read the posterior.

    Return a dict of the form {T: tuple of posterior and diagnostics} where T
    is the number of forward recursion.
    """

    posterior = {}
    for root, _, files in os.walk(path):
        if not root.endswith(f"posterior-{loss}"):
            continue
        for f in sorted(files):
            # match bb/tabpfn/copula-x-post.pickle
            if match := re.search(rf"{method}-(\d+)-post.pickle", f):
                T = int(match.group(1))
                posterior[T], _ = utils.read_from(f"{root}/{f}")
    return posterior


def read_max_T_post(path, method, loss):
    post = read_post(path, method, loss)
    if post:
        max_T = max(post.keys())
        return post[max_T]
    else:
        return None


# %%
###################################################################
###################################################################
###################### TRACE PLOT of REP 0 ########################
###################################################################
###################################################################

for date in DATES:
    print(date)
    experiment = load_experiment(f"../outputs/{date}", loss=loss)
    path = experiment.all_paths[0]
    name = utils.get_name(path)

    theta_true = experiment.theta_true
    mle, mle_opt = experiment.minimize_loss(
        experiment.all_train_data[0], theta_true, None
    )
    assert mle_opt.success
    theta_name = data_info.query("name == @name")["theta_name"].item()

    for method in ["tabpfn", "bb", "copula"]:
        trace, trace_info = read_trace(path, method, loss)
        if trace is None:
            continue

        trace = np.asarray(trace)  # Convert to numpy array if it's a JAX array
        assert mle.size == trace.shape[2]
        assert mle.size == theta_true.size
        dim_theta = theta_true.shape[0]

        # Define coordinate arrays
        N_idx = np.arange(
            trace_info["start"], trace_info["end"] + 1, trace_info["resolution"]
        )
        sample_idx = np.arange(trace.shape[1])
        param_idx = np.arange(dim_theta)

        # Create grids from coordinate arrays
        N_grid, sample_grid, param_grid = np.meshgrid(
            N_idx, sample_idx, param_idx, indexing="ij"
        )

        # Build the DataFrame
        df = pd.DataFrame(
            {
                "N": N_grid.flatten(),
                "sample": sample_grid.flatten(),
                "value": trace.flatten(),
                "parameter": np.array(theta_name)[param_grid.flatten()],
                "param_idx": param_grid.flatten(),
            }
        )

        # Create FacetGrid
        facet = sns.FacetGrid(
            df, col="parameter", col_wrap=4, height=2, aspect=1.8, sharey=False
        )
        facet.map_dataframe(
            sns.lineplot,
            x="N",
            y="value",
            units="sample",
            estimator=None,
            alpha=0.3,
            linewidth=0.3,
            color="black",
        )
        facet.set_titles("{col_name}")
        facet.set_ylabels(r"$\theta$ value")

        # Add reference lines
        for i, ax in enumerate(facet.axes.flat):
            ax.axhline(
                y=theta_true[i],
                linestyle="--",
                color="red",
                linewidth=1.5,
                label="Population",
            )
            ax.axhline(
                y=mle[i],
                linestyle="--",
                color="blue",
                linewidth=1.5,
                label="Empirical",
            )
            # ax.axvline(x=n_train, linestyle="--", color="black", linewidth=1.5)

        handles, labels = facet.axes.flat[0].get_legend_handles_labels()
        facet.figure.legend(handles, labels, loc="lower right")

        os.makedirs(f"{savedir}/trace", exist_ok=True)
        facet.savefig(f"{savedir}/trace/{name}-{method}-trace.pdf", bbox_inches="tight")
        plt.close(facet.figure)


# %%
########################################################################
########################################################################
########################## HISTOGRAM of REP 0 ##########################
########################################################################
########################################################################

# %%


for date in DATES:
    print(date)
    experiment = load_experiment(f"../outputs/{date}", loss=loss)
    path = experiment.all_paths[0]
    name = utils.get_name(path)

    theta_true = experiment.theta_true
    mle, mle_opt = experiment.minimize_loss(
        experiment.all_train_data[0], theta_true, None
    )
    assert mle_opt.success
    theta_name = data_info.query("name == @name")["theta_name"].item()

    ALPHA = 0.05
    assert theta_true.size == mle.size
    dim_theta = theta_true.size

    # Prepare a dataframe for seaborn
    data_list = []
    for method in POSTERIOR_NAMES.keys():
        if method == "gibbs":
            data = utils.read_from(f"{path}/posterior-{loss}/gibbs-post.pickle")[0]
        else:
            data = read_max_T_post(path, method, loss)
        if data is None:
            continue
        assert data.shape[-1] == dim_theta
        n_samples, n_params = data.shape
        temp_df = pd.DataFrame(
            {
                "parameter": np.repeat(theta_name, n_samples),
                "value": data.flatten(order="F"),  # Flatten column-by-column
                "method": method,
                "param_idx": np.repeat(np.arange(n_params), n_samples),
            }
        )
        data_list.append(temp_df)
    df = pd.concat(data_list, ignore_index=True)

    facet = sns.FacetGrid(
        df,
        col="parameter",
        hue="method",
        palette=COLOR_PALETTE,
        col_wrap=4,
        height=3,
        aspect=1.5,
        sharey=False,
        sharex=False,
    )

    # Add density plots
    facet.map_dataframe(
        sns.kdeplot, x="value", alpha=0.1, fill=True, common_norm=False, legend=True
    )
    facet.set_xlabels(r"$\theta$ value")
    facet.set_titles("{col_name}")

    # Add credible intervals
    n_col = facet._col_wrap
    for (i, j, k), data in facet.facet_data():
        method = facet.hue_names[k]
        samples = np.expand_dims(np.asarray(data["value"]), -1)
        mean, lower, upper = marginal_credible_interval(samples, ALPHA)
        ax = facet.facet_axis(i, j)
        ylim = ax.get_ylim()[1]
        y_pos = ylim * (0.95 - k * 0.05)
        ax.plot(mean, [y_pos], "o", markersize=8, color=COLOR_PALETTE[method])
        ax.hlines(y=y_pos, xmin=lower, xmax=upper, color=COLOR_PALETTE[method])

    # Add MLE and theta_true reference lines
    for i, ax in enumerate(facet.axes.flat):
        ax.axvline(x=theta_true[i], color="red", linestyle="--")
        ax.axvline(x=mle[i], color="blue", linestyle="--")

    # Create legend
    line_handles = [
        plt.Line2D([0], [0], color="red", linestyle="--", label="Population"),
        plt.Line2D([0], [0], color="blue", linestyle="--", label="Empirical"),
    ]
    density_handles = [
        plt.Line2D(
            [0], [0], color=COLOR_PALETTE[m], label=POSTERIOR_NAMES[m], marker="o"
        )
        for m in facet.hue_names
    ]
    facet.figure.legend(handles=line_handles + density_handles, loc="lower right")

    os.makedirs(f"{savedir}/density", exist_ok=True)
    facet.savefig(f"{savedir}/density/{name}-density.pdf", bbox_inches="tight")
    plt.close(facet.figure)

# %%


coverage = pd.read_csv("table/marginal-coverage.csv")

data_info["np_ratio"] = data_info["training_size"] / data_info["dim_theta"]
data_info = data_info.sort_values(
    by=["n_classes_in_y", "np_ratio"], ascending=[True, False]
)
synthetic_data = data_info[data_info["name"].str.contains("regression|classification")][
    "name"
]
regression_data = data_info.query("date.str.startswith('2025-07-0')")["name"]
classification_data = data_info.query("date.str.contains('2025-07-[56]')")["name"]

synthetic_data_xtick_label = {
    "regression-standard": r"$N(0, 1)$",
    "regression-t-5": r"$t_5$",
    "regression-t-4": r"$t_4$",
    "regression-t-3": r"$t_3$",
    "regression-dependent-0.25-0.5": r"$s_1$",
    "regression-dependent-0.05-0.25": r"$s_2$",
    "regression-dependent-0.01-0.1": r"$s_3$",
    "classification-standard": "Logistic",
    "classification-gmm-0": "GMM(0)",
    "classification-gmm--1": "GMM(-1)",
    "classification-gmm--2": "GMM(-2)",
}
# %%

plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="name",
    y="rate",
    hue="post_name",
    palette=COLOR_PALETTE,
    order=synthetic_data,
    width=0.5,
    gap=0.4,
)
plt.axhline(y=1 - ALPHA, color="black", linestyle="--")
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [POSTERIOR_NAMES[label] for label in labels]
plt.legend(handles, new_labels, loc="lower right")
plt.ylabel("Coverage")
plt.ylim(0.29, 1.02)
plt.xlabel("Dataset")
plt.xticks(
    ticks=np.arange(len(synthetic_data)),
    labels=[synthetic_data_xtick_label.get(name, name) for name in synthetic_data],
)
plt.savefig(
    f"{savedir}/synthetic-marginal-coverage.pdf",
    bbox_inches="tight",
)

# %%

plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="name",
    y="rate",
    hue="post_name",
    palette=COLOR_PALETTE,
    order=regression_data,
    width=0.5,
    gap=0.4,
)
plt.axhline(y=1 - ALPHA, color="black", linestyle="--")
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [POSTERIOR_NAMES[label] for label in labels]
plt.legend(handles, new_labels, loc="lower right")
plt.ylabel("Coverage")
plt.ylim(0.29, 1.02)
plt.xlabel("Dataset")
plt.savefig(
    f"{savedir}/regression-marginal-coverage.pdf",
    bbox_inches="tight",
)
# %%

plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="name",
    y="rate",
    hue="post_name",
    palette=COLOR_PALETTE,
    order=classification_data,
    width=0.5,
    gap=0.4,
)
plt.axhline(y=1 - ALPHA, color="black", linestyle="--")
handles, labels = plt.gca().get_legend_handles_labels()
new_labels = [POSTERIOR_NAMES[label] for label in labels]
plt.legend(handles, new_labels, loc="lower right")
plt.ylabel("Coverage")
plt.ylim(0.29, 1.02)
plt.xlabel("Dataset")
plt.savefig(
    f"{savedir}/classification-marginal-coverage.pdf",
    bbox_inches="tight",
)
# %%
fig = plt.figure(figsize=(7, 3))
for date in DATES:
    print(date)
    date_path = f"../outputs/{date}"
    experiment = load_experiment(date_path, loss)
    path_0 = experiment.all_paths[0]
    name = utils.get_name(experiment.all_paths[0])

    trace, trace_info = read_trace(path_0, "tabpfn", loss)
    l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
    N_idx = np.arange(
        0, trace_info["end"] - trace_info["start"] + 1, trace_info["resolution"]
    )
    plt.plot(N_idx, np.mean(l1, axis=-1), color="black", linewidth=1, alpha=0.5)


plt.ylabel(r"Scaled $L_1$")
plt.xlabel("N - n")
plt.axvline(x=N_idx[0], linestyle="--", color="black", linewidth=1.5)

fig.savefig(
    f"{savedir}/all-data-tabpfn-l1-0.pdf",
    bbox_inches="tight",
)


# %%
TITLE = {
    "2025-06-02": "Regression $N(0, 1)$",
    "2025-06-07-a": "Regression $t_5$",
    "2025-06-07-b": "Regression $t_4$",
    "2025-06-07-c": "Regression $t_3$",
    "2025-06-06-a": "Regression Dept. $s_1$",
    "2025-06-06-b": "Regression Dept. $s_2$",
    "2025-06-06-c": "Regression Dept. $s_3$",
    "2025-06-01": "Classification Logistic",
    "2025-06-05-a": "Classification GMM $a=0$",
    "2025-06-05-b": "Classification GMM $a=-1$",
    "2025-06-05-c": "Classification GMM $a=-2$",
    "2025-07-04": "concrete",
    "2025-07-01": "quake",
    "2025-07-02": "airfoil",
    "2025-07-05-a": "energy",
    "2025-07-08-a": "fish",
    "2025-07-03": "kin8nm",
    "2025-07-09-a": "auction",
    "2025-07-06-a": "grid",
    "2025-07-07-a": "abalone",
    "2025-07-54": "rice",
    "2025-07-57": "sepsis",
    "2025-07-60-a": "banknote",
    "2025-07-55": "mozilla",
    "2025-07-53": "skin",
    "2025-07-51": "blood",
    "2025-07-52": "phoneme",
    "2025-07-56-a": "telescope",
    "2025-07-58-b": "yeast",
    "2025-07-59-a": "wine",
}

n_plots = len(DATES)
ncols = 4
nrows = (n_plots + ncols - 1) // ncols

for method in ["tabpfn", "bb"]:
    # Set squeeze=False so 'axes' is always a 2D array
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 6, nrows * 3.5), squeeze=False)

    # Flatten the 2D array of axes into a 1D array for easy looping
    axes = axes.flatten()
    plot_idx = 0  # Use this index to track which axis to plot on

    for date in TITLE.keys():
        print(date)
        date_path = f"../outputs/{date}"
        experiment = load_experiment(date_path, loss)
        path_0 = experiment.all_paths[0]
        name = utils.get_name(experiment.all_paths[0])

        trace, trace_info = read_trace(path_0, method, loss)
        if trace is None:
            continue  # Skip this item

        l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
        N_idx = np.arange(
            trace_info["start"], trace_info["end"] + 1, trace_info["resolution"]
        )

        ax = axes[plot_idx]
        ax.plot(N_idx, l1, color="black", alpha=0.2, linewidth=0.3)
        ax.plot(N_idx, np.mean(l1, axis=-1), color="blue", linewidth=2)

        q1, q3 = np.quantile(l1[-1], [0.25, 0.75], axis=0)
        iqr = q3 - q1
        ax.set_ylim(bottom=0, top=q3 + 2 * iqr)
        ax.set_ylabel(r"Scaled $L_1$")
        ax.set_xlabel("N")
        ax.set_title(TITLE[date])
        ax.axvline(x=N_idx[0], linestyle="--", color="black", linewidth=1.5)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(
        f"{savedir}/all-data-{method}-full-l1-0.pdf",
        bbox_inches="tight",
    )

# %%
### CONCENTRATION OF TABMGP POSTERIOR

post_ls = []
for d, n in zip(
    ["2025-06-82", "2025-06-83", "2025-06-84", "2025-06-85"], [500, 1000, 1500, 2000]
):
    path = f"../outputs/{d}/name=regression-fixed dim_x=2 noise_std=0.1 resample_x=bb data={n} seed=1001"
    for root, _, files in os.walk(path):
        for f in files:
            if f == "tabpfn-2000-post.pickle":
                post_ls.append((n, utils.read_from(os.path.join(root, f))[0]))

exp2000 = load_experiment(path, loss=loss)
theta_true = exp2000.theta_true
dim_theta = theta_true.shape[0]
init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))


theta_name = ["Intercept", "x1", "x2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for d, ax in enumerate(axes.flat):
    # Density plot
    for n, post in post_ls:
        assert post.shape[1] == theta_true.size
        sns.kdeplot(data=post[:, d], alpha=0.1, fill=True, ax=ax, label=f"n={n}")
    ax.axvline(x=theta_true[d], color="red", linestyle="--", alpha=1, label="Population")
    ax.set_title(theta_name[d])

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right")
plt.tight_layout()
plt.savefig(f"{savedir}/toy-concentration.pdf")


# %%
# PLOT INTERCEPT DENSITY OF CONCRETE

path = "../outputs/2025-07-04/name=concrete resample_x=bb data=100 seed=1001" 
experiment = load_experiment(path, loss=loss)
theta_true = experiment.theta_true
train_data = experiment.all_train_data[0]
mle, mle_opt = experiment.minimize_loss(train_data, theta_true, None)

fig = plt.figure(figsize=(7, 3))
for i, method in enumerate(POSTERIOR_NAMES.keys()):
    if method == "gibbs":
        post = utils.read_from(f"{path}/posterior-{loss}/gibbs-post.pickle")[0][:, 0]
    else:
        post = read_max_T_post(path, method, loss)[:, 0]

    sns.kdeplot(post, fill=True, alpha=0.1, color=COLOR_PALETTE[method])
    mean, lower, upper = marginal_credible_interval(post[:, None], 0.05)
    ylim = plt.ylim()[1]
    y_pos = ylim * (0.95 - i * 0.05)
    plt.plot(mean, [y_pos], "o", markersize=8, color=COLOR_PALETTE[method])
    plt.hlines(y=y_pos, xmin=lower, xmax=upper, color=COLOR_PALETTE[method])

plt.axvline(x=theta_true[0], color="red", linestyle="--", alpha=1)
plt.axvline(x=mle[0], color="blue", linestyle="--", alpha=1)
plt.xlim(-0.4, 0.4)

# Create legend
line_handles = [
    plt.Line2D([0], [0], color="red", linestyle="--", label="Population"),
    plt.Line2D([0], [0], color="blue", linestyle="--", label="Empirical"),
]
density_handles =  [
    plt.Line2D([0], [0], color=COLOR_PALETTE[name], label=POSTERIOR_NAMES[name], marker="o")
    for name in POSTERIOR_NAMES.keys()
]
handles = line_handles + density_handles
plt.legend(handles=handles, loc="upper left")
plt.tight_layout()
fig.savefig(f"{savedir}/concrete-intercept-kde.pdf", bbox_inches="tight")


# %%
# PLOT ACID RESULTS FOR CLASSIFICATION

from scipy.special import logsumexp
input_dir = "../outputs/2025-06-97/name=classification-fixed dim_x=2 resample_x=bb data=100 seed=1001"
acid_dir = f"{input_dir}/acid"

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
cond_logpmf_y_x = {k: np.stack([dic[k] for dic in cond_logpmf_y_x]) for k in cond_logpmf_y_x[0]}

_, N_idx = compile_cond_logpmf(acid_eval_dir[0][1])

logpmf_two_step_cond_y_x = cond_logpmf_y_x["two_step_cond_y_x"]
logpmf_one_step_cond_y_x = cond_logpmf_y_x["one_step_cond_y_x"]


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

ncol = 4
nrow = 2
plt.figure(figsize=(5 * ncol, 3 * nrow))
for j in range(log_delta_pmf_cond_y_x.shape[0]):
    if j >= nrow * ncol:
        break
    plt.subplot(nrow, ncol, j + 1)
    plt.plot(
        N_idx,
        np.cumsum(np.exp(log_delta_pmf_cond_y_x[j, :, :, 1]), axis=-1).T,
        color="grey",
        alpha=0.3,
    )
    plt.title(rf"$x* = x[{j}]$")
    # cumsum over p(y = 1 | x*, z_1:N)
    plt.ylabel(r"$\sum_N |\Delta_i P(y = 1| x*)|$")
    plt.xlabel("N")
plt.tight_layout()
plt.savefig(f"{savedir}/classification-acid-sum-delta-y1.pdf")

# %%
