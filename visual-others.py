# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import logsumexp

import seaborn as sns
import utils
import pandas as pd
from experiment_setup import load_experiment, get_experiment_paths
from optimizer import Diagnostics, OptResult

from credible_set import marginal_credible_interval


import os
import re
import ast

from plot_settings import DATES, COLOR_PALETTE, POSTERIOR_NAMES, MARKER_SHAPES

jax.config.update("jax_enable_x64", True)


data_info = pd.read_csv("./data_info.csv")
data_info["theta_name"] = data_info["theta_name"].apply(ast.literal_eval)
output_dir = "../outputs"
savedir = "../paper/images"
loss = "likelihood"


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
# Histogram of the posterior for all methods and setups

for date in DATES:
    print(date)
    all_paths = get_experiment_paths(f"{output_dir}/{date}")
    data_name = utils.get_data_name(all_paths[0])
    _, functional, theta_true, processed_data = load_experiment(
        all_paths[0], loss="likelihood"
    )
    mle, mle_opt = functional.minimize_loss(processed_data, theta_true, None)
    assert mle_opt.success
    theta_name = data_info.query("name == @data_name")["theta_name"].item()

    ALPHA = 0.05
    assert theta_true.size == mle.size
    dim_theta = theta_true.size

    # Prepare a dataframe for seaborn
    data_list = []
    for method in POSTERIOR_NAMES.keys():
        if method == "gibbs-eb":
            data = utils.read_from(
                f"{all_paths[0]}/posterior-{loss}/gibbs-eb-post.pickle"
            )[0]
        else:
            data = read_max_T_post(all_paths[0], method, loss)
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
        ax.plot(
            mean,
            [y_pos],
            MARKER_SHAPES[method],
            markersize=8,
            color=COLOR_PALETTE[method],
        )
        ax.hlines(y=y_pos, xmin=lower, xmax=upper, color=COLOR_PALETTE[method])

    # Add MLE and theta_true reference lines
    for i, ax in enumerate(facet.axes.flat):
        ax.axvline(x=theta_true[i], color="black", linestyle="-")
        ax.axvline(x=mle[i], color="black", linestyle="--")

    # Create legend
    line_handles = [
        plt.Line2D(
            [0], [0], color="black", linestyle="-", label=r"$\theta(F^{\star})$"
        ),
        plt.Line2D([0], [0], color="black", linestyle="--", label=r"$\theta(F_n)$"),
    ]
    density_handles = [
        plt.Line2D(
            [0],
            [0],
            color=COLOR_PALETTE[m],
            label=POSTERIOR_NAMES[m],
            marker=MARKER_SHAPES[m],
        )
        for m in facet.hue_names
    ]
    facet.figure.legend(handles=line_handles + density_handles, loc="lower right")

    os.makedirs(f"{savedir}/density", exist_ok=True)
    facet.savefig(f"{savedir}/density/{data_name}-density.pdf", bbox_inches="tight")
    if data_name in ["classification-standard", "regression-standard", "abalone", "concrete", "telescope", "yeast"]:
        # these are the one showing up in appendix
        facet.savefig(f"{savedir}/{data_name}-density.pdf", bbox_inches="tight")
    plt.close(facet.figure)


# %%
# Histrogram of TabMGP for various n (concentration results)

post_ls = []
for d, n in zip(
    ["2025-06-82", "2025-06-83", "2025-06-84", "2025-06-85"], [500, 1000, 1500, 2000]
):
    path = f"{output_dir}/{d}/name=regression-fixed dim_x=2 noise_std=0.1 resample_x=bb data={n} seed=1001"
    for root, _, files in os.walk(path):
        for f in files:
            if f == "tabpfn-2000-post.pickle":
                post_ls.append((n, utils.read_from(os.path.join(root, f))[0]))

_, _, theta_true, _ = load_experiment(path, loss=loss)
dim_theta = theta_true.shape[0]


theta_name = ["Intercept", "x1", "x2"]
fig, axes = plt.subplots(1, 3, figsize=(15, 3))
for d, ax in enumerate(axes.flat):
    # Density plot
    for n, post in post_ls:
        assert post.shape[1] == theta_true.size
        sns.kdeplot(data=post[:, d], alpha=0.1, fill=True, ax=ax, label=f"n={n}")
    ax.axvline(
        x=theta_true[d],
        color="black",
        linestyle="-",
        alpha=1,
        label=r"$\theta(F^{\star})$",
    )
    ax.set_title(theta_name[d])

handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower right")
plt.tight_layout()
plt.savefig(f"{savedir}/toy-concentration.pdf")


# %%
# Histogram of the intercept of the 'concrete' setup

path = f"{output_dir}/2025-07-04/name=concrete resample_x=bb data=100 seed=1001"
_, functional, theta_true, train_data = load_experiment(path, loss=loss)
mle, _ = functional.minimize_loss(train_data, theta_true, None)

fig = plt.figure(figsize=(7, 3))
for i, method in enumerate(POSTERIOR_NAMES.keys()):
    if method == "gibbs-eb":
        post = utils.read_from(f"{path}/posterior-{loss}/gibbs-eb-post.pickle")[0][:, 0]
    else:
        post = read_max_T_post(path, method, loss)[:, 0]

    sns.kdeplot(post, fill=True, alpha=0.1, color=COLOR_PALETTE[method])
    mean, lower, upper = marginal_credible_interval(post[:, None], 0.05)
    ylim = plt.ylim()[1]
    y_pos = ylim * (0.95 - i * 0.05)
    plt.plot(
        mean, [y_pos], MARKER_SHAPES[method], markersize=8, color=COLOR_PALETTE[method]
    )
    plt.hlines(y=y_pos, xmin=lower, xmax=upper, color=COLOR_PALETTE[method])

plt.axvline(x=theta_true[0], color="black", linestyle="-", alpha=1)
plt.axvline(x=mle[0], color="black", linestyle="--", alpha=1)
plt.xlim(-0.4, 0.4)

# Create legend
line_handles = [
    plt.Line2D([0], [0], color="black", linestyle="-", label=r"$\theta(F^{\star})$"),
    plt.Line2D([0], [0], color="black", linestyle="--", label=r"$\theta(F_n)$"),
]
density_handles = [
    plt.Line2D(
        [0],
        [0],
        color=COLOR_PALETTE[name],
        label=POSTERIOR_NAMES[name],
        marker=MARKER_SHAPES[name],
    )
    for name in POSTERIOR_NAMES.keys()
]
handles = line_handles + density_handles
plt.legend(handles=handles, loc="upper left")
plt.tight_layout()
fig.savefig(f"{savedir}/concrete-intercept-kde.pdf", bbox_inches="tight")


# %%
# Boxplot of the marginal coverage (synthetic setups)

# Run table.py to get this table
coverage = pd.read_csv("table/marginal-coverage.csv")

# only include relevant posteriors
coverage = coverage[coverage["post_name"].isin(POSTERIOR_NAMES)]

ALPHA = 1 - coverage["ideal_rate"][0]

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

plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="data",
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
# Boxplot of the marginal coverage (linear regression, real data)
plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="data",
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

# Boxplot of the marginal coverage (logistic regression, real data)
plt.figure(figsize=(12, 4))
sns.boxplot(
    data=coverage,
    x="data",
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
