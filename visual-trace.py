# %%
import jax
import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
import utils
import pandas as pd
from utils import get_data_name
from experiment_setup import load_experiment, get_experiment_paths
from optimizer import Diagnostics, OptResult

import os
import re
import ast

from plot_settings import DATES, TITLE

jax.config.update("jax_enable_x64", True)


data_info = pd.read_csv("./data_info.csv")
data_info["theta_name"] = data_info["theta_name"].apply(ast.literal_eval)
output_dir = "../outputs"
savedir = "../paper/images"
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


# %%
# Standard trace plots for all methods and setups

for date in DATES:
    print(date)
    all_paths = get_experiment_paths(f"{output_dir}/{date}")
    data_name = get_data_name(all_paths[0])
    _, functional, theta_true, processed_data = load_experiment(
        all_paths[0], loss="likelihood"
    )
    mle, mle_opt = functional.minimize_loss(processed_data, theta_true, None)
    assert mle_opt.success
    theta_name = data_info.query("name == @data_name")["theta_name"].item()

    for method in ["tabpfn", "bb", "copula"]:
        trace, trace_info = read_trace(all_paths[0], method, loss)
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
            color="grey",
        )
        facet.set_titles("{col_name}")
        facet.set_ylabels(r"$\theta$ value")

        # Add reference lines
        for i, ax in enumerate(facet.axes.flat):
            ax.axhline(
                y=theta_true[i],
                linestyle="-",
                color="black",
                linewidth=1.5,
                label=r"$\theta(F^{\star})$",
            )
            ax.axhline(
                y=mle[i],
                linestyle="--",
                color="black",
                linewidth=1.5,
                label=r"$\theta(F_n)$",
            )
            # ax.axvline(x=n_train, linestyle="--", color="black", linewidth=1.5)

        handles, labels = facet.axes.flat[0].get_legend_handles_labels()
        facet.figure.legend(handles, labels, loc="lower right")

        os.makedirs(f"{savedir}/trace", exist_ok=True)
        facet.savefig(
            f"{savedir}/trace/{data_name}-{method}-trace.pdf", bbox_inches="tight"
        )
        plt.close(facet.figure)

# %%
# Expected l1-norm convergence plot (all setups)
fig = plt.figure(figsize=(7, 3))
for date in DATES:
    print(date)
    date_path = f"{output_dir}/{date}"
    all_paths = get_experiment_paths(date_path)
    data_name = utils.get_data_name(all_paths[0])

    trace, trace_info = read_trace(all_paths[0], "tabpfn", loss)
    l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
    N_idx = np.arange(
        0, trace_info["end"] - trace_info["start"] + 1, trace_info["resolution"]
    )
    plt.plot(N_idx, np.mean(l1, axis=-1), color="black", linewidth=1, alpha=0.5)


plt.ylabel(r"Scaled $L_1$")
plt.xlabel("N - n")
plt.xlim(0, 500)

fig.savefig(
    f"{savedir}/all-data-tabpfn-l1-0.pdf",
    bbox_inches="tight",
)


# %%
# Expected l1-norm convergence plot (individual setups)

n_plots = len(DATES)
ncols = 4
nrows = (n_plots + ncols - 1) // ncols

for method in ["tabpfn", "bb"]:
    # Set squeeze=False so 'axes' is always a 2D array
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 3.5), squeeze=False
    )

    # Flatten the 2D array of axes into a 1D array for easy looping
    axes = axes.flatten()
    plot_idx = 0  # Use this index to track which axis to plot on

    for date, data_name in TITLE.items():
        print(date)
        date_path = f"{output_dir}/{date}"
        all_paths = get_experiment_paths(date_path)
        trace, trace_info = read_trace(all_paths[0], method, loss)
        if trace is None:
            continue  # Skip this item

        l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
        N_idx = np.arange(
            trace_info["start"], trace_info["end"] + 1, trace_info["resolution"]
        )

        ax = axes[plot_idx]
        ax.plot(N_idx, l1, color="grey", alpha=0.2, linewidth=0.3)
        ax.plot(N_idx, np.mean(l1, axis=-1), color="black", linewidth=3)

        q1, q3 = np.quantile(l1[-1], [0.25, 0.75], axis=0)
        iqr = q3 - q1
        ax.set_ylim(bottom=0, top=q3 + 2 * iqr)
        ax.set_ylabel(r"Scaled $L_1$")
        ax.set_xlabel("N")
        ax.set_title(TITLE[date])
        ax.axvline(x=N_idx[0], linestyle="--", color="grey", linewidth=1.0)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    fig.savefig(
        f"{savedir}/all-data-{method}-full-l1-0.pdf",
        bbox_inches="tight",
    )

# %%
