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
from pathlib import Path

from plot_settings import IDS, TITLE, LINEAR_REGRESSION_IDS, LOGISTIC_REGRESSION_IDS

jax.config.update("jax_enable_x64", True)


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent

data_info = pd.read_csv(BASE_DIR / "data_info.csv")
data_info["theta_name"] = data_info["theta_name"].apply(ast.literal_eval)
output_dir = str(BASE_DIR / "outputs")
savedir = str(REPO_ROOT.parent / "paper" / "images")
loss = "likelihood"
SAVE_PLOTS = True


def read_trace(path):
    """
    Read trace from the given path that matches method-*-*-*-trace.pickle.

    Return a dict of the form {T: array of shape (samples, theta.size)} where T
    is the number of forward recursion.
    """

    # match bb/tabmgp/copula-x-post.pickle
    if match := re.search(r".+-(\d+)-(\d+)-(\d+)-trace.pickle", os.path.basename(path)):
        trace, _ = utils.read_from(path)
        info = {
            "start": int(match.group(1)),
            "end": int(match.group(2)),
            "resolution": int(match.group(3)),
        }
        return trace, info
    else:
        raise ValueError(f"{path} is not a valid trace path.")


# %%
# Standard trace plots for all methods and setups (only for the realisation of seed=1001)

for id in IDS:
    print(id)
    path_1001 = utils.get_matching_dirs(f"{output_dir}/{id}", "seed=1001")
    assert len(path_1001) == 1, (
        f"Expected exactly one path for seed=1001 in {id}, but found {len(path_1001)}."
    )
    path_1001 = path_1001[0]
    data_name = get_data_name(path_1001)
    _, functional, theta_true, processed_data = load_experiment(
        path_1001, loss="likelihood"
    )
    mle, mle_opt = functional.minimize_loss(processed_data, theta_true, None)
    assert mle_opt.success
    theta_name = data_info.query("name == @data_name")["theta_name"].item()

    for method in ["tabmgp", "bb", "copula"]:
        trace_path = utils.get_matching_files(
            f"{path_1001}/posterior-{loss}",
            rf"^{method}-\d+-\d+-\d+-trace\.pickle",
        )
        if len(trace_path) == 0:
            continue
        assert len(trace_path) == 1, (
            f"Expected exactly one trace file for {method} in {id}, but found {len(trace_path)}."
        )
        trace, trace_info = read_trace(trace_path[0])

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

        if SAVE_PLOTS:
            os.makedirs(f"{savedir}/trace", exist_ok=True)
            facet.savefig(
                f"{savedir}/trace/trace-{data_name}-{method}.pdf",
                bbox_inches="tight",
            )
        else:
            plt.show()
        plt.close(facet.figure)

# %%
# Expected l1-norm convergence plot (all setups)
fig = plt.figure(figsize=(7, 3))
for id in IDS:
    print(id)
    path_1001 = utils.get_matching_dirs(f"{output_dir}/{id}", "seed=1001")
    assert len(path_1001) == 1, (
        f"Expected exactly one path for seed=1001 in {id}, but found {len(path_1001)}."
    )
    path_1001 = path_1001[0]
    data_name = utils.get_data_name(path_1001)

    trace_path = utils.get_matching_files(
        f"{path_1001}/posterior-{loss}",
        rf"^tabmgp-\d+-\d+-\d+-trace\.pickle",
    )
    if len(trace_path) == 0:
        continue
    assert len(trace_path) == 1, (
        f"Expected exactly one trace file for tabmgp in {id}, but found {len(trace_path)}."
    )
    trace, trace_info = read_trace(trace_path[0])
    l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
    N_idx = np.arange(
        0, trace_info["end"] - trace_info["start"] + 1, trace_info["resolution"]
    )
    plt.plot(N_idx, np.mean(l1, axis=-1), color="black", linewidth=1, alpha=0.5)


plt.ylabel(r"Scaled $L_1$")
plt.xlabel("N - n")
plt.xlim(0, 500)

if SAVE_PLOTS:
    fig.savefig(
        f"{savedir}/l1-aggregate-0th-alldata-tabmgp.pdf",
        bbox_inches="tight",
    )
else:
    plt.show()


# %%
# Expected l1-norm convergence plot (individual setups)
for method in ["tabmgp", "bb"]:
    # Set squeeze=False so 'axes' is always a 2D array
    # for id, data_name in TITLE.items():
    for ids, ids_tag in [
        (LINEAR_REGRESSION_IDS, "linear"),
        (LOGISTIC_REGRESSION_IDS, "logistic"),
    ]:
        n_plots = len(ids)
        ncols = 4
        nrows = (n_plots + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 3.5), squeeze=False
        )

        # Flatten the 2D array of axes into a 1D array for easy looping
        axes = axes.flatten()
        plot_idx = 0  # Use this index to track which axis to plot on

        for id in ids:
            print(id)
            path_1001 = utils.get_matching_dirs(f"{output_dir}/{id}", "seed=1001")
            assert len(path_1001) == 1, (
                f"Expected exactly one path for seed=1001 in {id}, but found {len(path_1001)}."
            )
            path_1001 = path_1001[0]
            trace_path = utils.get_matching_files(
                f"{path_1001}/posterior-{loss}",
                rf"^{method}-\d+-\d+-\d+-trace\.pickle",
            )
            if len(trace_path) == 0:
                continue
            assert len(trace_path) == 1, (
                f"Expected exactly one trace file for {method} in {id}, but found {len(trace_path)}."
            )
            trace, trace_info = read_trace(trace_path[0])
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
            ax.set_title(TITLE[id])
            ax.axvline(x=N_idx[0], linestyle="--", color="grey", linewidth=1.0)
            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if SAVE_PLOTS:
            fig.savefig(
                f"{savedir}/l1-individual-0th-{ids_tag}-{method}.pdf",
                bbox_inches="tight",
            )
        else:
            plt.show()

# %%
# Extra long trace plots for selected setups
EXTRA_LONG_IDS = [
    "longroll-04",
    "longroll-05",
    "longroll-06",
    "longroll-01",
    "longroll-02",
    "longroll-03",
]
EXTRA_LONG_TITLE = {
    "longroll-04": "Classification GMM $a=0$",
    "longroll-05": "Classification GMM $a=-1$",
    "longroll-06": "Classification GMM $a=-2$",
    "longroll-01": "skin",
    "longroll-02": "yeast",
    "longroll-03": "wine",
}

# %%

# Extra long L1-norm convergence plot for selected setups (individual L1)
n_plots = len(EXTRA_LONG_IDS)
ncols = 3
nrows = (n_plots + ncols - 1) // ncols
for method in ["tabmgp"]:
    # Set squeeze=False so 'axes' is always a 2D array
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 6, nrows * 3.5), squeeze=False
    )

    # Flatten the 2D array of axes into a 1D array for easy looping
    axes = axes.flatten()
    plot_idx = 0  # Use this index to track which axis to plot on

    for id, data_name in EXTRA_LONG_TITLE.items():
        print(id)
        path_1001 = utils.get_matching_dirs(f"{output_dir}/{id}", "seed=1001")
        assert len(path_1001) == 1, (
            f"Expected exactly one path for seed=1001 in {id}, but found {len(path_1001)}."
        )
        path_1001 = path_1001[0]
        trace_path = utils.get_matching_files(
            f"{path_1001}/posterior-{loss}",
            rf"^{method}-\d+-\d+-\d+-trace\.pickle",
        )
        if len(trace_path) == 0:
            continue
        assert len(trace_path) == 1, (
            f"Expected exactly one trace file for {method} in {id}, but found {len(trace_path)}."
        )
        trace, trace_info = read_trace(trace_path[0])
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
        ax.set_title(EXTRA_LONG_TITLE[id])
        ax.axvline(x=N_idx[0], linestyle="--", color="grey", linewidth=1.0)
        plot_idx += 1

    for i in range(plot_idx, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(
            f"{savedir}/l1-individual-long-0th-nonconverge-{method}.pdf",
            bbox_inches="tight",
        )
    else:
        plt.show()

# %%

# Extra long L1-norm convergence plot for selected setups (expected L1)
fig = plt.figure(figsize=(7, 3))
for id in EXTRA_LONG_IDS:
    print(id)
    path_1001 = utils.get_matching_dirs(f"{output_dir}/{id}", "seed=1001")
    assert len(path_1001) == 1, (
        f"Expected exactly one path for seed=1001 in {id}, but found {len(path_1001)}."
    )
    path_1001 = path_1001[0]
    data_name = utils.get_data_name(path_1001)

    trace_path = utils.get_matching_files(
        f"{path_1001}/posterior-{loss}",
        r"^tabmgp-\d+-\d+-\d+-trace\.pickle",
    )
    if len(trace_path) == 0:
        continue
    assert len(trace_path) == 1, (
        f"Expected exactly one trace file for tabmgp in {id}, but found {len(trace_path)}."
    )
    trace, trace_info = read_trace(trace_path[0])
    l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
    N_idx = np.arange(
        0, trace_info["end"] - trace_info["start"] + 1, trace_info["resolution"]
    )
    plt.plot(N_idx, np.mean(l1, axis=-1), color="black", linewidth=1, alpha=0.5)


plt.ylabel(r"Scaled $L_1$")
plt.xlabel("N - n")
# plt.xlim(0, 500)

if SAVE_PLOTS:
    fig.savefig(
        f"{savedir}/extra-long-tabmgp-l1-0.pdf",
        bbox_inches="tight",
    )
else:
    plt.show()
# %%


# Expected l1-norm convergence plot, over 20 realisation of datasets
for method in ["tabmgp"]:
    for ids, ids_tag in [
        (LINEAR_REGRESSION_IDS, "linear"),
        (LOGISTIC_REGRESSION_IDS, "logistic"),
    ]:
        n_plots = len(ids)
        ncols = 4
        nrows = (n_plots + ncols - 1) // ncols

        # Set squeeze=False so 'axes' is always a 2D array
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(ncols * 6, nrows * 3.5), squeeze=False
        )

        # Flatten the 2D array of axes into a 1D array for easy looping
        axes = axes.flatten()
        plot_idx = 0  # Use this index to track which axis to plot on

        for id in ids:
            print(id)
            # seed that matches 1001, 1002, ..., 1020
            paths = utils.get_matching_dirs(
                f"{output_dir}/{id}", "seed=10(0[1-9]|1[0-9]|20)"
            )
            trace_path = utils.get_matching_files(
                f"{output_dir}/{id}",
                rf"^{method}-\d+-\d+-\d+-trace\.pickle",
                recursive=True,
            )
            trace_path = [s for s in trace_path if "posterior-likelihood" in s]
            assert len(trace_path) == 20, (
                f"Expected exactly 20 paths for seeds 1001-1020 in {id}, but found {len(trace_path)}."
            )

            exp_l1 = []
            for path in trace_path:
                trace, trace_info = read_trace(path)
                l1 = np.mean(np.abs((trace - trace[0])), axis=-1)
                exp_l1.append(np.mean(l1, axis=-1))  # expectation over rollouts
            exp_l1 = np.array(exp_l1)  # shape (realisations, N_idx)

            N_idx = np.arange(
                trace_info["start"], trace_info["end"] + 1, trace_info["resolution"]
            )
            ax = axes[plot_idx]
            ax.plot(N_idx, exp_l1.T, color="black", alpha=0.3, linewidth=1)
            # Expectation over realisation of datasets
            # ax.plot(N_idx, np.median(exp_l1, axis=0), color="black", linewidth=3)

            q1, q3 = np.quantile(exp_l1[:, -1], [0.25, 0.75], axis=0)
            iqr = q3 - q1
            ax.set_ylim(bottom=0, top=q3 + 2 * iqr)
            ax.set_ylabel(r"Scaled $L_1$")
            ax.set_xlabel("N")
            ax.set_title(TITLE[id])
            ax.axvline(x=N_idx[0], linestyle="--", color="grey", linewidth=1.0)
            plot_idx += 1

        for i in range(plot_idx, len(axes)):
            axes[i].axis("off")

        plt.tight_layout()
        if SAVE_PLOTS:
            fig.savefig(
                f"{savedir}/l1-expected-20reps-{ids_tag}-{method}.pdf",
                bbox_inches="tight",
            )
        else:
            plt.show()

# %%
