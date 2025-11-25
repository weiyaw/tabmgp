# %%
import jax
import jax.numpy as jnp
from jax import vmap
import chex
import numpy as np
import pandas as pd

import utils
from utils import (
    get_date_part,
    get_data_size,
    get_resample_x,
    format_decimal,
    get_data_name,
)

from functional import (
    LogisticRegression,
    LinearRegression,
    QuantileRegression,
)

from experiment_setup import load_experiment, get_experiment_paths


from credible_set import (
    joint_credible_set,
    coverage_probability,
    marginal_credible_interval,
    marginal_coverage,
)

from typing import Any
import os
import re
import ast

KeyArray = jax.Array
PyTree = Any

jax.config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=np.inf)


def get_functional(experiment):
    """
    Get the task type from the experiment.
    """
    if isinstance(experiment, LogisticRegression):
        if experiment.n_classes == 2:
            return "likelihood-binary"
        else:
            return "likelihood-multiclass"
    elif isinstance(experiment, LinearRegression):
        return "likelihood-gaussian"
    elif isinstance(experiment, QuantileRegression):
        return f"quantile-{experiment.tau}"
    else:
        raise ValueError("Unknown experiment type")


def read_post(path, loss):
    """
    Given a path, e.g. "... seed=1001", Read the posterior.
    Return a dict of {post_name: tuple of posterior and diagnostics} where post_name is the name of the posterior.
    """
    posterior = {}
    for root, _, files in os.walk(path):
        if not root.endswith(f"posterior-{loss}"):
            continue
        for f in sorted(files):
            # match all files that ends with -post.pickle. post_name is everything before this pattern
            if match := re.search(r"(.+)-post.pickle", f):
                post_name = match.group(1)
                posterior[post_name] = utils.read_from(f"{root}/{f}")
    return posterior


def get_coverage_given_posterior(posteriors, cov_type, alpha, true_value):
    """
    posteriors: list of posterior samples array
    cov_type: "sphere", "diag", "ellipsoid"
    alpha: confidence level (Type 1 error)
    true_value: 1D array representing the true parameter value
    """
    crs = [joint_credible_set(m, alpha, cov_type=cov_type) for m in posteriors]
    rate, in_out = coverage_probability(crs, true_value)
    radius = np.asarray([cr["radius"] for cr in crs])
    post_cov_trace = np.asarray([cr["trace"] for cr in crs])
    post_cov_rank = np.asarray([cr["cov_rank"] for cr in crs])

    return rate, radius, post_cov_trace, post_cov_rank


def get_posterior_details(post_key):
    # post_key: gibbs, bb-x, tabpfn-x, copula-x
    if post_key == "gibbs":
        return {"post_name": "gibbs", "T": -1}
    elif (
        post_key.startswith("bb-")
        or post_key.startswith("tabpfn-")
        or post_key.startswith("copula-")
    ):
        return {"post_name": post_key.split("-")[0], "T": int(post_key.split("-")[1])}
    else:
        raise ValueError("Unknown posterior name")


def get_algo_success_rate(diagnostic_ls):
    if len(diagnostic_ls) == 0:
        return None
    else:
        try:
            return np.mean(np.asarray([d.success for d in diagnostic_ls]))
        except ValueError:
            print("warning: samples might be missing in some seeds")
            return None


# The directory that contains all the experiment outputs
output_dir = "../outputs"

# %%
# Joint credible set coverage
loss = "likelihood"
rows = []
for date in DATES:
    print(f"Date: {date}")
    all_paths = get_experiment_paths(f"{output_dir}/{date}")
    all_dgps = [utils.read_from(f"{p}/dgp.pickle") for p in all_paths]

    preprocessor, functional, theta_true, _ = load_experiment(all_paths[0], loss=loss)
    all_train_data = [preprocessor.encode_data(dgp.train_data) for dgp in all_dgps]
    dim_theta = theta_true.shape[0]
    dim_x = all_train_data[0]["x"].shape[-1]

    # read all posterior and diagnostics as a dict for each seed
    all_post_diagnostics = [read_post(p, loss) for p in all_paths]
    assert all(
        [post.keys() == all_post_diagnostics[0].keys() for post in all_post_diagnostics]
    )

    # turn all posterior and diagnostics from a list of dict to a dict of list
    all_posterior = {
        k: [post[k][0] for post in all_post_diagnostics]
        for k in all_post_diagnostics[0].keys()
    }
    all_diagnostics = {
        k: [post[k][1] for post in all_post_diagnostics]
        for k in all_post_diagnostics[0].keys()
    }

    rank_x_ls = [np.linalg.matrix_rank(data["x"]) for data in all_train_data]

    for post_name, posterior in all_posterior.items():
        # For each type of posterior
        for alpha in [0.05, 0.2]:
            # For each alpha
            post_details = get_posterior_details(post_name)
            cov_type = "ellipsoid" if post_details["post_name"] == "gibbs" else "diag"
            rate, radius, post_cov_trace, post_cov_rank = get_coverage_given_posterior(
                posterior, cov_type, alpha, theta_true
            )
            algo_rate = get_algo_success_rate(all_diagnostics[post_name])
            row = {
                "data": get_data_name(all_paths[0]),
                "functional": get_functional(functional),
                **post_details,
                "cov_type": cov_type,
                "rate": format_decimal(rate, 2),
                "ideal_rate": 1 - alpha,
                "post_cov_trace_mean": format_decimal(np.mean(post_cov_trace), 4),
                "post_cov_trace_median": format_decimal(np.median(post_cov_trace), 4),
                "post_cov_trace_q3": format_decimal(
                    np.quantile(post_cov_trace, 0.75), 4
                ),
                "dim_theta": dim_theta,
                "post_cov_rank_mean": format_decimal(np.mean(post_cov_rank), 2),
                "training_set_size": get_data_size(all_paths[0]),
                "resample_x": get_resample_x(all_paths[0]),
                "dim_x": dim_x,
                "algo_success_rate": format_decimal(algo_rate, 2),
                "avg_rank_x": format_decimal(np.mean(rank_x_ls), 2),
                "date": get_date_part(all_paths[0]),
            }
            print(row)
            rows.append(row)

# flag all rows that are either Gibbs or have the maximum T
df = pd.DataFrame(rows)
df["max_T"] = df["T"].isna() | (
    df["T"] == df.groupby(["data", "post_name"])["T"].transform("max")
)
os.makedirs("table", exist_ok=True)
df.to_csv("./table/joint-coverage.csv")

# %%
# Marginal credible interval coverage

# Setup the variable name for each dimension of theta
data_info = pd.read_csv("./data_info.csv")
data_info["theta_name"] = data_info["theta_name"].apply(ast.literal_eval)

ALPHA = 0.05
loss = "likelihood"
marginal_ci_coverage_ls = []
for date in DATES:
    print(f"Date: {date}")
    all_paths = get_experiment_paths(f"{output_dir}/{date}")
    data_name = utils.get_data_name(all_paths[0])
    theta_name = data_info[data_info["name"] == data_name]["theta_name"].iloc[0]
    _, _, theta_true, _ = load_experiment(all_paths[0], loss=loss)
    dim_theta = theta_true.shape[0]

    # read all posterior and diagnostics as a dict for each seed
    all_post_diagnostics = [read_post(p, loss) for p in all_paths]
    assert all(
        [post.keys() == all_post_diagnostics[0].keys() for post in all_post_diagnostics]
    )

    # turn all posterior from a list of dict to a dict of list
    all_posterior = {
        k: [post[k][0] for post in all_post_diagnostics]
        for k in all_post_diagnostics[0].keys()
    }

    for post_name, posterior_ls in all_posterior.items():
        marginal_ci_ls = [marginal_credible_interval(p, ALPHA) for p in posterior_ls]
        coverage, width, winkler = marginal_coverage(marginal_ci_ls, theta_true, ALPHA)
        post_details = get_posterior_details(post_name)
        for c, w, wk, tn in zip(coverage, width, winkler, theta_name):
            marginal_ci_coverage_ls.append(
                {
                    "data": data_name,
                    "ideal_rate": 1 - ALPHA,
                    **post_details,
                    "theta_name": tn,
                    "rate": c,
                    "median_width": w,
                    "median_winkler": wk,
                }
            )
df = pd.DataFrame(marginal_ci_coverage_ls)
df["max_T"] = df["T"].isna() | (
    df["T"] == df.groupby(["data", "post_name"])["T"].transform("max")
)

os.makedirs("table", exist_ok=True)
df.to_csv("table/marginal-coverage.csv", index=False)


# %%
