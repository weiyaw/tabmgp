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
    get_name,
    get_data_size,
    get_resample_x,
    format_decimal,
)
from posterior import (
    load_experiment,
    Diagnostics,
    OptResult,
    Classification,
    Regression,
    QuantileRegression,
)

from credible_set import (
    joint_credible_set,
    coverage_probability,
    marginal_credible_interval,
    marginal_coverage,
)

from plot_settings import DATES, POSTERIOR_NAMES

from typing import Any
import os
import re

KeyArray = jax.Array
PyTree = Any

jax.config.update("jax_enable_x64", True)
np.set_printoptions(linewidth=np.inf)


def diagnose_minimizer(opt_state, **kwargs):
    if hasattr(opt_state, "success"):
        # jax.scipy and scipy BFGS success rate
        return opt_state.success
    elif isinstance(opt_state, dict) and "success" in opt_state:
        # acceptance rate of optax LBFGS
        return opt_state["success"]
    elif isinstance(opt_state, list):
        # jnp.linalg.lstsq, check the rank of A of the least squares Ax=b
        dim_theta = kwargs["dim_theta"]
        # there are dim_theta number of coef in the regression examples (include intercept, no log_std)
        return opt_state[1] == dim_theta
    elif isinstance(opt_state, dict) and "acceptance_rate" in opt_state:
        # acceptance rate of NUTS
        return opt_state["acceptance_rate"]


def get_loss(experiment):
    """
    Get the task type from the experiment.
    """
    if isinstance(experiment, Classification):
        if experiment.n_classes == 2:
            return "likelihood-binary"
        else:
            return "likelihood-multiclass"
    elif isinstance(experiment, Regression):
        return "likelihood-gaussian"
    elif isinstance(experiment, QuantileRegression):
        return f"quantile-{experiment.tau}"
    else:
        raise ValueError("Unknown experiment type")


def read_post(path, name, loss):
    """
    Given a path, e.g. "... seed=1001", Read the posterior.

    Return a dict of the form {T: tuple of posterior and diagnostics} where T
    is the number of forward recursion. For Gibbs, return the object directly.
    """
    posterior = {}
    for root, _, files in os.walk(path):
        if not root.endswith(f"posterior-{loss}"):
            continue
        for f in sorted(files):
            if name == "gibbs":
                # match gibbs-post.pickle
                if re.search(r"gibbs-post.pickle", f):
                    posterior = utils.read_from(f"{root}/{f}")
            else:
                # match bb/tabpfn/copula-x-post.pickle
                if match := re.search(rf"{name}-(\d+)-post.pickle", f):
                    T = int(match.group(1))
                    posterior[T] = utils.read_from(f"{root}/{f}")
    return posterior


# %%
loss = "likelihood"
for date in DATES:
    all_details = []
    all_posterior_ls = []

    print(f"Date: {date}")
    experiment = load_experiment(f"../outputs/{date}", loss=loss)
    path = experiment.all_paths[0]
    theta_true = experiment.theta_true
    dim_theta = theta_true.shape[0]
    init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))

    for name in POSTERIOR_NAMES.keys():
        # Read posterior for each type
        post_all = [read_post(p, name, loss) for p in experiment.all_paths]
        if not post_all or not post_all[0]:
            # if no posterior is found
            continue
        if name == "gibbs":
            # Gibbs posterior does not have T, so we skip the loop
            all_details.append({"post_name": name, "T": None, "max_T": True})
            all_posterior_ls.append(post_all)
            continue
        # For each type of posterior, we assume all paths have the same T
        max_T = max(post_all[0].keys())
        for T in post_all[0].keys():
            all_details.append({"post_name": name, "T": T, "max_T": T == max_T})
            all_posterior_ls.append([post[T] for post in post_all])
        pass

    dim_x = experiment.all_train_data[0]["x"].shape[-1]
    rank_x_ls = [np.linalg.matrix_rank(data["x"]) for data in experiment.all_train_data]

    for details, posterior_opt_ls in zip(all_details, all_posterior_ls):
        # For each type of posterior
        for alpha in [0.05, 0.2]:
            crs = []
            for m, _ in posterior_opt_ls:
                if details["post_name"] == "gibbs":
                    # we have more posterior samples from MCMC to afford using full cov
                    crs.append(joint_credible_set(m, alpha, cov_type="ellipsoid"))
                else:
                    crs.append(joint_credible_set(m, alpha, cov_type="diag"))

            rate, in_out = coverage_probability(crs, theta_true)
            radius = np.asarray([cr["radius"] for cr in crs])
            post_cov_trace = np.asarray([cr["trace"] for cr in crs])
            post_cov_rank = np.asarray([cr["cov_rank"] for cr in crs])

            # diagnose_minimizer(opt, dim_theta=theta_true.size)
            algo_success_ls = [
                diagnostics.success for _, diagnostics in posterior_opt_ls
            ]

            if len(algo_success_ls) == 0:
                algo_rate = None
            else:
                try:
                    algo_rate = np.mean(np.asarray([algo_success_ls]))
                except ValueError:
                    print("warning: samples might be missing in some seeds")
                    algo_rate = np.nan

            row = {
                "name": get_name(path),
                "loss": get_loss(experiment),
                **details,
                "rate": format_decimal(rate, 2),
                "ideal_rate": 1 - alpha,
                "post_cov_trace_mean": format_decimal(np.mean(post_cov_trace), 4),
                "post_cov_trace_median": format_decimal(np.median(post_cov_trace), 4),
                "post_cov_trace_q3": format_decimal(
                    np.quantile(post_cov_trace, 0.75), 4
                ),
                "dim_theta": experiment.theta_true.size,
                "post_cov_rank_mean": format_decimal(np.mean(post_cov_rank), 2),
                "training_set_size": get_data_size(path),
                "resample_x": get_resample_x(path),
                "dim_x": dim_x,
                "algo_success_rate": format_decimal(algo_rate),
                "avg_rank_x": format_decimal(np.mean(rank_x_ls), 2),
                "date": get_date_part(path),
            }
            print(row)

            df = pd.DataFrame([row])
            save_path = "./table/joint-coverage.csv"
            df.to_csv(
                save_path,
                mode="a",
                header=not pd.io.common.file_exists(save_path),
                index=False,
            )

# %%
#
data_info = utils.read_from(f"./data_info.pickle")
ALPHA = 0.05
loss = "likelihood"

def read_max_T_post(path, name, loss):
    post = read_post(path, name, loss)
    if post:
        max_T = max(post.keys())
        return post[max_T]
    else:
        return None, None


marginal_ci_coverage_ls = []
for date in DATES:
    print(date)
    experiment = load_experiment(f"../outputs/{date}", loss)
    name = utils.get_name(experiment.all_paths[0])
    theta_name = data_info[data_info["name"] == name]["theta_name"].iloc[0]
    theta_true = experiment.theta_true

    for post_name in POSTERIOR_NAMES.keys():
        marginal_ci_ls = []
        for p in experiment.all_paths:
            if post_name == "gibbs":
                max_T_post, _ = utils.read_from(f"{p}/posterior-{loss}/gibbs-post.pickle")
            else:
                max_T_post, _ = read_max_T_post(p, post_name, loss)

            if max_T_post is not None:
                marginal_ci_ls.append(marginal_credible_interval(max_T_post, ALPHA))

        if marginal_ci_ls:
            coverage, width, winkler = marginal_coverage(
                marginal_ci_ls, theta_true, ALPHA
            )
            for c, w, wk, tn in zip(coverage, width, winkler, theta_name):
                marginal_ci_coverage_ls.append(
                    {
                        "name": name,
                        "ideal_rate": 1 - ALPHA,
                        "post_name": post_name,
                        "theta_name": tn,
                        "rate": c,
                        "median_width": w,
                        "median_winkler": wk,
                    }
                )

df_results = pd.DataFrame(marginal_ci_coverage_ls)
df_results.to_csv("table/marginal-coverage.csv", index=False)


# %%
