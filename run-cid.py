import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import torch
import joblib
import math
from pathlib import Path
from tqdm import trange
from tqdm.auto import tqdm
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from timeit import default_timer as timer
import utils
import os

from functools import partial

from rollout import (
    TabPFNRegressorPredRule,
    TabPFNClassifierPredRule,
    assert_ppd_args_shape,
)

from dgp import DGPRegressionFixed, DGPClassificationFixed, DGPGamma

os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"


@eqx.filter_jit
def sample_x_truth(key, n, x_prev, get_x):
    """get_x is a function that returns (n, d) array"""
    return get_x(key, n)


@eqx.filter_jit
def sample_x_dirac(key, n, x_prev, x_mass):
    """x_mass is a (d, ) array"""
    return jnp.tile(x_mass, (n, 1))


def run_rollout(key, clf, x_new, sample_x, x_init, y_init, rollout_depth, save_path):
    """
    Run the rollout with some initial dataset x_init and y_init, then rollout up
    to rollout_depth.

    Parameters
    ----------
    key: jax.random.PRNGKey
        Random key for sampling.
    clf: TabPFNRegressorPPD
        PPD regressor.
    x_new : (m, d) array
        Query covariates.
    sample_x : Callable
        Function to sample the next covariate x. It will be called with
        arguments (key, n, x_prev).
    x_init : (n, d) array
        Initial dataset.
    y_init : (n,) array
        Initial dataset.
    rollout_depth : int
        Depth of the rollout.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    x_rollout : (rollout_depth, d) array
        Rolled out covariates.
    y_rollout : (rollout_depth,) array
        Rolled out targets.
    Saves rollout in {save_path}/rollout.pickle and return the rollout. It skips
    the computation if the file already exists.
    """
    assert_ppd_args_shape(x_new, x_init, y_init)
    n0 = x_init.shape[0]
    assert rollout_depth >= n0
    rollout_path = save_path / "rollout.pickle"
    if os.path.exists(rollout_path):
        logging.info("rollout exists")
        rollout = utils.read_from(rollout_path)
        x_rollout = rollout["x"]
        y_rollout = rollout["y"]
    else:
        start = timer()
        x_rollout = np.vstack([x_init, np.zeros((rollout_depth, x_init.shape[1]))])
        y_rollout = np.append(y_init, np.zeros((rollout_depth,), dtype=y_init.dtype))
        for i in trange(n0, rollout_depth + n0, desc="rollout", leave=False):
            loopkey = jr.fold_in(key, i)
            loopkey, subkey_x, subkey_y = jr.split(loopkey, 3)
            x_curr = sample_x(subkey_x, 1, None)  # we don't use x_prev for now
            y_curr = clf.sample(subkey_y, x_curr, x_rollout[:i], y_rollout[:i])
            x_rollout[i] = x_curr
            y_rollout[i] = y_curr
        assert x_rollout.shape[0] == y_rollout.shape[0] == rollout_depth + n0
        utils.write_to(rollout_path, {"x": x_rollout, "y": y_rollout})
        logging.info(f"rollout: {timer() - start:.2f} secs")
    return x_rollout, y_rollout


def compute_Fn(clf, x_rollout, y_rollout, x_new, t, n_points, save_path):
    """
    Compute F_n(x_new, t) along the rollout trajectory where
    F_n(x_new, t) = P( y <= t | x_new, x_{1:n}, y_{1:n})

    For each i point in n_points, it will evaluate F_n on x_rollout[:i]
    and y_rollout[:i].

    Parameters
    ----------
    clf: TabPFNRegressorPPD
        PPD regressor.
    x_rollout : (N, d) array
        Rolled out covariates.
    y_rollout : (N,) array
        Rolled out targets.
    x_new : (m, d) array
        Query covariates.
    t: (p, ) array
        Event of the PPD.
    n_points : list of int of length K
        List of points to evaluate F_n.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    It saves a dictionary with keys "F_n", "x_new", "t", and "n" in
    {save_path}/Fn.pickle. "F_n" is (K, p, m) array, i.e., F_n(x_new, t)
    together.
    """
    assert x_rollout.shape[0] == y_rollout.shape[0]
    assert x_rollout.shape[0] >= max(n_points), "Incomplete rollout"

    start = timer()
    F_n_all = []
    for n in n_points:
        x_prev, y_prev = x_rollout[:n], y_rollout[:n]
        F_n = clf.predict_event(t, x_new, x_prev, y_prev)  # (p, m)
        assert F_n.shape == (t.shape[0], x_new.shape[0])
        F_n_all.append(F_n)
    F_n_all = np.stack(F_n_all)
    utils.write_to(
        save_path / "Fn.pickle",
        {"F_n": F_n_all, "x_new": x_new, "t": t, "n": n_points},
    )
    logging.info(f"Fn: {timer() - start:.2f} secs")


def compute_Qn(clf, x_rollout, y_rollout, x_new, u, n_points, save_path):
    """
    Compute Q_n(x_new, u) along the rollout trajectory where
    Q_n(x_new, u) is the quantile (inverse cdf) function of F_n.

    For each i point in n_points, it will evaluate Q_n on x_rollout[:i]
    and y_rollout[:i].

    Parameters
    ----------
    clf: TabPFNRegressorPPD
        PPD regressor.
    x_rollout : (N, d) array
        Rolled out covariates.
    y_rollout : (N,) array
        Rolled out targets.
    x_new : (m, d) array
        Query covariates.
    u: (q, ) array
        Quantile of the PPD.
    n_points : list of int of length K
        List of points to evaluate Q_n.
    save_path : Path
        Path to save the computed bias.

    Returns
    -------
    It saves a dictionary with keys "Q_n", "x_new", "u", and "n" in
    {save_path}/Qn.pickle. "Q_n" is (K, q, m) array.
    """
    assert x_rollout.shape[0] == y_rollout.shape[0]
    assert x_rollout.shape[0] >= max(n_points), "Incomplete rollout"

    start = timer()
    Q_n_all = []
    for n in n_points:
        x_prev, y_prev = x_rollout[:n], y_rollout[:n]
        Q_n = clf.icdf(u, x_new, x_prev, y_prev)
        assert Q_n.shape == (u.shape[0], x_new.shape[0])
        Q_n_all.append(Q_n)
    Q_n_all = np.stack(Q_n_all)
    utils.write_to(
        save_path / "Qn.pickle",
        {"Q_n": Q_n_all, "x_new": x_new, "u": u, "n": n_points},
    )
    logging.info(f"Qn: {timer() - start:.2f} secs")


@hydra.main(version_base=None, config_path="conf", config_name="cid")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    n_estimators = int(cfg.n_estimators)
    sample_idx = int(cfg.sample_idx)
    resolution = int(cfg.resolution)
    rollout_length = int(cfg.rollout_length)
    seed = int(cfg.seed)
    rollout_x = cfg.rollout_x  # "truth" or "dirac:xx"
    n0 = cfg.data_size
    dgp_name = cfg.dgp.name
    dim_x = cfg.dgp.dim_x


    torch.set_num_threads(1)

    savedir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    logging.info(f"Experiment directory: {savedir}")

    # reproducibility
    key = jr.key(seed)
    key, key_sample, key_data = jr.split(key, 3)
    torch.manual_seed(seed)

    # Setup data
    if dgp_name == "gamma":
        shape, scale = cfg.dgp.shape, cfg.dgp.scale
        dgp = DGPGamma(key_data, n0, dim_x, shape, scale)
    elif dgp_name == "regression-fixed":
        noise_std = cfg.dgp.noise_std
        dgp = DGPRegressionFixed(key_data, n0, dim_x, noise_std)
    elif dgp_name == "classification-fixed":
        dgp = DGPClassificationFixed(key_data, n0, dim_x)

    utils.write_to(f"{savedir}/dgp.pickle", dgp)
    dim_x = dgp.train_data["x"].shape[-1]

    x_init, y_init = dgp.train_data["x"], dgp.train_data["y"]

    if rollout_x == "truth":
        sample_x = partial(sample_x_truth, get_x=dgp.get_x_data)
    elif rollout_x.startswith("dirac"):
        # extract dirac-xx
        mass = float(rollout_x.split(":")[1])
        x_mass = np.full((1, x_init.shape[1]), mass)
        sample_x = partial(sample_x_dirac, x_mass=x_mass)
    else:
        raise ValueError(f"Unknown rollout_x: {rollout_x}")

    # Setup prediction rule
    if dgp_name == "regression-fixed" or dgp_name == "gamma":
        pred_rule = TabPFNRegressorPredRule([False] * dim_x, n_estimators, False)
        t = np.linspace(y_init.min(), y_init.max(), 100)
    elif dgp_name == "classification-fixed":
        pred_rule = TabPFNClassifierPredRule([False] * dim_x, n_estimators, False)
        t = np.array([0, 1])

    x_new = np.tile(np.linspace(-1, 1, 5)[:, None], (1, dim_x))
    assert x_new.ndim == 2 and t.ndim == 1

    # ------------------------------------------------------------
    # 3.  Run A Single Path
    # ------------------------------------------------------------
    # Run one path (indexed by sample_idx). We rollout until the largest value
    # of n_points and compute F_n(x_new, t) term along the way.

    n_points = np.rint(np.linspace(n0, n0 + rollout_length, resolution)).astype(int)
    logging.info(f"Number of n_points: {len(n_points)}")

    key_sample = jr.fold_in(key_sample, sample_idx)
    save_path = savedir / f"sample-{sample_idx}"
    start = timer()

    key_path, _ = jr.split(key_sample)

    x_rollout, y_rollout = run_rollout(
        key_path, pred_rule, x_new, sample_x, x_init, y_init, max(n_points), save_path
    )

    compute_Fn(pred_rule, x_rollout, y_rollout, x_new, t, n_points, save_path)

    if isinstance(pred_rule, TabPFNRegressorPredRule):
        u = np.linspace(0.01, 0.99, 99)
        compute_Qn(pred_rule, x_rollout, y_rollout, x_new, u, n_points, save_path)
    logging.info(f"sample-{sample_idx}: {timer() - start:.2f} secs")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()

# %%
