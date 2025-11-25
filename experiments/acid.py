import os

from rollout import TabPFNClassifierPredRule, TabPFNRegressorPredRule
import jax
import jax.numpy as jnp
import chex
import numpy as np

from jax.typing import ArrayLike
from scipy.special import logsumexp, log_softmax

import torch

from timeit import default_timer as timer

from data import (
    OPENML_CLASSIFICATION,
    OPENML_BINARY_CLASSIFICATION,
    OPENML_REGRESSION,
)
import utils
import logging
import hydra

from omegaconf import DictConfig, OmegaConf

import warnings

jax.config.update("jax_enable_x64", True)

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


log = logging.getLogger(__name__)


class TabPFNRegresorPredRuleAcid(TabPFNRegressorPredRule):

    def log_prob(self, x_new: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
        pred_output = self.predict(x_new, output_type="full")
        logits = pred_output["logits"].cpu().numpy()
        return log_softmax(logits, axis=-1)


class TabPFNClassifierPredRuleAcid(TabPFNClassifierPredRule):

    def log_prob(self, x_new: np.ndarray) -> np.ndarray:
        return np.log(self.predict_proba(x_new))


def bin_logprob(x, minlength=0):
    # similar to np.bincount, but returns log probabilities instead of counts
    chex.assert_rank(x, 1)
    log_prob = np.log(np.bincount(x, minlength=minlength)) - np.log(x.size)
    assert np.allclose(logsumexp(log_prob), 0.0, atol=1e-4)
    return log_prob


def row_indices(x, y):
    """
    For each row in arry y, return the corresponding row indices in array x

    Args:
      x: A 2D NumPy array of unique rows.
      y: A 2D NumPy array where each row is a sample from x.

    Returns:
      A 1D NumPy array where the i-th element is the count of
      the i-th row of x in y.

    """
    # Create a mapping from a tuple-version of a row in x to its index.
    # Tuples are used because numpy arrays are not hashable.
    x = np.asarray(x)
    y = np.asarray(y)
    x_map = {tuple(row): i for i, row in enumerate(x)}

    # Use the map to find the index for each row in y.
    y_indices = [x_map[tuple(row)] for row in y]

    return np.array(y_indices)


def unique_rows(arr):
    """
    Provides a robust equivalent of np.unique(arr, axis=0) for object arrays.

    It works by converting rows to hashable tuples and using a set for
    efficient uniqueness checking.
    """
    if arr.ndim != 2:
        raise ValueError("Input array must be 2-dimensional.")

    seen = set()
    unique_rows = []
    index = []

    if not np.issubdtype(arr.dtype, object):
        # use the built-in method if it's usable
        return np.unique(arr, axis=0)

    for i, row in enumerate(arr):
        row_tuple = tuple(row)
        if row_tuple not in seen:
            seen.add(row_tuple)
            unique_rows.append(row)
            index.append(i)

    # Convert list of rows back to a NumPy array
    return np.array(unique_rows, dtype=object)


def delta_cond_logppd(key, ppd, x_eval, x_prev, y_prev, L):

    # One-step-ahead ppd (cond y | x_eval)
    ppd.fit(x_prev, y_prev)
    chex.assert_shape(x_eval, (1, None))
    one_step_cond_logpmf_y_x = ppd.log_prob(x_eval)
    chex.assert_shape(one_step_cond_logpmf_y_x, (1, None))

    # Draw samples from the one-step-ahead ppd to Monte-Carlo estimate
    # two-step-ahead ppd
    key, subkey = jax.random.split(key)
    batch_x_eval = np.tile(x_eval, (L, 1))
    y_new = ppd.sample(subkey, batch_x_eval)
    chex.assert_equal_shape_prefix([y_new, batch_x_eval], 1)

    # Monte-Carlo estimate of two-step-ahead ppd
    two_step_cond_logpmf_y_x_ls = []
    x_plus_1 = np.concatenate([x_prev, x_eval], axis=0)
    for i in range(L):
        y_plus_1 = np.concatenate([y_prev, y_new[i, np.newaxis]])
        ppd.fit(x_plus_1, y_plus_1)
        two_step_cond_logpmf_y_x = ppd.log_prob(x_eval)
        chex.assert_shape(two_step_cond_logpmf_y_x, (1, None))
        two_step_cond_logpmf_y_x_ls.append(two_step_cond_logpmf_y_x)
    two_step_cond_logpmf_y_x = np.stack(two_step_cond_logpmf_y_x_ls)
    two_step_cond_logpmf_y_x = logsumexp(two_step_cond_logpmf_y_x, axis=0) - np.log(L)

    # Check if the conditional pmf sum to 1
    chex.assert_shape(one_step_cond_logpmf_y_x, (1, None))
    chex.assert_shape(two_step_cond_logpmf_y_x, (1, None))
    if not np.allclose(logsumexp(one_step_cond_logpmf_y_x), 0.0, atol=1e-4):
        log.info(f"One-step cond pmf not summing to 1")
    if not np.allclose(logsumexp(two_step_cond_logpmf_y_x), 0.0, atol=1e-4):
        log.info(f"Two-step cond pmf not summing to 1")

    return one_step_cond_logpmf_y_x, two_step_cond_logpmf_y_x


def delta_joint_logppd(key, ppd, x_support, idx_prev, y_prev, L):
    chex.assert_rank(x_support, 2)
    chex.assert_equal_shape_prefix([idx_prev, y_prev], 1)
    n_uniq_x = x_support.shape[0]

    # One-step-ahead ppd (joint x, y)
    ppd.fit(x_support[idx_prev], y_prev)
    one_step_logpmf_x = bin_logprob(idx_prev, minlength=n_uniq_x)
    one_step_cond_logpmf_y_x = ppd.log_prob(x_support)
    chex.assert_shape(one_step_cond_logpmf_y_x, (n_uniq_x, None))
    one_step_joint_logpmf_y_x = one_step_cond_logpmf_y_x + one_step_logpmf_x[:, None]

    # Draw samples from the one-step-ahead ppd to Monte-Carlo estimate
    # two-step-ahead ppd
    key, subkey = jax.random.split(key)
    idx_new = jax.random.choice(subkey, a=idx_prev, shape=(L,), replace=True)
    key, subkey = jax.random.split(key)
    y_new = ppd.sample(subkey, x_support[idx_new])
    chex.assert_shape([y_new, idx_new], (L,))

    # Monte-Carlo estimate of two-step-ahead ppd
    two_step_joint_logpmf_y_x_ls = []
    for i in range(L):
        idx_plus_1 = np.concatenate([idx_prev, idx_new[i, np.newaxis]])
        y_plus_1 = np.concatenate([y_prev, y_new[i, np.newaxis]])
        ppd.fit(x_support[idx_plus_1], y_plus_1)
        two_step_logpmf_x = bin_logprob(idx_plus_1, minlength=n_uniq_x)
        two_step_cond_logpmf_y_x = ppd.log_prob(x_support)
        chex.assert_shape(two_step_cond_logpmf_y_x, (n_uniq_x, None))
        two_step_joint_logpmf_y_x_ls.append(
            two_step_cond_logpmf_y_x + two_step_logpmf_x[:, None]
        )
    two_step_joint_logpmf_y_x = np.stack(two_step_joint_logpmf_y_x_ls)
    two_step_joint_logpmf_y_x = logsumexp(two_step_joint_logpmf_y_x, axis=0) - np.log(L)

    # Check if the conditional pmf sum to 1
    chex.assert_shape(one_step_joint_logpmf_y_x, (n_uniq_x, None))
    chex.assert_shape(two_step_joint_logpmf_y_x, (n_uniq_x, None))
    if not np.allclose(logsumexp(one_step_joint_logpmf_y_x), 0.0, atol=1e-4):
        log.info(f"One-step joint pmf not summing to 1")
    if not np.allclose(logsumexp(two_step_joint_logpmf_y_x), 0.0, atol=1e-4):
        log.info(f"Two-step joint pmf not summing to 1")

    return one_step_joint_logpmf_y_x, two_step_joint_logpmf_y_x


@hydra.main(version_base=None, config_path="conf", config_name="diagnostics")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    path = cfg.expdir

    log.setLevel(logging.INFO)
    os.makedirs(f"{path}/acid-log", exist_ok=True)
    log.addHandler(
        logging.FileHandler(
            f"{path}/acid-log/sample-{cfg.sample_idx}.log", mode="w", delay=True
        )
    )
    log.info(f"Hydra version: {hydra.__version__}")
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    savedir = f"{path}/acid"
    main_key = jax.random.key(utils.get_seed(path) * 38)
    sample_key = jax.random.fold_in(main_key, cfg.sample_idx + 108)
    torch.manual_seed((cfg.sample_idx + 1) * 12)

    name = utils.get_name(path)
    dgp = utils.read_from(f"{path}/dgp.pickle")

    n_train = utils.get_n_data(dgp.train_data)
    N = n_train + cfg.recursion_length
    x_train = dgp.train_data["x"]
    y_train = dgp.train_data["y"]
    x_support = unique_rows(x_train)
    dim_x = x_support.shape[1]

    if name is not None and name.startswith("classification"):
        pfn_ppd = TabPFNClassifierPredRuleAcid([False] * dim_x)
    elif name is not None and name.startswith("regression"):
        pfn_ppd = TabPFNRegresorPredRuleAcid([False] * dim_x)
    elif name in OPENML_CLASSIFICATION + OPENML_BINARY_CLASSIFICATION:
        pfn_ppd = TabPFNClassifierPredRuleAcid(dgp.categorical_x)
    elif name in OPENML_REGRESSION:
        pfn_ppd = TabPFNRegresorPredRuleAcid(dgp.categorical_x)
    elif name == "gamma":
        pfn_ppd = TabPFNRegresorPredRuleAcid([False] * dim_x)
    else:
        raise ValueError(f"Unknown dgp name {name}")

    freq = (N - n_train) // cfg.resolution
    log.info(f"Computing {n_train}:{N}:{freq}")

    # Evaluate at a few x across the support
    x_eval_idx = np.linspace(0, x_support.shape[0] - 1, cfg.num_x_eval, dtype=int)
    start_loop = timer()
    for j in x_eval_idx:
        x_eval = x_support[np.newaxis, j]
        x_key = jax.random.fold_in(sample_key, j)
        one_step_cond_logpmf_y_x_over_time = []
        two_step_cond_logpmf_y_x_over_time = []
        log.info(
            f"Evaluating x[{j}]={np.array2string(x_eval.squeeze(), precision=4, separator=',')}"
        )
        x_prev = x_train
        y_prev = y_train
        start = timer()
        for i in range(n_train, N + 1):
            key = jax.random.fold_in(x_key, i)
            pfn_ppd.fit(x_prev, y_prev)
            key, subkey = jax.random.split(key)
            y_new = pfn_ppd.sample(subkey, x_eval)
            x_prev = np.append(x_prev, x_eval, axis=0)
            y_prev = np.append(y_prev, y_new, axis=0)

            if (i - n_train) % freq == 0:
                one_step_cond_logpmf_y_x, two_step_cond_logpmf_y_x = delta_cond_logppd(
                    subkey, pfn_ppd, x_eval, x_prev, y_prev, cfg.mc_samples
                )
                one_step_cond_logpmf_y_x_over_time.append(one_step_cond_logpmf_y_x)
                two_step_cond_logpmf_y_x_over_time.append(two_step_cond_logpmf_y_x)

                # Compute log of total variational distance: log sup_{A} | p2(A) - p1(A)|
                log_delta_cond_pmf_y_x, _ = logsumexp(
                    np.stack(
                        [two_step_cond_logpmf_y_x, one_step_cond_logpmf_y_x], axis=-1
                    ),
                    axis=-1,
                    b=np.asarray([1, -1]),
                    return_sign=True,
                )
                sup_log_delta_cond_pmf_y_x = np.max(log_delta_cond_pmf_y_x, axis=-1)

                log.info(
                    f"log of sup \u0394P(y | x=x[{j}], "
                    f"z_1:{i}): {np.squeeze(sup_log_delta_cond_pmf_y_x):.6f}"
                )
        cond_logpmf_y_x_over_time = {
            "one_step_cond_y_x": np.concatenate(one_step_cond_logpmf_y_x_over_time),
            "two_step_cond_y_x": np.concatenate(two_step_cond_logpmf_y_x_over_time),
        }
        chex.assert_shape(
            jax.tree.leaves(cond_logpmf_y_x_over_time), (cfg.resolution + 1, None)
        )

        # shape: (N_idx, dim_y)
        utils.write_to(
            f"{savedir}/x-eval-{j}/logpmf-{cfg.sample_idx}-{n_train}-{N}-{freq}.pickle",
            cond_logpmf_y_x_over_time,
            verbose=True,
        )

        log.info(f"Time for x=x[{j}]: {timer() - start:.2f} secs")

    log.info(f"acid {cfg.sample_idx} takes {timer() - start_loop:.2f} secs")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
