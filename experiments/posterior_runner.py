import logging
import os
import re
from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

import optimizer
import utils
from experiment_setup import load_experiment


METHOD_KEYS = {
    "bb": 49195,
    "bayes": 16005,
    "copula": 91501,
}


@dataclass(frozen=True)
class PosteriorContext:
    expdir: str
    loss: str
    savedir: str
    dgp: object
    preprocessor: object
    functional: object
    theta_true: Array
    train_data: dict[str, Array]
    n_train: int
    init_theta: Array


def normalize_forward_steps(forward_steps) -> list[int]:
    if isinstance(forward_steps, int):
        return [forward_steps]
    return [int(step) for step in forward_steps]


def method_key(seed: int, method: str) -> Array:
    if method not in METHOD_KEYS:
        raise ValueError(f"Unknown method key: {method}")
    return jax.random.fold_in(jax.random.key(seed), METHOD_KEYS[method])


def load_posterior_context(expdir: str, loss: str) -> PosteriorContext:
    dgp = utils.read_from(f"{expdir}/dgp.pickle")
    preprocessor, functional, theta_true, train_data = load_experiment(expdir, loss)
    n_train = utils.get_n_data(train_data)

    logging.info(f"dim_theta: {theta_true.size}")
    mle, mle_opt = functional.minimize_loss(train_data, theta_true, None)
    if hasattr(mle_opt, "success") and not mle_opt.success:
        logging.info("Optimization failed. MLE might be wrong. Use scipy.")
        mle = optimizer.scipy_mle(functional.loss, train_data, theta_true)

    return PosteriorContext(
        expdir=expdir,
        loss=loss,
        savedir=f"{expdir}/posterior-{loss}",
        dgp=dgp,
        preprocessor=preprocessor,
        functional=functional,
        theta_true=theta_true,
        train_data=train_data,
        n_train=n_train,
        init_theta=mle,
    )


def compile_rollout(expdir: str, rollout_dir_name: str) -> dict[str, Array]:
    rollout_dir = f"{expdir}/{rollout_dir_name}"
    rollout_paths = [
        p for p in os.listdir(rollout_dir) if re.match(r"rollout-\d+\.pickle", p)
    ]
    rollout_paths.sort(
        key=lambda p: int(re.search(r"rollout-(\d+)\.pickle", p).group(1))
    )
    rollouts = [utils.read_from(f"{rollout_dir}/{p}") for p in rollout_paths]
    if not rollouts:
        raise FileNotFoundError(f"No rollout-*.pickle files found in {rollout_dir}")
    return {k: np.stack([dic[k] for dic in rollouts]) for k in rollouts[0]}


def available_forward_steps(rollout: dict[str, Array], n_train: int) -> int:
    leaves = jax.tree.leaves(rollout)
    chex.assert_equal_shape_prefix(leaves, 2)
    return leaves[0].shape[1] - n_train
