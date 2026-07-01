import logging
import os
import re
from dataclasses import dataclass
from timeit import default_timer as timer

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


def save_mgp_posts(
    functional,
    rollout: dict[str, Array],
    init_theta: Array,
    savedir: str,
    run_name: str,
    n_train: int,
    forward_steps,
) -> None:
    max_t = available_forward_steps(rollout, n_train)
    for raw_t in forward_steps:
        t = int(raw_t)
        if t > max_t:
            raise ValueError(f"Requested forward step {t} exceeds max {max_t}")
        start = timer()
        chex.assert_equal_shape_prefix(jax.tree.leaves(rollout), 2)
        rollout_subset = jax.tree.map(lambda x: x[:, : n_train + t], rollout)
        post = functional.get_mgp(rollout_subset, init_theta)
        utils.write_to(f"{savedir}/{run_name}-{t}-post.pickle", post, verbose=True)
        logging.info(f"Diagnostics: {np.mean(post[1].success)}")
        logging.info(f"{run_name} posterior ({t}): {timer() - start:.2f} seconds")


def save_trace(
    functional,
    rollout: dict[str, Array],
    init_theta: Array,
    savedir: str,
    run_name: str,
    n_train: int,
    resolution: int,
    batch_size,
    require_final: bool = False,
) -> None:
    max_t = available_forward_steps(rollout, n_train)
    freq = max(max_t // int(resolution), 1)
    if require_final and max_t % freq != 0:
        raise AssertionError(
            "Trace will not include the final state. Adjust resolution."
        )

    start = timer()
    trace = functional.get_theta_trace(
        rollout,
        init_theta,
        start=n_train - 1,
        batch_size=batch_size,
        freq=freq,
    )
    utils.write_to(
        f"{savedir}/{run_name}-{n_train}-{max_t + n_train}-{freq}-trace.pickle",
        trace,
        verbose=True,
    )
    logging.info(f"{run_name} trace: {timer() - start:.2f} seconds")
