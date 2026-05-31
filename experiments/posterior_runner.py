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


def truncate_rollout(rollout: dict[str, Array], n: int) -> dict[str, Array]:
    leaves = jax.tree.leaves(rollout)
    chex.assert_equal_shape_prefix(leaves, 2)
    return jax.tree.map(lambda x: x[:, :n], rollout)


def available_forward_steps(rollout: dict[str, Array], n_train: int) -> int:
    leaves = jax.tree.leaves(rollout)
    chex.assert_equal_shape_prefix(leaves, 2)
    return leaves[0].shape[1] - n_train


def normalise_eval_t(eval_t) -> list[int]:
    return [int(t) for t in eval_t]


def save_mgp_posts(
    functional,
    rollout: dict[str, Array],
    init_theta: Array,
    savedir: str,
    method_name: str,
    n_train: int,
    eval_t,
    max_t_override: int | None = None,
) -> None:
    max_t = (
        available_forward_steps(rollout, n_train)
        if max_t_override is None
        else max_t_override
    )
    for t in filter(lambda x: x <= max_t, normalise_eval_t(eval_t)):
        start = timer()
        rollout_subset = truncate_rollout(rollout, n_train + t)
        post = functional.get_mgp(rollout_subset, init_theta)
        utils.write_to(f"{savedir}/{method_name}-{t}-post.pickle", post, verbose=True)
        logging.info(f"Diagnostics: {np.mean(post[1].success)}")
        logging.info(f"{method_name} posterior ({t}): {timer() - start:.2f} seconds")


def save_trace(
    functional,
    rollout: dict[str, Array],
    init_theta: Array,
    savedir: str,
    method_name: str,
    n_train: int,
    resolution: int,
    batch_size,
    require_final: bool = False,
    max_t_override: int | None = None,
) -> None:
    max_t = (
        available_forward_steps(rollout, n_train)
        if max_t_override is None
        else max_t_override
    )
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
        f"{savedir}/{method_name}-{n_train}-{max_t + n_train}-{freq}-trace.pickle",
        trace,
        verbose=True,
    )
    logging.info(f"{method_name} trace: {timer() - start:.2f} seconds")
