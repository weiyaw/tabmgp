import logging
import os
from timeit import default_timer as timer

import chex
import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

import baseline
import utils
from posterior_runner import (
    available_forward_steps,
    load_posterior_context,
    method_key,
    normalize_forward_steps,
)


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="bb")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    os.makedirs(f"{cfg.expdir}/logs", exist_ok=True)
    os.makedirs(f"{cfg.expdir}/configs", exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.expdir}/configs/{cfg.run_name}.yaml")

    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    forward_steps = normalize_forward_steps(cfg.forward_steps)
    max_steps = max(forward_steps)
    ctx = load_posterior_context(cfg.expdir, cfg.loss)
    bb_key = method_key(cfg.seed, "bb")

    logging.info(f"Run {cfg.run_name}: Bayesian bootstrap.")
    start = timer()
    _, subkey = jax.random.split(bb_key)
    full_rollout = baseline.bootstrap_many_samples(
        subkey, ctx.train_data, cfg.rollout_times, max_steps
    )
    logging.info(f"Shape of {cfg.run_name} rollout: {utils.tree_shape(full_rollout)}")
    chex.assert_tree_shape_prefix(
        full_rollout,
        (cfg.rollout_times, max_steps + ctx.n_train),
    )
    jax.block_until_ready(full_rollout)
    logging.info(f"{cfg.run_name} rollout: {timer() - start:.2f} seconds")

    n_steps = available_forward_steps(full_rollout, ctx.n_train)
    if max_steps > n_steps:
        raise ValueError(f"Requested forward step {max_steps} exceeds max {n_steps}")

    for t in forward_steps:
        start = timer()
        rollout_t = jax.tree.map(lambda x: x[:, : ctx.n_train + t], full_rollout)
        post = ctx.functional.get_mgp(rollout_t, ctx.init_theta)
        utils.write_to(
            f"{ctx.savedir}/{cfg.run_name}-{t}-post.pickle", post, verbose=True
        )
        logging.info(f"Diagnostics: {np.mean(post[1].success)}")
        logging.info(f"{cfg.run_name} posterior ({t}): {timer() - start:.2f} seconds")

    if cfg.trace:
        start = timer()
        freq = max(n_steps // int(cfg.resolution), 1)
        trace = ctx.functional.get_theta_trace(
            full_rollout,
            ctx.init_theta,
            start=ctx.n_train - 1,
            batch_size=cfg.batch,
            freq=freq,
        )
        utils.write_to(
            f"{ctx.savedir}/{cfg.run_name}-{ctx.n_train}-{ctx.n_train + n_steps}-{freq}-trace.pickle",
            trace,
            verbose=True,
        )
        logging.info(f"{cfg.run_name} trace: {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
