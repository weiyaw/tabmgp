import logging
from timeit import default_timer as timer

import chex
import hydra
import jax
from omegaconf import DictConfig, OmegaConf

import baseline
import utils
from posterior_runner import (
    load_posterior_context,
    method_key,
    save_mgp_posts,
    save_trace,
)


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="bb")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    ctx = load_posterior_context(cfg.expdir, cfg.loss)
    bb_key = method_key(cfg.seed, "bb")

    logging.info("Run Bayesian bootstrap (BB).")
    start = timer()
    _, subkey = jax.random.split(bb_key)
    full_rollout = baseline.bootstrap_many_samples(
        subkey, ctx.train_data, cfg.rollout_times, cfg.rollout_length
    )
    logging.info(f"Shape of BB rollout: {utils.tree_shape(full_rollout)}")
    chex.assert_tree_shape_prefix(
        full_rollout,
        (cfg.rollout_times, cfg.rollout_length + ctx.n_train),
    )
    jax.block_until_ready(full_rollout)
    logging.info(f"BB rollout: {timer() - start:.2f} seconds")

    save_mgp_posts(
        ctx.functional,
        full_rollout,
        ctx.init_theta,
        ctx.savedir,
        "bb",
        ctx.n_train,
        cfg.eval_t,
    )
    if cfg.trace:
        save_trace(
            ctx.functional,
            full_rollout,
            ctx.init_theta,
            ctx.savedir,
            "bb",
            ctx.n_train,
            cfg.resolution,
            cfg.batch,
        )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
