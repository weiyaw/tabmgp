import logging
import os
from timeit import default_timer as timer

import hydra
import jax
import numpy as np
from omegaconf import DictConfig, OmegaConf

import utils
from baseline import (
    copula_classification,
    copula_classification_tabpfn_init,
    copula_cregression,
    copula_cregression_tabpfn_init,
)
from functional import LinearRegression, LogisticRegression
from posterior_runner import (
    available_forward_steps,
    load_posterior_context,
    method_key,
    normalize_forward_steps,
)


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="copula")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    os.makedirs(f"{cfg.expdir}/logs", exist_ok=True)
    os.makedirs(f"{cfg.expdir}/configs", exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.expdir}/configs/{cfg.run_name}.yaml")

    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    if cfg.init not in ["std", "tabpfn"]:
        raise ValueError("cfg.init must be either 'std' or 'tabpfn'.")

    forward_steps = normalize_forward_steps(cfg.forward_steps)
    max_steps = max(forward_steps)
    ctx = load_posterior_context(cfg.expdir, cfg.loss)
    method_key(cfg.seed, "copula")

    logging.info(f"Run {cfg.run_name}: Copula MGP with {cfg.init} init.")
    start = timer()
    categorical_x = getattr(
        ctx.dgp,
        "categorical_x",
        [False] * ctx.dgp.train_data["x"].shape[-1],
    )

    if isinstance(ctx.functional, LinearRegression):
        if cfg.init == "tabpfn":
            full_rollout, _ = copula_cregression_tabpfn_init(
                ctx.dgp.train_data,
                categorical_x,
                cfg.rollout_times,
                max_steps,
                cfg.num_y_grid,
            )
        else:
            full_rollout, _ = copula_cregression(
                ctx.dgp.train_data,
                categorical_x,
                cfg.rollout_times,
                max_steps,
                cfg.num_y_grid,
            )
    elif (
        isinstance(ctx.functional, LogisticRegression) and ctx.functional.n_classes == 2
    ):
        if cfg.init == "tabpfn":
            full_rollout, _ = copula_classification_tabpfn_init(
                ctx.dgp.train_data,
                categorical_x,
                cfg.rollout_times,
                max_steps,
            )
        else:
            full_rollout, _ = copula_classification(
                ctx.dgp.train_data,
                categorical_x,
                cfg.rollout_times,
                max_steps,
            )
    elif (
        isinstance(ctx.functional, LogisticRegression) and ctx.functional.n_classes > 2
    ):
        logging.info(f"{cfg.run_name} not available for multiclass classification.")
        full_rollout = None
    else:
        raise NotImplementedError

    if full_rollout is None:
        return

    jax.block_until_ready(full_rollout)
    logging.info(f"{cfg.run_name} rollout: {timer() - start:.2f} seconds")
    full_rollout = ctx.preprocessor.encode_data(full_rollout)

    if isinstance(ctx.functional, LinearRegression):
        start = timer()
        post = ctx.functional.get_mgp(full_rollout, ctx.init_theta)
        utils.write_to(
            f"{ctx.savedir}/{cfg.run_name}-{max_steps}-post.pickle", post, verbose=True
        )
        logging.info(f"Diagnostics: {np.mean(post[1].success)}")
        logging.info(
            f"{cfg.run_name} posterior ({max_steps}): {timer() - start:.2f} seconds"
        )
        return

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
        if n_steps % freq != 0:
            raise AssertionError(
                "Trace will not include the final state. Adjust resolution."
            )
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
