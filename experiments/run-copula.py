import logging
import os
from timeit import default_timer as timer

import hydra
import jax
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
    load_posterior_context,
    method_key,
    normalize_forward_steps,
    save_mgp_posts,
    save_trace,
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

    save_mgp_posts(
        ctx.functional,
        full_rollout,
        ctx.init_theta,
        ctx.savedir,
        cfg.run_name,
        ctx.n_train,
        forward_steps,
    )
    if cfg.trace and isinstance(ctx.functional, LogisticRegression):
        save_trace(
            ctx.functional,
            full_rollout,
            ctx.init_theta,
            ctx.savedir,
            cfg.run_name,
            ctx.n_train,
            cfg.resolution,
            cfg.batch,
            require_final=True,
        )


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
