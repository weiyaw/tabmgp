import logging
from timeit import default_timer as timer

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

import baseline
import optimizer
import utils
from posterior_runner import load_posterior_context, method_key


jax.config.update("jax_enable_x64", True)


@hydra.main(version_base=None, config_path="conf", config_name="bayes")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    ctx = load_posterior_context(cfg.expdir, cfg.loss)
    bayes_key = method_key(cfg.seed, "bayes")

    logging.info("Run untempered Bayes posterior")
    start = timer()
    _, subkey = jax.random.split(bayes_key)
    prior_mean = jnp.zeros_like(ctx.init_theta)
    prior_cov = jnp.eye(len(ctx.init_theta)) * 10**2
    if cfg.eb:
        prior_mean = ctx.init_theta
        hessian = jax.hessian(ctx.functional.loss, argnums=1)(
            ctx.train_data, ctx.init_theta, None
        )
        prior_cov = jnp.linalg.inv(hessian)

    logging.info(f"Prior mean: {np.array(prior_mean)}")
    logging.info(f"Prior variance: {np.diag(prior_cov)}")

    def log_prior(theta):
        return jax.scipy.stats.multivariate_normal.logpdf(
            theta, mean=prior_mean, cov=prior_cov
        )

    def log_posterior(theta):
        return -ctx.functional.loss(ctx.train_data, theta, None) + log_prior(theta)

    samples, nuts_state = baseline.nuts_with_adapt(
        subkey,
        log_posterior,
        ctx.init_theta,
        init_step_size=cfg.step_size,
        n_warmup=cfg.n_warmup,
        n_samples=cfg.n_samples,
        n_chains=cfg.n_chains,
    )
    diagnostics = optimizer.Diagnostics(
        success=nuts_state["acceptance_rate"], state=nuts_state
    )

    post_name = "bayes-eb" if cfg.eb else "bayes"
    utils.write_to(
        f"{ctx.savedir}/{post_name}-post.pickle",
        (samples, diagnostics),
        verbose=True,
    )
    logging.info(f"Diagnostics: {np.mean(diagnostics.success):.2f}")
    logging.info(f"Bayes posterior: {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
