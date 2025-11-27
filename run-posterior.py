import logging
import os
import re

from timeit import default_timer as timer

import chex
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from omegaconf import DictConfig, OmegaConf

import baseline
from baseline import copula_classification, copula_cregression
import optimizer
import utils

from functional import (
    LogisticRegression,
    LinearRegression,
)

from experiment_setup import load_experiment

jax.config.update("jax_enable_x64", True)

# Evaluate martingale posterior with these many numbers of forward samples
EVAL_T = [250, 500, 1000, 2000, 3000, 4000, 5000]

# These are the magic numbers to reproduce the same key from the seed
BB_KEY = 49195
NUTS_KEY = 16005
COPULA_KEY = 91501


# Read rollout data (train data + forward samples) from the rollout
# directory and compile them into arrays
def compile_rollout(input_dir: str) -> dict[str, Array]:
    rollout_dir = f"{input_dir}/rollout"
    rollout_paths = [
        p for p in os.listdir(rollout_dir) if re.match(r"rollout-\d+\.pickle", p)
    ]
    # but it doesn't matter if the rollout samples are not in order
    rollout_paths.sort(
        key=lambda p: int(re.search(r"rollout-(\d+)\.pickle", p).group(1))
    )
    rollouts = [utils.read_from(f"{rollout_dir}/{p}") for p in rollout_paths]
    rollouts = {k: np.stack([dic[k] for dic in rollouts]) for k in rollouts[0]}
    return rollouts


def truncate_rollout(rollout, N):
    # Truncate up to N. Dim of leave is (n_samples, rollout_length, dim_theta)
    leaves = jax.tree.leaves(rollout)
    chex.assert_equal_shape_prefix(leaves, 2)
    return jax.tree.map(lambda x: x[:, :N], rollout)


@hydra.main(version_base=None, config_path="conf", config_name="posterior")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    path = cfg.expdir
    savedir = f"{path}/posterior-{cfg.loss}"

    # dir_paths = get_experiment_paths(path, verbose=False)
    dgp = utils.read_from(f"{path}/dgp.pickle")
    # experiment = load_experiment(path, cfg.loss)

    preprocessor, functional, theta_true, train_data = load_experiment(path, cfg.loss)

    logging.info(f"dim_theta: {theta_true.size}")

    # no rows are dropped during encoding/scaling
    n_train = utils.get_n_data(train_data)

    mle, mle_opt = functional.minimize_loss(train_data, theta_true, None)
    if hasattr(mle_opt, "success") and not mle_opt.success:
        # Backup optimizer
        logging.info("Optimization failed. MLE might be wrong. Use scipy.")
        mle = optimizer.scipy_mle(functional.loss, train_data, theta_true)
    init_theta = mle

    key = jax.random.key(cfg.seed)
    bb_key = jax.random.fold_in(key, BB_KEY)
    nuts_key = jax.random.fold_in(key, NUTS_KEY)

    # TabPFN
    if cfg.tabpfn:
        logging.info("Run TabPFN.")
        start = timer()
        tabpfn_full_rollout = preprocessor.encode_data(compile_rollout(path))
        tabpfn_T = tabpfn_full_rollout["x"].shape[1] - n_train
        logging.info(
            f"Shape of TabPFN rollout: {utils.tree_shape(tabpfn_full_rollout)}"
        )

        for T in filter(lambda t: t <= tabpfn_T, EVAL_T):
            start = timer()
            rollout_subset = truncate_rollout(tabpfn_full_rollout, n_train + T)
            tabpfn_post = functional.get_mgp(rollout_subset, init_theta)
            utils.write_to(
                f"{savedir}/tabpfn-{T}-post.pickle", tabpfn_post, verbose=True
            )
            logging.info(f"Diagnostics: {np.mean(tabpfn_post[1].success)}")
            logging.info(f"TabPFN posterior ({T}): {timer() - start:.2f} seconds")

        if cfg.trace:
            start = timer()
            tabpfn_freq = max(tabpfn_T // cfg.resolution, 1)
            tabpfn_trace = functional.get_theta_trace(
                tabpfn_full_rollout,
                init_theta,
                start=n_train - 1,  # start from MLE
                batch_size=cfg.batch,
                freq=tabpfn_freq,
            )
            utils.write_to(
                f"{savedir}/tabpfn-{n_train}-{tabpfn_T + n_train}-{tabpfn_freq}-trace.pickle",
                tabpfn_trace,
                verbose=True,
            )
            logging.info(f"TabPFN trace: {timer() - start:.2f} seconds")

    # Bayesian bootstrap
    if cfg.bb:
        logging.info("Run Bayesian bootstrap (BB).")
        start = timer()
        bb_key, subkey = jax.random.split(bb_key)
        bb_full_rollout = baseline.bootstrap_many_samples(
            subkey, train_data, cfg.bb_rollout_times, cfg.bb_rollout_length
        )
        bb_T = cfg.bb_rollout_length
        logging.info(f"Shape of BB rollout: {utils.tree_shape(bb_full_rollout)}")
        chex.assert_tree_shape_prefix(
            bb_full_rollout,
            (cfg.bb_rollout_times, cfg.bb_rollout_length + n_train),
        )
        jax.block_until_ready(bb_full_rollout)
        logging.info(f"BB rollout: {timer() - start:.2f} seconds")

        for T in filter(lambda t: t <= bb_T, EVAL_T):
            start = timer()
            rollout_subset = truncate_rollout(bb_full_rollout, n_train + T)
            bb_post = functional.get_mgp(rollout_subset, init_theta)
            utils.write_to(f"{savedir}/bb-{T}-post.pickle", bb_post, verbose=True)
            logging.info(f"Diagnostics: {np.mean(bb_post[1].success)}")
            logging.info(f"BB posterior ({T}): {timer() - start:.2f} seconds")

        if cfg.trace:
            start = timer()
            bb_freq = max(bb_T // cfg.resolution, 1)
            bb_trace = functional.get_theta_trace(
                bb_full_rollout,
                init_theta,
                start=n_train - 1,  # start from MLE
                batch_size=cfg.batch,
                freq=bb_freq,
            )
            utils.write_to(
                f"{savedir}/bb-{n_train}-{bb_T + n_train}-{bb_freq}-trace.pickle",
                bb_trace,
                verbose=True,
            )
            logging.info(f"BB trace: {timer() - start:.2f} seconds")

    # Copula
    if cfg.copula:
        logging.info("Run Bivariate Copula.")
        start = timer()
        copula_T = cfg.copula_rollout_length
        copula_B = cfg.copula_rollout_times
        copula_num_y_grid = cfg.copula_num_y_grid
        copula_key = jax.random.fold_in(key, COPULA_KEY)
        copula_freq = max(
            copula_T // cfg.resolution, 1
        )  # frequency to save the trace of pdf/cdf
        assert (
            copula_T % copula_freq == 0
        ), "Copula trace will not have the final logcdf/logpdf. Adjust resolution."

        # assert False, "need to use scaled dataset but without throwing away collinear data"
        if hasattr(dgp, "categorical_x"):
            categorical_x = dgp.categorical_x
        else:
            categorical_x = [False] * dgp.train_data["x"].shape[-1]

        if isinstance(functional, LinearRegression):
            copula_full_rollout, copula_obj = copula_cregression(
                dgp.train_data, categorical_x, copula_B, copula_T, copula_num_y_grid
            )

        elif isinstance(functional, LogisticRegression) and functional.n_classes == 2:
            copula_full_rollout, copula_obj = copula_classification(
                dgp.train_data, categorical_x, copula_B, copula_T
            )

        elif isinstance(functional, LogisticRegression) and functional.n_classes > 2:
            logging.info("Copula not available for multiclass classification.")
            copula_full_rollout = None
        else:
            raise NotImplementedError
        jax.block_until_ready(copula_full_rollout)
        logging.info(f"Copula rollout: {timer() - start:.2f} seconds")

        if copula_full_rollout is not None:
            copula_full_rollout = preprocessor.encode_data(copula_full_rollout)

            for T in filter(lambda t: t <= copula_T, EVAL_T):
                start = timer()
                rollout_subset = truncate_rollout(copula_full_rollout, n_train + T)
                copula_post = functional.get_mgp(rollout_subset, init_theta)
                utils.write_to(
                    f"{savedir}/copula-{T}-post.pickle", copula_post, verbose=True
                )
                logging.info(f"Diagnostics: {np.mean(copula_post[1].success)}")
                logging.info(f"Copula posterior ({T}): {timer() - start:.2f} seconds")

            if cfg.trace and isinstance(functional, LogisticRegression):
                start = timer()
                copula_trace = functional.get_theta_trace(
                    copula_full_rollout,
                    init_theta,
                    start=n_train - 1,  # start from MLE
                    batch_size=cfg.batch,
                    freq=copula_freq,
                )
                utils.write_to(
                    f"{savedir}/copula-{n_train}-{copula_T + n_train}-{copula_freq}-trace.pickle",
                    copula_trace,
                    verbose=True,
                )
                logging.info(f"Copula trace: {timer() - start:.2f} seconds")

    # Gibbs posterior (NUTS)
    if cfg.gibbs:
        logging.info("Run untempered Gibbs posterior")
        start = timer()
        nuts_key, subkey = jax.random.split(nuts_key)
        prior_mean = jnp.zeros_like(init_theta)
        prior_cov = jnp.eye(len(init_theta)) * 10**2
        if cfg.gibbs_eb:
            # Empirical Bayes: Estimate prior mean and variance
            prior_mean = init_theta
            H = jax.hessian(functional.loss, argnums=1)(train_data, init_theta, None)
            prior_cov = jnp.linalg.inv(H)

        logging.info(f"Prior mean: {np.array(prior_mean)}")
        logging.info(f"Prior variance: {np.diag(prior_cov)}")

        def log_prior(theta):
            return jax.scipy.stats.multivariate_normal.logpdf(
                theta, mean=prior_mean, cov=prior_cov
            )

        def log_posterior(theta):
            return -functional.loss(train_data, theta, None) + log_prior(theta)

        mcmc_init_theta = init_theta
        samples, nuts_state = baseline.nuts_with_adapt(
            subkey,
            log_posterior,
            mcmc_init_theta,
            init_step_size=cfg.gibbs_step_size,
            n_warmup=cfg.gibbs_n_warmup,
            n_samples=cfg.gibbs_n_samples,
            n_chains=cfg.gibbs_n_chains,
        )

        diagnostics = optimizer.Diagnostics(
            success=nuts_state["acceptance_rate"], state=nuts_state
        )

        if cfg.gibbs_eb:
            utils.write_to(
                f"{savedir}/gibbs-eb-post.pickle", (samples, diagnostics), verbose=True
            )
        else:
            utils.write_to(
                f"{savedir}/gibbs-post.pickle", (samples, diagnostics), verbose=True
            )
        logging.info(f"Diagnostics: {np.mean(diagnostics.success):.2f}")
        logging.info(f"Gibbs posterior: {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
