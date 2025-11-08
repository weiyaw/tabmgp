import logging
import os
import re
from abc import abstractmethod
from timeit import default_timer as timer
from typing import Callable

import chex
import equinox as eqx
import hydra
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from omegaconf import DictConfig, OmegaConf

import baseline
import optimizer
import utils
from copula import copula_classification, copula_cregression
from data import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, OPENML_REGRESSION
from functional import (
    Functional,
    LogisticRegression,
    LinearRegression,
    QuantileRegression,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer

jax.config.update("jax_enable_x64", True)

# Evaluate martingale posterior with these many numbers of forward samples
EVAL_T = [250, 500, 1000, 2000, 3000, 4000, 5000]


def make_x_encoder(
    categorical_x: list[bool], include_x: list[bool]
) -> ColumnTransformer:
    """
    One-hot encode categorical features and standardize numerical features.  We
    are going to fit an intercept, so drop the first category of each
    categorical feature to avoid multicollinearity.

    All the columns with False in include_x will be dropped.
    """

    numerical_x = [(not i) and j for i, j in zip(categorical_x, include_x)]
    categorical_x = [i and j for i, j in zip(categorical_x, include_x)]
    return make_column_transformer(
        (OneHotEncoder(drop="first", sparse_output=False), categorical_x),
        (StandardScaler(), numerical_x),
        remainder="drop",
        sparse_threshold=0,
        verbose_feature_names_out=False,
    )


class Preprocessor(eqx.Module):
    @abstractmethod
    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        """
        Encode the data (both x and y) into suitable format for the model.
        Args:
            data: A dict with keys "x" and "y", each with an array-like value.
        Returns:
            A dict with keys "x" and "y", each with a jax Array value.
        """
        pass


class DiscreteTarget(Preprocessor):
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: LabelEncoder

    def __init__(
        self,
        categorical_x: list[bool],
        include_x: list[bool],
        population_data: dict[str, ArrayLike],
    ):

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        # population_data = dgp.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = LabelEncoder().fit(population_data["y"])

    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        batch_shape = data["x"].shape[:-1]
        chex.assert_shape(data["x"], (*batch_shape, self.x_encoder.n_features_in_))
        chex.assert_shape(data["y"], batch_shape)
        flat_x = jnp.reshape(data["x"], (-1, self.x_encoder.n_features_in_))
        flat_y = jnp.reshape(data["y"], (-1,))
        x = jnp.reshape(self.x_encoder.transform(flat_x), (*batch_shape, -1))
        y = jnp.reshape(self.y_encoder.transform(flat_y), batch_shape)
        return {
            "x": jnp.asarray(x, dtype=jnp.float64),
            "y": jnp.asarray(y, dtype=jnp.int16),
        }


class ContinuousTarget(Preprocessor):
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: StandardScaler

    def __init__(
        self,
        categorical_x: list[bool],
        include_x: list[bool],
        population_data: dict[str, ArrayLike],
    ):

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        # population_data = dgp.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = StandardScaler().fit(population_data["y"][..., None])

    def encode_data(self, data: dict[str, ArrayLike]) -> dict[str, Array]:
        batch_shape = data["x"].shape[:-1]
        chex.assert_shape(data["x"], (*batch_shape, self.x_encoder.n_features_in_))
        chex.assert_shape(data["y"], batch_shape)
        flat_x = jnp.reshape(data["x"], (-1, self.x_encoder.n_features_in_))
        flat_y = jnp.reshape(data["y"], (-1, 1))
        x = jnp.reshape(self.x_encoder.transform(flat_x), (*batch_shape, -1))
        y = jnp.reshape(self.y_encoder.transform(flat_y), batch_shape)
        return {
            "x": jnp.asarray(x, dtype=jnp.float64),
            "y": jnp.asarray(y, dtype=jnp.float64),
        }


def get_experiment_paths(output_dir: str, verbose: bool = True) -> list[str]:
    """
    Recursively walk down the output_dir to get the paths of all directories (inclusive self) that matches the pattern.
    Args:
        output_dir: The directory containing the experiment results.
        verbose: Whether to print the seed in the paths.
    Returns:
        A list of paths to the experiment directories.
    """

    dir_pattern = re.compile(
        r"seed=(10[0-9][1-9]|10[1-9][0-9]|1100)$"
    )  # match any seed between 1001-1100
    paths = [p for p, _, _ in os.walk(output_dir) if dir_pattern.search(p)]
    paths.sort(key=lambda p: int(re.search(r"seed=(\d+)", p).group(1)))
    # Print the order of the paths (i.e. the seed)
    if verbose:
        logging.info(
            f"Path order: {[int(m.group(1)) for p in paths if (m := re.search(r"seed=(\d+)", p))]}"
        )
    return paths


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


# Use this mask to ignore collinear features in some real datasets. Feature with
# False will be ignored.
THETA_MASK = {
    "airfoil": [True, False, True, True, True],
    "concrete": [True, False, True, False, True, True, True, True],
    "energy": [False, False, True, False, True, True, True, True],
    "grid": [True] * 4 + [False] + [True] * 7,
    "abalone": [True, False, False, True, False, True, False, False],
    "fish": [True, True, True, True, True, False],
    "auction": [True] * 6 + [False],
    "banknote": [True, True, False, True],
    "rice": [False, False, False, True, True, False, True],
    "blood": [True, True, False, True],
    "skin": [True, False, True],
    "mozilla": [True, False, True, True, True],
    "telescope": [True] * 2 + [False] * 2 + [True] * 6,
    "yeast": [True] * 4 + [False] * 2 + [True] * 2,
    "wine": [True] * 6 + [False] * 2 + [True] * 3,
}


def load_experiment(
    experiment_dir: str, loss: str
) -> tuple[Preprocessor, Functional, Array, dict[str, Array]]:
    # Load the necessary components to compute posteriors

    if match := re.search(r"name=(\S+)", experiment_dir):
        exp_name = match.group(1)
    else:
        raise ValueError("No match in the directory name")

    dgp = utils.read_from(f"{experiment_dir}/dgp.pickle")
    population_data = dgp.get_population()

    if exp_name in THETA_MASK:
        include_x = THETA_MASK[exp_name]
    else:
        # Include all features by default
        include_x = [True] * dgp.train_data["x"].shape[1]

    if loss == "likelihood":
        # MLE functional
        if exp_name.startswith("regression-fixed") or exp_name in OPENML_REGRESSION:
            preprocessor = ContinuousTarget(
                dgp.categorical_x, include_x, population_data
            )
            functional = LinearRegression(l2=0.0)
        elif (
            exp_name.startswith("classification-fixed")
            or exp_name in OPENML_CLASSIFICATION
            or exp_name in OPENML_BINARY_CLASSIFICATION
        ):
            preprocessor = DiscreteTarget(dgp.categorical_x, include_x, population_data)
            n_classes = len(preprocessor.y_encoder.classes_)
            functional = LogisticRegression(n_classes=n_classes, l2=0.0)
        else:
            raise ValueError(f"{exp_name} not available for {loss} experiment.")
    elif loss.startswith("quantile"):
        # Quantile regression functional (only works for continuous target)
        tau = float(loss.split("-")[1])
        assert 0 < tau < 1, f"Quantile level must be in (0, 1), got {tau}"
        logging.info(f"Quantile level: {tau}")
        preprocessor = ContinuousTarget(dgp.categorical_x, include_x, population_data)
        if exp_name.startswith("regression-fixed") or exp_name in OPENML_REGRESSION:
            functional = QuantileRegression(tau, l2=0.0)
        else:
            raise ValueError(f"{exp_name} not available for {loss} experiment.")
    else:
        raise ValueError(f"Unknown loss: {loss}")

    # Normalised datasets with collinear features removed
    processed_data = preprocessor.encode_data(dgp.train_data)

    # Compute the dimension of theta
    if isinstance(functional, LogisticRegression):
        dim_theta = (processed_data["x"].shape[-1] + 1) * (functional.n_classes - 1)
    elif isinstance(functional, QuantileRegression) or isinstance(
        functional, LinearRegression
    ):
        dim_theta = processed_data["x"].shape[-1] + 1
    else:
        raise NotImplementedError

    # Compute true theta
    scaled_population_data = preprocessor.encode_data(population_data)
    init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))
    theta_true, opt_state = functional.minimize_loss(
        scaled_population_data, init_theta, None
    )

    if hasattr(opt_state, "success") and not opt_state.success:
        # Backup if our default optimizer fails
        logging.info("Optimization failed. theta_true might be wrong. Use scipy.")
        theta_true = optimizer.scipy_mle(
            functional.loss, scaled_population_data, init_theta
        )

    return preprocessor, functional, theta_true, processed_data


# These are the magic numbers to reproduce the same key from the seed
BB_KEY = 49195
NUTS_KEY = 16005
COPULA_KEY = 91501


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

    preprocessor, functional, theta_true, processed_data = load_experiment(
        path, cfg.loss
    )

    logging.info(f"dim_theta: {theta_true.size}")

    # no rows are dropped during encoding/scaling
    n_train = utils.get_n_data(processed_data)

    mle, mle_opt = functional.minimize_loss(processed_data, theta_true, None)
    if hasattr(mle_opt, "success") and not mle_opt.success:
        # Backup optimizer
        logging.info("Optimization failed. MLE might be wrong. Use scipy.")
        mle = optimizer.scipy_mle(functional.loss, processed_data, theta_true)
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
            subkey, processed_data, cfg.bb_rollout_times, cfg.bb_rollout_length
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

        def log_posterior(theta):
            log_prior = jnp.sum(jax.scipy.stats.norm.logpdf(theta, scale=10))
            return -functional.loss(processed_data, theta, None) + log_prior

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
        utils.write_to(
            f"{savedir}/gibbs-post.pickle", (samples, diagnostics), verbose=True
        )
        logging.info(f"Diagnostics: {np.mean(diagnostics.success):.2f}")
        logging.info(f"Gibbs posterior: {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
