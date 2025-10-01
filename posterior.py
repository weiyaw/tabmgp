# Python standard library
import os
import re
from abc import abstractmethod
from typing import Any, Callable
import logging
from timeit import default_timer as timer
from functools import partial
from dgp import *

# import matplotlib.pyplot as plt  # for diagnostic debugging

# Third-party imports
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


# JAX ecosystem
import jax
import jax.numpy as jnp
import equinox as eqx
import chex
import blackjax
from jax import vmap
from jaxtyping import Key
from jax import Array
from jax.typing import ArrayLike
from jax.scipy.special import expit
import optax
import optax.tree_utils as otu


# Local imports
import utils
from dgp import OPENML_REGRESSION, OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION
from utils import tree_shape, get_seed
from copula import copula_cregression, copula_classification

import hydra
from omegaconf import DictConfig, OmegaConf
from collections import namedtuple

PyTree = Any

jax.config.update("jax_enable_x64", True)

# for easy debugging
np.set_printoptions(linewidth=np.inf)  # Set to a specific width
np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=6, suppress=True)

OptResult = namedtuple("OptResult", ["success", "opt_state"])
Diagnostics = namedtuple("Diagnostics", ["success", "state"])


def scipy_mle(loss, data, init_theta):
    from scipy.optimize import minimize

    @jax.jit
    def scaled_loss(theta):
        return loss(data, theta, None)

    grad_scaled_loss = jax.jit(jax.grad(scaled_loss))
    mle = minimize(
        scaled_loss, init_theta, method="L-BFGS-B", jac=grad_scaled_loss, tol=1e-6
    )
    assert mle.success
    return mle.x


def run_opt(
    init_params: Array,
    fun: Callable,
    opt: optax.GradientTransformation,
    max_iter: int,
    tol: float,
) -> tuple[Array, OptResult]:
    """
    Copy from the L-BFGS example in Optax: https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html#l-bfgs-solver
    """

    value_and_grad_fun = optax.value_and_grad_from_state(fun)
    dim_theta = init_params.size

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, "count")
        grad = otu.tree_get(state, "grad")
        err = otu.tree_norm(grad, ord=2) / dim_theta
        return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

    init_carry = (init_params, opt.init(init_params))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    is_success = ~continuing_criterion((None, final_state))
    return final_params, OptResult(success=is_success, opt_state=final_state)


def run_lbfgs(
    loss: Callable,
    init_theta: Array,
    tol: float = 1e-6,
    max_iter: int = 1000,
    max_linesearch_steps: int = 100,
) -> tuple[Array, OptResult]:
    opt = optax.lbfgs(
        scale_init_precond=True,
        linesearch=optax.scale_by_zoom_linesearch(
            max_linesearch_steps=max_linesearch_steps,
            verbose=False,
            initial_guess_strategy="one",
        ),
    )
    return run_opt(init_theta, loss, opt, max_iter=max_iter, tol=tol)


########## PREPROCESSING DATA ##########
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


# Read recursion data (train data + forward samples) from the recursion
# directory and compile them into arrays
def compile_recursion(input_dir: str) -> dict[str, Array]:
    recursion_dir = f"{input_dir}/recursion"
    recursion_paths = [
        p for p in os.listdir(recursion_dir) if re.match(r"recursion-\d+\.pickle", p)
    ]
    # but it doesn't matter if the recursion samples are not in order
    recursion_paths.sort(
        key=lambda p: int(re.search(r"recursion-(\d+)\.pickle", p).group(1))
    )
    recursions = [utils.read_from(f"{recursion_dir}/{p}") for p in recursion_paths]
    recursions = {k: np.stack([dic[k] for dic in recursions]) for k in recursions[0]}
    return recursions


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


class DiscreteTarget(eqx.Module):
    all_train_data: list[dict[str, Array]]
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: LabelEncoder

    def __init__(self, all_dgps: list[DGP], include_x: list[bool]):

        dgp_0 = all_dgps[0]
        if hasattr(dgp_0, "categorical_x"):
            categorical_x = dgp_0.categorical_x
        else:
            categorical_x = [False] * dgp_0.dim_x

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        population_data = dgp_0.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = LabelEncoder().fit(population_data["y"])

        # From here onwards, encode_data is usable
        self.all_train_data = [self.encode_data(dgp.train_data) for dgp in all_dgps]

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


class ContinuousTarget(eqx.Module):
    all_train_data: list[dict[str, Array]]
    include_x: list[bool]
    x_encoder: ColumnTransformer
    y_encoder: StandardScaler

    def __init__(self, all_dgps: list[DGP], include_x: list[bool]):

        dgp_0 = all_dgps[0]
        if hasattr(dgp_0, "categorical_x"):
            categorical_x = dgp_0.categorical_x
        else:
            categorical_x = [False] * dgp_0.dim_x

        assert len(include_x) == len(categorical_x)
        self.include_x = include_x

        population_data = dgp_0.get_population()
        chex.assert_shape(population_data["x"], (None, len(categorical_x)))
        chex.assert_rank(population_data["y"], 1)

        self.x_encoder = make_x_encoder(categorical_x, include_x).fit(
            population_data["x"]
        )
        self.y_encoder = StandardScaler().fit(population_data["y"][..., None])

        # From here onwards, encode_data is usable
        self.all_train_data = [self.encode_data(dgp.train_data) for dgp in all_dgps]

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


########## POSTERIOR/LOSS DEFINITION ##########


class Posterior(eqx.Module):
    """
    Everything required for a posterior to be well-defined.  In fact, all we
    need is to define a loss function.
    """

    @abstractmethod
    def loss(self, data: dict[str, Array], theta: Array, weight: Array | None) -> Array:
        """
        Compute the loss function for the model given the data and parameters.
        Think of the loss in Gibbs posterior.
        """
        pass

    @eqx.filter_jit
    def minimize_loss(
        self, data: dict[str, Array], init_theta: Array, weight: Array | None
    ) -> tuple[Array, Diagnostics]:
        """
        Minimize the loss function for the model given the data and initial
        parameters.  Returns the optimized parameters and any auxiliary
        information.  This can be overridden by subclasses if a more efficient
        optimization is available, e.g., closed-form least squares solution.
        """

        scaled_loss = lambda x: self.loss(data, x, weight)
        minimizer, opt_state = run_lbfgs(scaled_loss, init_theta)
        diagnostics = Diagnostics(success=opt_state.success, state=opt_state)
        return minimizer, diagnostics

    @eqx.filter_jit
    def get_martingale_posterior(
        self,
        recursion_data: dict[str, Array],
        init_theta: Array,
        weight: Array | None = None,
    ) -> tuple[Array, Diagnostics]:
        """
        Compute the martingale posterior for a collection of recursion data
        (training + imputed data).  Dict has keys x, y, each with an array.
        This will return B posterior samples, where B is the number of recursion
        data in the collection.

        :param recursion_data: The data from training AND forward sampling.
            The 0th-dimension of the leave arrays is the number of posterior
            samples B.
        :param init_theta: Initial parameters to start the optimization from.
        :param weight: Optional weighting vector of shape (n_data, ) to weight
            each term in the loss.
        """

        return jax.vmap(self.minimize_loss, (0, None, None))(
            recursion_data, init_theta, weight
        )

    @eqx.filter_jit
    def get_trace_in_theta(
        self,
        recursion_data: dict[str, Array],
        init_theta: Array,
        start: int = 0,
        batch_size: int | None = 100,
        freq: int = 1,
    ) -> Array:
        """
        Inspect how theta behaves as we feed in more data points.

        :param recursion_data: The data from training AND forward sampling.
            The 0th-dimension of the leave Arrays is the number of posterior
            samples.
        :param init_theta: Initial parameters to start the optimization from.
        :param start: The starting point of the recursion.  Default is 0, which
            starts with 1 train_data.  Set it to (n_train_data - 1) will start
            from MLE.
        :param batch_size: The number of posterior to compute one at a time.
            This is to avoid OOM errors.
        :param freq: The frequency to compute the posterior.
        """
        n_data = utils.get_n_data(jax.tree.map(lambda x: x[0], recursion_data))
        masks = np.tri(n_data - start, n_data, start, dtype=np.bool)
        if freq > 1:
            # If freq is specified, we only take every `freq`-th mask
            masks = masks[::freq]

        # A mask is essentially a 0/1 weighting vector
        f = lambda m: self.get_martingale_posterior(
            recursion_data, init_theta, weight=m
        )

        # Avoid vmap to avoid OOM
        trace_post = jax.lax.map(f, masks, batch_size=batch_size)
        return trace_post


class Regression(Posterior):
    """
    l2: factor in front of L2 penalty (lambda in the writeup)
    """

    l2: float

    def log_model(self, dp: dict[str, Array], theta: Array) -> Array:
        # theta[0] is the log_std, theta[1:] is the beta
        x = dp["x"]
        y = dp["y"]
        dim_x = x.size
        x_with_1 = jnp.insert(x, 0, 1)  # fit an intercept
        chex.assert_shape(y, ())
        chex.assert_shape(x_with_1, (dim_x + 1,))
        chex.assert_shape(theta, (dim_x + 1,))
        pred = x_with_1 @ theta
        chex.assert_shape(pred, ())
        return jax.scipy.stats.norm.logpdf(y, loc=pred, scale=1.0)

    def loss(self, data: dict[str, Array], theta: Array, weight: Array | None) -> Array:
        n_data = utils.get_n_data(data)
        loglik_all_data = vmap(lambda dp: self.log_model(dp, theta))(data)
        chex.assert_shape(loglik_all_data, (n_data,))
        if weight is not None:
            chex.assert_shape(weight, (n_data,))
            loglik_all_data = weight * loglik_all_data

        beta = theta[1:]  # excluding intercept
        l2_penalty = 0.5 * self.l2 * jnp.sum(beta**2)
        return -jnp.sum(loglik_all_data) + l2_penalty

    def orthogonalize_data(
        self, data: dict[str, Array]
    ) -> tuple[dict[str, Array], Callable, Callable, Array]:
        n_data = utils.get_n_data(data)
        dim_x = data["x"].shape[-1]
        Q, R = np.linalg.qr(data["x"], mode="reduced")
        Q = Q * np.sqrt(n_data - 1)
        R = R / np.sqrt(n_data - 1)
        basis_data = {"x": Q, "y": data["y"]}

        def backward(theta: Array) -> Array:
            # theta is a vector with (dim_x + 2) elements. The first 2 elements
            # are log_std and the intercept. The rest are the coefs of each feature.
            non_features = theta[:2]  # log_std and intercept
            features = jnp.linalg.solve(R, theta[2:])
            chex.assert_shape(features, (dim_x,))
            return jnp.concatenate([non_features, features], axis=0)

        def forward(theta: Array) -> Array:
            non_features = theta[:2]  # log_std and intercept
            features = R @ theta[2:]
            chex.assert_shape(features, (dim_x,))
            return jnp.concatenate([non_features, features], axis=0)

        return basis_data, backward, forward, R

    @eqx.filter_jit
    def minimize_loss(
        self, data: dict[str, Array], init_theta: Array, weight: Array | None
    ) -> tuple[Array, Diagnostics]:
        # weighted least squares with L2 penalty
        x = data["x"]
        y = data["y"]
        dim_x = x.shape[-1]
        chex.assert_shape(x, (None, init_theta.size - 1))
        chex.assert_shape(y, (None,))
        chex.assert_equal_shape_prefix([x, y], 1)
        x_with_1 = jnp.insert(x, 0, 1, axis=-1)  # fit an intercept
        if weight is not None:
            chex.assert_equal_shape_prefix([x, y, weight], 1)
            # Broadcast weight across features
            x_with_1 = x_with_1 * weight[:, jnp.newaxis]
            y = y * weight
        penalty_matrix = jnp.diagflat(jnp.array([0, *jnp.full(dim_x, self.l2)]))
        lhs = (x_with_1.T @ x_with_1) + penalty_matrix
        rhs = x_with_1.T @ y
        ls_theta, resid, rank, s = jnp.linalg.lstsq(lhs, rhs, rcond=None)

        diagnostics = Diagnostics(
            success=init_theta.size == rank, state=(resid, rank, s)
        )

        return ls_theta, diagnostics


class QuantileRegression(Posterior):
    """
    tau: quantile level (0 < tau < 1)
    l2: factor in front of L2 penalty (lambda in the writeup)
    """

    tau: float
    l2: float

    def loss_per_dp(self, dp: dict[str, Array], theta: Array) -> Array:
        # a smooth version of pinball loss with expit
        x = dp["x"]
        y = dp["y"]
        dim_x = x.size
        x_with_1 = jnp.insert(x, 0, 1)  # fit an intercept
        chex.assert_shape(y, ())
        chex.assert_shape(x_with_1, (dim_x + 1,))
        chex.assert_shape(theta, (dim_x + 1,))
        pred = x_with_1 @ theta
        chex.assert_shape(pred, ())

        # Smooth version of jnp.where(pred < 0, (self.tau - 1) * pred, self.tau
        # * pred). This is exact if a = 0
        a = 0.01
        return pred * (self.tau - expit(-pred / a))

    def loss(self, data: dict[str, Array], theta: Array, weight: Array | None) -> Array:
        n_data = utils.get_n_data(data)
        loss_all_data = vmap(lambda dp: self.loss_per_dp(dp, theta))(data)
        chex.assert_shape(loss_all_data, (n_data,))
        if weight is not None:
            chex.assert_shape(weight, (n_data,))
            loss_all_data = weight * loss_all_data

        beta = theta[1:]  # excluding intercept
        l2_penalty = 0.5 * self.l2 * jnp.sum(beta**2)
        return jnp.sum(loss_all_data) + l2_penalty


class Classification(Posterior):
    """
    loss function follows
    https://scikit-learn.org/stable/modules/linear_model.html#regularized-logistic-loss

    l2: factor in front of L2 penalty (1/C in sklearn, lambda in the writeup)
    """

    n_classes: int
    l2: float

    def log_model(self, dp: dict[str, Array], theta: Array) -> Array:
        # Classification model with an intercept
        x = dp["x"]
        y = dp["y"]  # takes any values from 0, 1, ..., n_classes - 1
        chex.assert_shape(y, ())
        chex.assert_shape(x, (None,))
        dim_x = x.size
        x_with_1 = jnp.insert(x, 0, 1)  # fit an intercept
        # Set theta for 0th class to 0 to ensure identifiability
        chex.assert_shape(theta, ((dim_x + 1) * (self.n_classes - 1),))
        # reshape theta to (dim_x + 1, n_classes - 1), so the first (n_classes - 1) elements of
        # theta is the intercept, and so on.
        logits = x_with_1 @ jnp.reshape(theta, (dim_x + 1, self.n_classes - 1))
        chex.assert_shape(logits, (self.n_classes - 1,))
        probs = jax.nn.log_softmax(jnp.insert(logits, 0, 0))
        chex.assert_type(y, int)
        return probs[y]

    def loss(self, data: dict[str, Array], theta: Array, weight: Array | None) -> Array:
        n_data = utils.get_n_data(data)
        loglik_all_data = vmap(lambda dp: self.log_model(dp, theta))(data)
        chex.assert_shape(loglik_all_data, (n_data,))
        if weight is not None:
            chex.assert_shape(weight, (n_data,))
            loglik_all_data = weight * loglik_all_data
        # Penalty scalar following the scikit-learn, equivalant to variance in a Gaussian prior
        l2_penalty = 0.5 * self.l2 * jnp.sum(theta**2)
        return -jnp.sum(loglik_all_data) + l2_penalty


########## EXPERIMENT IMPLEMENTATION (COMBINE DATA PROCESSING AND POSTERIOR) ##########
class ClassificationExperiment(DiscreteTarget, Classification):

    all_paths: list[str]
    theta_true: jax.Array

    def __init__(self, all_experiments_dir: str, include_x: list[bool] | None = None):
        self.all_paths = get_experiment_paths(all_experiments_dir)
        all_dgps = [utils.read_from(f"{p}/dgp.pickle") for p in self.all_paths]
        # all_dgps = [read_dgp(f"{p}/dgp.eqx") for p in self.all_paths]

        include_x = include_x or [True] * all_dgps[0].train_data["x"].shape[-1]
        DiscreteTarget.__init__(self, all_dgps, include_x)
        Classification.__init__(self, n_classes=len(self.y_encoder.classes_), l2=0.0)

        dgp_0 = all_dgps[0]
        dim_theta = (self.all_train_data[0]["x"].shape[-1] + 1) * (self.n_classes - 1)
        population_data = self.encode_data(dgp_0.get_population())
        init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))
        self.theta_true, opt_state = self.minimize_loss(
            population_data, init_theta, None
        )

        if hasattr(opt_state, "success") and not opt_state.success:
            # see here for complete status.
            # https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.OptimizeResults.html
            logging.info("Optimization failed. theta_true might be wrong. Use scipy.")
            self.theta_true = scipy_mle(self.loss, population_data, init_theta)


class RegressionExperiment(ContinuousTarget, Regression):
    all_paths: list[str]
    theta_true: jax.Array

    def __init__(self, all_experiments_dir: str, include_x: list[bool] | None = None):
        self.all_paths = get_experiment_paths(all_experiments_dir)
        all_dgps = [utils.read_from(f"{p}/dgp.pickle") for p in self.all_paths]
        # all_dgps = [read_dgp(f"{p}/dgp.eqx") for p in self.all_paths]

        include_x = include_x or [True] * all_dgps[0].train_data["x"].shape[-1]
        ContinuousTarget.__init__(self, all_dgps, include_x)
        Regression.__init__(self, l2=0.0)

        # +1 for the intercept
        dim_theta = self.all_train_data[0]["x"].shape[-1] + 1

        dgp_0 = all_dgps[0]
        population_data = self.encode_data(dgp_0.get_population())
        init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))
        self.theta_true, _ = self.minimize_loss(population_data, init_theta, None)


class QuantileRegressionExperiment(ContinuousTarget, QuantileRegression):
    all_paths: list[str]
    theta_true: jax.Array

    def __init__(
        self,
        all_experiments_dir: str,
        tau: float,
        include_x: list[bool] | None = None,
    ):
        self.all_paths = get_experiment_paths(all_experiments_dir)
        all_dgps = [utils.read_from(f"{p}/dgp.pickle") for p in self.all_paths]
        # all_dgps = [read_dgp(f"{p}/dgp.eqx") for p in self.all_paths]

        include_x = include_x or [True] * all_dgps[0]["x"].shape[-1]
        ContinuousTarget.__init__(self, all_dgps, include_x)
        QuantileRegression.__init__(self, l2=0.0, tau=tau)

        # +1 for the intercept
        dim_theta = self.all_train_data[0]["x"].shape[-1] + 1

        dgp_0 = all_dgps[0]
        population_data = self.encode_data(dgp_0.get_population())
        init_theta = jax.random.normal(jax.random.key(1), (dim_theta,))
        self.theta_true, _ = self.minimize_loss(population_data, init_theta, None)


########## COMPETING BAYES METHODS ##########


def bootstrap(
    key: Key, data: dict[str, ArrayLike], recursion_length: int
) -> dict[str, ArrayLike]:
    # Bayesian bootstrap
    n_data = utils.get_n_data(data)
    data_idx = jnp.arange(n_data, dtype=jnp.int64)
    final_idx = jnp.concatenate([data_idx, jnp.full((recursion_length,), -1)])

    def body_fn(i, carry):
        key, final_idx = carry
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=n_data + i)
        final_idx = final_idx.at[n_data + i].set(final_idx[idx])
        return key, final_idx

    key, final_idx = jax.lax.fori_loop(0, recursion_length, body_fn, (key, final_idx))
    bootstrap_data = jax.tree.map(lambda x: x[final_idx], data)
    return bootstrap_data


@partial(jax.jit, static_argnames=["recursion_length", "recursion_times"])
def bootstrap_many_samples(
    key: Key,
    data: dict[str, ArrayLike],
    recursion_times: int,
    recursion_length: int,
) -> dict[str, ArrayLike]:
    # Bayesian bootstrap many times, each time with different keys
    keys = vmap(jax.random.fold_in, (None, 0))(key, jnp.arange(0, recursion_times))
    return vmap(lambda k: bootstrap(k, data, recursion_length))(keys)


def nuts_with_adapt(
    key: Key,
    log_posterior: Callable,
    init_theta: Array,
    init_step_size: float,
    n_warmup: int,
    n_samples: int,  # n_samples per chain
    n_chains: int,
):
    # Run multiple NUTS chains, starting from initial theta + some random noise
    chain_keys = jax.random.split(key, n_chains)
    samples_ls = []
    for ck in chain_keys:
        init_key, adapt_key, nuts_key = jax.random.split(ck, 3)
        # Add some random noise to the initial theta
        random_init_theta = init_theta + jax.random.normal(init_key, init_theta.shape)
        adapt_res, info = adapt_nuts(
            adapt_key, log_posterior, random_init_theta, init_step_size, n_warmup
        )
        samples = nuts_sampler(
            nuts_key, log_posterior, adapt_res.state, adapt_res.parameters, n_samples
        )
        samples_ls.append(samples)
    all_samples = jax.tree.map(lambda *x: jnp.concatenate(x, axis=0), *samples_ls)
    return all_samples


def adapt_nuts(key, log_posterior, init_theta, init_step_size, n_warmup):
    # Adapt the NUTS sampler to find the optimal step size
    adapt = blackjax.window_adaptation(
        blackjax.nuts, log_posterior, initial_step_size=init_step_size
    )
    key, subkey = jax.random.split(key)
    adapt_res, info = adapt.run(subkey, init_theta, num_steps=n_warmup)
    return adapt_res, info


@partial(jax.jit, static_argnames=["log_posterior", "n_samples"])
def nuts_sampler(key, log_posterior, init_state, nuts_parameters, n_samples):

    nuts = blackjax.nuts(log_posterior, **nuts_parameters)
    state = init_state  # use the warmed state from adaptation

    def scan_body(carry, _):
        state, key = carry
        key, subkey = jax.random.split(key)
        state, nuts_info = nuts.step(subkey, state)
        diagnostics = {
            "logdensity": state.logdensity,
            "is_divergent": nuts_info.is_divergent,
            "is_turning": nuts_info.is_turning,
            "energy": nuts_info.energy,
            "num_trajectory_expansions: ": nuts_info.num_trajectory_expansions,
            "num_integration_steps": nuts_info.num_integration_steps,
            "acceptance_rate": nuts_info.acceptance_rate,
        }
        return (state, key), (state.position, diagnostics)

    _, samples_diag = jax.lax.scan(scan_body, (state, key), None, length=n_samples)
    return samples_diag


# These are the magic numbers to reproduce the same key from the seed
BB_KEY = 49195
NUTS_KEY = 16005
COPULA_KEY = 91501

# Define the type for experiments
Experiment = (
    RegressionExperiment | ClassificationExperiment | QuantileRegressionExperiment
)

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


def load_experiment(experiment_dir: str, loss: str) -> Experiment:

    dir_paths = get_experiment_paths(experiment_dir, verbose=False)

    if match := re.search(r"name=(\S+)", dir_paths[0]):
        exp_name = match.group(1)
    else:
        raise ValueError("No match in the directory name")

    if loss == "likelihood":
        if exp_name in THETA_MASK:
            theta_mask = THETA_MASK[exp_name]
        else:
            theta_mask = None
        if exp_name.startswith("regression-fixed"):
            experiment = RegressionExperiment(experiment_dir)
        elif exp_name.startswith("classification-fixed"):
            experiment = ClassificationExperiment(experiment_dir)
        elif exp_name in OPENML_CLASSIFICATION:
            experiment = ClassificationExperiment(experiment_dir, theta_mask)
        elif exp_name in OPENML_BINARY_CLASSIFICATION:
            experiment = ClassificationExperiment(experiment_dir, theta_mask)
        elif exp_name in OPENML_REGRESSION:
            experiment = RegressionExperiment(experiment_dir, theta_mask)
        else:
            raise ValueError(f"{exp_name} not available for {loss} experiment.")
    elif loss.startswith("quantile"):
        tau = float(loss.split("-")[1])
        assert 0 < tau < 1, f"Quantile level must be in (0, 1), got {tau}"
        logging.info(f"Quantile level: {tau}")

        if exp_name.startswith("regression-fixed"):
            experiment = QuantileRegressionExperiment(experiment_dir, tau)
        elif exp_name in OPENML_REGRESSION:
            experiment = QuantileRegressionExperiment(experiment_dir, tau)
        else:
            raise ValueError(f"{exp_name} not available for {loss} experiment.")
    else:
        raise ValueError(f"Unknown loss: {loss}")

    logging.info(f"Data: {exp_name}, Experiment: {type(experiment).__name__}")
    return experiment


def truncate_recursion(recursion, N):
    # Truncate up to N. Dim of leave is (n_samples, recursion_length, dim_theta)
    leaves = jax.tree.leaves(recursion)
    chex.assert_equal_shape_prefix(leaves, 2)
    return jax.tree.map(lambda x: x[:, :N], recursion)


# Evaluate martingale posterior with these many numbers of forward samples
EVAL_T = [250, 500, 1000, 2000, 3000, 4000, 5000]


@hydra.main(version_base=None, config_path="conf", config_name="posterior")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    path = cfg.expdir
    savedir = f"{path}/posterior-{cfg.loss}"
    experiment = load_experiment(path, cfg.loss)
    logging.info(f"dim_theta: {experiment.theta_true.size}")

    # cfg.expdir must be a directory for a particular seed, and this step make sure the 0th path match the right seed
    assert os.path.basename(path) == os.path.basename(experiment.all_paths[0])

    train_data = experiment.all_train_data[0]  # this has already been encoded/scaled
    n_train = utils.get_n_data(train_data)

    theta_true = experiment.theta_true
    dim_theta = theta_true.shape[0]
    mle, mle_opt = experiment.minimize_loss(train_data, theta_true, None)
    if hasattr(mle_opt, "success") and not mle_opt.success:
        # see here for complete status.
        # https://docs.jax.dev/en/latest/_autosummary/jax.scipy.optimize.OptimizeResults.html
        logging.info("Optimization failed. MLE might be wrong. Use scipy.")
        mle = scipy_mle(experiment.loss, train_data, theta_true)
    init_theta = mle

    seed = get_seed(path)
    key = jax.random.key(seed)
    bb_key = jax.random.fold_in(key, BB_KEY)
    nuts_key = jax.random.fold_in(key, NUTS_KEY)

    # TabPFN
    if cfg.tabpfn:
        logging.info("Run TabPFN.")
        start = timer()
        tabpfn_full_recursion = experiment.encode_data(compile_recursion(path))
        tabpfn_T = tabpfn_full_recursion["x"].shape[1] - n_train
        logging.info(f"Shape of TabPFN recursion: {tree_shape(tabpfn_full_recursion)}")

        for T in filter(lambda t: t <= tabpfn_T, EVAL_T):
            start = timer()
            recursion_subset = truncate_recursion(tabpfn_full_recursion, n_train + T)
            tabpfn_post = experiment.get_martingale_posterior(
                recursion_subset, init_theta
            )
            utils.write_to(
                f"{savedir}/tabpfn-{T}-post.pickle", tabpfn_post, verbose=True
            )
            logging.info(f"Diagnostics: {np.mean(tabpfn_post[1].success)}")
            logging.info(f"TabPFN posterior ({T}): {timer() - start:.2f} seconds")

        if cfg.trace:
            start = timer()
            tabpfn_freq = max(tabpfn_T // cfg.resolution, 1)
            tabpfn_trace = experiment.get_trace_in_theta(
                tabpfn_full_recursion,
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
        bb_full_recursion = bootstrap_many_samples(
            subkey, train_data, cfg.bb_recursion_times, cfg.bb_recursion_length
        )
        bb_T = cfg.bb_recursion_length
        logging.info(f"Shape of BB recursion: {tree_shape(bb_full_recursion)}")
        chex.assert_tree_shape_prefix(
            bb_full_recursion,
            (cfg.bb_recursion_times, cfg.bb_recursion_length + n_train),
        )
        jax.block_until_ready(bb_full_recursion)
        logging.info(f"BB recursion: {timer() - start:.2f} seconds")

        for T in filter(lambda t: t <= bb_T, EVAL_T):
            start = timer()
            recursion_subset = truncate_recursion(bb_full_recursion, n_train + T)
            bb_post = experiment.get_martingale_posterior(recursion_subset, init_theta)
            utils.write_to(f"{savedir}/bb-{T}-post.pickle", bb_post, verbose=True)
            logging.info(f"Diagnostics: {np.mean(bb_post[1].success)}")
            logging.info(f"BB posterior ({T}): {timer() - start:.2f} seconds")

        if cfg.trace:
            start = timer()
            bb_freq = max(bb_T // cfg.resolution, 1)
            bb_trace = experiment.get_trace_in_theta(
                bb_full_recursion,
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
        copula_T = cfg.copula_recursion_length
        copula_B = cfg.copula_recursion_times
        copula_num_y_grid = cfg.copula_num_y_grid
        copula_key = jax.random.fold_in(key, COPULA_KEY)
        copula_freq = max(
            copula_T // cfg.resolution, 1
        )  # frequency to save the trace of pdf/cdf
        assert (
            copula_T % copula_freq == 0
        ), "Copula trace will not have the final logcdf/logpdf. Adjust resolution."
        n_train = utils.get_n_data(train_data)
        dgp = utils.read_from(f"{experiment.all_paths[0]}/dgp.pickle")

        # assert False, "need to use scaled dataset but without throwing away collinear data"
        if hasattr(dgp, "categorical_x"):
            categorical_x = dgp.categorical_x
        else:
            categorical_x = [False] * dgp.train_data["x"].shape[-1]

        if isinstance(experiment, Regression):
            copula_full_recursion, copula_obj = copula_cregression(
                dgp.train_data, categorical_x, copula_B, copula_T, copula_num_y_grid
            )

        elif isinstance(experiment, Classification) and experiment.n_classes == 2:
            copula_full_recursion, copula_obj = copula_classification(
                dgp.train_data, categorical_x, copula_B, copula_T
            )

        elif isinstance(experiment, Classification) and experiment.n_classes > 2:
            logging.info("Copula not available for multiclass classification.")
            copula_full_recursion = None
        else:
            raise NotImplementedError
        jax.block_until_ready(copula_full_recursion)
        logging.info(f"Copula recursion: {timer() - start:.2f} seconds")

        if copula_full_recursion is not None:
            copula_full_recursion = experiment.encode_data(copula_full_recursion)

            for T in filter(lambda t: t <= copula_T, EVAL_T):
                start = timer()
                recursion_subset = truncate_recursion(
                    copula_full_recursion, n_train + T
                )
                copula_post = experiment.get_martingale_posterior(
                    recursion_subset, init_theta
                )
                utils.write_to(
                    f"{savedir}/copula-{T}-post.pickle", copula_post, verbose=True
                )
                logging.info(f"Diagnostics: {np.mean(copula_post[1].success)}")
                logging.info(f"Copula posterior ({T}): {timer() - start:.2f} seconds")

            if cfg.trace and isinstance(experiment, Classification):
                start = timer()
                copula_trace = experiment.get_trace_in_theta(
                    copula_full_recursion,
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
            return -experiment.loss(train_data, theta, None) + log_prior

        mcmc_init_theta = init_theta
        samples, nuts_state = nuts_with_adapt(
            subkey,
            log_posterior,
            mcmc_init_theta,
            init_step_size=cfg.gibbs_step_size,
            n_warmup=cfg.gibbs_n_warmup,
            n_samples=cfg.gibbs_n_samples,
            n_chains=cfg.gibbs_n_chains,
        )

        diagnostics = Diagnostics(
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
