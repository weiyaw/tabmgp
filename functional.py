# JAX ecosystem
import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.special import expit

import chex
import equinox as eqx
import numpy as np

from abc import abstractmethod
from typing import Any

from jaxtyping import Array

import utils
from optimizer import Diagnostics, run_lbfgs

PyTree = Any


class Functional(eqx.Module):
    """
    Logic to compute the functional.  All we need to define is the loss
    function.  Alternatively, we can override the default minimize_loss if there
    is a closed-form solution.
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
    def get_mgp(
        self,
        rollout_data: dict[str, Array],
        init_theta: Array,
        weight: Array | None = None,
    ) -> tuple[Array, Diagnostics]:
        """
        Compute the martingale posterior for a collection of rollout data
        (training + imputed data).  Dict has keys x, y, each with an array.
        This will return B posterior samples, where B is the number of rollout
        data in the collection.

        :param rollout_data: The data from training AND forward sampling.
            The 0th-dimension of the leave arrays is the number of posterior
            samples B.
        :param init_theta: Initial parameters to start the optimization from.
        :param weight: Optional weighting vector of shape (n_data, ) to weight
            each term in the loss.
        """

        return jax.vmap(self.minimize_loss, (0, None, None))(
            rollout_data, init_theta, weight
        )

    @eqx.filter_jit
    def get_theta_trace(
        self,
        rollout_data: dict[str, Array],
        init_theta: Array,
        start: int = 0,
        batch_size: int | None = 100,
        freq: int = 1,
    ) -> Array:
        """
        Inspect how theta behaves as we feed in more data points.

        :param rollout_data: The data from training AND forward sampling.
            The 0th-dimension of the leave Arrays is the number of posterior
            samples.
        :param init_theta: Initial parameters to start the optimization from.
        :param start: The starting point of the rollout.  Default is 0, which
            starts with 1 train_data.  Set it to (n_train_data - 1) will start
            from MLE.
        :param batch_size: The number of posterior to compute one at a time.
            This is to avoid OOM errors.
        :param freq: The frequency to compute the posterior.
        """
        n_data = utils.get_n_data(jax.tree.map(lambda x: x[0], rollout_data))
        masks = np.tri(n_data - start, n_data, start, dtype=np.bool)
        if freq > 1:
            # If freq is specified, we only take every `freq`-th mask
            masks = masks[::freq]

        # A mask is essentially a 0/1 weighting vector
        f = lambda m: self.get_mgp(rollout_data, init_theta, weight=m)

        # Avoid vmap to avoid OOM
        trace_post = jax.lax.map(f, masks, batch_size=batch_size)
        return trace_post


class LinearRegression(Functional):
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


class QuantileRegression(Functional):
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


class LogisticRegression(Functional):
    """
    loss function follows
    https://scikit-learn.org/stable/modules/linear_model.html#regularized-logistic-loss

    l2: factor in front of L2 penalty (1/C in sklearn, lambda in the writeup)
    """

    n_classes: int
    l2: float

    def log_model(self, dp: dict[str, Array], theta: Array) -> Array:
        # Logistic regression model with an intercept
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
