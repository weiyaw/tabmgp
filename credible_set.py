import jax
import jax.numpy as jnp
import chex
import numpy as np
from jax import vmap

from functools import partial


@partial(jax.jit, static_argnames=["cov_type"])
def joint_credible_set(samples, alpha, cov_type="ellipsoid"):
    """
    Compute the confidence region using Mahalanobis distance.

    Args:
        samples: 2D array of samples (n_samples x n_dimensions)
        alpha: confidence level (Type 1 error)
        cov_type: "sphere", "diag", "ellipsoid"

    Returns:
        dict with keys: "mu" (mean), "cov" (covariance matrix), "radius" (critical value)
    """
    # samples, _ = jax.tree.flatten(samples)
    # samples = [s if s.ndim == 2 else s[:, jnp.newaxis] for s in samples]
    # samples = jnp.concatenate(samples, axis=1)
    chex.assert_shape(samples, (None, None))

    # Calculate sample mean and covariance
    mu = jnp.mean(samples, axis=0)
    if cov_type == "sphere":
        # cov is an identity, represented as a vector of ones
        cov = jnp.ones(samples.shape[1])
        cov_rank = samples.shape[1]
    elif cov_type == "diag":
        # cov is a diagonal matrix, represented as a vector of variances
        cov = jnp.var(samples, axis=0)
        cov_rank = jnp.sum(cov > 1e-6)
    elif cov_type == "ellipsoid":
        # cov is a full covariance matrix
        cov = jnp.cov(samples, rowvar=False)
        cov_rank = jnp.linalg.matrix_rank(cov)
    else:
        raise ValueError("cov_type must be 'sphere', 'diag', or 'ellipsoid'")

    sq_dist = vmap(mahalanobis_sq, (0, None, None))(samples, mu, cov)

    # Find the critical value
    radius = jnp.quantile(sq_dist, 1 - alpha)

    return {
        "mu": mu,
        "cov": cov,
        "cov_rank": cov_rank,
        "radius": jnp.sqrt(radius),
        "trace": jnp.sum(cov) if cov.ndim == 1 else jnp.trace(cov),
    }


# def mahalanobis_sq(x, mu, var):
def mahalanobis_sq(x, mu, cov):
    """
    the most general form

    """
    chex.assert_equal_shape([x, mu])
    chex.assert_rank(x, {0, 1})  # x must be scalar or 1D

    centered = x - mu
    if cov.ndim == 1:
        chex.assert_shape(cov, x.shape)
        prec_centered = (1 / cov) * centered
        mahalanobis_sq = jnp.sum(centered * prec_centered)
    elif cov.ndim == 2:
        chex.assert_shape(cov, (x.shape[0], x.shape[0]))
        mahalanobis_sq = centered @ jnp.linalg.solve(cov, centered)
    else:
        raise ValueError("cov must be 1D or 2D")

    return mahalanobis_sq


# Companion function to check if a point is covered
@jax.jit
def is_covered(cr, true_value):
    """
    Check if true_value is within the confidence region using Mahalanobis distance.

    Args:
        cr: confidence region dict with keys: mu, cov, radius
        true_value: 1D array representing the true parameter value

    Returns:
        bool: True if the true value is within the region, False otherwise
    """
    mu = cr["mu"]
    cov = cr["cov"]
    radius = cr["radius"]

    # Compute Mahalanobis distance
    sq_distance = mahalanobis_sq(true_value, mu, cov)

    return jnp.sqrt(sq_distance) < radius


def coverage_probability(cr_ls, true_value, use_vmap=False):
    """
    cr_ls: list of credible, or 'arrays' of credible set
    # true_value: 1D array
    # use_vmap: whether to use vmap for vectorization
    # return: coverage probability
    """
    true_value = jax.flatten_util.ravel_pytree(true_value)[0]
    if use_vmap:
        in_or_out = jax.vmap(is_covered, (0, None))(cr_ls, true_value)
    else:
        in_or_out = jnp.array([is_covered(cr, true_value) for cr in cr_ls])
    chex.assert_shape(in_or_out, (None,))
    return jnp.mean(in_or_out, axis=0), in_or_out


def marginal_credible_interval(samples, alpha):
    # samples: array of shape (num_samples, dim)
    assert samples.ndim == 2
    lower = np.quantile(samples, alpha / 2, axis=0)
    upper = np.quantile(samples, 1 - alpha / 2, axis=0)
    mean = np.mean(samples, axis=0)
    return mean, lower, upper


def marginal_coverage(marginal_ci_ls, theta_true, alpha):
    means = np.array([mean for mean, _, _ in marginal_ci_ls])
    lower = np.array([lower for _, lower, _ in marginal_ci_ls])
    upper = np.array([upper for _, _, upper in marginal_ci_ls])

    cover = np.asarray((lower < theta_true) & (upper > theta_true))
    median_width = np.median(upper - lower, axis=0)

    winkler_score = np.median(
        (upper - lower)
        + (2 / alpha) * (lower - theta_true) * (theta_true < lower)
        + (2 / alpha) * (theta_true - upper) * (theta_true > upper),
        axis=0,
    )
    return np.mean(cover, axis=0), median_width, winkler_score
