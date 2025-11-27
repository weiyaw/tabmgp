from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import vmap, jit

from jaxtyping import Array, ArrayLike, PRNGKeyArray

import numpy as np
import blackjax
import logging


import utils

from pr_copula.main_copula_regression_conditional import (
    fit_copula_cregression,
    predictive_resample_cregression,
)

from pr_copula.main_copula_classification import (
    fit_copula_classification,
    predictive_resample_classification,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision for JAX
########## COMPETING BAYES METHODS ##########

# Bayesian bootstrap
def bootstrap(
    key: PRNGKeyArray, data: dict[str, ArrayLike], rollout_length: int
) -> dict[str, ArrayLike]:
    # Bayesian bootstrap
    n_data = utils.get_n_data(data)
    data_idx = jnp.arange(n_data, dtype=jnp.int64)
    final_idx = jnp.concatenate([data_idx, jnp.full((rollout_length,), -1)])

    def body_fn(i, carry):
        key, final_idx = carry
        key, subkey = jax.random.split(key)
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=n_data + i)
        final_idx = final_idx.at[n_data + i].set(final_idx[idx])
        return key, final_idx

    key, final_idx = jax.lax.fori_loop(0, rollout_length, body_fn, (key, final_idx))
    bootstrap_data = jax.tree.map(lambda x: x[final_idx], data)
    return bootstrap_data


@partial(jax.jit, static_argnames=["rollout_length", "rollout_times"])
def bootstrap_many_samples(
    key: PRNGKeyArray,
    data: dict[str, ArrayLike],
    rollout_times: int,
    rollout_length: int,
) -> dict[str, ArrayLike]:
    # Bayesian bootstrap many times, each time with different keys
    keys = vmap(jax.random.fold_in, (None, 0))(key, jnp.arange(0, rollout_times))
    return vmap(lambda k: bootstrap(k, data, rollout_length))(keys)


# Standard Bayes
def nuts_with_adapt(
    key: PRNGKeyArray,
    log_posterior: Callable[..., Array],
    init_theta: Array,
    init_step_size: float,
    n_warmup: int,
    n_samples: int,  # n_samples per chain
    n_chains: int,
) -> Any:
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


def adapt_nuts(
    key: PRNGKeyArray,
    log_posterior: Callable[..., Array],
    init_theta: Array,
    init_step_size: float,
    n_warmup: int,
) -> tuple[Any, Any]:
    # Adapt the NUTS sampler to find the optimal step size
    adapt = blackjax.window_adaptation(
        blackjax.nuts, log_posterior, initial_step_size=init_step_size
    )
    key, subkey = jax.random.split(key)
    adapt_res, info = adapt.run(subkey, init_theta, num_steps=n_warmup)
    return adapt_res, info


@partial(jax.jit, static_argnames=["log_posterior", "n_samples"])
def nuts_sampler(
    key: PRNGKeyArray,
    log_posterior: Callable[..., Array],
    init_state: Any,
    nuts_parameters: dict[str, Any],
    n_samples: int,
) -> tuple[Array, dict[str, Array]]:

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



# Copula
class XEncoder(ColumnTransformer):
    def __init__(self, categorical_x: list[bool]):
        self.categorical_x = categorical_x
        self.numerical_x = [not i for i in categorical_x]
        transformers = [
            ("onehot", OneHotEncoder(sparse_output=False), self.categorical_x),
            ("standardscaler", StandardScaler(), self.numerical_x),
        ]
        ColumnTransformer.__init__(
            self, transformers, remainder="passthrough", sparse_threshold=0
        )

    def inverse_transform(self, X_transformed):
        # Split transformed features back
        X_cat = X_transformed[:, self.output_indices_["onehot"]]
        X_num = X_transformed[:, self.output_indices_["standardscaler"]]

        # Inverse transform categorical (one-hot -> original)
        if X_cat.shape[-1] > 0:
            cat_transformer = self.named_transformers_["onehot"]
            X_cat_inv = cat_transformer.inverse_transform(X_cat)
        else:
            X_cat_inv = X_transformed[:, :0]  # Empty array with correct shape

        # Inverse transform numerical (standardized -> original)
        if X_num.shape[-1] > 0:
            num_transformer = self.named_transformers_["standardscaler"]
            X_num_inv = num_transformer.inverse_transform(X_num)
        else:
            X_num_inv = X_transformed[:, :0]  # Empty array with correct shape

        # Reconstruct original feature order
        common_dtype = np.result_type(X_num_inv, X_cat_inv)
        X_inv = np.empty(
            (X_transformed.shape[0], len(self.categorical_x)), dtype=common_dtype
        )
        cat_idx = num_idx = 0
        for i, is_cat in enumerate(self.categorical_x):
            if is_cat:
                X_inv[:, i] = X_cat_inv[:, cat_idx]
                cat_idx += 1
            else:
                X_inv[:, i] = X_num_inv[:, num_idx]
                num_idx += 1

        return X_inv


def batched_inverse_transform(encoder, batched_X):
    """Batched version of inverse_transform for ndarrays (..., n_features)"""
    original_shape = batched_X.shape
    batch_dims = original_shape[:-1]
    n_features = original_shape[-1]

    # Reshape to 2D for processing
    X_flat = batched_X.reshape(-1, n_features)

    # Apply inverse transform
    if isinstance(encoder, LabelEncoder):
        X_inv_flat = encoder.inverse_transform(X_flat.squeeze(-1))
    else:
        X_inv_flat = encoder.inverse_transform(X_flat)

    # Reshape back to original batch dimensions
    return X_inv_flat.reshape(*batch_dims, -1)


def copula_classification(
    train_data: dict[str, ArrayLike], categorical_x: list[bool], B: int, T: int
):

    x_encoder = XEncoder(categorical_x).fit(train_data["x"])
    y_encoder = LabelEncoder().fit(train_data["y"])

    x = x_encoder.transform(train_data["x"])
    y = y_encoder.transform(train_data["y"])

    copula_classification_obj = fit_copula_classification(
        jnp.array(y), jnp.array(x), single_x_bandwidth=False, n_perm_optim=10, n_perm=10
    )
    n = len(y)
    logging.info("Bandwidth is {}".format(copula_classification_obj.rho_opt))
    logging.info("Bandwidth is {}".format(copula_classification_obj.rho_x_opt))
    logging.info("Preq loglik is {}".format(copula_classification_obj.preq_loglik / n))

    # Predict Yplot
    _, _, y_samp, x_samp, _ = predictive_resample_classification(
        copula_classification_obj, y, x, x, B_postsamples=B, T_fwdsamples=T
    )
    y_samp = batched_inverse_transform(y_encoder, y_samp.astype(int))
    x_samp = batched_inverse_transform(x_encoder, x_samp)

    recursion_data = {"y": y_samp.squeeze(), "x": x_samp}

    return recursion_data, copula_classification_obj


def copula_cregression(
    train_data: dict[str, ArrayLike],
    categorical_x: list[bool],
    B: int,
    T: int,
    y_grid_size: int = 100,
):

    x_encoder = XEncoder(categorical_x).fit(train_data["x"])
    y_encoder = StandardScaler().fit(train_data["y"][..., np.newaxis])

    x = x_encoder.transform(train_data["x"])
    y = y_encoder.transform(train_data["y"][..., np.newaxis]).squeeze()
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)

    y_pr = np.linspace(np.min(y), np.max(y), y_grid_size)
    x_pr = x
    range_y_pr = jnp.arange(jnp.shape(y_pr)[0])
    range_x_pr = jnp.arange(jnp.shape(x_pr)[0])
    ind_y_grid, ind_x_grid = jnp.meshgrid(range_y_pr, range_x_pr, indexing="ij")
    x_pr_grid = x_pr[ind_x_grid.reshape(-1)]
    y_pr_grid = y_pr[ind_y_grid.reshape(-1)]

    n = len(y)
    copula_cregression_obj = fit_copula_cregression(
        y, x, single_x_bandwidth=False, n_perm_optim=10, n_perm=10
    )
    logging.info("Bandwidth is {}".format(copula_cregression_obj.rho_opt))
    logging.info("Bandwidth is {}".format(copula_cregression_obj.rho_x_opt))
    logging.info("Preq loglik is {}".format(copula_cregression_obj.preq_loglik / n))

    logcdf, _ = predictive_resample_cregression(
        copula_cregression_obj, x, y_pr_grid, x_pr_grid, B_postsamples=B, T_fwdsamples=T
    )
    logcdf = logcdf.reshape(B, np.shape(y_pr)[0], np.shape(x_pr)[0])
    logcdf = jnp.moveaxis(logcdf, 1, 2)  # move y to the last axis

    @jit
    @vmap
    @vmap
    def icdf(u, logcdf):
        """
        Given the logcdf of y_pr and a uniform variate, find the quantile.
        """
        log_u = jnp.log(u)  # a uniform
        return jnp.asarray(y_pr)[jnp.searchsorted(logcdf, log_u)]

    U_KEY = jax.random.PRNGKey(10592)
    subkeys = jax.random.split(U_KEY, 5)  # sample 5 y from each x

    y_samp = []
    x_samp = []
    # repeat a few times
    for k in subkeys:
        u = jax.random.uniform(k, (B, np.shape(x_pr)[0]))
        y_samp.append(icdf(u, logcdf))
        x_samp.append(jnp.tile(x_pr[np.newaxis, :], (B, 1, 1)))

    y_samp = jnp.concatenate(y_samp, axis=1)
    x_samp = jnp.concatenate(x_samp, axis=1)

    y_samp = batched_inverse_transform(y_encoder, y_samp[..., np.newaxis])
    x_samp = batched_inverse_transform(x_encoder, x_samp)

    recursion_data = {"y": y_samp.squeeze(-1), "x": x_samp}

    return recursion_data, copula_cregression_obj
