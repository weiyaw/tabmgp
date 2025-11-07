# %%
import jax
import jax.numpy as jnp
from jax import jit, vmap
from jax.typing import ArrayLike
import numpy as np
import logging

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


### This script is intended as a shim to use the Copula implementation from Fong
### et al. (2023). The main functions are copula_cregression and
### copula_classification.


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
