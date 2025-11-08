from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import vmap

from jaxtyping import Array, ArrayLike, PRNGKeyArray

import blackjax
import utils

########## COMPETING BAYES METHODS ##########


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
