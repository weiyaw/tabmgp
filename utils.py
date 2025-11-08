import logging
import os
import pickle
import re
import subprocess

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array


def get_n_data(data):
    # return the number of samples in 'data' which is a dictionary of arrays
    # with the same leading dimension

    leading_dims = [x.shape[0] for x in data.values()]
    assert all([x == leading_dims[0] for x in leading_dims])
    return leading_dims[0]


def get_tree_lead_dim(tree):
    # Return the leading dimensions of a PyTree, assuming that all leaves have
    # the same leading dimension

    leaves = jax.tree.leaves(tree)
    chex.assert_equal_shape_prefix(leaves, 1)
    return leaves[0].shape[0]


def tree_shape(tree):
    return jax.tree.map(lambda x: jnp.shape(x), tree)


def githash():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def write_to_local(path, obj, verbose=False):
    # write to local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        logging.info(f"Write to local:/{path}")


def write_to(path, obj, verbose=False):
    write_to_local(path, obj, verbose=verbose)


def read_from_gs(bucket_name, path):
    # bucket: bucket name
    # path: path to the file
    # obj: the object to save

    bucket = get_bucket(bucket_name)
    blob = bucket.blob(path)
    with blob.open("rb") as f:
        obj = pickle.load(f)
    return obj


def read_from_local(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_from(path):
    if path.startswith("gs://"):
        # read from gcs
        bucket_name, path = path.split("/", 3)[2:]
        return read_from_gs(bucket_name, path)
    else:
        # read from local
        return read_from_local(path)


def print_dgp(d):
    return " ".join([f"{k}={v}" for k, v in d.items()])


def get_data_size(path):
    match = re.search(r"data=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_resample_x(path):
    match = re.search(r"resample_x=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_seed(path):
    match = re.search(r"seed=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_dim_x(path):
    match = re.search(r"dim_x=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_date_part(path):
    match = re.search(r"outputs/([^/]+)/", path)
    if match:
        return match.group(1)
    return None


def format_decimal(x, decimals=2):
    return f"{x:.{decimals}f}"


def get_name(path):
    match = re.search(r"name=([^ ]+)", path)
    if match:
        if match.group(1) == "classification-fixed":
            return "classification-standard"
        elif match.group(1) == "classification-fixed-gmm":
            match2 = re.search(r"a=([^ ]+)", path)
            return f"classification-gmm-{match2.group(1)}"
        elif match.group(1) == "regression-fixed-dependent":
            match2 = re.search(r"s_small=([^ ]+)", path)
            match3 = re.search(r"s_mod=([^ ]+)", path)
            return f"regression-dependent-{match2.group(1)}-{match3.group(1)}"
        elif match.group(1) == "regression-fixed":
            return "regression-standard"
        elif match.group(1) == "regression-fixed-non-normal":
            match2 = re.search(r"df=([^ ]+)", path)
            return f"regression-t-{match2.group(1)}"
        return match.group(1)
    return None


class OptDiagnostic(eqx.Module):
    """
    Diagnostic information for optimization
    """

    steps: int  # total number of steps taken
    converged: bool  # whether the optimization converged
    final_value: Array | None = None  # final value of the objective function
    path: Array | None = None  # temporarily storing the gradient norm to detect nan


# def run_opt(init_params, fun, opt, max_iter, tol):
#     # LBFGS from optax
#     # https://optax.readthedocs.io/en/stable/_collections/examples/lbfgs.html#l-bfgs-solver
#     value_and_grad_fun = optax.value_and_grad_from_state(fun)

#     def step(carry):
#         params, state = carry
#         value, grad = value_and_grad_fun(params, state=state)
#         updates, state = opt.update(
#             grad, state, params, value=value, grad=grad, value_fn=fun
#         )
#         params = optax.apply_updates(params, updates)
#         return params, state

#     def continuing_criterion(carry):
#         _, state = carry
#         iter_num = otu.tree_get(state, "count")
#         grad = otu.tree_get(state, "grad")
#         err = otu.tree_l2_norm(grad)
#         return (iter_num == 0) | ((iter_num < max_iter) & (err >= tol))

#     init_carry = (init_params, opt.init(init_params))
#     final_params, final_state = jax.lax.while_loop(
#         continuing_criterion, step, init_carry
#     )
#     return final_params, final_state


def newton_descent(f, x0, step, tol, max_iter):
    # Newton-Rahpson to find the argmin of f(x). Stop at convergence.

    def loop_body(state):
        k, x, x_best, f_best, _ = state
        hess_f = jax.hessian(f)(x)  # Jacobian of f wrt x
        grad_f = jax.grad(f)(x)
        delta_x, resid, rank, s = jnp.linalg.lstsq(hess_f, -grad_f)
        x_next = x + delta_x
        x_best, f_best = jax.lax.cond(
            f(x_next) < f_best,
            lambda: (x_next, f(x_next)),
            lambda: (x_best, f_best),
        )
        gnorm = jnp.linalg.norm(jax.grad(f)(x_next))
        return (k + 1, x_next, x_best, f_best, gnorm)

    def loop_cond(state):
        # Terminate when x is no longer moving or when max_iter is reached
        k, _, _, _, gnorm = state
        return jnp.logical_and(gnorm > tol, k < max_iter)

    init_state = (0, x0, x0, f(x0), jnp.inf)
    total_k, x_final, x_best, f_best, gnorm = jax.lax.while_loop(
        loop_cond, loop_body, init_state
    )

    return x_best, OptDiagnostic(total_k, total_k < max_iter, f_best, gnorm)
