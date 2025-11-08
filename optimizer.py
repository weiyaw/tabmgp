from collections import namedtuple
from typing import Any, Callable

import jax
import optax
import optax.tree_utils as otu
from jaxtyping import Array

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


########## FUNCTIONAL/LOSS DEFINITION ##########
