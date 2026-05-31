import logging
import os
import re
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

import utils
import optimizer
from dgp import OPENML_BINARY_CLASSIFICATION, OPENML_CLASSIFICATION, OPENML_REGRESSION
from functional import (
    Functional,
    LogisticRegression,
    LinearRegression,
    QuantileRegression,
)
from preprocessor import (
    Preprocessor,
    DiscreteTarget,
    ContinuousTarget,
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


def get_experiment_name(experiment_dir: str) -> str:
    if match := re.search(r"name=(\S+)", experiment_dir):
        return match.group(1)
    else:
        raise ValueError("No match in the directory name")


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
    print(
        f"Path order: {[int(m.group(1)) for p in paths if (m := re.search(r'seed=(\d+)', p))]}"
    )
    return paths


def get_include_x(exp_name: str, dgp: object) -> list[bool]:
    if exp_name in THETA_MASK:
        return THETA_MASK[exp_name]
    else:
        # Include all features by default
        return [True] * dgp.train_data["x"].shape[1]


def create_preprocessor_and_functional(
    loss: str,
    exp_name: str,
    dgp: object,
    include_x: list[bool],
    population_data: dict[str, ArrayLike],
) -> tuple[Preprocessor, Functional]:
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

    return preprocessor, functional


def compute_true_theta(
    functional: Functional,
    preprocessor: Preprocessor,
    population_data: dict[str, ArrayLike],
    processed_data: dict[str, Array],
) -> Array:
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

    return theta_true


def load_experiment(
    experiment_dir: str, loss: str
) -> tuple[Preprocessor, Functional, Array, dict[str, Array]]:
    # This function returns the preprocessor, functional, true theta, and
    # pre-processed data

    exp_name = get_experiment_name(experiment_dir)

    dgp = utils.read_from(f"{experiment_dir}/dgp.pickle")
    population_data = dgp.get_population()

    include_x = get_include_x(exp_name, dgp)

    preprocessor, functional = create_preprocessor_and_functional(
        loss, exp_name, dgp, include_x, population_data
    )

    # Normalised datasets with collinear features removed
    processed_data = preprocessor.encode_data(dgp.train_data)

    theta_true = compute_true_theta(
        functional, preprocessor, population_data, processed_data
    )

    return preprocessor, functional, theta_true, processed_data
