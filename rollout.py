# %%
import warnings
from typing import Callable

import jax
import numpy as np
import torch
from jaxtyping import Array, PRNGKeyArray
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


class TabPFNRegressorPredRule(TabPFNRegressor):

    def __init__(
        self,
        categorical_x: list[bool],
        n_estimators: int = 8,  # this is the default in 2.0.6
        average_before_softmax: bool = False,
    ):
        categorical_features_indices = [i for i, c in enumerate(categorical_x) if c]
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=1.0,
            categorical_features_indices=categorical_features_indices,
            fit_mode="low_memory",
            model_path="tabpfn-v2-regressor.ckpt",
        )

    def sample(
        self,
        key: PRNGKeyArray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        # Sample from predictive density
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        y_new = self.bardist_sample(key, bardist.icdf, pred_output["logits"])
        return np.squeeze(y_new)

    def bardist_sample(
        self,
        key: PRNGKeyArray,
        bardist_icdf: Callable,
        logits: np.ndarray,
        t: float = 1.0,
    ) -> np.ndarray:
        """Samples values from the bar distribution. A modified version of
        https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model/bar_distribution.py#L576

        Temperature t.
        """
        p_cdf = jax.random.uniform(key, shape=logits.shape[:-1])
        return np.array(
            [
                bardist_icdf(logits[i, :] / t, p).cpu()
                for i, p in enumerate(p_cdf.tolist())
            ],
        )


class TabPFNClassifierPredRule(TabPFNClassifier):

    def __init__(
        self,
        categorical_x: list[bool],
        n_estimators: int = 4,  # this is the default in 2.0.6
        average_before_softmax: bool = False,
    ):
        categorical_features_indices = [i for i, c in enumerate(categorical_x) if c]
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=1.0,
            categorical_features_indices=categorical_features_indices,
            fit_mode="low_memory",
            model_path="tabpfn-v2-classifier.ckpt",
        )

    def sample(
        self,
        key: PRNGKeyArray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new).squeeze()
        idx_new = jax.random.choice(key, a=self.classes_.size, p=probs_new)
        y_new = self.classes_[idx_new]

        # we use jax to sample from a categorical distribution in the PPD
        # resampling step.
        y_new = y_new.squeeze() if isinstance(y_new, np.ndarray) else y_new
        return y_new


def get_x_new(key: PRNGKeyArray, x: Array) -> Array:
    # For now, we draw x_new uniformly from x
    idx = jax.random.randint(key, shape=(), minval=0, maxval=x.shape[0])
    return x[None, idx]


def forward_sampling(
    key: PRNGKeyArray,
    one_step_ahead: Callable[[PRNGKeyArray, Array, Array, Array], Array],
    x_train: Array,
    y_train: Array,
    rollout_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate forward rollouts by iteratively sampling x and y.

    Args:
        key: Base PRNG key for reproducible sampling.
        one_step_pred_rule: Callable producing y predictions given a key,
            candidate x, and all previous x/y observations.
        x_train: Observed feature matrix used to seed the rollout.
        y_train: Observed targets aligned with x_train.
        rollout_length: Number of additional samples to generate.

    Returns:
        Tuple containing the concatenated x and y arrays (train + rollout).
    """

    assert x_train.shape[0] == y_train.shape[0]
    dim_x = x_train.shape[1]
    x_full = np.concatenate([x_train, np.full((rollout_length, dim_x), -1.0)])
    y_full = np.concatenate([y_train, np.full(rollout_length, -1.0)])

    for i in tqdm(range(len(x_train), len(x_train) + rollout_length)):
        # This loop performs forward sampling
        x_prev = x_full[:i]  # contains i number of data points
        y_prev = y_full[:i]  # contains i number of data points

        # one-step-ahead prediction of x
        rkey = jax.random.fold_in(key, i)
        rkey, subkey = jax.random.split(rkey)
        x_new = get_x_new(subkey, x_prev)
        x_full[i] = x_new

        # one-step-ahead prediction of y | x
        rkey, subkey = jax.random.split(rkey)
        y_full[i] = one_step_ahead(subkey, x_new, x_prev, y_prev)

    return x_full, y_full

