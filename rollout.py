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


def assert_ppd_args_shape(x_new, x_prev, y_prev):
    assert x_new.ndim == 2, "x_new must be 2D array"
    assert x_prev.ndim == 2, "x_prev must be 2D array"
    assert y_prev.ndim == 1, "y_prev must be 1D array"
    assert (
        x_prev.shape[0] == y_prev.shape[0]
    ), "x_prev and y_prev must have same number of samples"
    assert (
        x_prev.shape[1] == x_new.shape[1]
    ), "x_prev and x_new must have same number of features"
    assert y_prev.ndim == 1, "y_prev must be 1D array"


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
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]  # (m, num_of_bins)

        EPS = 1e-5
        all_u = jax.random.uniform(
            key, shape=logits.shape[0], minval=EPS, maxval=1 - EPS
        )  # icdf doesn't like u that are too close to 0 and 1

        y_new = np.array(
            [bardist.icdf(l, float(u)).cpu() for l, u in zip(logits, all_u)],
        )
        return np.squeeze(y_new)

    def icdf(
        self, u: np.ndarray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> np.ndarray:
        """
        Return inverse CDF of P(Y <= t | X = x_new, x_prev, y_prev) given a
        value u between [0, 1].

        Parameters
        ----------
        u : (p, ) array
            Values between [0, 1].
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            Inverse CDF values. Each row corresponds to a value of u, and each
            column corresponds to a value of x_new. Shape: (p, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]  # (m, num_of_bins)

        all_u = np.atleast_1d(u)
        assert all_u.ndim == 1, "u must be 1D array"

        # For each u, compute for all x_new
        results = [[bardist.icdf(l, float(u)).cpu() for l in logits] for u in all_u]
        return np.array(results)

    def predict_event(
        self, t: np.ndarray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> np.ndarray:
        """
        Return P(Y <= t | X = x_new, x_prev, y_prev).

        Parameters
        ----------
        t : (p, ) array
            Events of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y <= t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        return self.cdf(t, x_new, x_prev, y_prev)

    def cdf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """
        Return P(Y <= t | X = x_new, prev data).

        Parameters
        ----------
        t : (p, ) array
            Events of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y <= t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            pred_output = self.predict(x_new, output_type="full")

        logits = pred_output["logits"]  # shape: (m, num_of_bins)
        bardist = pred_output["criterion"]

        # t must be a 1D float array
        t = np.atleast_1d(t)
        assert t.ndim == 1 and t.dtype == float

        bardist.borders = bardist.borders.cpu()
        results = []
        for single_t in t:
            # Evaluate the predictive CDF at a single t for each x_new query point.
            ys = torch.full(
                (logits.shape[0], 1),
                float(single_t),
                dtype=torch.float32,
            )
            # cdf returns (m, 1) or (m,), squeeze to ensure (m,)
            cdf_val = bardist.cdf(logits.cpu(), ys).squeeze(-1)
            results.append(cdf_val.numpy())

        # Stack to get (p, m)
        return np.stack(results)


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
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new).squeeze()
        idx_new = jax.random.choice(key, a=self.classes_.size, p=probs_new)
        y_new = self.classes_[idx_new]

        # we use jax to sample from a categorical distribution in the PPD
        # resampling step.
        y_new = y_new.squeeze() if isinstance(y_new, np.ndarray) else y_new
        return y_new

    def pmf(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = t | X = x_new, prev data).

        Parameters
        ----------
        t: (p, ) array
            Event of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y = t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """

        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)

        # Use logits for higher precision
        logits = self.predict_logits(x_new)  # shape: (m, num_classes)

        # Convert to float64 for precision
        logits = logits.astype(np.float64)

        # Compute softmax in float64
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # t must be a 1D integer array
        t = np.atleast_1d(t)
        assert t.ndim == 1 and t.dtype == int

        def predict_event_single_t(single_t: int) -> np.ndarray:
            # Create a mask for the class: (num_classes,)
            matches = self.classes_ == single_t
            # Dot product selects the column or results in 0 if no match
            # probs: (m, num_classes), matches: (num_classes,) -> (m,)
            return np.dot(probs, matches.astype(np.float64))

        event_prob = np.array([predict_event_single_t(ti) for ti in t])
        return event_prob

    def predict_event(
        self,
        t: np.ndarray,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
    ) -> np.ndarray:
        """Return P(Y = t | X = x_new, prev data).

        Parameters
        ----------
        t: (p, ) array
            Event of the PPD.
        x_new : (m, d) array
            Query covariates.
        x_prev : (n, d) array
            Historical covariates.
        y_prev : (n,) array
            Historical targets.

        Return:
        -------
        np.ndarray
            P(Y = t | X = x_new, prev data). Each row corresponds to a value of t, and each column corresponds to a value of x_new.
            Shape: (p, m)
        """
        return self.pmf(t, x_new, x_prev, y_prev)


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
