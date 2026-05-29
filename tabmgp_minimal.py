# %%
import warnings
from typing import Callable

import numpy as np
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
    assert x_prev.shape[0] == y_prev.shape[0], (
        "x_prev and y_prev must have same number of samples"
    )
    assert x_prev.shape[1] == x_new.shape[1], (
        "x_prev and x_new must have same number of features"
    )
    assert y_prev.ndim == 1, "y_prev must be 1D array"


class TabPFNRegressorPPD(TabPFNRegressor):
    def __init__(
        self,
        *,
        n_estimators: int = 8,  # this is the default in 2.0.6
        average_before_softmax: bool = False,
        softmax_temperature: float = 1.0,
        fit_mode: str = "low_memory",
        model_path: str = "tabpfn-v2-regressor.ckpt",
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=softmax_temperature,
            fit_mode=fit_mode,
            model_path=model_path,
            **kwargs,
        )

    def _predict_full(
        self, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> dict:
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in cast",
                category=RuntimeWarning,
            )
            return self.predict(x_new, output_type="full")

    def sample(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw samples from the one-step-ahead predictive distribution.

        Parameters
        ----------
        x_new:
            Query covariates with shape (m, d).
        x_prev:
            Previous covariates with shape (n, d).
        y_prev:
            Previous targets with shape (n,).
        rng:
            Optional NumPy random source. May be None, an integer seed, or an
            np.random.Generator. Passing a Generator advances its stream;
            passing the same integer seed repeats the same draw.

        Returns
        -------
        np.ndarray
            Sampled targets for x_new, squeezed to match the existing rollout
            behavior.
        """
        rng = np.random.default_rng(rng)
        pred_output = self._predict_full(x_new, x_prev, y_prev)
        bardist = pred_output["criterion"]
        logits = pred_output["logits"]  # (m, num_of_bins)

        EPS = 1e-5
        # icdf doesn't like u that are too close to 0 and 1.
        all_u = rng.uniform(EPS, 1 - EPS, size=logits.shape[0])

        y_new = np.array(
            [bardist.icdf(l, float(u)).cpu() for l, u in zip(logits, all_u)],
        )
        return np.squeeze(y_new)


class TabPFNClassifierPPD(TabPFNClassifier):
    def __init__(
        self,
        *,
        n_estimators: int = 4,  # this is the default in 2.0.6
        average_before_softmax: bool = False,
        softmax_temperature: float = 1.0,
        fit_mode: str = "low_memory",
        model_path: str = "tabpfn-v2-classifier.ckpt",
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=softmax_temperature,
            fit_mode=fit_mode,
            model_path=model_path,
            **kwargs,
        )

    def sample(
        self,
        x_new: np.ndarray,
        x_prev: np.ndarray,
        y_prev: np.ndarray,
        *,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw samples from the one-step-ahead predictive distribution.

        Parameters
        ----------
        x_new:
            Query covariates with shape (m, d).
        x_prev:
            Previous covariates with shape (n, d).
        y_prev:
            Previous targets with shape (n,).
        rng:
            Optional NumPy random source. May be None, an integer seed, or an
            np.random.Generator. Passing a Generator advances its stream;
            passing the same integer seed repeats the same draw.

        Returns
        -------
        np.ndarray
            Sampled targets for x_new, squeezed to match the existing rollout
            behavior.
        """
        rng = np.random.default_rng(rng)
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
        probs_new = np.asarray(self.predict_proba(x_new))
        if probs_new.ndim == 1:
            probs_new = probs_new[None, :]

        idx_new = np.array(
            [rng.choice(self.classes_.size, p=probs_i) for probs_i in probs_new]
        )
        y_new = self.classes_[idx_new]

        return np.squeeze(y_new)


def get_x_new(
    x: np.ndarray, *, rng: np.random.Generator | int | None = None
) -> np.ndarray:
    # For now, we draw x_new uniformly from x
    rng = np.random.default_rng(rng)
    idx = rng.integers(x.shape[0])
    return x[None, idx]


def forward_sampling(
    one_step_ahead: Callable[..., np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    rollout_length: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate forward rollouts by iteratively sampling x and y.

    Args:
        one_step_ahead: Callable producing y predictions given candidate x,
            all previous x/y observations, and an optional keyword-only rng.
        x_train: Observed feature matrix used to seed the rollout.
        y_train: Observed targets aligned with x_train.
        rollout_length: Number of additional samples to generate.
        rng: Optional NumPy random source. May be None, an integer seed, or an
            np.random.Generator. Passing a Generator advances its stream;
            passing the same integer seed repeats the same rollout.

    Returns:
        Tuple containing the concatenated x and y arrays (train + rollout).
    """
    rng = np.random.default_rng(rng)

    assert x_train.shape[0] == y_train.shape[0]
    dim_x = x_train.shape[1]
    x_full = np.concatenate([x_train, np.full((rollout_length, dim_x), -1.0)])
    y_full = np.concatenate([y_train, np.full(rollout_length, -1.0)])

    for i in tqdm(range(len(x_train), len(x_train) + rollout_length)):
        # This loop performs forward sampling
        x_prev = x_full[:i]  # contains i number of data points
        y_prev = y_full[:i]  # contains i number of data points

        # one-step-ahead prediction of x
        x_new = get_x_new(x_prev, rng=rng)
        x_full[i] = x_new[0]

        # one-step-ahead prediction of y | x
        y_full[i] = one_step_ahead(x_new, x_prev, y_prev, rng=rng)

    return x_full, y_full
