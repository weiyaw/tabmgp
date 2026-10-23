# %%
from typing import Callable

import numpy as np
from tabpfn import TabPFNClassifier, TabPFNRegressor
from tqdm import tqdm

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
        n_estimators: int = 1,
        average_before_softmax: bool = False,
        softmax_temperature: float = 1.0,
        fit_mode: str = "low_memory",
        model_path: str | None = None,
        memory_saving_mode=False,
        inference_config={"FINGERPRINT_FEATURE": False},
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=softmax_temperature,
            fit_mode=fit_mode,
            model_path=model_path,
            memory_saving_mode=memory_saving_mode,
            inference_config=inference_config,
            **kwargs,
        )

    def _predict_full(
        self, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> dict:
        assert_ppd_args_shape(x_new, x_prev, y_prev)
        self.fit(x_prev, y_prev)
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

    def sample_batched(
        self,
        x_new_list: list[np.ndarray],
        x_prev_list: list[np.ndarray],
        y_prev_list: list[np.ndarray],
        *,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw one sample per context from one batched predict.

        Parameters
        ----------
        x_new_list:
            B query covariates, each with shape (m, d).
        x_prev_list:
            B previous covariates, each with shape (n, d).
        y_prev_list:
            B previous targets, each with shape (n,).
        rng:
            Optional NumPy random source, as in sample.

        Returns
        -------
        np.ndarray
            Sampled targets for each x_new. Shape: (B, m).
        """
        rng = np.random.default_rng(rng)
        for x_new, x_prev, y_prev in zip(x_new_list, x_prev_list, y_prev_list):
            assert_ppd_args_shape(x_new, x_prev, y_prev)
        outputs = self.predict_batched(
            list(x_prev_list),
            list(y_prev_list),
            list(x_new_list),
            output_type="full",
            quantiles=[],
        )

        EPS = 1e-5
        y_new = []
        for output in outputs:
            bardist = output["criterion"]
            logits = output["logits"]  # (m, num_of_bins)
            # icdf doesn't like u that are too close to 0 and 1.
            all_u = rng.uniform(EPS, 1 - EPS, size=logits.shape[0])
            y_new.append(
                [float(bardist.icdf(l, float(u))) for l, u in zip(logits, all_u)]
            )
        return np.array(y_new)


class TabPFNClassifierPPD(TabPFNClassifier):
    def __init__(
        self,
        *,
        n_estimators: int = 1,  # this is the default in 2.0.6
        average_before_softmax: bool = False,
        softmax_temperature: float = 1.0,
        fit_mode: str = "low_memory",
        model_path: str | None = None,
        inference_config={"FINGERPRINT_FEATURE": False},
        **kwargs,
    ):
        super().__init__(
            n_estimators=n_estimators,
            average_before_softmax=average_before_softmax,
            softmax_temperature=softmax_temperature,
            fit_mode=fit_mode,
            model_path=model_path,
            inference_config=inference_config,
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

    def sample_batched(
        self,
        x_new_list: list[np.ndarray],
        x_prev_list: list[np.ndarray],
        y_prev_list: list[np.ndarray],
        *,
        rng: np.random.Generator | int | None = None,
    ) -> np.ndarray:
        """Draw one sample per context from one batched predict.

        All contexts must share the same set of classes.

        Parameters
        ----------
        x_new_list:
            B query covariates, each with shape (m, d).
        x_prev_list:
            B previous covariates, each with shape (n, d).
        y_prev_list:
            B previous targets, each with shape (n,).
        rng:
            Optional NumPy random source, as in sample.

        Returns
        -------
        np.ndarray
            Sampled targets for each x_new. Shape: (B, m).
        """
        rng = np.random.default_rng(rng)
        for x_new, x_prev, y_prev in zip(x_new_list, x_prev_list, y_prev_list):
            assert_ppd_args_shape(x_new, x_prev, y_prev)
        probs = self.predict_proba_batched(
            list(x_prev_list), list(y_prev_list), list(x_new_list)
        )  # (B, m, num_of_classes)
        # predict_proba_batched does not fit self, so classes_ is not set.
        classes = np.unique(y_prev_list[0])

        idx_new = np.array(
            [
                [rng.choice(classes.size, p=probs_i) for probs_i in probs_new]
                for probs_new in probs
            ]
        )
        return classes[idx_new]


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
    forward_steps: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate forward rollouts by iteratively sampling x and y.

    Args:
        one_step_ahead: Callable producing y predictions given candidate x,
            all previous x/y observations, and an optional keyword-only rng.
        x_train: Observed feature matrix used to seed the rollout.
        y_train: Observed targets aligned with x_train.
        forward_steps: Number of additional samples to generate.
        rng: Optional NumPy random source. May be None, an integer seed, or an
            np.random.Generator. Passing a Generator advances its stream;
            passing the same integer seed repeats the same rollout.

    Returns:
        Tuple containing the concatenated x and y arrays (train + rollout).
    """
    rng = np.random.default_rng(rng)

    assert x_train.shape[0] == y_train.shape[0]
    dim_x = x_train.shape[1]
    x_full = np.concatenate([x_train, np.full((forward_steps, dim_x), -1.0)])
    y_full = np.concatenate([y_train, np.full(forward_steps, -1.0)])

    for i in tqdm(range(len(x_train), len(x_train) + forward_steps)):
        # This loop performs forward sampling
        x_prev = x_full[:i]  # contains i number of data points
        y_prev = y_full[:i]  # contains i number of data points

        # one-step-ahead prediction of x
        x_new = get_x_new(x_prev, rng=rng)
        x_full[i] = x_new[0]

        # one-step-ahead prediction of y | x
        y_full[i] = one_step_ahead(x_new, x_prev, y_prev, rng=rng)

    return x_full, y_full


def forward_sampling_batched(
    one_step_ahead_batched: Callable[..., np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    forward_steps: int,
    n_rollouts: int,
    *,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate several forward rollouts together, one batched call per step.

    Args:
        one_step_ahead_batched: Callable producing y predictions given lists of
            candidate x and all previous x/y observations, one entry per
            rollout, and an optional keyword-only rng (e.g. sample_batched).
        x_train: Observed feature matrix used to seed every rollout.
        y_train: Observed targets aligned with x_train.
        forward_steps: Number of additional samples to generate.
        n_rollouts: Number of rollouts.
        rng: Optional NumPy random source, as in forward_sampling.

    Returns:
        Tuple containing the concatenated x and y arrays (train + rollout),
        with shapes (n_rollouts, n + forward_steps, d) and
        (n_rollouts, n + forward_steps).
    """
    rng = np.random.default_rng(rng)

    assert x_train.shape[0] == y_train.shape[0]
    dim_x = x_train.shape[1]
    x_full = np.concatenate([x_train, np.full((forward_steps, dim_x), -1.0)])
    y_full = np.concatenate([y_train, np.full(forward_steps, -1.0)])
    x_full = np.tile(x_full, (n_rollouts, 1, 1))
    y_full = np.tile(y_full, (n_rollouts, 1))

    for i in tqdm(range(len(x_train), len(x_train) + forward_steps)):
        # This loop performs forward sampling for all rollouts at once
        x_prev = x_full[:, :i]  # contains i number of data points
        y_prev = y_full[:, :i]  # contains i number of data points

        # one-step-ahead prediction of x
        x_new = [get_x_new(x, rng=rng) for x in x_prev]
        x_full[:, i] = np.concatenate(x_new)

        # one-step-ahead prediction of y | x
        y_new = one_step_ahead_batched(x_new, list(x_prev), list(y_prev), rng=rng)
        y_full[:, i] = y_new[:, 0]

    return x_full, y_full
