import jax
import numpy as np
import pytest

from tabmgp import TabPFNClassifierPPD, TabPFNRegressorPPD, forward_sampling


class _Scalar:
    def __init__(self, value):
        self.value = value

    def __float__(self):
        return self.value


class _FakeBarDist:
    def icdf(self, logits, u):
        return _Scalar(float(logits[0] + u))


class _FakeRegressorPPD(TabPFNRegressorPPD):
    def __init__(self, logits):
        self.logits = np.asarray(logits)

    def _predict_full(self, x_new, x_prev, y_prev):
        return {
            "criterion": _FakeBarDist(),
            "logits": self.logits[: x_new.shape[0]],
        }


class _FakeClassifierPPD(TabPFNClassifierPPD):
    def __init__(self, probs):
        self.probs = np.asarray(probs)
        self.classes_ = np.array([0, 1])

    def fit(self, x_prev, y_prev):
        return self

    def predict_proba(self, x_new):
        return self.probs[: x_new.shape[0]]


def test_classifier_sample_returns_vector_for_multiple_rows():
    x_new = np.zeros((3, 2))
    x_prev = np.zeros((4, 2))
    y_prev = np.array([0, 1, 0, 1])
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    y_new = _FakeClassifierPPD(probs).sample(
        jax.random.key(0), x_new, x_prev, y_prev
    )

    assert y_new.shape == (3,)
    np.testing.assert_array_equal(y_new, np.array([0, 1, 0]))


def test_classifier_sample_returns_vector_for_single_row():
    x_new = np.zeros((1, 2))
    x_prev = np.zeros((4, 2))
    y_prev = np.array([0, 1, 0, 1])

    y_new = _FakeClassifierPPD(np.array([[0.0, 1.0]])).sample(
        jax.random.key(0), x_new, x_prev, y_prev
    )

    assert y_new.shape == (1,)
    np.testing.assert_array_equal(y_new, np.array([1]))


def test_regressor_sample_returns_vector_for_single_row():
    x_new = np.zeros((1, 2))
    x_prev = np.zeros((4, 2))
    y_prev = np.array([0.0, 1.0, 0.0, 1.0])

    y_new = _FakeRegressorPPD(np.array([[2.0, 3.0]])).sample(
        jax.random.key(0), x_new, x_prev, y_prev
    )

    assert y_new.shape == (1,)


def test_forward_sampling_accepts_single_row_sample_vector():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0.0, 1.0])

    def one_step_ahead(key, x_new, x_prev, y_prev):
        assert x_new.shape == (1, 2)
        return np.array([float(y_prev[-1] + 1.0)])

    x_full, y_full = forward_sampling(
        jax.random.key(0), one_step_ahead, x_train, y_train, forward_steps=2
    )

    assert x_full.shape == (4, 2)
    assert y_full.shape == (4,)
    np.testing.assert_array_equal(y_full[-2:], np.array([2.0, 3.0]))


def test_forward_sampling_resumes_deterministically():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0.0, 1.0])
    key = jax.random.key(0)

    def one_step_ahead(key, x_new, x_prev, y_prev):
        return np.asarray([jax.random.uniform(key)])

    x_full, y_full = forward_sampling(
        key, one_step_ahead, x_train, y_train, forward_steps=4
    )
    x_partial, y_partial = forward_sampling(
        key, one_step_ahead, x_train, y_train, forward_steps=2
    )
    x_resumed, y_resumed = forward_sampling(
        key, one_step_ahead, x_partial, y_partial, forward_steps=2
    )

    np.testing.assert_array_equal(x_resumed, x_full)
    np.testing.assert_array_equal(y_resumed, y_full)


def test_forward_sampling_rejects_scalar_sample():
    x_train = np.array([[0.0, 0.0], [1.0, 1.0]])
    y_train = np.array([0.0, 1.0])

    def one_step_ahead(key, x_new, x_prev, y_prev):
        return np.array(2.0)

    with pytest.raises(AssertionError):
        forward_sampling(
            jax.random.key(0), one_step_ahead, x_train, y_train, forward_steps=1
        )
