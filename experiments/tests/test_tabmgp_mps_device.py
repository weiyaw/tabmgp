import jax
import numpy as np
import pytest
import torch

from tabmgp import TabPFNRegressorPPD


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS device is not available",
)
def test_regressor_sample_and_icdf_on_mps_device():
    x_prev = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [0.2, 0.8],
        ]
    )
    y_prev = np.array([0.0, 1.0, 1.0, 2.0, 0.5, 1.2])
    x_new = np.array([[0.1, 0.2], [0.8, 0.7]])

    regressor = TabPFNRegressorPPD(
        categorical_features_indices=[],
        device="mps",
        n_estimators=1,
        model_path="tabpfn-v2-regressor.ckpt",
    )

    y_new = regressor.sample(jax.random.key(0), x_new, x_prev, y_prev)
    q_new = regressor.icdf(np.array([0.25, 0.75]), x_new, x_prev, y_prev)

    assert y_new.shape == (2,)
    assert q_new.shape == (2, 2)
    assert np.isfinite(y_new).all()
    assert np.isfinite(q_new).all()
