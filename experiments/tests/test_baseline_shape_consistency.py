import numpy as np

from baseline import (
    copula_classification,
    copula_classification_tabpfn_init,
    copula_cregression,
    copula_cregression_tabpfn_init,
)


def test_cregression_tabpfn_init_shape_matches_copula():
    rng = np.random.default_rng(0)
    n = 12
    d = 2
    b = 2
    t = 2

    train_data = {
        "x": rng.normal(size=(n, d)),
        "y": rng.normal(size=n),
    }

    out_copula, _ = copula_cregression(
        train_data, categorical_x=[False] * d, B=b, T=t
    )
    out_tabpfn, _ = copula_cregression_tabpfn_init(
        train_data, categorical_x=[False] * d, B=b, T=t
    )

    assert out_copula["y"].shape == out_tabpfn["y"].shape
    assert out_copula["x"].shape == out_tabpfn["x"].shape


def test_classification_tabpfn_init_shape_matches_copula():
    rng = np.random.default_rng(1)
    n = 24
    d = 2
    b = 2
    t = 3

    train_data = {
        "x": rng.normal(size=(n, d)),
        "y": rng.binomial(1, 0.5, size=n),
    }

    out_copula, _ = copula_classification(
        train_data, categorical_x=[False] * d, B=b, T=t
    )
    out_tabpfn, _ = copula_classification_tabpfn_init(
        train_data, categorical_x=[False] * d, B=b, T=t
    )

    assert out_copula["y"].shape == out_tabpfn["y"].shape
    assert out_copula["x"].shape == out_tabpfn["x"].shape
