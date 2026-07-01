from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

import utils


RUN_TABMGP_PATH = Path(__file__).resolve().parents[1] / "run-tabmgp.py"
SPEC = spec_from_file_location("run_tabmgp", RUN_TABMGP_PATH)
run_tabmgp = module_from_spec(SPEC)
SPEC.loader.exec_module(run_tabmgp)


def _run(tmp_path, monkeypatch, initial_rollout=None, fail_on_sampling=False):
    train_data = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "y": np.array([0.0, 1.0]),
    }
    ctx = SimpleNamespace(
        n_train=2,
        dgp=SimpleNamespace(train_data=train_data),
        preprocessor=SimpleNamespace(encode_data=lambda data: data),
        functional=object(),
        init_theta=None,
        savedir=str(tmp_path / "posterior-likelihood"),
    )
    pred_rule = SimpleNamespace(
        sample=lambda key, x_new, x_prev, y_prev: np.array([len(y_prev)])
    )
    rollout_path = tmp_path / "tabmgp-rollout" / "rollout-0.pickle"
    if initial_rollout is not None:
        utils.write_to(rollout_path, initial_rollout)

    monkeypatch.setattr(run_tabmgp, "load_posterior_context", lambda *args: ctx)
    monkeypatch.setattr(run_tabmgp, "get_experiment_name", lambda *args: "test")
    monkeypatch.setattr(run_tabmgp, "make_pred_rule", lambda *args: pred_rule)
    monkeypatch.setattr(run_tabmgp, "save_mgp_posts", lambda *args: None)
    if fail_on_sampling:
        monkeypatch.setattr(
            run_tabmgp,
            "forward_sampling",
            lambda *args: pytest.fail("forward_sampling should not be called"),
        )

    cfg = OmegaConf.create(
        {
            "expdir": str(tmp_path),
            "loss": "likelihood",
            "forward_steps": [2, 3],
            "seed": 1,
            "rollout_times": 1,
            "run_name": "tabmgp",
            "trace": False,
        }
    )
    run_tabmgp.main.__wrapped__(cfg)
    return utils.read_from(rollout_path)


def test_fresh_rollout_reaches_target_length(tmp_path, monkeypatch):
    rollout = _run(tmp_path, monkeypatch)

    assert len(rollout["y"]) == 5


def test_partial_rollout_is_extended_without_changing_prefix(
    tmp_path, monkeypatch
):
    initial_rollout = {
        "x": np.array([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]),
        "y": np.array([0.0, 1.0, 2.0]),
    }

    rollout = _run(tmp_path, monkeypatch, initial_rollout)

    assert len(rollout["y"]) == 5
    np.testing.assert_array_equal(rollout["x"][:3], initial_rollout["x"])
    np.testing.assert_array_equal(rollout["y"][:3], initial_rollout["y"])


@pytest.mark.parametrize("length", [5, 6])
def test_complete_or_overlength_rollout_is_untouched(
    tmp_path, monkeypatch, length
):
    initial_rollout = {
        "x": np.arange(length * 2, dtype=float).reshape(length, 2),
        "y": np.arange(length, dtype=float),
    }

    rollout = _run(
        tmp_path,
        monkeypatch,
        initial_rollout,
        fail_on_sampling=True,
    )

    np.testing.assert_array_equal(rollout["x"], initial_rollout["x"])
    np.testing.assert_array_equal(rollout["y"], initial_rollout["y"])
