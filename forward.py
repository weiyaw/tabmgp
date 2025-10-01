# %%
from tabpfn import TabPFNClassifier, TabPFNRegressor

import torch
import jax
import numpy as np
from tqdm import tqdm

from timeit import default_timer as timer
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Callable
import warnings

import utils
from dgp import (
    load_dgp,
    OPENML_CLASSIFICATION,
    OPENML_BINARY_CLASSIFICATION,
    OPENML_REGRESSION,
)

KeyArray = jax.Array
Array = jax.Array

from dgp import *

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


class TabPFNRegresorPPD(TabPFNRegressor):

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
        )

    def sample(
        self, key: KeyArray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> tuple[np.ndarray, dict]:
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
        del pred_output["criterion"]
        pred_output["logits"] = pred_output["logits"].numpy()
        return np.squeeze(y_new), pred_output

    def bardist_sample(
        self, key: KeyArray, bardist_icdf: Callable, logits: np.ndarray, t: float = 1.0
    ) -> np.ndarray:
        """Samples values from the bar distribution. A modified version of
        https://github.com/PriorLabs/TabPFN/blob/main/src/tabpfn/model/bar_distribution.py#L576

        Temperature t.
        """
        p_cdf = jax.random.uniform(key, shape=logits.shape[:-1])
        return np.array(
            [bardist_icdf(logits[i, :] / t, p) for i, p in enumerate(p_cdf.tolist())],
        )


class TabPFNClassifierPPD(TabPFNClassifier):

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
        )

    def sample(
        self, key: KeyArray, x_new: np.ndarray, x_prev: np.ndarray, y_prev: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        self.fit(x_prev, y_prev)
        probs_new = self.predict_proba(x_new).squeeze()
        idx_new = jax.random.choice(key, a=self.classes_.size, p=probs_new)
        y_new = self.classes_[idx_new]

        # we use jax to sample from a categorical distribution in the PPD
        # resampling step.
        y_new = y_new.squeeze() if isinstance(y_new, np.ndarray) else y_new
        return y_new, {"probs": probs_new}


def bootstrap(key: KeyArray, data: Array | np.ndarray, n: int) -> np.ndarray:
    # Bayesian bootstrap
    n_data = data.shape[0]
    bootstrap_data = np.concatenate([data, np.zeros((n, data.shape[1]))])

    for i in range(n):
        key, subkey = jax.random.split(key)
        # resample with replacement
        idx = jax.random.randint(subkey, shape=(), minval=0, maxval=n_data + i)
        bootstrap_data[n_data + i] = bootstrap_data[idx]
    return bootstrap_data


@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    outdir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    torch.manual_seed(cfg.seed * 71)
    key = jax.random.key(cfg.seed * 37)
    key, data_key, resample_key = jax.random.split(key, 3)

    dgp = load_dgp(cfg, data_key)
    # save all the infomation about dgp
    utils.write_to(f"{outdir}/dgp.pickle", dgp)

    dim_x = dgp.train_data["x"].shape[-1]

    if cfg.dgp.name.startswith("regression-fixed"):
        ppd = TabPFNRegresorPPD(
            [False] * dim_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name.startswith("classification-fixed"):
        ppd = TabPFNClassifierPPD(
            [False] * dim_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_CLASSIFICATION:
        ppd = TabPFNClassifierPPD(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_BINARY_CLASSIFICATION:
        ppd = TabPFNClassifierPPD(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_REGRESSION:
        ppd = TabPFNRegresorPPD(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    else:
        raise ValueError(f"Unknown experiment name: {cfg.dgp.name}")

    n_train = utils.get_n_data(dgp.train_data)
    recursion_length = cfg.recursion_length
    recursion_times = cfg.recursion_times
    n_total_size = n_train + recursion_length

    def get_x_new(key, x_prev, option):
        # This is the PPD for x. It should return an array of shape (1, dim_x)
        if option == "bb":
            # Future x are drawn with Bayesian bootstrap
            idx = jax.random.randint(key, shape=(), minval=0, maxval=x_prev.shape[0])
            return x_prev[None, idx]
        elif option == "truth":
            # Future x are drawn from truth
            return dgp.get_x_data(key, 1)
        else:
            raise ValueError(f"Unknown resample_x option: {option}")

    for b in range(recursion_times):
        start = timer()
        bkey = jax.random.fold_in(resample_key, b)

        recursion_path = f"{outdir}/recursion/recursion-{b}.pickle"
        aux_path = f"{outdir}/aux/aux-{b}.pickle"
        if os.path.exists(recursion_path) and os.path.exists(aux_path):
            # Continue recursion if it has already exists
            aux_ls = utils.read_from(aux_path)
            recursion_existing = utils.read_from(recursion_path)
            x_existing = recursion_existing["x"]
            y_existing = recursion_existing["y"]
            recursion_start = x_existing.shape[0]
            recursion_left = n_total_size - recursion_start
        else:
            # Otherwise, start afresh
            aux_ls = []
            x_existing = dgp.train_data["x"]
            y_existing = dgp.train_data["y"]
            recursion_start = n_train
            recursion_left = recursion_length

        x_full = np.concatenate([x_existing, np.full((recursion_left, dim_x), -1.0)])
        y_full = np.concatenate([y_existing, np.full(recursion_left, -1.0)])

        for i in tqdm(range(recursion_start, n_total_size)):
            # This loop performs forward sampling
            x_all_prev = x_full[:i]  # contains i number of data points
            y_all_prev = y_full[:i]  # contains i number of data points

            # one-step-ahead prediction of x
            rkey = jax.random.fold_in(bkey, i)
            rkey, subkey = jax.random.split(rkey)
            x_new = get_x_new(subkey, x_all_prev, cfg.resample_x)
            x_full[i] = x_new

            # one-step-ahead prediction of y | x
            rkey, subkey = jax.random.split(rkey)
            y_full[i], aux = ppd.sample(subkey, x_new, x_all_prev, y_all_prev)
            aux_ls.append(aux)

        if recursion_left == 0:
            logging.info(f"Sample {b} untouched.")
        else:
            # Save if we actually performed recursion
            logging.info(f"Sample {b} takes {timer() - start:.4f} seconds")
            utils.write_to(aux_path, aux_ls, verbose=False)
            utils.write_to(recursion_path, {"x": x_full, "y": y_full}, verbose=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("print_dgp", utils.print_dgp)
    main()

# %%
