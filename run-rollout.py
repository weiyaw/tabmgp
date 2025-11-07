# %%


import torch
import jax
from jax import Array
from jax.typing import ArrayLike
from timeit import default_timer as timer
import logging
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import warnings

import utils

from rollout import forward_sampling, TabPFNClassifierPredRule, TabPFNRegressorPredRule

from data import (
    load_dgp,
    OPENML_CLASSIFICATION,
    OPENML_BINARY_CLASSIFICATION,
    OPENML_REGRESSION,
)

from data import *

warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


@hydra.main(version_base=None, config_path="conf", config_name="rollout")
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

    # Setup data
    dgp = load_dgp(cfg, data_key)
    utils.write_to(f"{outdir}/dgp.pickle", dgp)
    dim_x = dgp.train_data["x"].shape[-1]

    # Setup prediction rule
    if cfg.dgp.name.startswith("regression-fixed"):
        pred_rule = TabPFNRegressorPredRule(
            [False] * dim_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name.startswith("classification-fixed"):
        pred_rule = TabPFNClassifierPredRule(
            [False] * dim_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_CLASSIFICATION:
        pred_rule = TabPFNClassifierPredRule(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_BINARY_CLASSIFICATION:
        pred_rule = TabPFNClassifierPredRule(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    elif cfg.dgp.name in OPENML_REGRESSION:
        pred_rule = TabPFNRegressorPredRule(
            dgp.categorical_x, cfg.n_estimators, cfg.average_before_softmax
        )
    else:
        raise ValueError(f"Unknown experiment name: {cfg.dgp.name}")

    for b in range(cfg.rollout_times):
        start = timer()
        bkey = jax.random.fold_in(resample_key, b)

        rollout_path = f"{outdir}/rollout/rollout-{b}.pickle"
        if os.path.exists(rollout_path):
            # Continue rollout if it has already exists
            logging.info(f"Sample {b} untouched.")
            continue

        x_full, y_full = forward_sampling(
            bkey,
            pred_rule,
            dgp.train_data["x"],
            dgp.train_data["y"],
            cfg.rollout_length,
        )

        logging.info(f"Sample {b} takes {timer() - start:.4f} seconds")
        utils.write_to(rollout_path, {"x": x_full, "y": y_full}, verbose=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("print_dgp", utils.print_dgp)
    main()

# %%
