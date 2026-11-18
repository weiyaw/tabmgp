import logging
import os
import warnings
from timeit import default_timer as timer

import hydra
import jax
import chex
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

import utils
from dgp import (
    OPENML_BINARY_CLASSIFICATION,
    OPENML_CLASSIFICATION,
    OPENML_REGRESSION,
)
from experiment_setup import get_experiment_name
from posterior_runner import (
    available_forward_steps,
    compile_rollout,
    load_posterior_context,
    normalize_forward_steps,
)
from tabmgp import TabPFNClassifierPPD, TabPFNRegressorPPD, forward_sampling


jax.config.update("jax_enable_x64", True)
warnings.filterwarnings(
    "ignore",
    message="Running on CPU with more than 200 samples may be slow.",
    category=UserWarning,
)


def make_pred_rule(cfg: DictConfig, dgp, exp_name: str):
    if exp_name.startswith("regression-fixed"):
        return TabPFNRegressorPPD(
            categorical_features_indices=[],
            n_estimators=cfg.n_estimators,
            average_before_softmax=cfg.average_before_softmax,
            model_path="tabpfn-v2-regressor.ckpt",
        )
    if exp_name.startswith("classification-fixed"):
        return TabPFNClassifierPPD(
            categorical_features_indices=[],
            n_estimators=cfg.n_estimators,
            average_before_softmax=cfg.average_before_softmax,
            model_path="tabpfn-v2-classifier.ckpt",
        )
    if exp_name in OPENML_CLASSIFICATION or exp_name in OPENML_BINARY_CLASSIFICATION:
        categorical_features_indices = [i for i, c in enumerate(dgp.categorical_x) if c]
        return TabPFNClassifierPPD(
            categorical_features_indices=categorical_features_indices,
            n_estimators=cfg.n_estimators,
            average_before_softmax=cfg.average_before_softmax,
            model_path="tabpfn-v2-classifier.ckpt",
        )
    if exp_name in OPENML_REGRESSION:
        categorical_features_indices = [i for i, c in enumerate(dgp.categorical_x) if c]
        return TabPFNRegressorPPD(
            categorical_features_indices=categorical_features_indices,
            n_estimators=cfg.n_estimators,
            average_before_softmax=cfg.average_before_softmax,
            model_path="tabpfn-v2-regressor.ckpt",
        )
    raise ValueError(f"Unknown experiment name: {exp_name}")


@hydra.main(version_base=None, config_path="conf", config_name="tabmgp")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    os.makedirs(f"{cfg.expdir}/logs", exist_ok=True)
    os.makedirs(f"{cfg.expdir}/configs", exist_ok=True)
    OmegaConf.save(cfg, f"{cfg.expdir}/configs/{cfg.run_name}.yaml")

    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    forward_steps = normalize_forward_steps(cfg.forward_steps)
    max_steps = max(forward_steps)
    ctx = load_posterior_context(cfg.expdir, cfg.loss)
    rollout_length = ctx.n_train + max_steps
    torch.manual_seed(cfg.seed * 71)
    key = jax.random.key(cfg.seed * 37)
    _, _, resample_key = jax.random.split(key, 3)
    pred_rule = make_pred_rule(cfg, ctx.dgp, get_experiment_name(cfg.expdir))

    for b in range(cfg.rollout_times):
        start = timer()
        bkey = jax.random.fold_in(resample_key, b)
        rollout_path = f"{cfg.expdir}/tabmgp-rollout/rollout-{b}.pickle"
        if os.path.exists(rollout_path):
            rollout = utils.read_from(rollout_path)
            x_init, y_init = rollout["x"], rollout["y"]
        else:
            x_init = ctx.dgp.train_data["x"]
            y_init = ctx.dgp.train_data["y"]

        chex.assert_equal_shape_prefix([x_init, y_init], 1)
        required_steps = rollout_length - len(y_init)
        if required_steps <= 0:
            logging.info(f"Sample {b} untouched.")
            continue

        x_full, y_full = forward_sampling(
            bkey,
            pred_rule.sample,
            x_init,
            y_init,
            required_steps,
        )
        logging.info(f"Sample {b} takes {timer() - start:.4f} seconds")
        utils.write_to(rollout_path, {"x": x_full, "y": y_full}, verbose=True)

    full_rollout = ctx.preprocessor.encode_data(
        compile_rollout(cfg.expdir, "tabmgp-rollout")
    )
    logging.info(f"Shape of {cfg.run_name} rollout: {utils.tree_shape(full_rollout)}")
    n_steps = available_forward_steps(full_rollout, ctx.n_train)
    if max_steps > n_steps:
        raise ValueError(f"Requested forward step {max_steps} exceeds max {n_steps}")

    for t in forward_steps:
        start = timer()
        rollout_t = jax.tree.map(lambda x: x[:, : ctx.n_train + t], full_rollout)
        post = ctx.functional.get_mgp(rollout_t, ctx.init_theta)
        utils.write_to(
            f"{ctx.savedir}/{cfg.run_name}-{t}-post.pickle", post, verbose=True
        )
        logging.info(f"Diagnostics: {np.mean(post[1].success)}")
        logging.info(f"{cfg.run_name} posterior ({t}): {timer() - start:.2f} seconds")

    if cfg.trace:
        start = timer()
        freq = max(n_steps // int(cfg.resolution), 1)
        trace = ctx.functional.get_theta_trace(
            full_rollout,
            ctx.init_theta,
            start=ctx.n_train - 1,
            batch_size=cfg.batch,
            freq=freq,
        )
        utils.write_to(
            f"{ctx.savedir}/{cfg.run_name}-{ctx.n_train}-{ctx.n_train + n_steps}-{freq}-trace.pickle",
            trace,
            verbose=True,
        )
        logging.info(f"{cfg.run_name} trace: {timer() - start:.2f} seconds")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
