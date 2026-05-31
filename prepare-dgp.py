import logging
import os

import hydra
import jax
import torch
from omegaconf import DictConfig, OmegaConf

import utils
from dgp import load_dgp


@hydra.main(version_base=None, config_path="conf", config_name="dgp")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    outdir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    torch.manual_seed(cfg.seed * 71)
    key = jax.random.key(cfg.seed * 37)
    _, data_key, _ = jax.random.split(key, 3)

    dgp = load_dgp(cfg, data_key)
    utils.write_to(f"{outdir}/dgp.pickle", dgp, verbose=True)
    OmegaConf.save(cfg, f"{outdir}/dgp.yaml")


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("print_dgp", utils.print_dgp)
    main()
