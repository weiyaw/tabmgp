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
    outdir = os.path.relpath(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    os.makedirs(f"{outdir}/logs", exist_ok=True)
    os.makedirs(f"{outdir}/configs", exist_ok=True)
    OmegaConf.save(cfg, f"{outdir}/configs/{cfg.run_name}.yaml")

    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed * 71)
    key = jax.random.key(cfg.seed * 37)
    _, data_key, _ = jax.random.split(key, 3)

    dgp = load_dgp(cfg, data_key)
    utils.write_to(f"{outdir}/dgp.pickle", dgp, verbose=True)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    OmegaConf.register_new_resolver("print_dgp", utils.print_dgp)
    main()
