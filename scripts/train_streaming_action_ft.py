import sys
from multiprocessing.reduction import ForkingPickler

import hydra
import torch
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
