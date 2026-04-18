import hydra
from omegaconf import DictConfig
import pathlib
import sys
from multiprocessing.reduction import ForkingPickler

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()

import torch

for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]

ROOT_DIR = str(pathlib.Path(__file__).parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
