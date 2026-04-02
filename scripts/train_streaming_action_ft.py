import sys
from multiprocessing.reduction import ForkingPickler
from typing import Any

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, dataloader

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers


default_collate_func = dataloader.default_collate


def default_collate_override(batch) -> Any:
    dataloader._use_shared_memory = False
    return default_collate_func(batch)


setattr(dataloader, "default_collate", default_collate_override)
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
