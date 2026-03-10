if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import numpy as np
import random
import hydra
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import pathlib
import copy
import tqdm
from typing import Union, List

from oat.workspace.base_workspace import BaseWorkspace
from oat.dataset.base_dataset import BaseDataset
from oat.common.checkpoint_util import TopKCheckpointManager
from oat.common.hydra_util import register_new_resolvers
from oat.tokenizer.fast.tokenizer_wrapper import FASTTok

register_new_resolvers()


class TrainFASTTokWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, output_dir=None, lazy_instantiation=True):
        super().__init__(cfg, output_dir=output_dir)
        """
        Lazy instantiation allows deferring model, optimizer, and ema creation
        until after the seed has been set.
        1. If lazy_instantiation is False, model, optimizer, and ema are created immediately.
           This is useful for checkpoint loading, where we need to create the model
           before loading the state dict.
        2. If lazy_instantiation is True, model, optimizer, and ema are created in run().
           This is useful for normal training, where we want to seed the creation of these
           objects.
        """
        if lazy_instantiation:
            self.model = None
        else:
            self.model = hydra.utils.instantiate(cfg.tokenizer)

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # set seed
        seed = int(cfg.training.seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model after seeding
        self.model: FASTTok = hydra.utils.instantiate(cfg.tokenizer)
        
        # configure dataset
        dataset: BaseDataset = hydra.utils.instantiate(
            cfg.task.tokenizer.dataset)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        # configure normalizer
        normalizer = dataset.get_normalizer()
        self.model.set_normalizer(normalizer)

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, "checkpoints"),
            **cfg.checkpoint.topk
        )

        # train
        action_data: List[np.ndarray] = []
        for data in dataset:
            action_data.append(data['action'].numpy())
        self.model.fast_tok = self.model.fast_tok.fit(
            action_data,
            scale=cfg.training.fast_scale,
            vocab_size=cfg.training.fast_vocab_size,
            time_horizon=cfg.horizon,
            action_dim=action_data[0].shape[-1]
        )
        
        # reconstruction eval
        self.model.eval()
        step_log = dict()
        with torch.inference_mode():
            loss_info = torch.zeros(2, dtype=torch.float32)
            with tqdm.tqdm(
                val_dataloader,
                desc="Reconstruction eval",
                leave=False,
                mininterval=cfg.training.tqdm_interval_sec
            ) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    samples = batch['action']   # (B, T, D)
                    reconst_samples = self.model.detokenize(self.model.tokenize(samples))
                    mse = F.mse_loss(reconst_samples, samples).item()

                    batch_size = samples.shape[0]
                    loss_info[0] += mse * batch_size
                    loss_info[1] += batch_size

                    if (cfg.training.max_reconst_steps is not None) \
                        and batch_idx >= (cfg.training.max_reconst_steps - 1):
                        break

            step_log['test_reconst_mse'] = (loss_info[0] / loss_info[1]).item()
            print(f"Reconstruction MSE: {step_log['test_reconst_mse']}")
        
        # checkpoint
        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint()
        if cfg.checkpoint.save_last_snapshot:
            self.save_snapshot()
        metric_dict = dict()
        for k, v in step_log.items():
            new_key = k.replace('/', '_')
            metric_dict[new_key] = v
        topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
        if topk_ckpt_path is not None:
            self.save_checkpoint(path=topk_ckpt_path)
            fast_save_path = os.path.join(
                os.path.dirname(topk_ckpt_path), 
                cfg.checkpoint.fast_save_name
            )
            self.model.fast_tok.save_pretrained(fast_save_path)
        else:
            raise ValueError("No checkpoint path to save.")

        
@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainFASTTokWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
