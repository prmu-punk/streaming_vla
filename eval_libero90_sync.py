import argparse
import json
import pathlib
import sys

import torch
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from vla_qwen3 import Qwen3VLA
from workspace.train_libero90_sync import rollout_eval_sync


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location=vla.device)
    vla.load_state_dict(payload["model"], strict=True)
    vla.eval()

    metrics = rollout_eval_sync(vla, cfg)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
