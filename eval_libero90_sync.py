import argparse
import json
import pathlib
import sys

import torch
from omegaconf import OmegaConf

ROOT_DIR = str(pathlib.Path(__file__).parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from model.vla_qwen3 import Qwen3VLA
from workspace.libero_rollout import evaluate_libero_rollouts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n-eval", type=int, default=10)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location="cpu")
    vla.load_state_dict(payload["model"], strict=True)
    vla.eval()

    metrics = evaluate_libero_rollouts(
        vla=vla,
        cfg=cfg,
        task_name=str(args.task),
        n_eval=int(args.n_eval),
        save_videos=bool(args.save_videos),
        output_dir=str(pathlib.Path.cwd()),
        temperature=float(args.temperature),
        top_k=args.top_k,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
