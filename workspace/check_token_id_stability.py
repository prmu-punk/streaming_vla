from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict

import torch
from omegaconf import OmegaConf


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from model.vla_qwen3 import Qwen3VLA


def collect_token_info(vla: Qwen3VLA) -> Dict[str, Any]:
    tokenizer = vla.processor.tokenizer
    action_tokens = [f"<act_oat_{i}>" for i in range(min(8, vla.action_tokenizer.codebook_size))]
    return {
        "state_placeholder_token": vla.state_placeholder_token,
        "state_placeholder_token_id": int(vla.state_placeholder_token_id),
        "act_eos_hf_id": int(vla.action_tokenizer.act_eos_hf_id),
        "action_token_ids_head": {
            token: int(tokenizer.convert_tokens_to_ids(token)) for token in action_tokens
        },
        "vocab_size": int(len(tokenizer)),
    }


def load_once(config_path: str, checkpoint_path: str) -> Dict[str, Any]:
    vla = Qwen3VLA(config_path=config_path)
    payload = torch.load(checkpoint_path, map_location=vla.device)
    vla.load_state_dict(payload["model"], strict=True)
    info = collect_token_info(vla)
    info["checkpoint_epoch"] = int(payload.get("epoch", -1))
    info["checkpoint_global_step"] = int(payload.get("global_step", 0))
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    vla_config_path = str(cfg.model.vla_config_path)

    first = load_once(vla_config_path, str(args.checkpoint))
    second = load_once(vla_config_path, str(args.checkpoint))

    summary = {
        "first": first,
        "second": second,
        "equal": first == second,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
