from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_dataset_obs_runtime_mse import (  # noqa: E402
    NumpyEncoder,
    _build_model,
    _episode_image,
    _episode_index_from_dataset_index,
    _episode_proprio,
    _resolve_device,
)
from experiments.libero.eval_dataset_obs_sync_mse import (  # noqa: E402
    _resolve_action_horizon,
)
from experiments.libero.eval_first_obs_runtime_ckpt_mse import _drive_first_chunk  # noqa: E402
from experiments.libero.eval_libero_policy_utils import _postprocess_libero_action_chunk  # noqa: E402
from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.async_streaming_runtime import StreamingRuntime  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the first-observation action chunk from runtime cold-start versus "
            "model.infer_action() on the same checkpoint and dataset sample."
        )
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--task-suite-name", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--rand-device", default=None)
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def _load_dataset_and_cfg(args: argparse.Namespace):
    overrides = [
        f"task={args.task_config}",
        f"ckpt={args.ckpt}",
        f"mixed_precision={args.mixed_precision}",
        f"seed={int(args.seed)}",
    ]
    if args.device is not None:
        overrides.append(f"EVALUATION.device={args.device}")
    if args.num_inference_steps is not None:
        overrides.append(f"EVALUATION.num_inference_steps={int(args.num_inference_steps)}")
    if args.action_horizon is not None:
        overrides.append(f"EVALUATION.action_horizon={int(args.action_horizon)}")
    if args.rand_device is not None:
        overrides.append(f"EVALUATION.rand_device={args.rand_device}")
    if args.task_suite_name is not None:
        overrides.append(f"EVALUATION.task_suite_name={args.task_suite_name}")
    if args.task_id is not None:
        overrides.append(f"EVALUATION.task_id={int(args.task_id)}")

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="sim_libero", overrides=overrides)

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    return cfg, dataset


def _infer_chunk_env_action(
    *,
    model,
    cfg,
    processor,
    prompt: str,
    input_image: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    seed: int,
    rand_device: str,
    tiled: bool,
) -> np.ndarray:
    with torch.no_grad():
        video_context, video_context_mask = model.encode_prompt(prompt)
        out = model.infer_action(
            prompt=None,
            input_image=input_image,
            action_horizon=int(action_horizon),
            proprio=proprio,
            context=video_context,
            context_mask=video_context_mask,
            num_inference_steps=int(num_inference_steps),
            sigma_shift=sigma_shift,
            seed=int(seed),
            rand_device=str(rand_device),
            tiled=bool(tiled),
        )
    pred_chunk = out["action"].detach().to(device="cpu", dtype=torch.float32).unsqueeze(0)
    env_chunk = np.asarray(
        _postprocess_libero_action_chunk(pred_chunk, processor=processor, cfg=cfg),
        dtype=np.float32,
    )
    if env_chunk.ndim == 3:
        env_chunk = env_chunk[0]
    if env_chunk.ndim != 2:
        raise ValueError(f"Expected infer env chunk [T, D], got shape {tuple(env_chunk.shape)}")
    return env_chunk


def _runtime_chunk_env_action(
    *,
    model,
    cfg,
    processor,
    prompt: str,
    input_image: torch.Tensor,
    proprio: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: float | None,
    seed: int,
    rand_device: str,
    tiled: bool,
) -> np.ndarray:
    with torch.no_grad():
        video_context, video_context_mask = model.encode_prompt(prompt)
    video_context = video_context.to(device="cpu", dtype=model.torch_dtype)
    video_context_mask = video_context_mask.to(device="cpu", dtype=torch.bool)
    action_postprocess = lambda x: _postprocess_libero_action_chunk(x, processor=processor, cfg=cfg)
    runtime = StreamingRuntime(
        video_model=model,
        action_model=model,
        video_context=video_context,
        video_context_mask=video_context_mask,
        action_context=video_context,
        action_context_mask=video_context_mask,
        action_postprocess=action_postprocess,
        action_horizon=int(action_horizon),
        num_inference_steps=int(num_inference_steps),
        sigma_shift=sigma_shift,
        rand_device=str(rand_device),
        tiled=bool(tiled),
        seed=int(seed),
    )
    return _drive_first_chunk(
        runtime,
        input_image=input_image,
        proprio=proprio,
        num_inference_steps=int(num_inference_steps),
        action_horizon=int(action_horizon),
    )


@torch.no_grad()
def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg, dataset = _load_dataset_and_cfg(args)
    episode_index = (
        int(args.episode_index)
        if args.episode_index is not None
        else _episode_index_from_dataset_index(dataset, int(args.dataset_index))
    )
    payload = dataset._load_episode_cache(int(episode_index))
    device = _resolve_device(cfg, args)
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = _build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    action_horizon = int(_resolve_action_horizon(cfg))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False) or bool(args.tiled))
    sigma_shift_cfg = cfg.EVALUATION.get("sigma_shift")
    sigma_shift = None if sigma_shift_cfg is None else float(sigma_shift_cfg)

    input_image = _episode_image(dataset, payload, 0)
    proprio = _episode_proprio(dataset, payload, 0)
    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))

    infer_env_chunk = _infer_chunk_env_action(
        model=model,
        cfg=cfg,
        processor=processor,
        prompt=prompt,
        input_image=input_image,
        proprio=proprio,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=int(args.seed),
        rand_device=rand_device,
        tiled=tiled,
    )
    runtime_env_chunk = _runtime_chunk_env_action(
        model=model,
        cfg=cfg,
        processor=processor,
        prompt=prompt,
        input_image=input_image,
        proprio=proprio,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        seed=int(args.seed),
        rand_device=rand_device,
        tiled=tiled,
    )

    diff = runtime_env_chunk - infer_env_chunk
    per_position_mse = [float(np.mean(np.square(diff[pos]))) for pos in range(int(action_horizon))]

    return {
        "ckpt": str(args.ckpt),
        "task_config": str(args.task_config),
        "dataset_index": int(args.dataset_index),
        "episode_index": int(episode_index),
        "instruction": str(payload["instruction"]),
        "seed": int(args.seed),
        "device": str(device),
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(num_inference_steps),
        "infer_vs_runtime": {
            "chunk_env_action_mse": float(np.mean(np.square(diff))),
            "chunk_env_action_mae": float(np.mean(np.abs(diff))),
            "chunk_env_action_max_abs": float(np.max(np.abs(diff))),
            "first_env_action_mse": float(np.mean(np.square(diff[0]))),
            "first_env_action_mae": float(np.mean(np.abs(diff[0]))),
            "per_position_env_action_mse": per_position_mse,
        },
        "infer_env_chunk": infer_env_chunk.astype(float).tolist(),
        "runtime_env_chunk": runtime_env_chunk.astype(float).tolist(),
        "diff_env_chunk": diff.astype(float).tolist(),
    }


def main() -> None:
    args = _parse_args()
    result = run_compare(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(json.dumps(result["infer_vs_runtime"], indent=2, cls=NumpyEncoder))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
