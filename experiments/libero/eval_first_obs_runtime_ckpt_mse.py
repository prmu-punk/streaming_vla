from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir

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
    _array_stats,
    _build_padded_raw_chunk,
    _dataset_action_to_env_action,
    _resolve_action_horizon,
)
from experiments.libero.eval_libero_policy_utils import _postprocess_libero_action_chunk  # noqa: E402
from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.async_streaming_runtime import StreamingRuntime  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from hydra.utils import instantiate  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two checkpoints on the first observation only, using the runtime's "
            "first-chunk inference path, and measure chunk MSE against dataset GT."
        )
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--task-suite-name", default=None)
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--ft-ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--baseline-mode", choices=["runtime", "sync"], default="runtime")
    parser.add_argument("--baseline-num-inference-steps", type=int, default=None)
    parser.add_argument("--baseline-action-horizon", type=int, default=None)
    parser.add_argument("--ft-num-inference-steps", type=int, default=None)
    parser.add_argument("--ft-action-horizon", type=int, default=None)
    parser.add_argument("--rand-device", default=None)
    parser.add_argument("--tiled", action="store_true")
    return parser.parse_args()


def _load_dataset_and_cfg(args: argparse.Namespace):
    overrides = [
        f"task={args.task_config}",
        f"ckpt={args.ft_ckpt}",
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


def _drive_first_chunk(
    runtime: StreamingRuntime,
    *,
    input_image: torch.Tensor,
    proprio: torch.Tensor,
    num_inference_steps: int,
    action_horizon: int,
) -> np.ndarray:
    runtime.start()
    try:
        runtime.reset_for_formal_phase(env_step=0)
        runtime.submit_observation(
            input_image=input_image,
            proprio=proprio,
            env_step=0,
            obs_index=0,
            obs_timestamp_ms=0.0,
            trigger_job=False,
        )

        max_iters = max(8, int(num_inference_steps) * 4 + int(action_horizon) * 2)
        for _ in range(max_iters):
            runtime.poll()
            completed = runtime.completed_jobs()
            pending = runtime.pending_jobs()
            if completed >= int(num_inference_steps) and pending == 0:
                break
            time.sleep(0.0005)

        chunk: list[np.ndarray] = []
        for env_step in range(int(action_horizon)):
            action = runtime.get_action(int(env_step), count_miss=False)
            if action is None:
                raise RuntimeError(
                    f"Missing first-chunk action at env_step={env_step}. "
                    f"completed_jobs={runtime.completed_jobs()} pending_jobs={runtime.pending_jobs()}"
                )
            chunk.append(np.asarray(action, dtype=np.float32))
        return np.stack(chunk, axis=0)
    finally:
        runtime.stop()


def _compute_chunk_metrics(
    *,
    pred_env_chunk: np.ndarray,
    target_raw_action: np.ndarray,
    action_horizon: int,
    binarize_gripper: bool,
) -> dict[str, Any]:
    target_raw_chunk, action_is_pad_np = _build_padded_raw_chunk(
        target_raw_action,
        env_step=0,
        action_horizon=int(action_horizon),
    )
    target_env_chunk = _dataset_action_to_env_action(
        target_raw_chunk,
        binarize_gripper=bool(binarize_gripper),
    )

    valid_mask = ~action_is_pad_np
    chunk_env_sqerr = np.square(pred_env_chunk[valid_mask] - target_env_chunk[valid_mask])
    chunk_env_mse = float(np.mean(chunk_env_sqerr)) if bool(np.any(valid_mask)) else None
    chunk_env_mse_per_position = [
        (None if bool(action_is_pad_np[pos]) else float(np.mean(np.square(pred_env_chunk[pos] - target_env_chunk[pos]))))
        for pos in range(int(action_horizon))
    ]

    first_pred = np.asarray(pred_env_chunk[0], dtype=np.float32)
    first_target = np.asarray(target_env_chunk[0], dtype=np.float32)
    first_diff = first_pred - first_target

    return {
        "first_env_action_mse": float(np.mean(np.square(first_diff))),
        "first_env_action_mae": float(np.mean(np.abs(first_diff))),
        "chunk_env_action_mse": chunk_env_mse,
        "chunk_env_action_mse_per_position": chunk_env_mse_per_position,
        "pred_env_chunk": pred_env_chunk.astype(float).tolist(),
        "target_env_chunk": target_env_chunk.astype(float).tolist(),
    }


@torch.no_grad()
def _predict_first_chunk_sync(
    *,
    model,
    processor,
    cfg,
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
        raise ValueError(f"Expected sync env chunk [T, D], got shape {tuple(env_chunk.shape)}")
    return env_chunk


@torch.no_grad()
def _evaluate_ckpt(
    *,
    cfg,
    ckpt: str,
    dataset,
    payload: dict[str, Any],
    input_image: torch.Tensor,
    proprio: torch.Tensor,
    seed: int,
    device: str,
    mode: str,
    action_horizon: int,
    num_inference_steps: int,
    rand_device: str,
    tiled: bool,
) -> dict[str, Any]:
    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", "bf16")))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = _build_model(cfg, ckpt=str(ckpt), model_dtype=model_dtype, device=device)
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
    if mode == "sync":
        pred_env_chunk = _predict_first_chunk_sync(
            model=model,
            processor=processor,
            cfg=cfg,
            prompt=prompt,
            input_image=input_image,
            proprio=proprio,
            action_horizon=int(action_horizon),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=sigma_shift,
            seed=int(seed),
            rand_device=str(rand_device),
            tiled=bool(tiled),
        )
    else:
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
        pred_env_chunk = _drive_first_chunk(
            runtime,
            input_image=input_image,
            proprio=proprio,
            num_inference_steps=int(num_inference_steps),
            action_horizon=int(action_horizon),
        )

    metrics = _compute_chunk_metrics(
        pred_env_chunk=pred_env_chunk,
        target_raw_action=payload["action_raw"]["default"].detach().cpu().numpy().astype(np.float32),
        action_horizon=int(action_horizon),
        binarize_gripper=bool(cfg.EVALUATION.get("binarize_gripper", False)),
    )

    return {
        "ckpt": str(ckpt),
        "mode": str(mode),
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(num_inference_steps),
        **metrics,
    }


@torch.no_grad()
def run_first_obs_runtime_ckpt_mse(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg, dataset = _load_dataset_and_cfg(args)

    episode_index = (
        int(args.episode_index)
        if args.episode_index is not None
        else _episode_index_from_dataset_index(dataset, int(args.dataset_index))
    )
    payload = dataset._load_episode_cache(int(episode_index))

    device = _resolve_device(cfg, args)
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    default_action_horizon = int(_resolve_action_horizon(cfg))
    default_num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False) or bool(args.tiled))

    baseline_action_horizon = int(
        args.baseline_action_horizon
        if args.baseline_action_horizon is not None
        else (args.action_horizon if args.action_horizon is not None else default_action_horizon)
    )
    baseline_num_inference_steps = int(
        args.baseline_num_inference_steps
        if args.baseline_num_inference_steps is not None
        else (args.num_inference_steps if args.num_inference_steps is not None else default_num_inference_steps)
    )
    ft_action_horizon = int(
        args.ft_action_horizon
        if args.ft_action_horizon is not None
        else (args.action_horizon if args.action_horizon is not None else default_action_horizon)
    )
    ft_num_inference_steps = int(
        args.ft_num_inference_steps
        if args.ft_num_inference_steps is not None
        else (args.num_inference_steps if args.num_inference_steps is not None else default_num_inference_steps)
    )

    input_image = _episode_image(dataset, payload, 0)
    proprio = _episode_proprio(dataset, payload, 0)

    baseline = _evaluate_ckpt(
        cfg=cfg,
        ckpt=str(args.baseline_ckpt),
        dataset=dataset,
        payload=payload,
        input_image=input_image,
        proprio=proprio,
        seed=int(args.seed),
        device=device,
        mode=str(args.baseline_mode),
        action_horizon=baseline_action_horizon,
        num_inference_steps=baseline_num_inference_steps,
        rand_device=rand_device,
        tiled=tiled,
    )
    ft = _evaluate_ckpt(
        cfg=cfg,
        ckpt=str(args.ft_ckpt),
        dataset=dataset,
        payload=payload,
        input_image=input_image,
        proprio=proprio,
        seed=int(args.seed),
        device=device,
        mode="runtime",
        action_horizon=ft_action_horizon,
        num_inference_steps=ft_num_inference_steps,
        rand_device=rand_device,
        tiled=tiled,
    )

    baseline_pos = [v for v in baseline["chunk_env_action_mse_per_position"] if v is not None]
    ft_pos = [v for v in ft["chunk_env_action_mse_per_position"] if v is not None]
    per_position_delta: list[float | None] = []
    for base_v, ft_v in zip(
        baseline["chunk_env_action_mse_per_position"],
        ft["chunk_env_action_mse_per_position"],
    ):
        if base_v is None or ft_v is None:
            per_position_delta.append(None)
        else:
            per_position_delta.append(float(ft_v - base_v))
    result = {
        "task_config": str(args.task_config),
        "dataset_index": int(args.dataset_index),
        "episode_index": int(episode_index),
        "instruction": str(payload["instruction"]),
        "seed": int(args.seed),
        "device": str(device),
        "baseline_mode": str(args.baseline_mode),
        "baseline_action_horizon": int(baseline_action_horizon),
        "baseline_num_inference_steps": int(baseline_num_inference_steps),
        "ft_action_horizon": int(ft_action_horizon),
        "ft_num_inference_steps": int(ft_num_inference_steps),
        "baseline": baseline,
        "ft": ft,
        "delta": {
            "first_env_action_mse_ft_minus_baseline": float(ft["first_env_action_mse"] - baseline["first_env_action_mse"]),
            "chunk_env_action_mse_ft_minus_baseline": (
                None
                if baseline["chunk_env_action_mse"] is None or ft["chunk_env_action_mse"] is None
                else float(ft["chunk_env_action_mse"] - baseline["chunk_env_action_mse"])
            ),
            "per_position_env_action_mse_ft_minus_baseline": per_position_delta,
            "baseline_chunk_env_action_mse_mean": _array_stats(baseline_pos),
            "ft_chunk_env_action_mse_mean": _array_stats(ft_pos),
        },
    }
    return result


def main() -> None:
    args = _parse_args()
    result = run_first_obs_runtime_ckpt_mse(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote first-obs runtime CKPT MSE comparison to: {output_path}")
    print(json.dumps(result["delta"], indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
