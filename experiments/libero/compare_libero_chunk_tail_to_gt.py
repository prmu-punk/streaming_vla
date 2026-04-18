from __future__ import annotations

import argparse
import copy
import inspect
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_policy_utils import (  # noqa: E402
    _obs_to_model_input,
    _postprocess_libero_action_chunk,
)
from experiments.libero.eval_libero_single import (  # noqa: E402
    NumpyEncoder,
    _build_eval_model,
    _configure_egl_device,
    _mixed_precision_to_model_dtype,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
)
from experiments.libero.libero_utils import (  # noqa: E402
    invert_gripper_action,
)
from experiments.libero.compare_libero_native_async_utils import (  # noqa: E402
    collect_episode_obs_trace,
    clone_obs_dict,
    run_native_async_anchor_chunk,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402
from libero.libero import benchmark  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one-chunk LIBERO action error to GT at matched anchor timesteps "
            "between a streaming/mixed-cache policy and a synchronous baseline."
        )
    )
    parser.add_argument("--streaming-task-config", required=True)
    parser.add_argument("--streaming-ckpt", required=True)
    parser.add_argument("--baseline-task-config", required=True)
    parser.add_argument("--baseline-ckpt", required=True)
    parser.add_argument("--task-suite-name", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--trial-idx", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--async-video-device", default=None)
    parser.add_argument("--async-action-device", default=None)
    parser.add_argument("--render-gpu-id", type=int, default=0)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--tail-fraction", type=float, default=0.25)
    parser.add_argument("--segment", choices=["tail", "head"], default="tail")
    parser.add_argument("--anchor-steps", type=str, default=None)
    parser.add_argument("--num-anchors", type=int, default=5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gt-episode-index", type=int, default=None)
    parser.add_argument("--binarize-gripper", action="store_true")
    return parser.parse_args()


def _compose_cfg(task_config_name: str, ckpt: str, args: argparse.Namespace) -> DictConfig:
    overrides = [
        f"task={task_config_name}",
        f"ckpt={ckpt}",
        f"EVALUATION.task_suite_name={args.task_suite_name}",
        f"EVALUATION.task_id={int(args.task_id)}",
        "EVALUATION.num_trials=1",
        f"gpu_id={int(args.render_gpu_id)}",
        f"mixed_precision={args.mixed_precision}",
    ]
    if args.device is not None:
        overrides.append(f"EVALUATION.device={args.device}")
    if args.num_inference_steps is not None:
        overrides.append(f"EVALUATION.num_inference_steps={int(args.num_inference_steps)}")
    if args.action_horizon is not None:
        overrides.append(f"EVALUATION.action_horizon={int(args.action_horizon)}")
    if args.seed is not None:
        overrides.append(f"seed={int(args.seed)}")

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="sim_libero", overrides=overrides)


def _build_processor(cfg: DictConfig) -> tuple[FastWAMProcessor, Path]:
    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    return processor, dataset_stats_path


def _force_eval_model_dtype(model: torch.nn.Module, *, device: str, model_dtype: torch.dtype) -> torch.nn.Module:
    return model.to(device=device, dtype=model_dtype).eval()


def _normalize_runtime_device(device: str, *, render_gpu_id: int) -> str:
    resolved = str(device)
    if resolved == "cuda":
        return f"cuda:{int(render_gpu_id)}"
    return resolved


def _resolve_action_horizon(cfg: DictConfig) -> int:
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if horizon_cfg is None else int(horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"Resolved action horizon must be positive, got {action_horizon}")
    return action_horizon


def _resolve_input_hw(cfg: DictConfig) -> tuple[int, int]:
    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    return int(video_size[1]), int(video_size[0])


def _resolve_task_dataset_root(cfg: DictConfig, task_suite_name: str) -> Path:
    dataset_dirs = [Path(str(v)).resolve() for v in cfg.data.train.get("dataset_dirs", [])]
    for dataset_dir in dataset_dirs:
        if task_suite_name in dataset_dir.name:
            return dataset_dir
    raise FileNotFoundError(
        f"Failed to find dataset dir for suite={task_suite_name}. Candidates: {[str(v) for v in dataset_dirs]}"
    )


def _find_task_episode_indices(dataset_root: Path, task_description: str) -> list[int]:
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    if not episodes_path.exists():
        raise FileNotFoundError(f"Missing episodes metadata: {episodes_path}")

    matched: list[int] = []
    with open(episodes_path, "r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            tasks = list(payload.get("tasks", []))
            if len(tasks) > 0 and str(tasks[0]) == str(task_description):
                matched.append(int(payload["episode_index"]))
    if not matched:
        raise ValueError(f"No GT episodes found for task description: {task_description}")
    return matched


def _read_episode_actions(dataset_root: Path, episode_index: int) -> np.ndarray:
    parquet_path = dataset_root / "data" / f"chunk-{episode_index // 1000:03d}" / f"episode_{episode_index:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing episode parquet: {parquet_path}")
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected episode actions with shape [T, D], got {actions.shape} from {parquet_path}")
    return actions


def _dataset_action_to_env_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = converted[..., -1] * 2.0 - 1.0
    converted = invert_gripper_action(converted)
    if binarize_gripper:
        converted[..., -1] = np.sign(converted[..., -1])
    return converted


def _resolve_anchor_steps(
    *,
    args: argparse.Namespace,
    gt_actions: np.ndarray,
    action_horizon: int,
    obs_stride_env_steps: int,
) -> list[int]:
    max_anchor = int(gt_actions.shape[0]) - int(action_horizon)
    if max_anchor < 0:
        raise ValueError(
            f"GT episode length {gt_actions.shape[0]} is shorter than action horizon {action_horizon}."
        )
    if int(obs_stride_env_steps) <= 0:
        raise ValueError(f"obs_stride_env_steps must be positive, got {obs_stride_env_steps}")
    if args.anchor_steps is not None:
        raw_anchors = [int(v.strip()) for v in str(args.anchor_steps).split(",") if v.strip()]
    else:
        num_anchors = max(1, int(args.num_anchors))
        legal_anchors = list(range(0, max_anchor + 1, int(obs_stride_env_steps)))
        if not legal_anchors:
            raise ValueError(
                f"No legal anchors available for max_anchor={max_anchor} and obs_stride_env_steps={obs_stride_env_steps}."
            )
        raw_indices = np.linspace(0, len(legal_anchors) - 1, num=num_anchors, dtype=np.int64).tolist()
        raw_anchors = [int(legal_anchors[int(idx)]) for idx in raw_indices]

    normalized_anchors: list[int] = []
    for raw_anchor in raw_anchors:
        if raw_anchor < 0 or raw_anchor > max_anchor:
            raise ValueError(f"Anchor step {raw_anchor} is out of valid range [0, {max_anchor}].")
        legal_anchor = (int(raw_anchor) // int(obs_stride_env_steps)) * int(obs_stride_env_steps)
        normalized_anchors.append(int(legal_anchor))

    deduped = sorted(set(normalized_anchors))
    for step in deduped:
        if step < 0 or step > max_anchor:
            raise ValueError(f"Anchor step {step} is out of valid range [0, {max_anchor}].")
        if step % int(obs_stride_env_steps) != 0:
            raise ValueError(f"Anchor step {step} must align with obs stride {obs_stride_env_steps}.")
    if not deduped:
        raise ValueError("No valid anchor steps were resolved.")
    return deduped


def _encode_obs(
    obs_dict: dict,
    *,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    return _obs_to_model_input(
        obs_dict,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device=device,
        dtype=dtype,
    )


def _build_chunk_record(*, source: str, trigger_env_step: int, action_chunk: np.ndarray) -> dict[str, Any]:
    return {
        "source": str(source),
        "chunk_index": 0,
        "trigger_env_step": int(trigger_env_step),
        "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
        "chunk_len": int(action_chunk.shape[0]),
        "action_dim": int(action_chunk.shape[1]),
    }


def _resolve_streaming_submit_obs(*, obs_history: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(obs_history) == 0:
        raise ValueError("obs_history must be non-empty when resolving streaming submit obs.")

    history_index = len(obs_history) - 1
    selected_obs = _clone_obs_dict(obs_history[history_index])
    meta = {
        "requested_source": "current",
        "applied_source": "current",
        "history_index": int(history_index),
        "history_size": int(len(obs_history)),
        "effective_history_lag": 0,
    }
    return selected_obs, meta


def _run_sync_anchor_chunk(
    *,
    model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    action_horizon: int,
    input_w: int,
    input_h: int,
    episode_obs_trace: list[dict[str, Any]],
    task_description: str,
    anchor_step: int,
) -> dict[str, Any]:
    obs = _clone_obs_dict(episode_obs_trace[int(anchor_step)])
    image, proprio, _ = _encode_obs(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=str(model.device),
        dtype=model.torch_dtype,
    )
    prompt = DEFAULT_PROMPT.format(task=task_description)
    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": int(action_horizon),
        "proprio": proprio,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))),
        "sigma_shift": (None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))),
        "seed": (None if cfg.get("seed") is None else int(cfg.seed)),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    if "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = int(cfg.data.train.num_frames)
    with torch.no_grad():
        pred = model.infer_action(**infer_kwargs)
    action_chunk = _postprocess_libero_action_chunk(pred["action"], processor=processor, cfg=cfg)
    return {
        "task_description": task_description,
        "chunk_record": _build_chunk_record(
            source="sync",
            trigger_env_step=int(anchor_step),
            action_chunk=np.asarray(action_chunk, dtype=np.float32),
        ),
    }


def _run_streaming_anchor_chunk(
    *,
    task,
    initial_state: Any,
    model: torch.nn.Module,
    action_model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    action_horizon: int,
    input_w: int,
    input_h: int,
    render_gpu_device_id: int,
    gt_actions_env: np.ndarray,
    anchor_step: int,
    args: argparse.Namespace,
    episode_obs_trace: list[dict[str, Any]],
    task_description: str,
) -> dict[str, Any]:
    def _selector(
        phase: str,
        env_step: int,
        current_obs: dict[str, Any],
        obs_history: list[dict[str, Any]],
        formal_obs_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if phase == "bootstrap":
            return clone_obs_dict(current_obs), {
                "requested_source": "current",
                "applied_source": "current",
                "history_index": 0,
                "history_size": 1,
                "effective_history_lag": 0,
            }
        submit_obs, submit_meta = _resolve_streaming_submit_obs(obs_history=obs_history)
        return submit_obs, submit_meta

    native_out = run_native_async_anchor_chunk(
        task=task,
        initial_state=initial_state,
        model=model,
        action_model=action_model,
        processor=processor,
        cfg=cfg,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        render_gpu_device_id=render_gpu_device_id,
        gt_actions_env=gt_actions_env,
        anchor_step=anchor_step,
        submit_selector=_selector,
        seed=(None if cfg.get("seed") is None else int(cfg.seed)),
        submit_anchor_if_aligned=True,
        episode_obs_trace=episode_obs_trace,
        task_description_override=task_description,
    )
    return {
        "task_description": native_out["task_description"],
        "chunk_record": _build_chunk_record(
            source="streaming",
            trigger_env_step=int(anchor_step),
            action_chunk=np.asarray(native_out["action_chunk"], dtype=np.float32),
        ),
        "trigger_obs_index": int(native_out["trigger_obs_index"]),
        "submission_trace": list(native_out["submission_trace"]),
        "runtime_stats": native_out["runtime_stats"],
    }


def _summarize_chunk_metrics(
    *,
    chunk_records: list[dict[str, Any]],
    gt_actions: np.ndarray,
    tail_fraction: float,
    segment: str,
) -> dict[str, Any]:
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError(f"tail_fraction must be in (0, 1], got {tail_fraction}")
    if str(segment) not in {"tail", "head"}:
        raise ValueError(f"segment must be 'tail' or 'head', got {segment}")

    per_chunk: list[dict[str, Any]] = []
    valid_mae: list[float] = []
    valid_rmse: list[float] = []
    valid_l2_step: list[float] = []
    valid_segment_steps: list[int] = []

    for record in chunk_records:
        start = int(record["trigger_env_step"])
        pred_chunk = np.asarray(record["action_chunk"], dtype=np.float32)
        pred_len = int(pred_chunk.shape[0])
        segment_len = max(1, min(int(math.floor(pred_len * float(tail_fraction))), pred_len))
        if str(segment) == "tail":
            segment_start, segment_end = pred_len - segment_len, pred_len
        else:
            segment_start, segment_end = 0, segment_len

        gt_valid = gt_actions[start : start + pred_len]
        valid_len = int(gt_valid.shape[0])
        row = {
            "source": str(record["source"]),
            "chunk_index": int(record["chunk_index"]),
            "trigger_env_step": int(start),
            "pred_len": int(pred_len),
            "valid_len": int(valid_len),
            "segment": str(segment),
            "segment_start": int(segment_start),
            "segment_end": int(segment_end),
            "segment_steps": 0,
            "mae": None,
            "rmse": None,
            "mean_step_l2": None,
            "per_dim_abs_mean": None,
        }
        effective_end = min(valid_len, segment_end)
        if effective_end <= segment_start:
            per_chunk.append(row)
            continue

        pred_valid = pred_chunk[:valid_len]
        diff = pred_valid[segment_start:effective_end] - gt_valid[segment_start:effective_end]
        abs_diff = np.abs(diff)
        mae = float(np.mean(abs_diff))
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        mean_step_l2 = float(np.mean(np.linalg.norm(diff, axis=-1)))
        row.update(
            {
                "segment_steps": int(diff.shape[0]),
                "mae": mae,
                "rmse": rmse,
                "mean_step_l2": mean_step_l2,
                "per_dim_abs_mean": np.mean(abs_diff, axis=0).tolist(),
            }
        )
        per_chunk.append(row)
        valid_mae.append(mae)
        valid_rmse.append(rmse)
        valid_l2_step.append(mean_step_l2)
        valid_segment_steps.append(int(diff.shape[0]))

    return {
        "num_chunks_total": int(len(chunk_records)),
        "num_chunks_with_valid_segment": int(len(valid_mae)),
        "mean_mae": None if not valid_mae else float(np.mean(valid_mae)),
        "mean_rmse": None if not valid_rmse else float(np.mean(valid_rmse)),
        "mean_step_l2": None if not valid_l2_step else float(np.mean(valid_l2_step)),
        "mean_segment_steps": None if not valid_segment_steps else float(np.mean(valid_segment_steps)),
        "per_chunk": per_chunk,
    }


def _build_paired_summary(
    streaming_metrics: dict[str, Any],
    sync_metrics: dict[str, Any],
) -> dict[str, Any]:
    stream_by_step = {
        int(item["trigger_env_step"]): item
        for item in streaming_metrics["per_chunk"]
        if item["mae"] is not None
    }
    sync_by_step = {
        int(item["trigger_env_step"]): item
        for item in sync_metrics["per_chunk"]
        if item["mae"] is not None
    }
    common_steps = sorted(set(stream_by_step.keys()) & set(sync_by_step.keys()))
    rows: list[dict[str, Any]] = []
    for step in common_steps:
        s_item = stream_by_step[step]
        b_item = sync_by_step[step]
        rows.append(
            {
                "anchor_step": int(step),
                "streaming_mae": float(s_item["mae"]),
                "sync_mae": float(b_item["mae"]),
                "streaming_rmse": float(s_item["rmse"]),
                "sync_rmse": float(b_item["rmse"]),
                "streaming_better_mae": bool(float(s_item["mae"]) < float(b_item["mae"])),
                "mae_delta_sync_minus_streaming": float(float(b_item["mae"]) - float(s_item["mae"])),
            }
        )
    return {
        "num_common_anchor_steps": int(len(common_steps)),
        "mean_sync_minus_streaming_mae": (
            None if not rows else float(np.mean([row["mae_delta_sync_minus_streaming"] for row in rows]))
        ),
        "streaming_better_fraction": (
            None if not rows else float(np.mean([1.0 if row["streaming_better_mae"] else 0.0 for row in rows]))
        ),
        "rows": rows,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, cls=NumpyEncoder)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, cls=NumpyEncoder))
            f.write("\n")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
    start_time = time.time()

    if args.seed is not None:
        set_global_seed(int(args.seed), get_worker_init_fn=False)

    streaming_cfg = _compose_cfg(args.streaming_task_config, args.streaming_ckpt, args)
    baseline_cfg = _compose_cfg(args.baseline_task_config, args.baseline_ckpt, args)
    _configure_egl_device(streaming_cfg)
    _configure_egl_device(baseline_cfg)

    model_dtype = _mixed_precision_to_model_dtype(args.mixed_precision)
    base_device = _resolve_eval_device(streaming_cfg) if args.device is None else str(args.device)
    base_device = _normalize_runtime_device(base_device, render_gpu_id=int(args.render_gpu_id))
    streaming_video_device = str(args.async_video_device) if args.async_video_device is not None else base_device
    streaming_action_device = str(args.async_action_device) if args.async_action_device is not None else base_device
    baseline_device = _resolve_eval_device(baseline_cfg) if args.device is None else str(args.device)
    baseline_device = _normalize_runtime_device(baseline_device, render_gpu_id=int(args.render_gpu_id))

    logging.info("Loading streaming video model on %s", streaming_video_device)
    streaming_model = _force_eval_model_dtype(
        _build_eval_model(streaming_cfg, model_dtype=model_dtype, device=streaming_video_device),
        device=streaming_video_device,
        model_dtype=model_dtype,
    )
    logging.info("Loading streaming action model on %s", streaming_action_device)
    streaming_action_model = _force_eval_model_dtype(
        _build_eval_model(streaming_cfg, model_dtype=model_dtype, device=streaming_action_device),
        device=streaming_action_device,
        model_dtype=model_dtype,
    )
    logging.info("Loading baseline model on %s", baseline_device)
    baseline_model = _force_eval_model_dtype(
        _build_eval_model(baseline_cfg, model_dtype=model_dtype, device=baseline_device),
        device=baseline_device,
        model_dtype=model_dtype,
    )

    streaming_processor, streaming_stats_path = _build_processor(streaming_cfg)
    baseline_processor, baseline_stats_path = _build_processor(baseline_cfg)

    action_horizon = _resolve_action_horizon(streaming_cfg)
    if action_horizon != _resolve_action_horizon(baseline_cfg):
        raise ValueError("Streaming and baseline action horizon mismatch.")
    input_w, input_h = _resolve_input_hw(streaming_cfg)
    obs_stride_env_steps = int(streaming_cfg.EVALUATION.get("async_obs_stride_env_steps", 3))

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(int(args.task_id))
    task_description = str(task.language)
    initial_states = list(task_suite.get_task_init_states(int(args.task_id)))
    if not initial_states:
        raise ValueError("No initial states available for the requested task.")
    resolved_trial_idx = int(args.trial_idx) % len(initial_states)
    initial_state = initial_states[resolved_trial_idx]

    dataset_root = _resolve_task_dataset_root(streaming_cfg, args.task_suite_name)
    matched_episode_indices = _find_task_episode_indices(dataset_root, task_description)
    gt_episode_index = (
        int(args.gt_episode_index)
        if args.gt_episode_index is not None
        else int(matched_episode_indices[resolved_trial_idx % len(matched_episode_indices)])
    )
    gt_actions_raw = _read_episode_actions(dataset_root, gt_episode_index)
    gt_actions_env = _dataset_action_to_env_action(
        gt_actions_raw,
        binarize_gripper=bool(args.binarize_gripper or streaming_cfg.EVALUATION.get("binarize_gripper", False)),
    )
    episode_obs_trace, episode_task_description = collect_episode_obs_trace(
        task=task,
        initial_state=initial_state,
        cfg=streaming_cfg,
        gt_actions_env=gt_actions_env,
        render_gpu_device_id=int(args.render_gpu_id),
    )

    anchor_steps = _resolve_anchor_steps(
        args=args,
        gt_actions=gt_actions_env,
        action_horizon=action_horizon,
        obs_stride_env_steps=obs_stride_env_steps,
    )
    logging.info("Evaluating anchors at env steps: %s", anchor_steps)

    streaming_records: list[dict[str, Any]] = []
    sync_records: list[dict[str, Any]] = []
    anchor_details: list[dict[str, Any]] = []
    for anchor_step in anchor_steps:
        print(f"[anchor-debug] compare_chunk anchor={int(anchor_step)} streaming begin", flush=True)
        logging.info("Running matched single-chunk evaluation at anchor=%d", anchor_step)
        stream_out = _run_streaming_anchor_chunk(
            task=task,
            initial_state=initial_state,
            model=streaming_model,
            action_model=streaming_action_model,
            processor=streaming_processor,
            cfg=streaming_cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            render_gpu_device_id=int(args.render_gpu_id),
            gt_actions_env=gt_actions_env,
            anchor_step=int(anchor_step),
            args=args,
            episode_obs_trace=episode_obs_trace,
            task_description=episode_task_description,
        )
        print(f"[anchor-debug] compare_chunk anchor={int(anchor_step)} streaming done", flush=True)
        print(f"[anchor-debug] compare_chunk anchor={int(anchor_step)} sync begin", flush=True)
        sync_out = _run_sync_anchor_chunk(
            model=baseline_model,
            processor=baseline_processor,
            cfg=baseline_cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            episode_obs_trace=episode_obs_trace,
            task_description=episode_task_description,
            anchor_step=int(anchor_step),
        )
        print(f"[anchor-debug] compare_chunk anchor={int(anchor_step)} sync done", flush=True)
        streaming_records.append(stream_out["chunk_record"])
        sync_records.append(sync_out["chunk_record"])
        anchor_details.append(
            {
                "anchor_step": int(anchor_step),
                "streaming_trigger_obs_index": int(stream_out["trigger_obs_index"]),
                "streaming_submission_trace": list(stream_out["submission_trace"]),
                "streaming_runtime_stats": stream_out["runtime_stats"],
                "streaming_chunk_len": int(stream_out["chunk_record"]["chunk_len"]),
                "sync_chunk_len": int(sync_out["chunk_record"]["chunk_len"]),
            }
        )

    streaming_metrics = _summarize_chunk_metrics(
        chunk_records=streaming_records,
        gt_actions=gt_actions_env,
        tail_fraction=float(args.tail_fraction),
        segment=str(args.segment),
    )
    sync_metrics = _summarize_chunk_metrics(
        chunk_records=sync_records,
        gt_actions=gt_actions_env,
        tail_fraction=float(args.tail_fraction),
        segment=str(args.segment),
    )
    paired_summary = _build_paired_summary(streaming_metrics, sync_metrics)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_mode": "matched_anchor_single_chunk",
        "task_suite_name": str(args.task_suite_name),
        "task_id": int(args.task_id),
        "task_description": task_description,
        "trial_idx": int(resolved_trial_idx),
        "gt_episode_index": int(gt_episode_index),
        "gt_alignment_mode": ("explicit_episode_index" if args.gt_episode_index is not None else "task_ordinal"),
        "gt_dataset_root": str(dataset_root),
        "matched_task_episode_count": int(len(matched_episode_indices)),
        "streaming_ckpt": str(args.streaming_ckpt),
        "baseline_ckpt": str(args.baseline_ckpt),
        "streaming_task_config": str(args.streaming_task_config),
        "baseline_task_config": str(args.baseline_task_config),
        "async_video_device": str(streaming_video_device),
        "async_action_device": str(streaming_action_device),
        "streaming_dataset_stats_path": str(streaming_stats_path),
        "baseline_dataset_stats_path": str(baseline_stats_path),
        "action_horizon": int(action_horizon),
        "segment": str(args.segment),
        "tail_fraction": float(args.tail_fraction),
        "async_obs_stride_env_steps": int(obs_stride_env_steps),
        "anchor_steps": [int(v) for v in anchor_steps],
        "num_anchors": int(len(anchor_steps)),
        "streaming_anchor_eval": {
            "metrics": {k: v for k, v in streaming_metrics.items() if k != "per_chunk"},
        },
        "baseline_anchor_eval": {
            "metrics": {k: v for k, v in sync_metrics.items() if k != "per_chunk"},
        },
        "paired_summary": paired_summary,
        "duration_sec": float(time.time() - start_time),
    }

    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "streaming_chunks.jsonl", streaming_metrics["per_chunk"])
    _write_jsonl(output_dir / "sync_chunks.jsonl", sync_metrics["per_chunk"])
    _write_jsonl(output_dir / "paired_common_steps.jsonl", paired_summary["rows"])
    _write_jsonl(output_dir / "anchors.jsonl", anchor_details)

    logging.info("Wrote summary to %s", output_dir / "summary.json")
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
