from __future__ import annotations

import argparse
import copy
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
            "Compare matched-anchor single-chunk tail error to GT between standard streaming "
            "and future-only noise-corrupted streaming."
        )
    )
    parser.add_argument("--streaming-task-config", required=True)
    parser.add_argument("--streaming-ckpt", required=True)
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
    table = pq.read_table(parquet_path, columns=["action"])
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"Expected [T, D] actions, got {actions.shape}")
    return actions


def _dataset_action_to_env_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = converted[..., -1] * 2.0 - 1.0
    converted = invert_gripper_action(converted)
    if binarize_gripper:
        converted[..., -1] = np.sign(converted[..., -1])
    return converted


def _resolve_anchor_steps(*, args: argparse.Namespace, gt_actions: np.ndarray, action_horizon: int) -> list[int]:
    max_anchor = int(gt_actions.shape[0]) - int(action_horizon)
    obs_stride_env_steps = int(args.obs_stride_env_steps)
    if obs_stride_env_steps <= 0:
        raise ValueError(f"obs_stride_env_steps must be positive, got {obs_stride_env_steps}")
    if args.anchor_steps is not None:
        raw_anchors = [int(v.strip()) for v in str(args.anchor_steps).split(",") if v.strip()]
    else:
        legal_anchors = list(range(0, max_anchor + 1, obs_stride_env_steps))
        if not legal_anchors:
            raise ValueError(
                f"No legal anchors available for max_anchor={max_anchor} and obs_stride_env_steps={obs_stride_env_steps}."
            )
        raw_indices = np.linspace(0, len(legal_anchors) - 1, num=max(1, int(args.num_anchors)), dtype=np.int64).tolist()
        raw_anchors = [int(legal_anchors[int(idx)]) for idx in raw_indices]
    normalized = [int((int(v) // obs_stride_env_steps) * obs_stride_env_steps) for v in raw_anchors]
    deduped = sorted(set(normalized))
    for step in deduped:
        if step < 0 or step > max_anchor:
            raise ValueError(f"Anchor step {step} out of range [0, {max_anchor}]")
        if step % obs_stride_env_steps != 0:
            raise ValueError(f"Anchor step {step} must align with obs stride {obs_stride_env_steps}")
    return deduped

def _clone_obs_dict(obs_dict: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(obs_dict)


def _select_wrong_obs(
    *,
    obs_history: list[dict[str, Any]],
    rng: np.random.Generator,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(obs_history) <= 0:
        raise ValueError("obs_history must be non-empty when constructing wrong observation noise.")

    base_obs = _clone_obs_dict(obs_history[-1])
    replaced_keys: list[str] = []
    for image_key in ("agentview_image", "robot0_eye_in_hand_image"):
        if image_key not in base_obs:
            continue
        image = np.asarray(base_obs[image_key])
        noise_image = rng.integers(0, 256, size=image.shape, dtype=np.uint8)
        if image.dtype != np.uint8:
            noise_image = noise_image.astype(image.dtype, copy=False)
        base_obs[image_key] = noise_image
        replaced_keys.append(str(image_key))

    if not replaced_keys:
        raise ValueError("Failed to find visual observation keys to replace with noise.")

    return base_obs, {
        "sample_scope": "noise",
        "history_size": int(len(obs_history)),
        "noise_replaced_keys": replaced_keys,
    }


def _build_chunk_record(*, source: str, trigger_env_step: int, action_chunk: np.ndarray) -> dict[str, Any]:
    return {
        "source": str(source),
        "chunk_index": 0,
        "trigger_env_step": int(trigger_env_step),
        "action_chunk": np.asarray(action_chunk, dtype=np.float32).tolist(),
        "chunk_len": int(action_chunk.shape[0]),
        "action_dim": int(action_chunk.shape[1]),
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
    mode: str,
    seed: int | None,
    episode_obs_trace: list[dict[str, Any]],
    task_description: str,
) -> dict[str, Any]:
    run_seed = None if seed is None else int(seed)
    rng = np.random.default_rng(None if run_seed is None else run_seed + int(anchor_step))

    def _selector(
        phase: str,
        env_step: int,
        current_obs: dict[str, Any],
        obs_history: list[dict[str, Any]],
        formal_obs_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if phase == "bootstrap":
            return _clone_obs_dict(current_obs), {
                "applied_source": "current",
                "sample_scope": "history",
                "history_index": 0,
                "history_size": 1,
            }
        if phase == "scheduled_post_trigger" and str(mode) == "wrong":
            submit_obs, meta = _select_wrong_obs(
                obs_history=obs_history,
                rng=rng,
            )
            return submit_obs, {"applied_source": "noise", **meta}
        history_index = int(len(obs_history) - 1)
        return _clone_obs_dict(current_obs), {
            "applied_source": "current",
            "sample_scope": "history",
            "history_index": history_index,
            "history_size": int(len(obs_history)),
        }

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
        seed=run_seed,
        submit_anchor_if_aligned=True,
        episode_obs_trace=episode_obs_trace,
        task_description_override=task_description,
    )
    return {
        "task_description": native_out["task_description"],
        "chunk_record": _build_chunk_record(
            source=str(mode),
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
        segment_start, segment_end = ((pred_len - segment_len, pred_len) if str(segment) == "tail" else (0, segment_len))
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


def _build_paired_summary(standard_metrics: dict[str, Any], wrong_metrics: dict[str, Any]) -> dict[str, Any]:
    standard_by_step = {int(item["trigger_env_step"]): item for item in standard_metrics["per_chunk"] if item["mae"] is not None}
    wrong_by_step = {int(item["trigger_env_step"]): item for item in wrong_metrics["per_chunk"] if item["mae"] is not None}
    common_steps = sorted(set(standard_by_step.keys()) & set(wrong_by_step.keys()))
    rows: list[dict[str, Any]] = []
    for step in common_steps:
        s_item = standard_by_step[step]
        w_item = wrong_by_step[step]
        rows.append(
            {
                "anchor_step": int(step),
                "standard_mae": float(s_item["mae"]),
                "wrong_obs_mae": float(w_item["mae"]),
                "standard_rmse": float(s_item["rmse"]),
                "wrong_obs_rmse": float(w_item["rmse"]),
                "wrong_obs_worse_mae": bool(float(w_item["mae"]) > float(s_item["mae"])),
                "mae_delta_wrong_minus_standard": float(float(w_item["mae"]) - float(s_item["mae"])),
                "rmse_delta_wrong_minus_standard": float(float(w_item["rmse"]) - float(s_item["rmse"])),
            }
        )
    return {
        "num_common_anchor_steps": int(len(common_steps)),
        "mean_wrong_minus_standard_mae": (
            None if not rows else float(np.mean([row["mae_delta_wrong_minus_standard"] for row in rows]))
        ),
        "wrong_obs_worse_fraction": (
            None if not rows else float(np.mean([1.0 if row["wrong_obs_worse_mae"] else 0.0 for row in rows]))
        ),
        "rows": rows,
    }


def _summarize_layer_source_stats(layer_source_stats: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not layer_source_stats:
        return []
    rows: list[dict[str, Any]] = []
    for step in list(layer_source_stats.get("per_step", [])):
        rows.append(
            {
                "denoise_step": int(step["denoise_step"]),
                "mode_counts": dict(step.get("mode_counts", {})),
                "frontier_stats_by_mode": dict(step.get("frontier_stats_by_mode", {})),
                "latest_offset_counts": dict(step.get("latest_offset_counts", {})),
                "older_offset_counts": dict(step.get("older_offset_counts", {})),
            }
        )
    return rows


def _summarize_runtime_stats(runtime_stats: dict[str, Any] | None) -> dict[str, Any]:
    if not runtime_stats:
        return {}
    timing_ms = dict(runtime_stats.get("timing_ms", {}))
    return {
        "submitted_obs": int(runtime_stats.get("submitted_obs", 0)),
        "submitted_jobs": int(runtime_stats.get("submitted_jobs", 0)),
        "completed_jobs": int(runtime_stats.get("completed_jobs", 0)),
        "action_job_avg_ms": timing_ms.get("action_job", {}).get("avg_ms"),
        "action_job_wall_avg_ms": timing_ms.get("action_job_wall", {}).get("avg_ms"),
        "layer_source_per_step": _summarize_layer_source_stats(runtime_stats.get("layer_source_stats")),
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
    _configure_egl_device(streaming_cfg)
    model_dtype = _mixed_precision_to_model_dtype(args.mixed_precision)
    args.obs_stride_env_steps = int(streaming_cfg.EVALUATION.get("async_obs_stride_env_steps", 3))
    base_device = _resolve_eval_device(streaming_cfg) if args.device is None else str(args.device)
    base_device = _normalize_runtime_device(base_device, render_gpu_id=int(args.render_gpu_id))
    video_device = str(args.async_video_device) if args.async_video_device is not None else base_device
    action_device = str(args.async_action_device) if args.async_action_device is not None else base_device

    streaming_model = _force_eval_model_dtype(
        _build_eval_model(streaming_cfg, model_dtype=model_dtype, device=video_device),
        device=video_device,
        model_dtype=model_dtype,
    )
    streaming_action_model = (
        streaming_model
        if action_device == video_device
        else _force_eval_model_dtype(
            _build_eval_model(streaming_cfg, model_dtype=model_dtype, device=action_device),
            device=action_device,
            model_dtype=model_dtype,
        )
    )
    processor, dataset_stats_path = _build_processor(streaming_cfg)
    action_horizon = _resolve_action_horizon(streaming_cfg)
    input_w, input_h = _resolve_input_hw(streaming_cfg)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(int(args.task_id))
    task_description = str(task.language)
    initial_states = list(task_suite.get_task_init_states(int(args.task_id)))
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
    anchor_steps = _resolve_anchor_steps(args=args, gt_actions=gt_actions_env, action_horizon=action_horizon)
    standard_records: list[dict[str, Any]] = []
    wrong_records: list[dict[str, Any]] = []
    anchor_details: list[dict[str, Any]] = []
    for anchor_step in anchor_steps:
        print(f"[anchor-debug] compare_future_wrong anchor={int(anchor_step)} standard begin", flush=True)
        standard_out = _run_streaming_anchor_chunk(
            task=task,
            initial_state=initial_state,
            model=streaming_model,
            action_model=streaming_action_model,
            processor=processor,
            cfg=streaming_cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            render_gpu_device_id=int(args.render_gpu_id),
            gt_actions_env=gt_actions_env,
            anchor_step=int(anchor_step),
            mode="standard",
            seed=None if args.seed is None else int(args.seed),
            episode_obs_trace=episode_obs_trace,
            task_description=episode_task_description,
        )
        print(f"[anchor-debug] compare_future_wrong anchor={int(anchor_step)} standard done", flush=True)
        print(f"[anchor-debug] compare_future_wrong anchor={int(anchor_step)} wrong begin", flush=True)
        wrong_out = _run_streaming_anchor_chunk(
            task=task,
            initial_state=initial_state,
            model=streaming_model,
            action_model=streaming_action_model,
            processor=processor,
            cfg=streaming_cfg,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            render_gpu_device_id=int(args.render_gpu_id),
            gt_actions_env=gt_actions_env,
            anchor_step=int(anchor_step),
            mode="wrong",
            seed=None if args.seed is None else int(args.seed),
            episode_obs_trace=episode_obs_trace,
            task_description=episode_task_description,
        )
        print(f"[anchor-debug] compare_future_wrong anchor={int(anchor_step)} wrong done", flush=True)
        standard_records.append(standard_out["chunk_record"])
        wrong_records.append(wrong_out["chunk_record"])
        anchor_details.append(
            {
                "anchor_step": int(anchor_step),
                "standard_trigger_obs_index": int(standard_out["trigger_obs_index"]),
                "wrong_trigger_obs_index": int(wrong_out["trigger_obs_index"]),
                "standard_chunk_len": int(standard_out["chunk_record"]["chunk_len"]),
                "wrong_chunk_len": int(wrong_out["chunk_record"]["chunk_len"]),
                "standard_runtime": _summarize_runtime_stats(standard_out["runtime_stats"]),
                "wrong_runtime": _summarize_runtime_stats(wrong_out["runtime_stats"]),
            }
        )

    standard_metrics = _summarize_chunk_metrics(
        chunk_records=standard_records,
        gt_actions=gt_actions_env,
        tail_fraction=float(args.tail_fraction),
        segment=str(args.segment),
    )
    wrong_metrics = _summarize_chunk_metrics(
        chunk_records=wrong_records,
        gt_actions=gt_actions_env,
        tail_fraction=float(args.tail_fraction),
        segment=str(args.segment),
    )
    paired_summary = _build_paired_summary(standard_metrics, wrong_metrics)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "evaluation_mode": "matched_anchor_standard_vs_future_wrong_single_chunk",
        "task_suite_name": str(args.task_suite_name),
        "task_id": int(args.task_id),
        "task_description": task_description,
        "trial_idx": int(resolved_trial_idx),
        "gt_episode_index": int(gt_episode_index),
        "gt_alignment_mode": ("explicit_episode_index" if args.gt_episode_index is not None else "task_ordinal"),
        "gt_dataset_root": str(dataset_root),
        "matched_task_episode_count": int(len(matched_episode_indices)),
        "streaming_ckpt": str(args.streaming_ckpt),
        "streaming_task_config": str(args.streaming_task_config),
        "async_video_device": str(video_device),
        "async_action_device": str(action_device),
        "streaming_dataset_stats_path": str(dataset_stats_path),
        "action_horizon": int(action_horizon),
        "segment": str(args.segment),
        "tail_fraction": float(args.tail_fraction),
        "async_obs_stride_env_steps": int(args.obs_stride_env_steps),
        "anchor_steps": [int(v) for v in anchor_steps],
        "num_anchors": int(len(anchor_steps)),
        "standard_anchor_eval": {"metrics": {k: v for k, v in standard_metrics.items() if k != "per_chunk"}},
        "wrong_obs_anchor_eval": {"metrics": {k: v for k, v in wrong_metrics.items() if k != "per_chunk"}},
        "paired_summary": paired_summary,
        "duration_sec": float(time.time() - start_time),
    }
    _write_json(output_dir / "summary.json", summary)
    _write_jsonl(output_dir / "standard_chunks.jsonl", standard_metrics["per_chunk"])
    _write_jsonl(output_dir / "wrong_obs_chunks.jsonl", wrong_metrics["per_chunk"])
    _write_jsonl(output_dir / "paired_common_steps.jsonl", paired_summary["rows"])
    _write_json(output_dir / "anchors.json", {"anchors": anchor_details})
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
