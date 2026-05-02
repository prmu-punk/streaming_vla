from __future__ import annotations

import argparse
import json
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

from experiments.libero.build_xt_replay import (  # noqa: E402
    _batched,
    _layer_keys_from_schedule,
    _load_schedule_pool,
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from experiments.libero.libero_utils import invert_gripper_action  # noqa: E402
from fastwam.models.wan22.streaming_cache import CacheSnapshot  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> float:
    pred_f = pred.detach().float()
    target_f = target.detach().float()
    err = (pred_f - target_f).pow(2)
    if action_is_pad is None:
        return float(err.mean().item())
    pad = action_is_pad.detach().to(device=err.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    valid = (~pad.bool()).to(dtype=err.dtype)
    while valid.ndim < err.ndim:
        valid = valid.unsqueeze(-1)
    denom = valid.sum() * err.shape[-1]
    if float(denom.item()) <= 0.0:
        return float(err.mean().item())
    return float((err * valid).sum().div(denom).item())


def _masked_mse_per_position(
    pred: torch.Tensor,
    target: torch.Tensor,
    action_is_pad: torch.Tensor | None = None,
) -> list[float | None]:
    pred_f = pred.detach().float()
    target_f = target.detach().float()
    err = (pred_f - target_f).pow(2).mean(dim=-1)  # [B, T]
    if action_is_pad is None:
        return [float(v) for v in err.mean(dim=0).tolist()]
    pad = action_is_pad.detach().to(device=err.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    valid = (~pad.bool()).to(dtype=err.dtype)
    per_pos: list[float | None] = []
    for pos in range(int(err.shape[1])):
        denom = float(valid[:, pos].sum().item())
        if denom <= 0.0:
            per_pos.append(None)
        else:
            per_pos.append(float((err[:, pos] * valid[:, pos]).sum().div(valid[:, pos].sum()).item()))
    return per_pos


def _array_mse(pred: np.ndarray, target: np.ndarray) -> float:
    pred_f = np.asarray(pred, dtype=np.float32)
    target_f = np.asarray(target, dtype=np.float32)
    return float(np.mean(np.square(pred_f - target_f)))


def _normalized_action_to_env_action(
    action: torch.Tensor,
    *,
    processor,
    binarize_gripper: bool,
) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected normalized action [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged action key in processor.shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    action_cpu = action.detach().to(dtype=torch.float32, device="cpu")
    env_action = normalizer.backward(action_cpu).numpy()[0]
    env_action[..., -1] = env_action[..., -1] * 2.0 - 1.0
    env_action = invert_gripper_action(env_action)
    if bool(binarize_gripper):
        env_action[..., -1] = np.sign(env_action[..., -1])
    return np.asarray(env_action, dtype=np.float32)


def _source_summary(layer_keys: list[str]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for key in layer_keys:
        counts[key] = counts.get(key, 0) + 1
    total = max(len(layer_keys), 1)
    return {
        "counts": counts,
        "ratios": {key: float(value) / float(total) for key, value in sorted(counts.items())},
    }


def _summarize_schedule_pattern(per_step: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    mixed_steps = 0
    frontier_values = []
    for step in per_step:
        frontier_values.append(int(step["frontier"]))
        step_counts = dict(step["source_summary"]["counts"])
        if len(step_counts) > 1:
            mixed_steps += 1
        for key, value in step_counts.items():
            counts[str(key)] = counts.get(str(key), 0) + int(value)
    total = max(sum(counts.values()), 1)
    return {
        "mixed_steps": int(mixed_steps),
        "source_counts": counts,
        "source_ratios": {key: float(value) / float(total) for key, value in sorted(counts.items())},
        "frontier_mean": (
            None if len(frontier_values) == 0 else float(sum(frontier_values) / float(len(frontier_values)))
        ),
        "frontier_min": None if len(frontier_values) == 0 else int(min(frontier_values)),
        "frontier_max": None if len(frontier_values) == 0 else int(max(frontier_values)),
    }


def _mean_per_position(rows: list[dict[str, Any]], key: str) -> list[float | None] | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    length = len(values[0])
    out: list[float | None] = []
    for idx in range(length):
        bucket = [float(v[idx]) for v in values if v[idx] is not None]
        out.append(None if len(bucket) == 0 else float(sum(bucket) / float(len(bucket))))
    return out


@torch.no_grad()
def _run_sync_baseline_for_loaded_sample(
    model,
    batch: dict[str, Any],
    *,
    seed: int,
    rand_device: str,
    infer_steps: int,
    processor=None,
    binarize_gripper: bool = False,
) -> dict[str, Any]:
    target_action = batch["target_action"]
    action_is_pad = batch.get("action_is_pad", None)
    infer_out = model.infer_action(
        prompt=None,
        input_image=batch["obs_cur"],
        action_horizon=int(target_action.shape[1]),
        proprio=batch.get("proprio_t", None),
        context=batch["context"],
        context_mask=batch["context_mask"],
        num_inference_steps=int(infer_steps),
        seed=int(seed),
        rand_device=str(rand_device),
        tiled=False,
    )
    pred_action = infer_out["action"].detach().float()
    target_action_eval = target_action.detach().to(device=pred_action.device, dtype=torch.float32)
    action_is_pad_eval = (
        None
        if action_is_pad is None
        else action_is_pad.detach().to(device=pred_action.device)
    )
    result = {
        "final_mse": _masked_mse(pred_action, target_action_eval, action_is_pad_eval),
        "final_mse_per_position": _masked_mse_per_position(pred_action, target_action_eval, action_is_pad_eval),
    }
    if processor is not None:
        pred_env_action = _normalized_action_to_env_action(
            pred_action,
            processor=processor,
            binarize_gripper=bool(binarize_gripper),
        )
        target_env_action = _normalized_action_to_env_action(
            target_action_eval,
            processor=processor,
            binarize_gripper=bool(binarize_gripper),
        )
        env_mse = _array_mse(pred_env_action, target_env_action)
        result["final_env_action_mse"] = float(env_mse)
        result["final_env_action_mse_per_position"] = (
            np.mean(np.square(pred_env_action - target_env_action), axis=-1).astype(float).tolist()
        )
    return result


@torch.no_grad()
def _run_schedule_for_loaded_sample(
    model,
    sample: dict[str, Any],
    batch: dict[str, Any],
    *,
    schedule: dict[str, Any],
    schedule_index: int,
    seed: int,
    rand_device: str,
    infer_steps: int,
    include_per_step: bool = True,
    processor=None,
    binarize_gripper: bool = False,
) -> dict[str, Any]:
    target_action = batch["target_action"]
    action_is_pad = batch.get("action_is_pad", None)
    action_horizon = int(target_action.shape[1])
    resolved_context, resolved_context_mask = model._resolve_streaming_condition_inputs(
        prompt=None,
        context=batch["context"],
        context_mask=batch["context_mask"],
        proprio=batch["proprio_t"],
    )
    batch = dict(batch)
    batch["context"] = resolved_context
    batch["context_mask"] = resolved_context_mask
    job = model.start_action_job(
        action_horizon=action_horizon,
        context=batch["context"],
        context_mask=batch["context_mask"],
        trigger_obs_index=0,
        num_inference_steps=infer_steps,
        seed=int(seed),
        rand_device=str(rand_device),
    )
    caches = model._build_selected_video_cache_payload(
        batch,
        required_cache_keys=["prev", "cur", "next", "next2"],
    )

    steps = list(schedule["steps"])
    profile_trigger_obs_index = int(schedule.get("trigger_obs_index", 0))
    per_step: list[dict[str, Any]] = []
    initial_mse = _masked_mse(job.latents_action, target_action, action_is_pad)

    while not job.done:
        step_idx = int(job.current_step_idx)
        if step_idx >= len(steps):
            raise ValueError(
                f"Schedule {schedule_index} has only {len(steps)} steps, "
                f"but inference expects {infer_steps} steps."
            )
        step = steps[step_idx]
        if int(step["denoise_step"]) != step_idx:
            raise ValueError(
                f"Schedule {schedule_index} denoise step mismatch: expected {step_idx}, "
                f"got {step['denoise_step']}."
            )
        layer_keys, source_offsets = _layer_keys_from_schedule(
            model,
            step=step,
            trigger_obs_index=profile_trigger_obs_index,
        )
        if len(layer_keys) != int(model.mot.num_layers):
            raise ValueError(
                f"Schedule {schedule_index} step {step_idx} has {len(layer_keys)} layers, "
                f"expected {model.mot.num_layers}."
            )
        cache_layers = model._compose_replay_layer_cache(
            caches=caches,
            layer_cache_keys=[",".join(layer_keys)],
        )
        snapshot = CacheSnapshot(
            version=step_idx,
            obs_timestamp_ms=0.0,
            frontier=int(step.get("frontier", model.mot.num_layers)),
            video_seq_len=int(caches["video_seq_len"]),
            tokens_per_frame=int(caches["tokens_per_frame"]),
            cache_layers=cache_layers,
            context=batch["context"],
            context_mask=batch["context_mask"],
            obs_index=int(max(source_offsets)),
            layer_version_ids=[step_idx] * int(model.mot.num_layers),
            layer_obs_indices=source_offsets,
            layer_obs_timestamps_ms=[0.0] * int(model.mot.num_layers),
            layer_ready_events=[None] * int(model.mot.num_layers),
        )
        mse_before = _masked_mse(job.latents_action, target_action, action_is_pad)
        model.step_action_job(job, snapshot=snapshot)
        mse_after = _masked_mse(job.latents_action, target_action, action_is_pad)
        per_step.append(
            {
                "denoise_step": step_idx,
                "timestep": float(job.timesteps[step_idx].detach().float().item()),
                "delta": float(job.deltas[step_idx].detach().float().item()),
                "mode": str(step.get("mode", "")),
                "frontier": int(step.get("frontier", model.mot.num_layers)),
                "mse_before": mse_before,
                "mse_after": mse_after,
                "source_offsets_min": int(min(source_offsets)),
                "source_offsets_max": int(max(source_offsets)),
                "source_summary": _source_summary(layer_keys),
            }
        )

    final_action = job.latents_action.detach().float()
    target_action_f = target_action.detach().float()
    final_mse = _masked_mse(final_action, target_action_f, action_is_pad)
    result = {
        "schedule_index": int(schedule_index),
        "episode_index": int(sample.get("episode_idx", -1)),
        "env_step": int(sample.get("raw_action_start", -1)),
        "seed": int(seed),
        "rand_device": str(rand_device),
        "num_inference_steps": int(infer_steps),
        "action_shape": list(final_action.shape),
        "initial_mse": float(initial_mse),
        "final_mse": float(final_mse),
        "final_mse_per_position": _masked_mse_per_position(final_action, target_action_f, action_is_pad),
        "final_rmse": float(final_mse ** 0.5),
        "final_unmasked_mse": float((final_action - target_action_f).pow(2).mean().item()),
        "schedule_pattern": _summarize_schedule_pattern(per_step),
    }
    if processor is not None:
        pred_env_action = _normalized_action_to_env_action(
            final_action,
            processor=processor,
            binarize_gripper=bool(binarize_gripper),
        )
        target_env_action = _normalized_action_to_env_action(
            target_action_f,
            processor=processor,
            binarize_gripper=bool(binarize_gripper),
        )
        env_mse = _array_mse(pred_env_action, target_env_action)
        result.update(
            {
                "final_env_action_mse": float(env_mse),
                "final_env_action_mse_per_position": np.mean(
                    np.square(pred_env_action - target_env_action),
                    axis=-1,
                ).astype(float).tolist(),
                "final_env_action_rmse": float(env_mse ** 0.5),
                "final_env_action_per_dim_mse": np.mean(
                    np.square(pred_env_action - target_env_action),
                    axis=0,
                ).astype(float).tolist(),
            }
        )
    if include_per_step:
        result["per_step"] = per_step
    return result


@torch.no_grad()
def run_one_schedule_denoise_mse(
    cfg,
    *,
    ckpt: str,
    baseline_cfg=None,
    baseline_ckpt: str | None = None,
    schedule_path: str | Path,
    schedule_index: int,
    dataset_index: int,
    seed: int,
    device: str | None,
    rand_device: str,
    num_inference_steps: int | None,
) -> dict[str, Any]:
    set_global_seed(max(int(seed), 1))
    schedules = _load_schedule_pool(schedule_path)
    if not 0 <= int(schedule_index) < len(schedules):
        raise IndexError(f"schedule_index={schedule_index} out of range [0, {len(schedules) - 1}]")
    schedule = schedules[int(schedule_index)]

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    if not 0 <= int(dataset_index) < len(dataset):
        raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset) - 1}]")
    sample = dataset[int(dataset_index)]

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    model.eval()
    baseline_model = None
    baseline_weight_ckpt = None
    if baseline_ckpt is not None:
        baseline_cfg = cfg if baseline_cfg is None else baseline_cfg
        baseline_model = instantiate(baseline_cfg.model, model_dtype=model_dtype, device=model_device)
        baseline_weight_ckpt = resolve_weight_checkpoint(baseline_ckpt)
        baseline_model.load_checkpoint(str(baseline_weight_ckpt))
        baseline_model.eval()

    batch = model._extract_streaming_episode_batch(_batched(sample))
    processor = getattr(getattr(dataset, "lerobot_dataset", None), "processor", None)
    binarize_gripper = bool(cfg.get("EVALUATION", {}).get("binarize_gripper", False))
    infer_steps = (
        int(num_inference_steps)
        if num_inference_steps is not None
        else int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    )
    result = _run_schedule_for_loaded_sample(
        model,
        sample,
        batch,
        schedule=schedule,
        schedule_index=int(schedule_index),
        seed=int(seed),
        rand_device=str(rand_device),
        infer_steps=int(infer_steps),
        include_per_step=True,
        processor=processor,
        binarize_gripper=binarize_gripper,
    )
    result.update(
        {
            "ckpt": str(weight_ckpt),
            "schedule_path": str(Path(schedule_path).resolve()),
            "dataset_index": int(dataset_index),
            "device": model_device,
            "num_schedules": int(len(schedules)),
        }
    )
    if baseline_model is not None and baseline_weight_ckpt is not None:
        result["sync_baseline"] = _run_sync_baseline_for_loaded_sample(
            baseline_model,
            batch,
            seed=int(seed),
            rand_device=str(rand_device),
            infer_steps=int(infer_steps),
            processor=processor,
            binarize_gripper=binarize_gripper,
        )
        result["sync_baseline"]["ckpt"] = str(baseline_weight_ckpt)
    del model, baseline_model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def scan_schedule_denoise_mse(
    cfg,
    *,
    ckpt: str,
    baseline_cfg=None,
    baseline_ckpt: str | None = None,
    schedule_path: str | Path,
    dataset_index: int,
    seed: int,
    device: str | None,
    rand_device: str,
    num_inference_steps: int | None,
    scan_start: int,
    scan_limit: int | None,
    scan_stride: int,
    top_k: int,
) -> dict[str, Any]:
    set_global_seed(max(int(seed), 1))
    schedules = _load_schedule_pool(schedule_path)
    start = max(int(scan_start), 0)
    stride = max(int(scan_stride), 1)
    indices = list(range(start, len(schedules), stride))
    if scan_limit is not None:
        indices = indices[: max(int(scan_limit), 0)]
    if len(indices) == 0:
        raise ValueError("No schedules selected for scan.")

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    if not 0 <= int(dataset_index) < len(dataset):
        raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset) - 1}]")
    sample = dataset[int(dataset_index)]

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    model.eval()
    baseline_model = None
    baseline_weight_ckpt = None
    if baseline_ckpt is not None:
        baseline_cfg = cfg if baseline_cfg is None else baseline_cfg
        baseline_model = instantiate(baseline_cfg.model, model_dtype=model_dtype, device=model_device)
        baseline_weight_ckpt = resolve_weight_checkpoint(baseline_ckpt)
        baseline_model.load_checkpoint(str(baseline_weight_ckpt))
        baseline_model.eval()
    batch = model._extract_streaming_episode_batch(_batched(sample))
    processor = getattr(getattr(dataset, "lerobot_dataset", None), "processor", None)
    binarize_gripper = bool(cfg.get("EVALUATION", {}).get("binarize_gripper", False))
    infer_steps = (
        int(num_inference_steps)
        if num_inference_steps is not None
        else int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    )

    rows: list[dict[str, Any]] = []
    for schedule_index in indices:
        row = _run_schedule_for_loaded_sample(
            model,
            sample,
            batch,
            schedule=schedules[int(schedule_index)],
            schedule_index=int(schedule_index),
            seed=int(seed),
            rand_device=str(rand_device),
            infer_steps=int(infer_steps),
            include_per_step=False,
            processor=processor,
            binarize_gripper=binarize_gripper,
        )
        if baseline_model is not None:
            row["sync_baseline"] = _run_sync_baseline_for_loaded_sample(
                baseline_model,
                batch,
                seed=int(seed),
                rand_device=str(rand_device),
                infer_steps=int(infer_steps),
                processor=processor,
                binarize_gripper=binarize_gripper,
            )
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda item: float(item["final_mse"]), reverse=True)
    final_mses = [float(row["final_mse"]) for row in rows]
    final_env_mses = [
        float(row["final_env_action_mse"])
        for row in rows
        if row.get("final_env_action_mse", None) is not None
    ]
    mixed_steps = [int(row["schedule_pattern"]["mixed_steps"]) for row in rows]
    top = rows_sorted[: max(int(top_k), 0)]
    result = {
        "ckpt": str(weight_ckpt),
        "schedule_path": str(Path(schedule_path).resolve()),
        "dataset_index": int(dataset_index),
        "episode_index": int(sample.get("episode_idx", -1)),
        "env_step": int(sample.get("raw_action_start", -1)),
        "seed": int(seed),
        "device": model_device,
        "rand_device": str(rand_device),
        "num_schedules_total": int(len(schedules)),
        "num_schedules_scanned": int(len(rows)),
        "scan_start": int(start),
        "scan_stride": int(stride),
        "num_inference_steps": int(infer_steps),
        "summary": {
            "final_mse_mean": float(sum(final_mses) / float(len(final_mses))),
            "final_mse_min": float(min(final_mses)),
            "final_mse_max": float(max(final_mses)),
            "final_mse_per_position_mean": _mean_per_position(rows, "final_mse_per_position"),
            "final_env_action_mse_mean": _mean(final_env_mses),
            "final_env_action_mse_min": None if not final_env_mses else float(min(final_env_mses)),
            "final_env_action_mse_max": None if not final_env_mses else float(max(final_env_mses)),
            "mixed_steps_mean": float(sum(mixed_steps) / float(len(mixed_steps))),
        },
        "worst": top,
        "best": sorted(rows, key=lambda item: float(item["final_mse"]))[: max(int(top_k), 0)],
    }
    if baseline_model is not None and baseline_weight_ckpt is not None:
        baseline_rows = [row["sync_baseline"] for row in rows]
        baseline_mses = [float(row["final_mse"]) for row in baseline_rows]
        result["sync_baseline"] = {
            "ckpt": str(baseline_weight_ckpt),
            "summary": {
                "final_mse_mean": float(sum(baseline_mses) / float(len(baseline_mses))),
                "final_mse_min": float(min(baseline_mses)),
                "final_mse_max": float(max(baseline_mses)),
                "final_mse_per_position_mean": _mean_per_position(baseline_rows, "final_mse_per_position"),
                "final_env_action_mse_mean": _mean(
                    [float(row["final_env_action_mse"]) for row in baseline_rows if row.get("final_env_action_mse") is not None]
                ),
                "final_env_action_mse_per_position_mean": _mean_per_position(
                    baseline_rows, "final_env_action_mse_per_position"
                ),
            },
        }
    del model, baseline_model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


@torch.no_grad()
def scan_random_schedule_sample_pairs(
    cfg,
    *,
    ckpt: str,
    baseline_cfg=None,
    baseline_ckpt: str | None = None,
    schedule_path: str | Path,
    seed: int,
    device: str | None,
    rand_device: str,
    num_inference_steps: int | None,
    num_pairs: int,
    top_k: int,
) -> dict[str, Any]:
    set_global_seed(max(int(seed), 1))
    schedules = _load_schedule_pool(schedule_path)
    if len(schedules) == 0:
        raise ValueError(f"No schedules available: {schedule_path}")

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    if len(dataset) <= 0:
        raise ValueError("Dataset is empty.")

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    model.eval()
    baseline_model = None
    baseline_weight_ckpt = None
    if baseline_ckpt is not None:
        baseline_cfg = cfg if baseline_cfg is None else baseline_cfg
        baseline_model = instantiate(baseline_cfg.model, model_dtype=model_dtype, device=model_device)
        baseline_weight_ckpt = resolve_weight_checkpoint(baseline_ckpt)
        baseline_model.load_checkpoint(str(baseline_weight_ckpt))
        baseline_model.eval()
    processor = getattr(getattr(dataset, "lerobot_dataset", None), "processor", None)
    binarize_gripper = bool(cfg.get("EVALUATION", {}).get("binarize_gripper", False))
    infer_steps = (
        int(num_inference_steps)
        if num_inference_steps is not None
        else int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    )

    pair_count = max(int(num_pairs), 1)
    generator = torch.Generator(device="cpu").manual_seed(max(int(seed), 1) + 29021)
    rows: list[dict[str, Any]] = []
    for pair_idx in range(pair_count):
        dataset_index = int(torch.randint(low=0, high=len(dataset), size=(1,), generator=generator).item())
        schedule_index = int(torch.randint(low=0, high=len(schedules), size=(1,), generator=generator).item())
        sample = dataset[int(dataset_index)]
        batch = model._extract_streaming_episode_batch(_batched(sample))
        row = _run_schedule_for_loaded_sample(
            model,
            sample,
            batch,
            schedule=schedules[int(schedule_index)],
            schedule_index=int(schedule_index),
            seed=int(seed) + int(pair_idx) * 1009 + int(dataset_index),
            rand_device=str(rand_device),
            infer_steps=int(infer_steps),
            include_per_step=False,
            processor=processor,
            binarize_gripper=binarize_gripper,
        )
        if baseline_model is not None:
            row["sync_baseline"] = _run_sync_baseline_for_loaded_sample(
                baseline_model,
                batch,
                seed=int(seed) + int(pair_idx) * 1009 + int(dataset_index),
                rand_device=str(rand_device),
                infer_steps=int(infer_steps),
                processor=processor,
                binarize_gripper=binarize_gripper,
            )
        row.update(
            {
                "pair_index": int(pair_idx),
                "dataset_index": int(dataset_index),
                "trigger_obs_idx": int(sample.get("trigger_obs_idx", -1)),
                "raw_action_start": int(sample.get("raw_action_start", -1)),
            }
        )
        rows.append(row)

    final_mses = [float(row["final_mse"]) for row in rows]
    final_env_mses = [
        float(row["final_env_action_mse"])
        for row in rows
        if row.get("final_env_action_mse", None) is not None
    ]
    mixed_steps = [float(row["schedule_pattern"]["mixed_steps"]) for row in rows]
    result = {
        "ckpt": str(weight_ckpt),
        "schedule_path": str(Path(schedule_path).resolve()),
        "seed": int(seed),
        "device": model_device,
        "rand_device": str(rand_device),
        "num_schedules_total": int(len(schedules)),
        "num_dataset_samples_total": int(len(dataset)),
        "num_pairs": int(len(rows)),
        "num_inference_steps": int(infer_steps),
        "pair_mode": "random_schedule_random_sample",
        "summary": {
            "final_mse_mean": _mean(final_mses),
            "final_mse_min": float(min(final_mses)),
            "final_mse_max": float(max(final_mses)),
            "final_mse_per_position_mean": _mean_per_position(rows, "final_mse_per_position"),
            "final_env_action_mse_mean": _mean(final_env_mses),
            "final_env_action_mse_min": None if not final_env_mses else float(min(final_env_mses)),
            "final_env_action_mse_max": None if not final_env_mses else float(max(final_env_mses)),
            "mixed_steps_mean": _mean(mixed_steps),
        },
        "worst": sorted(rows, key=lambda item: float(item["final_mse"]), reverse=True)[: max(int(top_k), 0)],
        "best": sorted(rows, key=lambda item: float(item["final_mse"]))[: max(int(top_k), 0)],
    }
    if baseline_model is not None and baseline_weight_ckpt is not None:
        baseline_rows = [row["sync_baseline"] for row in rows]
        baseline_mses = [float(row["final_mse"]) for row in baseline_rows]
        result["sync_baseline"] = {
            "ckpt": str(baseline_weight_ckpt),
            "summary": {
                "final_mse_mean": _mean(baseline_mses),
                "final_mse_min": float(min(baseline_mses)),
                "final_mse_max": float(max(baseline_mses)),
                "final_mse_per_position_mean": _mean_per_position(baseline_rows, "final_mse_per_position"),
                "final_env_action_mse_mean": _mean(
                    [float(row["final_env_action_mse"]) for row in baseline_rows if row.get("final_env_action_mse") is not None]
                ),
                "final_env_action_mse_per_position_mean": _mean_per_position(
                    baseline_rows, "final_env_action_mse_per_position"
                ),
            },
        }
    del model, baseline_model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _mean(values: list[float]) -> float | None:
    if len(values) == 0:
        return None
    return float(sum(values) / float(len(values)))


def _episode_dataset_indices(dataset, episode_index: int) -> list[int]:
    if not hasattr(dataset, "sample_index"):
        raise ValueError("Episode scan requires a dataset with `sample_index` metadata.")
    indices = []
    for dataset_index, row in enumerate(list(dataset.sample_index)):
        sample_episode_index = int(row[0])
        if sample_episode_index == int(episode_index):
            indices.append(int(dataset_index))
    if len(indices) == 0:
        raise ValueError(f"No streaming chunks found for episode_index={episode_index}.")
    return indices


def _select_episode_schedule_index(
    *,
    schedules: list[dict[str, Any]],
    env_step_to_schedule_indices: dict[int, list[int]],
    schedule_mode: str,
    base_schedule_index: int,
    chunk_ordinal: int,
    raw_action_start: int,
    generator: torch.Generator,
) -> tuple[int, bool]:
    mode = str(schedule_mode).strip().lower()
    if mode == "fixed":
        return int(base_schedule_index) % len(schedules), False
    if mode == "cycle":
        return (int(base_schedule_index) + int(chunk_ordinal)) % len(schedules), False
    if mode == "random":
        idx = int(torch.randint(low=0, high=len(schedules), size=(1,), generator=generator).item())
        return idx, False
    if mode == "env_step":
        matches = env_step_to_schedule_indices.get(int(raw_action_start), [])
        if matches:
            return int(matches[int(chunk_ordinal) % len(matches)]), False
        return (int(base_schedule_index) + int(chunk_ordinal)) % len(schedules), True
    raise ValueError(f"Unsupported episode schedule mode: {schedule_mode}")


@torch.no_grad()
def scan_episode_denoise_mse(
    cfg,
    *,
    ckpt: str,
    baseline_cfg=None,
    baseline_ckpt: str | None = None,
    schedule_path: str | Path,
    episode_index: int,
    seed: int,
    device: str | None,
    rand_device: str,
    num_inference_steps: int | None,
    schedule_index: int,
    episode_schedule_mode: str,
    episode_chunk_start: int,
    episode_chunk_limit: int | None,
    episode_chunk_stride: int,
    top_k: int,
) -> dict[str, Any]:
    set_global_seed(max(int(seed), 1))
    schedules = _load_schedule_pool(schedule_path)
    if len(schedules) == 0:
        raise ValueError(f"No schedules available: {schedule_path}")
    env_step_to_schedule_indices: dict[int, list[int]] = {}
    for idx, schedule in enumerate(schedules):
        trigger_env_step = int(schedule.get("trigger_env_step", -1))
        env_step_to_schedule_indices.setdefault(trigger_env_step, []).append(int(idx))

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)
    all_dataset_indices = _episode_dataset_indices(dataset, int(episode_index))
    chunk_start = max(int(episode_chunk_start), 0)
    chunk_stride = max(int(episode_chunk_stride), 1)
    selected_dataset_indices = all_dataset_indices[chunk_start::chunk_stride]
    if episode_chunk_limit is not None:
        selected_dataset_indices = selected_dataset_indices[: max(int(episode_chunk_limit), 0)]
    if len(selected_dataset_indices) == 0:
        raise ValueError("No episode chunks selected for scan.")

    mixed_precision = _normalize_mixed_precision(str(cfg.mixed_precision))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model_device = str(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    model.eval()
    baseline_model = None
    baseline_weight_ckpt = None
    if baseline_ckpt is not None:
        baseline_cfg = cfg if baseline_cfg is None else baseline_cfg
        baseline_model = instantiate(baseline_cfg.model, model_dtype=model_dtype, device=model_device)
        baseline_weight_ckpt = resolve_weight_checkpoint(baseline_ckpt)
        baseline_model.load_checkpoint(str(baseline_weight_ckpt))
        baseline_model.eval()
    processor = getattr(getattr(dataset, "lerobot_dataset", None), "processor", None)
    binarize_gripper = bool(cfg.get("EVALUATION", {}).get("binarize_gripper", False))
    infer_steps = (
        int(num_inference_steps)
        if num_inference_steps is not None
        else int(cfg.model.streaming.streaming_train.get("infer_num_inference_steps", 10))
    )
    generator = torch.Generator(device="cpu").manual_seed(max(int(seed), 1) + 17017)

    rows: list[dict[str, Any]] = []
    schedule_fallbacks = 0
    for chunk_ordinal, dataset_index in enumerate(selected_dataset_indices):
        sample = dataset[int(dataset_index)]
        raw_action_start = int(sample.get("raw_action_start", -1))
        selected_schedule_index, used_fallback = _select_episode_schedule_index(
            schedules=schedules,
            env_step_to_schedule_indices=env_step_to_schedule_indices,
            schedule_mode=str(episode_schedule_mode),
            base_schedule_index=int(schedule_index),
            chunk_ordinal=int(chunk_ordinal),
            raw_action_start=int(raw_action_start),
            generator=generator,
        )
        if used_fallback:
            schedule_fallbacks += 1
        batch = model._extract_streaming_episode_batch(_batched(sample))
        row = _run_schedule_for_loaded_sample(
            model,
            sample,
            batch,
            schedule=schedules[int(selected_schedule_index)],
            schedule_index=int(selected_schedule_index),
            seed=int(seed) + int(dataset_index),
            rand_device=str(rand_device),
            infer_steps=int(infer_steps),
            include_per_step=False,
            processor=processor,
            binarize_gripper=binarize_gripper,
        )
        if baseline_model is not None:
            row["sync_baseline"] = _run_sync_baseline_for_loaded_sample(
                baseline_model,
                batch,
                seed=int(seed) + int(dataset_index),
                rand_device=str(rand_device),
                infer_steps=int(infer_steps),
                processor=processor,
                binarize_gripper=binarize_gripper,
            )
        row.update(
            {
                "dataset_index": int(dataset_index),
                "chunk_ordinal": int(chunk_ordinal),
                "trigger_obs_idx": int(sample.get("trigger_obs_idx", -1)),
                "raw_action_start": int(raw_action_start),
                "schedule_fallback": bool(used_fallback),
            }
        )
        rows.append(row)

    final_mses = [float(row["final_mse"]) for row in rows]
    final_rmses = [float(row["final_rmse"]) for row in rows]
    final_env_mses = [
        float(row["final_env_action_mse"])
        for row in rows
        if row.get("final_env_action_mse", None) is not None
    ]
    mixed_steps = [int(row["schedule_pattern"]["mixed_steps"]) for row in rows]
    rows_worst = sorted(rows, key=lambda item: float(item["final_mse"]), reverse=True)
    rows_best = sorted(rows, key=lambda item: float(item["final_mse"]))
    result = {
        "ckpt": str(weight_ckpt),
        "schedule_path": str(Path(schedule_path).resolve()),
        "episode_index": int(episode_index),
        "seed": int(seed),
        "device": model_device,
        "rand_device": str(rand_device),
        "num_schedules_total": int(len(schedules)),
        "num_episode_chunks_total": int(len(all_dataset_indices)),
        "num_episode_chunks_scanned": int(len(rows)),
        "episode_chunk_start": int(chunk_start),
        "episode_chunk_stride": int(chunk_stride),
        "episode_schedule_mode": str(episode_schedule_mode),
        "schedule_fallbacks": int(schedule_fallbacks),
        "num_inference_steps": int(infer_steps),
        "summary": {
            "final_mse_mean": _mean(final_mses),
            "final_mse_min": float(min(final_mses)),
            "final_mse_max": float(max(final_mses)),
            "final_rmse_mean": _mean(final_rmses),
            "final_mse_per_position_mean": _mean_per_position(rows, "final_mse_per_position"),
            "final_env_action_mse_mean": _mean(final_env_mses),
            "final_env_action_mse_min": None if not final_env_mses else float(min(final_env_mses)),
            "final_env_action_mse_max": None if not final_env_mses else float(max(final_env_mses)),
            "mixed_steps_mean": _mean([float(v) for v in mixed_steps]),
        },
        "worst": rows_worst[: max(int(top_k), 0)],
        "best": rows_best[: max(int(top_k), 0)],
    }
    if baseline_model is not None and baseline_weight_ckpt is not None:
        baseline_rows = [row["sync_baseline"] for row in rows]
        baseline_mses = [float(row["final_mse"]) for row in baseline_rows]
        result["sync_baseline"] = {
            "ckpt": str(baseline_weight_ckpt),
            "summary": {
                "final_mse_mean": _mean(baseline_mses),
                "final_mse_min": float(min(baseline_mses)),
                "final_mse_max": float(max(baseline_mses)),
                "final_mse_per_position_mean": _mean_per_position(baseline_rows, "final_mse_per_position"),
                "final_env_action_mse_mean": _mean(
                    [float(row["final_env_action_mse"]) for row in baseline_rows if row.get("final_env_action_mse") is not None]
                ),
                "final_env_action_mse_per_position_mean": _mean_per_position(
                    baseline_rows, "final_env_action_mse_per_position"
                ),
            },
        }
    del model, baseline_model, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one collected async schedule as a full denoise trajectory and report action-chunk MSE."
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--baseline-task-config", default=None)
    parser.add_argument("--baseline-ckpt", default=None)
    parser.add_argument("--schedule-path", required=True)
    parser.add_argument("--schedule-index", type=int, default=0)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan-start", type=int, default=0)
    parser.add_argument("--scan-limit", type=int, default=None)
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--scan-top-k", type=int, default=20)
    parser.add_argument("--random-pairs", action="store_true")
    parser.add_argument("--num-random-pairs", type=int, default=128)
    parser.add_argument("--episode", action="store_true")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument(
        "--episode-schedule-mode",
        choices=["env_step", "cycle", "fixed", "random"],
        default="env_step",
    )
    parser.add_argument("--episode-chunk-start", type=int, default=0)
    parser.add_argument("--episode-chunk-limit", type=int, default=None)
    parser.add_argument("--episode-chunk-stride", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with initialize_config_dir(version_base="1.3", config_dir=str((project_root / "configs").resolve())):
        cfg = compose(
            config_name="train",
            overrides=[
                f"task={args.task_config}",
            ],
        )
        baseline_cfg = None
        if args.baseline_ckpt is not None:
            baseline_task_config = args.task_config if args.baseline_task_config is None else args.baseline_task_config
            baseline_cfg = compose(
                config_name="train",
                overrides=[
                    f"task={baseline_task_config}",
                ],
            )
    if args.random_pairs:
        result = scan_random_schedule_sample_pairs(
            cfg,
            ckpt=str(args.ckpt),
            baseline_cfg=baseline_cfg,
            baseline_ckpt=args.baseline_ckpt,
            schedule_path=args.schedule_path,
            seed=int(args.seed),
            device=args.device,
            rand_device=str(args.rand_device),
            num_inference_steps=args.num_inference_steps,
            num_pairs=int(args.num_random_pairs),
            top_k=int(args.scan_top_k),
        )
    elif args.episode:
        result = scan_episode_denoise_mse(
            cfg,
            ckpt=str(args.ckpt),
            baseline_cfg=baseline_cfg,
            baseline_ckpt=args.baseline_ckpt,
            schedule_path=args.schedule_path,
            episode_index=int(args.episode_index),
            seed=int(args.seed),
            device=args.device,
            rand_device=str(args.rand_device),
            num_inference_steps=args.num_inference_steps,
            schedule_index=int(args.schedule_index),
            episode_schedule_mode=str(args.episode_schedule_mode),
            episode_chunk_start=int(args.episode_chunk_start),
            episode_chunk_limit=args.episode_chunk_limit,
            episode_chunk_stride=int(args.episode_chunk_stride),
            top_k=int(args.scan_top_k),
        )
    elif args.scan:
        result = scan_schedule_denoise_mse(
            cfg,
            ckpt=str(args.ckpt),
            baseline_cfg=baseline_cfg,
            baseline_ckpt=args.baseline_ckpt,
            schedule_path=args.schedule_path,
            dataset_index=int(args.dataset_index),
            seed=int(args.seed),
            device=args.device,
            rand_device=str(args.rand_device),
            num_inference_steps=args.num_inference_steps,
            scan_start=int(args.scan_start),
            scan_limit=args.scan_limit,
            scan_stride=int(args.scan_stride),
            top_k=int(args.scan_top_k),
        )
    else:
        result = run_one_schedule_denoise_mse(
            cfg,
            ckpt=str(args.ckpt),
            baseline_cfg=baseline_cfg,
            baseline_ckpt=args.baseline_ckpt,
            schedule_path=args.schedule_path,
            schedule_index=int(args.schedule_index),
            dataset_index=int(args.dataset_index),
            seed=int(args.seed),
            device=args.device,
            rand_device=str(args.rand_device),
            num_inference_steps=args.num_inference_steps,
        )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote schedule denoise MSE to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
