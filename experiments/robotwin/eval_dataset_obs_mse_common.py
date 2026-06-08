from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf, open_dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.async_streaming_runner import AsyncStreamingRunner  # noqa: E402
from fastwam.utils.async_streaming_runtime import ProfiledRuntime  # noqa: E402
from fastwam.utils.pytorch_utils import set_global_seed  # noqa: E402


def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ChunkOffsetRuntime(ProfiledRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._action_source_cache: dict[int, list[int]] = defaultdict(list)
        self.last_served_action_source_env_step: int | None = None

    def reset_for_formal_phase(self, *, env_step: int = 0) -> None:
        super().reset_for_formal_phase(env_step=env_step)
        self._action_source_cache.clear()
        self.last_served_action_source_env_step = None

    def _publish_released_actions(
        self,
        released_actions: np.ndarray,
        *,
        released_env_steps: list[int],
        source_env_step: int,
        release_obs_index: int = -1,
        release_obs_env_step: int = -1,
        job_id: int = -1,
        phase_id: int = -1,
    ) -> int:
        del release_obs_index, release_obs_env_step, job_id, phase_id
        dropped = 0
        current_env_step = int(self._current_env_step)
        if released_actions.ndim == 1:
            released_actions = released_actions[None, :]
        for i, target_step in enumerate(released_env_steps):
            if int(target_step) < current_env_step:
                dropped += 1
                continue
            target_step = int(target_step)
            self._ensembler.action_cache[target_step] = [np.asarray(released_actions[i], dtype=np.float32)]
            self._action_source_cache[target_step].append(int(source_env_step))
        self._dropped_prefix_actions += dropped
        return dropped

    def get_action(self, env_step: int, *, count_miss: bool = True) -> np.ndarray | None:
        action = super().get_action(env_step, count_miss=count_miss)
        if action is None:
            return None
        sources = self._action_source_cache.get(int(env_step), [])
        self.last_served_action_source_env_step = None if len(sources) == 0 else int(sources[-1])
        self._action_source_cache.pop(int(env_step), None)
        return action


def compose_robotwin_cfg(args):
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
    if getattr(args, "replan_steps", None) is not None:
        overrides.append(f"EVALUATION.replan_steps={int(args.replan_steps)}")
    if args.rand_device is not None:
        overrides.append(f"EVALUATION.rand_device={args.rand_device}")
    if getattr(args, "control_dt_ms", None) is not None:
        overrides.append(f"EVALUATION.async_control_dt_ms={float(args.control_dt_ms)}")
    if getattr(args, "obs_stride_env_steps", None) is not None:
        overrides.append(f"EVALUATION.async_obs_stride_env_steps={int(args.obs_stride_env_steps)}")
    if bool(getattr(args, "tiled", False)):
        overrides.append("EVALUATION.tiled=true")

    with initialize_config_dir(version_base="1.3", config_dir=str((PROJECT_ROOT / "configs").resolve())):
        return compose(config_name="sim_robotwin", overrides=overrides)


def build_model(cfg, *, ckpt: str, model_dtype: torch.dtype, device: str):
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    return model.to(device).eval()


def resolve_action_horizon(cfg) -> int:
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if horizon_cfg is not None:
        return int(horizon_cfg)
    if cfg.data.train.get("action_horizon", None) is not None:
        return int(cfg.data.train.action_horizon)
    return int(cfg.data.train.num_frames) - 1


def resolve_device(cfg, args) -> str:
    device = str(args.device or cfg.EVALUATION.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return "cuda:0" if device == "cuda" else device


def instantiate_dataset(cfg):
    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    return instantiate(sample_cfg.data.train)


def episode_index_from_dataset_index(dataset, dataset_index: int) -> int:
    if not hasattr(dataset, "sample_index"):
        raise ValueError("Expected StreamingRobotEpisodeDataset with sample_index.")
    if not 0 <= int(dataset_index) < len(dataset.sample_index):
        raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset.sample_index) - 1}]")
    episode_idx, _trigger_obs_idx, _raw_action_start = dataset.sample_index[int(dataset_index)]
    return int(episode_idx)


def episode_image(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    image = dataset._query_episode_images(
        dataset=payload["dataset"],
        local_episode_idx=int(payload["local_episode_idx"]),
        raw_frame_indices=[int(raw_frame_index)],
    )
    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError(f"Expected one processed image frame [1,C,H,W], got {tuple(image.shape)}")
    return image[0].detach().to(device="cpu", dtype=torch.float32)


def episode_proprio(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    action_raw = {key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1] for key, tensor in payload["action_raw"].items()}
    state_raw = {key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1] for key, tensor in payload["state_raw"].items()}
    _action, state = dataset._normalize_action_and_state(action_raw=action_raw, state_raw=state_raw)
    return state.squeeze(0).detach().to(device="cpu", dtype=torch.float32)


def raw_action_chunk_to_normalized(raw_chunk: np.ndarray, *, processor) -> torch.Tensor:
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged action key in processor.shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    raw_chunk_t = torch.from_numpy(np.asarray(raw_chunk, dtype=np.float32)).unsqueeze(0)
    return normalizer.forward(raw_chunk_t).squeeze(0).to(dtype=torch.float32, device="cpu")


def normalized_action_to_raw(action: torch.Tensor, *, processor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected normalized action [B, T, D], got {tuple(action.shape)}")
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError("Expected a single merged action key in processor.shape_meta['action'].")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    return normalizer.backward(action.detach().to(dtype=torch.float32, device="cpu")).numpy()[0].astype(np.float32)


def robotwin_action_postprocess(actions: torch.Tensor, *, processor, cfg) -> np.ndarray:
    del cfg
    return normalized_action_to_raw(actions, processor=processor)


def build_padded_raw_chunk(raw_action: np.ndarray, *, env_step: int, action_horizon: int) -> tuple[np.ndarray, np.ndarray]:
    action_dim = int(raw_action.shape[1])
    target = np.zeros((int(action_horizon), action_dim), dtype=np.float32)
    pad = np.ones((int(action_horizon),), dtype=bool)
    end = min(int(env_step) + int(action_horizon), int(raw_action.shape[0]))
    valid = max(0, int(end - int(env_step)))
    if valid > 0:
        target[:valid] = raw_action[int(env_step) : int(end)]
        pad[:valid] = False
    return target, pad


def array_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {"count": int(arr.shape[0]), "mean": float(np.mean(arr)), "min": float(np.min(arr)), "max": float(np.max(arr))}


def masked_mse(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> float:
    err = (pred.detach().float() - target.detach().float()).pow(2)
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


def masked_mse_per_position(pred: torch.Tensor, target: torch.Tensor, action_is_pad: torch.Tensor | None = None) -> list[float | None]:
    err = (pred.detach().float() - target.detach().float()).pow(2).mean(dim=-1)
    if action_is_pad is None:
        return [float(v) for v in err.mean(dim=0).tolist()]
    pad = action_is_pad.detach().to(device=err.device)
    if pad.ndim == 1:
        pad = pad.unsqueeze(0)
    valid = (~pad.bool()).to(dtype=err.dtype)
    out: list[float | None] = []
    for pos in range(int(err.shape[1])):
        denom = float(valid[:, pos].sum().item())
        out.append(None if denom <= 0.0 else float((err[:, pos] * valid[:, pos]).sum().div(valid[:, pos].sum()).item()))
    return out


def mean_per_position(rows: list[dict[str, Any]], key: str) -> list[float | None] | None:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    if not values:
        return None
    out: list[float | None] = []
    for idx in range(len(values[0])):
        bucket = [float(v[idx]) for v in values if v[idx] is not None]
        out.append(None if len(bucket) == 0 else float(sum(bucket) / float(len(bucket))))
    return out


def sync_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    head = rows[: min(8, len(rows))]
    next8 = rows[8:16]
    tail = rows[16:]
    if len(rows) > 0 and "mse" in rows[0]:
        pred_arr = np.asarray([row["pred_action"] for row in rows], dtype=np.float32)
        target_arr = np.asarray([row["target_action"] for row in rows], dtype=np.float32)
        return {
            "env_action_mse": array_stats([float(row["mse"]) for row in rows]),
            "env_action_mae": array_stats([float(row["mae"]) for row in rows]),
            "head8_env_action_mse": array_stats([float(row["mse"]) for row in head]),
            "next8_env_action_mse": array_stats([float(row["mse"]) for row in next8]),
            "tail_env_action_mse": array_stats([float(row["mse"]) for row in tail]),
            "chunk_position_env_action_mse": position_stats(
                rows,
                action_horizon=max(int(row.get("action_horizon", 0)) for row in rows),
                key="mse",
                position_key="chunk_position",
                output_key="chunk_position",
            ),
            "per_dim_mse": np.mean(np.square(pred_arr - target_arr), axis=0).astype(float).tolist(),
        }
    return {
        "chunk_mse": array_stats([float(row["chunk_mse"]) for row in rows]),
        "env_action_mse": array_stats([float(row["chunk_env_action_mse"]) for row in rows if row.get("chunk_env_action_mse") is not None]),
        "head8_env_action_mse": array_stats([float(row["chunk_env_action_mse"]) for row in head if row.get("chunk_env_action_mse") is not None]),
        "next8_env_action_mse": array_stats([float(row["chunk_env_action_mse"]) for row in next8 if row.get("chunk_env_action_mse") is not None]),
        "tail_env_action_mse": array_stats([float(row["chunk_env_action_mse"]) for row in tail if row.get("chunk_env_action_mse") is not None]),
        "chunk_offset_env_action_mse": mean_per_position(rows, "chunk_env_action_mse_per_position"),
    }


def position_stats(
    rows: list[dict[str, Any]],
    *,
    action_horizon: int,
    key: str,
    position_key: str,
    output_key: str,
) -> list[dict[str, float | int | None]]:
    buckets: list[list[float]] = [[] for _ in range(int(action_horizon))]
    for row in rows:
        position = row.get(position_key, None)
        if position is None:
            continue
        idx = int(position)
        if 0 <= idx < int(action_horizon):
            buckets[idx].append(float(row[key]))
    return [{output_key: int(idx), **array_stats(values)} for idx, values in enumerate(buckets)]


def runtime_chunk_offset_stats(rows: list[dict[str, Any]], *, action_horizon: int, key: str) -> list[dict[str, float | int | None]]:
    return position_stats(
        rows,
        action_horizon=action_horizon,
        key=key,
        position_key="chunk_offset",
        output_key="chunk_offset",
    )


def runtime_summary(rows: list[dict[str, Any]], *, action_horizon: int) -> dict[str, Any]:
    pred_arr = np.asarray([row["pred_action"] for row in rows], dtype=np.float32) if len(rows) > 0 else np.empty((0, 0))
    target_arr = np.asarray([row["target_action"] for row in rows], dtype=np.float32) if len(rows) > 0 else np.empty((0, 0))
    head = rows[: min(8, len(rows))]
    next8 = rows[8:16]
    tail = rows[16:]
    return {
        "env_action_mse": array_stats([float(row["mse"]) for row in rows]),
        "env_action_mae": array_stats([float(row["mae"]) for row in rows]),
        "head8_env_action_mse": array_stats([float(row["mse"]) for row in head]),
        "next8_env_action_mse": array_stats([float(row["mse"]) for row in next8]),
        "tail_env_action_mse": array_stats([float(row["mse"]) for row in tail]),
        "chunk_offset_env_action_mse": runtime_chunk_offset_stats(rows, action_horizon=action_horizon, key="mse"),
        "per_dim_mse": [] if len(rows) == 0 else np.mean(np.square(pred_arr - target_arr), axis=0).astype(float).tolist(),
        "initial_miss_steps": int(sum(1 for row in rows if bool(row["had_initial_miss"]))),
        "submitted_obs_steps": int(sum(1 for row in rows if bool(row["submitted_obs"]))),
    }


@torch.no_grad()
def run_sync_mse(args) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg = compose_robotwin_cfg(args)
    dataset = instantiate_dataset(cfg)
    episode_index = int(args.episode_index) if args.episode_index is not None else episode_index_from_dataset_index(dataset, int(args.dataset_index))
    payload = dataset._load_episode_cache(int(episode_index))
    raw_action = payload["action_raw"]["default"].detach().cpu().numpy().astype(np.float32)
    raw_num_steps = int(raw_action.shape[0])
    max_steps = min(max(int(args.max_steps), 1), raw_num_steps)

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", args.mixed_precision)))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = resolve_device(cfg, args)
    model = build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))
    context, context_mask = model.encode_prompt(prompt)
    context = context.to(device="cpu", dtype=model.torch_dtype)
    context_mask = context_mask.to(device="cpu", dtype=torch.bool)

    action_horizon = int(resolve_action_horizon(cfg))
    replan_steps = int(cfg.EVALUATION.get("replan_steps", action_horizon))
    replan_steps = max(1, min(int(replan_steps), int(action_horizon)))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False) or bool(getattr(args, "tiled", False)))

    rows: list[dict[str, Any]] = []
    replan_env_step = 0
    while replan_env_step < max_steps:
        model.reset_streaming_state()
        image = episode_image(dataset, payload, int(replan_env_step))
        proprio = episode_proprio(dataset, payload, int(replan_env_step))
        infer_out = model.infer_action(
            prompt=None,
            input_image=image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            seed=int(args.seed) + int(replan_env_step),
            rand_device=rand_device,
            tiled=tiled,
        )
        pred_action = infer_out["action"].detach().to(device="cpu", dtype=torch.float32)
        target_raw_chunk, action_is_pad_np = build_padded_raw_chunk(raw_action, env_step=int(replan_env_step), action_horizon=action_horizon)
        target_action = raw_action_chunk_to_normalized(target_raw_chunk, processor=processor)
        action_is_pad = torch.from_numpy(action_is_pad_np)

        pred_raw_chunk = normalized_action_to_raw(pred_action, processor=processor)
        chunk_mse = masked_mse(pred_action.unsqueeze(0), target_action.unsqueeze(0), action_is_pad.unsqueeze(0))
        valid_mask = ~action_is_pad_np
        chunk_env_mse = float(np.mean(np.square(pred_raw_chunk[valid_mask] - target_raw_chunk[valid_mask]))) if bool(np.any(valid_mask)) else None
        chunk_mse_per_position = masked_mse_per_position(pred_action.unsqueeze(0), target_action.unsqueeze(0), action_is_pad.unsqueeze(0))
        chunk_env_action_mse_per_position = [
            None if bool(action_is_pad_np[pos]) else float(np.mean(np.square(pred_raw_chunk[pos] - target_raw_chunk[pos])))
            for pos in range(int(action_horizon))
        ]

        n_exec = min(int(replan_steps), int(action_horizon), int(max_steps - replan_env_step))
        for chunk_pos in range(n_exec):
            if bool(action_is_pad_np[chunk_pos]):
                continue
            env_step = int(replan_env_step + chunk_pos)
            pred = np.asarray(pred_raw_chunk[chunk_pos], dtype=np.float32)
            target = np.asarray(raw_action[env_step], dtype=np.float32)
            diff = pred - target
            rows.append(
                {
                    "env_step": int(env_step),
                    "source_env_step": int(replan_env_step),
                    "chunk_position": int(chunk_pos),
                    "action_horizon": int(action_horizon),
                    "replan_steps": int(replan_steps),
                    "mse": float(np.mean(np.square(diff))),
                    "mae": float(np.mean(np.abs(diff))),
                    "per_dim_sqerr": np.square(diff).astype(float).tolist(),
                    "pred_action": pred.astype(float).tolist(),
                    "target_action": target.astype(float).tolist(),
                    "chunk_mse": float(chunk_mse),
                    "chunk_rmse": float(chunk_mse ** 0.5),
                    "chunk_mse_per_position": chunk_mse_per_position,
                    "chunk_env_action_mse": chunk_env_mse,
                    "chunk_env_action_mse_per_position": chunk_env_action_mse_per_position,
                }
            )
        replan_env_step += int(replan_steps)

    skip_initial_steps = max(0, int(args.skip_initial_steps))
    return {
        "ckpt": str(resolve_weight_checkpoint(str(args.ckpt))),
        "task_config": str(args.task_config),
        "dataset_index": int(args.dataset_index),
        "episode_index": int(episode_index),
        "instruction": str(payload["instruction"]),
        "seed": int(args.seed),
        "device": str(device),
        "mixed_precision": str(mixed_precision),
        "num_episode_raw_actions": int(raw_num_steps),
        "num_compared_steps": int(len(rows)),
        "action_horizon": int(action_horizon),
        "replan_steps": int(replan_steps),
        "num_inference_steps": int(num_inference_steps),
        "tiled": bool(tiled),
        "summary": sync_summary(rows),
        "summary_after_skip": {"skip_initial_steps": int(skip_initial_steps), **sync_summary(rows[skip_initial_steps:])},
        "per_step": rows,
    }


@torch.no_grad()
def run_runtime_mse(args) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg = compose_robotwin_cfg(args)
    dataset = instantiate_dataset(cfg)
    episode_index = int(args.episode_index) if args.episode_index is not None else episode_index_from_dataset_index(dataset, int(args.dataset_index))
    payload = dataset._load_episode_cache(int(episode_index))
    raw_action = payload["action_raw"]["default"].detach().cpu().numpy().astype(np.float32)
    raw_num_steps = int(raw_action.shape[0])
    max_steps = min(max(int(args.max_steps), 1), raw_num_steps)

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", args.mixed_precision)))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = resolve_device(cfg, args)
    video_model = build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)
    action_model = video_model
    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))
    video_context, video_context_mask = video_model.encode_prompt(prompt)
    action_context, action_context_mask = video_context, video_context_mask
    video_context = video_context.to(device="cpu", dtype=video_model.torch_dtype)
    video_context_mask = video_context_mask.to(device="cpu", dtype=torch.bool)
    action_context = action_context.to(device="cpu", dtype=action_model.torch_dtype)
    action_context_mask = action_context_mask.to(device="cpu", dtype=torch.bool)

    obs_stride_env_steps = int(cfg.EVALUATION.get("async_obs_stride_env_steps", cfg.data.train.get("obs_stride", 3)))
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    action_horizon = int(resolve_action_horizon(cfg))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False))

    action_postprocess = lambda x: robotwin_action_postprocess(x, processor=processor, cfg=cfg)
    runtime = ChunkOffsetRuntime(
        video_model=video_model,
        action_model=action_model,
        video_context=video_context,
        video_context_mask=video_context_mask,
        action_context=action_context,
        action_context_mask=action_context_mask,
        action_postprocess=action_postprocess,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        rand_device=rand_device,
        tiled=tiled,
        seed=int(args.seed),
    )

    rows: list[dict[str, Any]] = []
    runtime_started = False
    try:
        runtime.start()
        runtime_started = True
        runner = AsyncStreamingRunner(runtime=runtime, obs_stride_env_steps=obs_stride_env_steps, control_dt_ms=control_dt_ms)
        runtime.reset_for_formal_phase(env_step=0)
        runner.start_formal_phase(obs_index_start=0)
        runner.prime_formal_observation(input_image=episode_image(dataset, payload, 0), proprio=episode_proprio(dataset, payload, 0), env_step=0)

        for env_step in range(max_steps):
            step_start = time.perf_counter()
            image = episode_image(dataset, payload, int(env_step))
            proprio = episode_proprio(dataset, payload, int(env_step))
            submitted_obs = runner.maybe_submit_formal_observation(input_image=image, proprio=proprio, env_step=int(env_step))
            action = runtime.get_action(int(env_step), count_miss=False)
            had_initial_miss = action is None
            if action is None:
                action = runner.wait_for_action(env_step=int(env_step), proprio=proprio)
            pred = np.asarray(action, dtype=np.float32)
            target = np.asarray(raw_action[int(env_step)], dtype=np.float32)
            diff = pred - target
            source_env_step = runtime.last_served_action_source_env_step
            rows.append(
                {
                    "env_step": int(env_step),
                    "source_env_step": None if source_env_step is None else int(source_env_step),
                    "chunk_offset": None if source_env_step is None else int(env_step) - int(source_env_step),
                    "submitted_obs": bool(submitted_obs),
                    "had_initial_miss": bool(had_initial_miss),
                    "mse": float(np.mean(np.square(diff))),
                    "mae": float(np.mean(np.abs(diff))),
                    "per_dim_sqerr": np.square(diff).astype(float).tolist(),
                    "pred_action": pred.astype(float).tolist(),
                    "target_action": target.astype(float).tolist(),
                }
            )
            if not bool(args.no_realtime_pacing):
                sleep_s = max(0.0, float(control_dt_ms) / 1000.0 - float(time.perf_counter() - step_start))
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        runtime.wait_until_idle()
        runtime_stats = runtime.stats()
    finally:
        if runtime_started:
            runtime.stop()

    skip_initial_steps = max(0, int(args.skip_initial_steps))
    return {
        "ckpt": str(resolve_weight_checkpoint(str(args.ckpt))),
        "task_config": str(args.task_config),
        "dataset_index": int(args.dataset_index),
        "episode_index": int(episode_index),
        "instruction": str(payload["instruction"]),
        "seed": int(args.seed),
        "device": str(device),
        "mixed_precision": str(mixed_precision),
        "num_episode_raw_actions": int(raw_num_steps),
        "num_compared_steps": int(len(rows)),
        "action_horizon": int(action_horizon),
        "num_inference_steps": int(num_inference_steps),
        "obs_stride_env_steps": int(obs_stride_env_steps),
        "control_dt_ms": float(control_dt_ms),
        "realtime_pacing": not bool(args.no_realtime_pacing),
        "summary": runtime_summary(rows, action_horizon=action_horizon),
        "summary_after_skip": {"skip_initial_steps": int(skip_initial_steps), **runtime_summary(rows[skip_initial_steps:], action_horizon=action_horizon)},
        "runtime_stats": runtime_stats,
        "per_step": rows,
    }
