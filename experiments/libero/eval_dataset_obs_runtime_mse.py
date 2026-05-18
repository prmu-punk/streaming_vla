from __future__ import annotations

import argparse
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

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.build_xt_replay import (  # noqa: E402
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    resolve_weight_checkpoint,
)
from experiments.libero.eval_libero_policy_utils import _postprocess_libero_action_chunk  # noqa: E402
from experiments.libero.libero_utils import invert_gripper_action  # noqa: E402
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
    ) -> int:
        dropped = 0
        current_env_step = int(self._current_env_step)
        if released_actions.ndim == 1:
            released_actions = released_actions[None, :]
        for i, target_step in enumerate(released_env_steps):
            if target_step < current_env_step:
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
        trigger_sources = self._action_source_cache.get(int(env_step), [])
        self.last_served_action_source_env_step = None if len(trigger_sources) == 0 else int(trigger_sources[-1])
        if int(env_step) in self._action_source_cache:
            del self._action_source_cache[int(env_step)]
        return action


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teacher-force dataset observations through the real async runtime and compare the "
            "actions that would be served at each env step against the paired dataset actions."
        )
    )
    parser.add_argument("--task-config", default="libero_streaming_action_ft_2cam224_1e-4")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--rand-device", default=None)
    parser.add_argument("--control-dt-ms", type=float, default=None)
    parser.add_argument("--obs-stride-env-steps", type=int, default=None)
    parser.add_argument(
        "--no-realtime-pacing",
        action="store_true",
        help="Feed dataset frames as fast as possible. Default behavior sleeps each step to mimic control dt.",
    )
    return parser.parse_args()


def _compose_cfg(args: argparse.Namespace):
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
    if args.control_dt_ms is not None:
        overrides.append(f"EVALUATION.async_control_dt_ms={float(args.control_dt_ms)}")
    if args.obs_stride_env_steps is not None:
        overrides.append(f"EVALUATION.async_obs_stride_env_steps={int(args.obs_stride_env_steps)}")

    config_dir = str((project_root / "configs").resolve())
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="sim_libero", overrides=overrides)


def _resolve_device(cfg, args: argparse.Namespace) -> str:
    device = str(args.device or cfg.EVALUATION.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if device == "cuda":
        device = "cuda:0"
    return device


def _build_model(cfg, *, ckpt: str, model_dtype: torch.dtype, device: str):
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    weight_ckpt = resolve_weight_checkpoint(ckpt)
    model.load_checkpoint(str(weight_ckpt))
    return model.to(device).eval()


def _resolve_action_horizon(cfg) -> int:
    horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if horizon_cfg is not None:
        return int(horizon_cfg)
    if cfg.data.train.get("action_horizon", None) is not None:
        return int(cfg.data.train.action_horizon)
    return int(cfg.data.train.num_frames) - 1


def _dataset_action_to_env_action(action: np.ndarray, *, binarize_gripper: bool) -> np.ndarray:
    converted = np.asarray(action, dtype=np.float32).copy()
    converted[..., -1] = converted[..., -1] * 2.0 - 1.0
    converted = invert_gripper_action(converted)
    if bool(binarize_gripper):
        converted[..., -1] = np.sign(converted[..., -1])
    return np.asarray(converted, dtype=np.float32)


def _array_stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _slice_rows(rows: list[dict[str, Any]], start: int, end: int | None = None) -> list[dict[str, Any]]:
    start = max(0, int(start))
    if end is None:
        return rows[start:]
    return rows[start : max(start, int(end))]


def _rows_stat(rows: list[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    return _array_stats([float(row[key]) for row in rows])


def _chunk_offset_stats(
    rows: list[dict[str, Any]],
    *,
    action_horizon: int,
    key: str,
) -> list[dict[str, float | int | None]]:
    buckets: list[list[float]] = [[] for _ in range(int(action_horizon))]
    for row in rows:
        chunk_offset = row.get("chunk_offset", None)
        if chunk_offset is None:
            continue
        idx = int(chunk_offset)
        if 0 <= idx < int(action_horizon):
            buckets[idx].append(float(row[key]))
    stats: list[dict[str, float | int | None]] = []
    for idx, values in enumerate(buckets):
        entry = {"chunk_offset": int(idx)}
        entry.update(_array_stats(values))
        stats.append(entry)
    return stats


def _episode_index_from_dataset_index(dataset, dataset_index: int) -> int:
    if not hasattr(dataset, "sample_index"):
        raise ValueError("Expected StreamingRobotEpisodeDataset with sample_index.")
    if not 0 <= int(dataset_index) < len(dataset.sample_index):
        raise IndexError(f"dataset_index={dataset_index} out of range [0, {len(dataset.sample_index) - 1}]")
    episode_idx, _trigger_obs_idx, _raw_action_start = dataset.sample_index[int(dataset_index)]
    return int(episode_idx)


def _episode_image(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    image = dataset._query_episode_images(
        dataset=payload["dataset"],
        local_episode_idx=int(payload["local_episode_idx"]),
        raw_frame_indices=[int(raw_frame_index)],
    )
    if image.ndim != 4 or int(image.shape[0]) != 1:
        raise ValueError(f"Expected one processed image frame [1,C,H,W], got {tuple(image.shape)}")
    return image[0].detach().to(device="cpu", dtype=torch.float32)


def _episode_proprio(dataset, payload: dict[str, Any], raw_frame_index: int) -> torch.Tensor:
    action_raw = {
        key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1]
        for key, tensor in payload["action_raw"].items()
    }
    state_raw = {
        key: tensor[int(raw_frame_index) : int(raw_frame_index) + 1]
        for key, tensor in payload["state_raw"].items()
    }
    _action, state = dataset._normalize_action_and_state(action_raw=action_raw, state_raw=state_raw)
    return state.squeeze(0).detach().to(device="cpu", dtype=torch.float32)


@torch.no_grad()
def run_dataset_obs_runtime_mse(args: argparse.Namespace) -> dict[str, Any]:
    set_global_seed(max(int(args.seed), 1), get_worker_init_fn=False)
    cfg = _compose_cfg(args)

    sample_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(sample_cfg.data.train):
        sample_cfg.data.train.pop("trajectory_replay_key", None)
    dataset = instantiate(sample_cfg.data.train)

    episode_index = (
        int(args.episode_index)
        if args.episode_index is not None
        else _episode_index_from_dataset_index(dataset, int(args.dataset_index))
    )
    payload = dataset._load_episode_cache(int(episode_index))
    raw_action = payload["action_raw"]["default"].detach().cpu().numpy().astype(np.float32)
    raw_num_steps = int(raw_action.shape[0])
    max_steps = min(max(int(args.max_steps), 1), raw_num_steps)

    mixed_precision = _normalize_mixed_precision(str(cfg.get("mixed_precision", args.mixed_precision)))
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    device = _resolve_device(cfg, args)
    video_model = _build_model(cfg, ckpt=str(args.ckpt), model_dtype=model_dtype, device=device)
    action_model = video_model

    processor = dataset.lerobot_dataset.processor
    if processor is None:
        raise ValueError("Dataset processor is missing.")

    prompt = DEFAULT_PROMPT.format(task=str(payload["instruction"]))
    with torch.no_grad():
        video_context, video_context_mask = video_model.encode_prompt(prompt)
        if action_model is video_model:
            action_context, action_context_mask = video_context, video_context_mask
        else:
            action_context, action_context_mask = action_model.encode_prompt(prompt)
    video_context = video_context.to(device="cpu", dtype=video_model.torch_dtype)
    video_context_mask = video_context_mask.to(device="cpu", dtype=torch.bool)
    action_context = action_context.to(device="cpu", dtype=action_model.torch_dtype)
    action_context_mask = action_context_mask.to(device="cpu", dtype=torch.bool)

    obs_stride_env_steps = int(
        cfg.EVALUATION.get(
            "async_obs_stride_env_steps",
            cfg.data.train.get("obs_stride", 3),
        )
    )
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    action_horizon = int(_resolve_action_horizon(cfg))
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False))
    binarize_gripper = bool(cfg.EVALUATION.get("binarize_gripper", False))

    action_postprocess = lambda x: _postprocess_libero_action_chunk(x, processor=processor, cfg=cfg)
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

        image0 = _episode_image(dataset, payload, 0)
        proprio0 = _episode_proprio(dataset, payload, 0)

        runner = AsyncStreamingRunner(
            runtime=runtime,
            obs_stride_env_steps=obs_stride_env_steps,
            control_dt_ms=control_dt_ms,
        )
        runtime.reset_for_formal_phase(env_step=0)
        runner.start_formal_phase(obs_index_start=0)
        runner.prime_formal_observation(
            input_image=image0,
            proprio=proprio0,
            env_step=0,
        )

        for env_step in range(max_steps):
            step_start = time.perf_counter()
            image = _episode_image(dataset, payload, int(env_step))
            proprio = _episode_proprio(dataset, payload, int(env_step))
            submitted_obs = runner.maybe_submit_formal_observation(
                input_image=image,
                proprio=proprio,
                env_step=int(env_step),
            )
            action = runtime.get_action(int(env_step), count_miss=False)
            had_initial_miss = action is None
            if action is None:
                action = runner.wait_for_action(env_step=int(env_step), proprio=proprio)

            pred = np.asarray(action, dtype=np.float32)
            target = _dataset_action_to_env_action(
                raw_action[int(env_step)],
                binarize_gripper=binarize_gripper,
            )
            diff = pred - target
            source_env_step = runtime.last_served_action_source_env_step
            chunk_offset = None if source_env_step is None else int(env_step) - int(source_env_step)
            rows.append(
                {
                    "env_step": int(env_step),
                    "source_env_step": None if source_env_step is None else int(source_env_step),
                    "chunk_offset": chunk_offset,
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
                elapsed_s = float(time.perf_counter() - step_start)
                sleep_s = max(0.0, float(control_dt_ms) / 1000.0 - elapsed_s)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

        runtime.wait_until_idle()
        runtime_stats = runtime.stats()
    finally:
        if runtime_started:
            runtime.stop()

    mses = [float(row["mse"]) for row in rows]
    maes = [float(row["mae"]) for row in rows]
    head = rows[: min(8, len(rows))]
    next8 = _slice_rows(rows, 8, 16)
    tail = _slice_rows(rows, 16, None)
    pred_arr = np.asarray([row["pred_action"] for row in rows], dtype=np.float32)
    target_arr = np.asarray([row["target_action"] for row in rows], dtype=np.float32)
    result = {
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
        "summary": {
            "env_action_mse": _array_stats(mses),
            "env_action_mae": _array_stats(maes),
            "head8_env_action_mse": _rows_stat(head, "mse"),
            "next8_env_action_mse": _rows_stat(next8, "mse"),
            "tail_env_action_mse": _rows_stat(tail, "mse"),
            "chunk_offset_env_action_mse": _chunk_offset_stats(
                rows,
                action_horizon=action_horizon,
                key="mse",
            ),
            "per_dim_mse": (
                []
                if len(rows) == 0
                else np.mean(np.square(pred_arr - target_arr), axis=0).astype(float).tolist()
            ),
            "initial_miss_steps": int(sum(1 for row in rows if bool(row["had_initial_miss"]))),
            "submitted_obs_steps": int(sum(1 for row in rows if bool(row["submitted_obs"]))),
        },
        "runtime_stats": runtime_stats,
        "per_step": rows,
    }
    return result


def main() -> None:
    args = _parse_args()
    result = run_dataset_obs_runtime_mse(args)
    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, cls=NumpyEncoder)
    print(f"Wrote dataset-observation runtime MSE to: {output_path}")
    print(json.dumps(result["summary"], indent=2, cls=NumpyEncoder))


if __name__ == "__main__":
    main()
