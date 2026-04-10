from __future__ import annotations

from collections import defaultdict
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_rollout_profiled import run_single_episode_async
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_async_runtime_devices(cfg: DictConfig, fallback_device: str) -> tuple[str, str]:
    video_device = cfg.EVALUATION.get("async_video_device")
    action_device = cfg.EVALUATION.get("async_action_device")
    resolved_video_device = fallback_device if video_device is None else str(video_device)
    resolved_action_device = fallback_device if action_device is None else str(action_device)
    return resolved_video_device, resolved_action_device


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)


def _build_eval_model(cfg: DictConfig, *, model_dtype: torch.dtype, device: str) -> torch.nn.Module:
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    if cfg.get("ckpt") is not None:
        _load_model_checkpoint(model, str(cfg.ckpt))
    else:
        logging.warning("No checkpoint provided; using randomly initialized weights.")
    return model.to(device).eval()


def _configure_egl_device(cfg: DictConfig) -> int:
    gpu_id = int(cfg.get("gpu_id", 0))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(gpu_id)
    return gpu_id


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    if cfg.get("ckpt") is not None:
        ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
        for parent in list(ckpt.parents)[:4]:
            candidates.append(parent / "dataset_stats.json")

    dataset_dirs = [str(v) for v in cfg.data.train.get("dataset_dirs", [])]
    if any("libero" in v for v in dataset_dirs):
        candidates.append(project_root / "checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json")
    elif any("robotwin" in v for v in dataset_dirs):
        candidates.append(project_root / "checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved
    raise FileNotFoundError(
        "Failed to locate dataset_stats.json. Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )


def _resolve_task_ids(task_suite, cfg: DictConfig) -> list[int]:
    n_tasks = int(task_suite.n_tasks)
    explicit = cfg.EVALUATION.get("collect_task_ids", None)
    if explicit is not None:
        if isinstance(explicit, str):
            tokens = [token.strip() for token in explicit.split(",") if token.strip()]
            task_ids = [int(token) for token in tokens]
        else:
            task_ids = [int(v) for v in list(explicit)]
    else:
        if bool(cfg.EVALUATION.get("collect_all_tasks", True)):
            task_ids = list(range(n_tasks))
        else:
            task_ids = [int(cfg.EVALUATION.task_id)]
    task_ids = sorted(set(task_ids))
    for task_id in task_ids:
        if task_id < 0 or task_id >= n_tasks:
            raise ValueError(f"Task id {task_id} out of range [0, {n_tasks - 1}] for suite.")
    return task_ids


def _summarize_int_samples(samples: list[int]) -> dict[str, float | int | None]:
    if len(samples) == 0:
        return {
            "count": 0,
            "avg": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _safe_close_env(env) -> None:
    if hasattr(env, "close"):
        try:
            env.close()
        except Exception:
            logging.exception("Failed to close LIBERO env cleanly.")


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg
    render_gpu_device_id = _configure_egl_device(cfg)

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    async_video_device, async_action_device = _resolve_async_runtime_devices(cfg, model_device)
    action_model: Optional[torch.nn.Module] = None
    if async_action_device != async_video_device:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)
        action_model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_action_device)
    else:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task_ids = _resolve_task_ids(task_suite, cfg)
    num_trials = int(cfg.EVALUATION.num_trials)
    mode_prob_threshold = float(cfg.EVALUATION.get("collect_mode_prob_threshold", 0.0))

    step_mode_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    step_frontier_samples: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    step_latest_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    step_older_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    step_age_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    total_episodes = 0
    task_episode_counts: dict[int, int] = defaultdict(int)
    task_successes: dict[int, int] = defaultdict(int)
    resolved_action_model = model if action_model is None else action_model

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = list(task_suite.get_task_init_states(task_id))
        while len(initial_states) < num_trials:
            initial_states.extend(initial_states[: (num_trials - len(initial_states))])
        env, task_description = get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            cfg.get("seed"),
            render_gpu_device_id=render_gpu_device_id,
        )
        try:
            for episode_idx in range(num_trials):
                success, _, runtime_summary = run_single_episode_async(
                    env=env,
                    initial_state=initial_states[episode_idx],
                    task_description=task_description,
                    video_model=model,
                    action_model=resolved_action_model,
                    processor=processor,
                    cfg=cfg,
                    episode_idx=episode_idx,
                    action_horizon=action_horizon,
                    input_w=input_w,
                    input_h=input_h,
                    collect_replay=False,
                )
                total_episodes += 1
                task_episode_counts[int(task_id)] += 1
                if success:
                    task_successes[int(task_id)] += 1

                layer_source_stats = runtime_summary.get("layer_source_stats", {})
                for row in layer_source_stats.get("per_step", []):
                    denoise_step = int(row["denoise_step"])
                    for mode, count in dict(row.get("mode_counts", {})).items():
                        step_mode_counts[denoise_step][str(mode)] += int(count)
                    for mode, values in dict(row.get("frontier_samples_by_mode", {})).items():
                        step_frontier_samples[denoise_step][str(mode)].extend(int(v) for v in list(values))
                    for key, count in dict(row.get("latest_offset_counts", {})).items():
                        step_latest_offset_counts[denoise_step][int(key)] += int(count)
                    for key, count in dict(row.get("older_offset_counts", {})).items():
                        step_older_offset_counts[denoise_step][int(key)] += int(count)
                    for key, count in dict(row.get("age_counts", {})).items():
                        step_age_counts[denoise_step][str(key)] += int(count)
        finally:
            _safe_close_env(env)

    per_step: list[dict[str, Any]] = []
    suggested_distribution: dict[str, list[dict[str, Any]]] = {}
    for denoise_step in sorted(step_mode_counts.keys()):
        mode_counts = dict(step_mode_counts[denoise_step])
        samples = int(sum(mode_counts.values()))
        if samples <= 0:
            continue
        mode_probs = {mode: float(count) / float(samples) for mode, count in mode_counts.items()}
        frontier_samples_by_mode = {
            mode: [int(v) for v in values]
            for mode, values in dict(step_frontier_samples[denoise_step]).items()
        }
        frontier_stats_by_mode = {
            mode: _summarize_int_samples(values)
            for mode, values in frontier_samples_by_mode.items()
        }
        latest_offset_counts = {
            str(key): int(value)
            for key, value in sorted(dict(step_latest_offset_counts[denoise_step]).items(), key=lambda kv: kv[0])
        }
        latest_offset_probs = {
            key: float(value) / float(samples)
            for key, value in latest_offset_counts.items()
        }
        older_offset_counts_int = dict(step_older_offset_counts[denoise_step])
        older_total = int(sum(older_offset_counts_int.values()))
        older_offset_counts = {
            str(key): int(value)
            for key, value in sorted(older_offset_counts_int.items(), key=lambda kv: kv[0])
        }
        if older_total > 0:
            older_offset_probs = {
                key: float(value) / float(older_total)
                for key, value in older_offset_counts.items()
            }
        else:
            older_offset_probs = {}
        age_counts = dict(step_age_counts[denoise_step])
        age_total = int(sum(age_counts.values()))
        if age_total > 0:
            age_probs = {key: float(value) / float(age_total) for key, value in age_counts.items()}
        else:
            age_probs = {key: 0.0 for key in age_counts.keys()}

        per_step.append(
            {
                "denoise_step": int(denoise_step),
                "samples": int(samples),
                "mode_counts": mode_counts,
                "mode_probs": mode_probs,
                "frontier_samples_by_mode": frontier_samples_by_mode,
                "frontier_stats_by_mode": frontier_stats_by_mode,
                "latest_offset_counts": latest_offset_counts,
                "latest_offset_probs": latest_offset_probs,
                "older_offset_counts": older_offset_counts,
                "older_offset_probs": older_offset_probs,
                "age_counts": age_counts,
                "age_probs": age_probs,
            }
        )

        entries: list[dict[str, Any]] = []
        sorted_modes = sorted(mode_probs.items(), key=lambda kv: -kv[1])
        for mode, prob in sorted_modes:
            if prob < mode_prob_threshold:
                continue
            entry: dict[str, Any] = {"mode": str(mode), "prob": float(prob)}
            if "_to_" in mode:
                stats = frontier_stats_by_mode.get(mode, {})
                if stats.get("count", 0) > 0:
                    p10 = int(math.floor(float(stats["p10"])))
                    p90 = int(math.ceil(float(stats["p90"])))
                    entry["frontier_min"] = int(max(0, p10))
                    entry["frontier_max"] = int(max(entry["frontier_min"], p90))
            entries.append(entry)
        if len(entries) == 0:
            mode, prob = sorted_modes[0]
            entries.append({"mode": str(mode), "prob": float(prob)})
        suggested_distribution[str(int(denoise_step))] = entries

    task_summary = {
        str(task_id): {
            "episodes": int(task_episode_counts[int(task_id)]),
            "successes": int(task_successes[int(task_id)]),
        }
        for task_id in task_ids
    }
    output = {
        "task_suite": str(cfg.EVALUATION.task_suite_name),
        "task_ids": [int(v) for v in task_ids],
        "num_trials_per_task": int(num_trials),
        "total_episodes": int(total_episodes),
        "mode_prob_threshold": float(mode_prob_threshold),
        "task_summary": task_summary,
        "layer_source_distribution": {
            "enabled": True,
            "per_step": per_step,
        },
        "suggested_training_distribution": suggested_distribution,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": float(time.time() - start_time),
    }

    output_dir = Path(cfg.EVALUATION.output_dir) / str(cfg.EVALUATION.task_suite_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = cfg.EVALUATION.get("collect_output_file", None)
    if output_name is None:
        output_name = f"gpu{int(cfg.gpu_id)}_layer_distribution.json"
    output_path = output_dir / str(output_name)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)

    print(
        f"Layer distribution collection done | suite={cfg.EVALUATION.task_suite_name} "
        f"tasks={len(task_ids)} episodes={total_episodes} output={output_path}"
    )
    return output


if __name__ == "__main__":
    main()
