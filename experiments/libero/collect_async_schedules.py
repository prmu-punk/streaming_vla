import logging
import os
import sys
import time
import atexit
import faulthandler
from pathlib import Path
from typing import Optional

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_rollout_profiled import run_single_task
from experiments.libero.eval_libero_single_profiled import (
    _build_eval_model,
    _configure_egl_device,
    _mixed_precision_to_model_dtype,
    _resolve_async_runtime_devices,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _validate_visualize_future_video_cfg,
)
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from fastwam.utils import async_streaming_runtime as _runtime_mod
from fastwam.utils import async_streaming_workers as _workers_mod
from libero.libero import benchmark

def _register_resolver_if_needed(name: str, fn) -> None:
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, fn)


_register_resolver_if_needed("eval", eval)
_register_resolver_if_needed("max", lambda x: max(x))
_register_resolver_if_needed("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _maybe_enable_worker_faulthandler(worker_name: str) -> None:
    fault_dir_raw = os.environ.get("FASTWAM_FAULT_DIR", "").strip()
    if not fault_dir_raw:
        return
    enable_raw = str(os.environ.get("FASTWAM_CHILD_FAULTHANDLER", "1")).strip().lower()
    if enable_raw in {"0", "false", "no", "n"}:
        return

    handle = None
    try:
        fault_dir = Path(fault_dir_raw)
        fault_dir.mkdir(parents=True, exist_ok=True)
        fault_file = fault_dir / f"{worker_name}_pid{os.getpid()}_faulthandler.log"
        handle = open(fault_file, "a", encoding="utf-8")
        faulthandler.enable(file=handle, all_threads=True)
    except Exception:
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        return

    def _flush_and_close() -> None:
        try:
            handle.flush()
        except Exception:
            pass
        try:
            os.fsync(handle.fileno())
        except Exception:
            pass
        try:
            handle.close()
        except Exception:
            pass

    atexit.register(_flush_and_close)


def _libero_video_worker_loop_with_fault(**kwargs):
    _maybe_enable_worker_faulthandler("streaming_video_worker")
    _workers_mod._video_worker_loop(**kwargs)


def _libero_action_worker_loop_profiled_with_fault(**kwargs):
    _maybe_enable_worker_faulthandler("streaming_action_worker")
    _workers_mod._action_worker_loop_profiled(**kwargs)


_runtime_mod._video_worker_loop = _libero_video_worker_loop_with_fault
_runtime_mod._action_worker_loop_profiled = _libero_action_worker_loop_profiled_with_fault


def _collect_schedules(results: dict) -> list[dict]:
    schedules: list[dict] = []
    for episode in list(results.get("async_runtime_episodes", [])):
        for job in list(episode.get("job_records", [])):
            steps = list(job.get("job_layer_source_steps", []))
            if len(steps) == 0:
                continue
            schedules.append(
                {
                    "task_id": int(results.get("task_id", -1)),
                    "episode_idx": int(episode.get("episode_idx", -1)),
                    "success": bool(episode.get("success", False)),
                    "job_id": int(job.get("job_id", -1)),
                    "trigger_env_step": int(job.get("trigger_env_step", -1)),
                    "trigger_obs_index": int(job.get("trigger_obs_index", -1)),
                    "steps": steps,
                }
            )
    return schedules


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def main(cfg: DictConfig):
    start_time = time.time()
    PartialState().config = cfg
    render_gpu_device_id = _configure_egl_device(cfg)

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)
    _validate_visualize_future_video_cfg(cfg)

    output_path = Path(str(cfg.EVALUATION.get("schedule_output_path", "data/trajectory_replay/libero_async_schedule/schedule.pt")))
    os.environ.setdefault("FASTWAM_FAULT_DIR", str((output_path.parent / "faults").resolve()))
    shard_id = int(cfg.EVALUATION.get("schedule_shard_id", 0))
    num_shards = int(cfg.EVALUATION.get("schedule_num_shards", 1))
    if num_shards <= 0 or shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard settings: shard_id={shard_id}, num_shards={num_shards}")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    async_video_device, async_action_device = _resolve_async_runtime_devices(cfg, model_device)

    action_model: Optional[torch.nn.Module] = None
    if async_action_device != async_video_device:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)
        action_model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_action_device)
    else:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)

    dataset_stats = load_dataset_stats_from_json(str(_resolve_dataset_stats_path(cfg)))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    action_horizon = int(cfg.data.train.num_frames) - 1 if action_horizon_cfg is None else int(action_horizon_cfg)
    video_size = cfg.data.train.get("video_size", [224, 224])
    input_h, input_w = int(video_size[0]), int(video_size[1])

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task_ids = [int(cfg.EVALUATION.task_id)]
    if cfg.EVALUATION.get("task_ids", None) is not None:
        task_ids = [int(v) for v in list(cfg.EVALUATION.task_ids)]

    requested_trials = int(cfg.EVALUATION.num_trials)
    all_schedules: list[dict] = []
    task_summaries: list[dict] = []
    video_dir = output_path.parent / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    for task_id in task_ids:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        while len(initial_states) < requested_trials:
            initial_states.extend(initial_states[: (requested_trials - len(initial_states))])
        shard_states = [
            state
            for trial_idx, state in enumerate(initial_states[:requested_trials])
            if int(trial_idx) % num_shards == shard_id
        ]
        if len(shard_states) == 0:
            continue
        with open_dict(cfg):
            cfg.EVALUATION.task_id = int(task_id)
            cfg.EVALUATION.num_trials = int(len(shard_states))
        results = run_single_task(
            task=task,
            initial_states=shard_states,
            model=model,
            action_model=action_model,
            processor=processor,
            cfg=cfg,
            video_dir=video_dir,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=async_video_device,
            action_device=async_action_device,
            render_gpu_device_id=render_gpu_device_id,
        )
        schedules = _collect_schedules(results)
        all_schedules.extend(schedules)
        task_summaries.append(
            {
                "task_id": int(task_id),
                "successes": int(results.get("successes", 0)),
                "num_trials": int(len(shard_states)),
                "num_schedules": int(len(schedules)),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "meta": {
                "task_suite": str(cfg.EVALUATION.task_suite_name),
                "task_ids": task_ids,
                "num_trials": requested_trials,
                "ckpt": None if cfg.get("ckpt") is None else str(cfg.ckpt),
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_s": float(time.time() - start_time),
                "num_schedules": int(len(all_schedules)),
                "shard_id": int(shard_id),
                "num_shards": int(num_shards),
                "task_summaries": task_summaries,
            },
            "schedules": all_schedules,
        },
        output_path,
    )
    logging.info("Saved %d async schedule traces to %s", len(all_schedules), output_path)
    print(f"Saved {len(all_schedules)} async schedule traces to {output_path}")


if __name__ == "__main__":
    main()
