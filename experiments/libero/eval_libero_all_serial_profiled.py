import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import hydra
import torch
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.eval_libero_rollout_profiled import run_single_task
from experiments.libero.eval_libero_single_profiled import (
    NumpyEncoder,
    _build_eval_model,
    _configure_egl_device,
    _mixed_precision_to_model_dtype,
    _resolve_async_runtime_devices,
    _resolve_dataset_stats_path,
    _resolve_eval_device,
    _select_initial_states,
    _validate_visualize_future_video_cfg,
)
from experiments.libero.summarize_results import summarize_results
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from fastwam.utils.pytorch_utils import set_global_seed
from libero.libero import benchmark


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_all_serial(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg
    render_gpu_device_id = _configure_egl_device(cfg)

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError("Only env_num=1 is supported in eval_libero_all_serial_profiled.py.")

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    async_video_device, async_action_device = _resolve_async_runtime_devices(cfg, model_device)

    action_model: Optional[torch.nn.Module] = None
    if async_action_device != async_video_device:
        logging.info(
            "Async dual-device runtime enabled: video_device=%s action_device=%s",
            async_video_device,
            async_action_device,
        )
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)
        action_model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_action_device)
    else:
        model = _build_eval_model(cfg, model_dtype=model_dtype, device=async_video_device)

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FastWAMProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])

    output_root = Path(cfg.EVALUATION.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    suite_names_cfg = cfg.EVALUATION.get("task_suite_names")
    if suite_names_cfg is not None:
        task_suite_names = [str(x) for x in suite_names_cfg]
    else:
        task_suite_names = [str(x) for x in cfg.MULTIRUN.get("task_suite_names", [cfg.EVALUATION.task_suite_name])]
    if len(task_suite_names) == 0:
        raise ValueError("No task suites provided.")

    benchmark_dict = benchmark.get_benchmark_dict()
    total_tasks = 0
    total_successes = 0
    total_episodes = 0

    all_tasks: list[tuple[str, int]] = []
    for suite_name in task_suite_names:
        if suite_name not in benchmark_dict:
            raise ValueError(f"Unknown LIBERO suite: {suite_name}")
        task_suite = benchmark_dict[suite_name]()
        n_tasks = int(task_suite.n_tasks)
        logging.info("Running suite %s with %d tasks (serial).", suite_name, n_tasks)
        for task_id in range(n_tasks):
            all_tasks.append((suite_name, task_id))

    if len(all_tasks) == 0:
        raise ValueError("No tasks found in selected suites.")

    pbar = tqdm(total=len(all_tasks), desc="All LIBERO Tasks", dynamic_ncols=True)
    for suite_name in task_suite_names:
        task_suite = benchmark_dict[suite_name]()
        video_dir = output_root / suite_name / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)

        for task_id in range(n_tasks):
            task_start = time.time()
            cfg.EVALUATION.task_suite_name = suite_name
            cfg.EVALUATION.task_id = int(task_id)

            task = task_suite.get_task(task_id)
            initial_states = _select_initial_states(
                task_suite.get_task_init_states(task_id),
                num_trials=int(cfg.EVALUATION.num_trials),
                seed=(None if cfg.get("seed") is None else int(cfg.seed)),
                task_suite_name=str(suite_name),
                task_id=int(task_id),
            )

            task_results = run_single_task(
                task=task,
                initial_states=initial_states,
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

            result = {
                "task_suite": suite_name,
                "task_id": int(task_id),
                "task_description": task_results.get("task_description"),
                "action_horizon": int(action_horizon),
                "ckpt_loaded": bool(cfg.get("ckpt") is not None),
                "successes": int(task_results["successes"]),
                "total_episodes": int(cfg.EVALUATION.num_trials),
                "gpu_id": int(cfg.gpu_id),
                "success_episodes": task_results.get("success_episodes", []),
                "failure_episodes": task_results.get("failure_episodes", []),
                "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration": float(time.time() - task_start),
            }

            for key in ("async_runtime_episodes", "async_runtime_summary", "async_video_device", "async_action_device"):
                if key in task_results:
                    result[key] = task_results[key]

            suite_output_dir = output_root / suite_name
            suite_output_dir.mkdir(parents=True, exist_ok=True)
            output_file = suite_output_dir / f"gpu{cfg.gpu_id}_task{task_id}_results.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, cls=NumpyEncoder)

            total_tasks += 1
            total_successes += int(result["successes"])
            total_episodes += int(result["total_episodes"])
            running_sr = 0.0 if total_episodes == 0 else float(total_successes) / float(total_episodes) * 100.0
            logging.info(
                "[%s task=%d] %d/%d success, duration=%.2fs",
                suite_name,
                task_id,
                result["successes"],
                result["total_episodes"],
                result["duration"],
            )
            pbar.update(1)
            pbar.set_postfix_str(
                f"SR={running_sr:.2f}% ({total_successes}/{total_episodes}), "
                f"last={suite_name}:{task_id}={result['successes']}/{result['total_episodes']}"
            )
    pbar.close()

    summarize_results(str(output_root))

    total_duration = time.time() - start_time
    overall_sr = 0.0 if total_episodes == 0 else float(total_successes) / float(total_episodes) * 100.0
    logging.info(
        "Completed all tasks serially. tasks=%d episodes=%d successes=%d overall_success_rate=%.2f%% total_time=%.2fs",
        total_tasks,
        total_episodes,
        total_successes,
        overall_sr,
        total_duration,
    )
    print(
        f"Done. tasks={total_tasks}, episodes={total_episodes}, successes={total_successes}, "
        f"overall_success_rate={overall_sr:.2f}%, output_dir={output_root}"
    )


if __name__ == "__main__":
    eval_all_serial()
