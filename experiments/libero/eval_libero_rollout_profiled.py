from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from experiments.libero.eval_libero_policy_utils import _obs_to_model_input, _postprocess_libero_action_chunk
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env, save_rollout_video
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.utils.async_streaming_runtime import StreamingRuntime
from fastwam.utils.async_streaming_runner import AsyncStreamingRunner


def _summarize_ms(samples_ms: list[float]) -> dict[str, float | int | None]:
    if len(samples_ms) == 0:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(np.max(arr)),
    }


def _step_env_with_min_dt(env, action, *, min_step_dt_s: float) -> tuple[dict, float, bool, dict, float]:
    t0 = time.perf_counter()
    obs, reward, done, info = env.step(action)
    elapsed_s = time.perf_counter() - t0
    if min_step_dt_s > 0.0 and elapsed_s < min_step_dt_s:
        time.sleep(min_step_dt_s - elapsed_s)
        elapsed_s = min_step_dt_s
    return obs, reward, done, info, elapsed_s * 1000.0


def _get_max_steps(task_suite_name: str) -> int:
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name not in suite_steps:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return suite_steps[task_suite_name]


def _summarize_async_runtime_episodes(episodes: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    
    if len(episodes) == 0:
        return None

    scalar_keys = [
        "submitted_obs",
        "submitted_jobs",
        "completed_jobs",
        "actions_served",
        "actions_missed",
        "dropped_prefix_actions",
    ]
    summary: dict[str, Any] = {"num_episodes": int(len(episodes))}
    for key in scalar_keys:
        values = [float(ep[key]) for ep in episodes]
        summary[f"{key}_total"] = float(np.sum(values))
        summary[f"{key}_mean"] = float(np.mean(values))

    timing_keys = ["video_refresh", "action_job", "action_job_wall", "action_step", "snapshot_copy", "env_step"]
    timing_summary: dict[str, Any] = {}
    for key in timing_keys:
        counts = [int(ep.get("timing_ms", {}).get(key, {}).get("count", 0)) for ep in episodes]
        avg_values = [ep.get("timing_ms", {}).get(key, {}).get("avg_ms") for ep in episodes]
        weighted_sum = 0.0
        total_count = 0
        for count, avg_value in zip(counts, avg_values):
            if avg_value is None or count <= 0:
                continue
            weighted_sum += float(avg_value) * int(count)
            total_count += int(count)
        timing_summary[key] = {
            "count_total": int(total_count),
            "avg_ms": (None if total_count == 0 else float(weighted_sum / total_count)),
        }
    summary["timing_ms"] = timing_summary
    return summary


def run_single_episode_async(
    env,
    initial_state,
    task_description: str,
    video_model: torch.nn.Module,
    action_model: torch.nn.Module,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    collect_replay: bool = True,
) -> tuple[bool, list, dict[str, Any]]:
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        raise ValueError("Async LIBERO rollout does not support visualize_future_video=true.")

    if not hasattr(video_model, "start_action_job") or not hasattr(action_model, "start_action_job"):
        raise ValueError("Async LIBERO rollout requires a FastWAMStreaming-style model.")

    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    obs_stride_env_steps = int(cfg.EVALUATION.get("async_obs_stride_env_steps", 3))
    trigger_every_n_obs = int(cfg.EVALUATION.get("async_action_trigger_every_n_obs", 3))
    video_layers_per_chunk = int(cfg.EVALUATION.get("async_video_layers_per_chunk", 2))
    force_first_job = bool(cfg.EVALUATION.get("async_force_first_job", True))
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    min_step_dt_s = max(0.0, control_dt_ms / 1000.0)
    warmup_action_jobs = int(cfg.EVALUATION.get("async_warmup_action_jobs", 0))
    if warmup_action_jobs < 0:
        raise ValueError(f"`async_warmup_action_jobs` must be >= 0, got {warmup_action_jobs}.")

    prompt = DEFAULT_PROMPT.format(task=task_description)
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

    action_postprocess = lambda x: _postprocess_libero_action_chunk(x, processor=processor, cfg=cfg)
    runtime = StreamingRuntime(
        video_model=video_model,
        action_model=action_model,
        video_context=video_context,
        video_context_mask=video_context_mask,
        action_context=action_context,
        action_context_mask=action_context_mask,
        action_postprocess=action_postprocess,
        action_horizon=action_horizon,
        num_inference_steps=int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10))),
        sigma_shift=(None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))),
        rand_device=str(cfg.EVALUATION.get("rand_device", "cpu")),
        tiled=bool(cfg.EVALUATION.get("tiled", False)),
        action_trigger_every_n_obs=trigger_every_n_obs,
        video_layers_per_chunk=video_layers_per_chunk,
        seed=(None if cfg.get("seed") is None else int(cfg.seed)),
        profile=True,
    )

    replay_images = []
    runtime_started = False
    done = False
    env_step_samples_ms: list[float] = []

    env.reset()
    obs = env.set_init_state(initial_state)
    try:
        runtime.start()
        runtime_started = True

        def _encode_obs(obs_dict: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
            return _obs_to_model_input(
                obs_dict,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
                device="cpu",
                dtype=torch.float32,
            )

        image, proprio, imgs = _encode_obs(obs)
        runtime.bootstrap_sync(input_image=image, obs_index=0, obs_timestamp_ms=0.0)
        runtime.wait_until_idle()
        runtime.reset_for_formal_phase(env_step=0)
        runner = AsyncStreamingRunner(
            runtime=runtime,
            obs_stride_env_steps=obs_stride_env_steps,
            control_dt_ms=control_dt_ms,
            force_first_job=force_first_job,
        )
        formal_obs_index_start = 0
        if warmup_action_jobs > 0:
            warmup_span = max(1, warmup_action_jobs) * max(1, action_horizon)
            warmup_start = -int(warmup_span)
            formal_obs_index_start = runner.run_warmup(
                input_image=image,
                proprio=proprio,
                warmup_action_jobs=warmup_action_jobs,
                start_env_step=warmup_start,
                start_obs_index=warmup_start,
            )
            runtime.reset_for_formal_phase(env_step=0)
        runner.start_formal_phase(obs_index_start=formal_obs_index_start)

        t = 0
        executed_env_steps = 0
        pbar = tqdm(total=max_steps, desc=f"Episode {episode_idx + 1} (async)")
        while executed_env_steps < max_steps:
            runner.maybe_submit_formal_observation(
                input_image=image,
                proprio=proprio,
                env_step=t,
            )
            action = runtime.get_action(t)
            if collect_replay:
                replay_images.append(imgs.copy())
            if action is None:
                action = runner.wait_for_action(env_step=t, proprio=proprio)

            obs, _, done, _, elapsed_ms = _step_env_with_min_dt(
                env,
                action.tolist(),
                min_step_dt_s=min_step_dt_s,
            )
            env_step_samples_ms.append(float(elapsed_ms))
            executed_env_steps += 1
            pbar.update(1)
            if done:
                break
            t += 1
            image, proprio, imgs = _encode_obs(obs)
        pbar.close()
    finally:
        if runtime_started:
            runtime.stop()

    runtime_summary = runtime.stats()
    runtime_summary.setdefault("timing_ms", {})
    runtime_summary["timing_ms"]["env_step"] = _summarize_ms(env_step_samples_ms)
    runtime_summary.setdefault("timing_samples_ms", {})
    runtime_summary["timing_samples_ms"]["env_step"] = [float(v) for v in env_step_samples_ms]
    logging.info(
        "Async runtime stats | episode=%s submitted_obs=%s submitted_jobs=%s completed_jobs=%s "
        "actions_served=%s actions_missed=%s dropped_prefix_actions=%s",
        episode_idx,
        runtime_summary["submitted_obs"],
        runtime_summary["submitted_jobs"],
        runtime_summary["completed_jobs"],
        runtime_summary["actions_served"],
        runtime_summary["actions_missed"],
        runtime_summary["dropped_prefix_actions"],
    )
    return bool(done), replay_images, runtime_summary


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    action_model: Optional[torch.nn.Module],
    processor: FastWAMProcessor,
    cfg: DictConfig,
    video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    action_device: str,
    render_gpu_device_id: int,
) -> dict:
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        render_gpu_device_id=render_gpu_device_id,
    )
    resolved_action_model = model if action_model is None else action_model
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
        "action_horizon": int(action_horizon),
        "async_runtime_episodes": [],
        "async_runtime_summary": None,
        "async_video_device": str(model_device),
        "async_action_device": str(action_device),
    }
    try:
        for trial_idx in range(int(cfg.EVALUATION.num_trials)):
            save_video = bool(cfg.EVALUATION.get("save_video", True))
            success, replay_images, runtime_summary = run_single_episode_async(
                env=env,
                initial_state=initial_states[trial_idx],
                task_description=task_description,
                video_model=model,
                action_model=resolved_action_model,
                processor=processor,
                cfg=cfg,
                episode_idx=trial_idx,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                collect_replay=save_video,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)

            results["async_runtime_episodes"].append(
                {
                    "episode_idx": int(trial_idx),
                    "success": bool(success),
                    **runtime_summary,
                }
            )

            if save_video:
                save_rollout_video(
                    video_dir,
                    replay_images,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    success=success,
                    task_description=task_description,
                )
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                logging.exception("Failed to close LIBERO env cleanly.")

    results["async_runtime_summary"] = _summarize_async_runtime_episodes(results["async_runtime_episodes"])
    return results
