from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

project_root = Path(__file__).resolve().parents[2]

from experiments.libero.eval_libero_policy_utils import (  # noqa: E402
    _obs_to_model_input,
    _postprocess_libero_action_chunk,
)
from experiments.libero.eval_libero_rollout import _step_env_with_min_dt  # noqa: E402
from experiments.libero.libero_utils import LIBERO_ENV_RESOLUTION, get_libero_env  # noqa: E402
from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # noqa: E402
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from fastwam.utils.async_streaming_runtime import ProfiledRuntime  # noqa: E402
from fastwam.utils.async_streaming_runner import AsyncStreamingRunner  # noqa: E402


def clone_obs_dict(obs_dict: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(obs_dict)


def step_env_compat(env, action, *, min_step_dt_s: float) -> tuple[dict, float, bool, dict]:
    out = _step_env_with_min_dt(env, action, min_step_dt_s=min_step_dt_s)
    if len(out) == 4:
        obs, reward, done, info = out
        return obs, reward, done, info
    if len(out) == 5:
        obs, reward, done, info, _elapsed_ms = out
        return obs, reward, done, info
    raise ValueError(f"Unexpected _step_env_with_min_dt return length: {len(out)}")


def collect_episode_obs_trace(
    *,
    task,
    initial_state: Any,
    cfg: DictConfig,
    gt_actions_env: np.ndarray,
    render_gpu_device_id: int,
) -> tuple[list[dict[str, Any]], str]:
    print(
        f"[anchor-debug] collect_episode_obs_trace start total_steps={int(gt_actions_env.shape[0])}",
        flush=True,
    )
    env, task_description = get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        cfg.get("seed"),
        render_gpu_device_id=render_gpu_device_id,
    )
    try:
        env.reset()
        obs = env.set_init_state(initial_state)
        obs_trace: list[dict[str, Any]] = [clone_obs_dict(obs)]
        for env_step in range(int(gt_actions_env.shape[0])):
            if env_step == 0 or (env_step + 1) % 100 == 0:
                print(
                    f"[anchor-debug] collect_episode_obs_trace progress step={env_step + 1}/{int(gt_actions_env.shape[0])}",
                    flush=True,
                )
            obs, _, done, _ = step_env_compat(
                env,
                np.asarray(gt_actions_env[env_step], dtype=np.float32).tolist(),
                min_step_dt_s=0.0,
            )
            obs_trace.append(clone_obs_dict(obs))
            if done:
                print(
                    f"[anchor-debug] collect_episode_obs_trace env_done step={env_step + 1} collected={len(obs_trace)}",
                    flush=True,
                )
                break
        print(
            f"[anchor-debug] collect_episode_obs_trace done collected={len(obs_trace)}",
            flush=True,
        )
        return obs_trace, str(task_description)
    finally:
        if hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass


def encode_obs_cpu(
    obs_dict: dict,
    *,
    cfg: DictConfig,
    processor: FastWAMProcessor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    return _obs_to_model_input(
        obs_dict,
        cfg=cfg,
        processor=processor,
        width=width,
        height=height,
        device="cpu",
        dtype=torch.float32,
    )


SubmissionSelector = Callable[[str, int, dict[str, Any], list[dict[str, Any]], int], tuple[dict[str, Any], dict[str, Any]]]


def _assert_single_formal_job_runtime_stats(*, runtime_stats: dict[str, Any], anchor_step: int) -> None:
    submitted_jobs = int(runtime_stats.get("submitted_jobs", 0))
    completed_jobs = int(runtime_stats.get("completed_jobs", 0))
    job_records = list(runtime_stats.get("job_records", []))
    if submitted_jobs != 1 or completed_jobs != 1 or len(job_records) != 1:
        raise RuntimeError(
            "Single-chunk anchor evaluation expected exactly one formal action job, "
            f"but anchor_step={anchor_step} saw submitted_jobs={submitted_jobs}, "
            f"completed_jobs={completed_jobs}, job_records={len(job_records)}."
        )
    layer_source_stats = dict(runtime_stats.get("layer_source_stats", {}))
    bad_steps: list[tuple[int, int]] = []
    for step in list(layer_source_stats.get("per_step", [])):
        denoise_step = int(step.get("denoise_step", -1))
        samples = int(step.get("samples", 0))
        if samples != 1:
            bad_steps.append((denoise_step, samples))
    if bad_steps:
        bad_steps_str = ", ".join(f"step={step}:samples={samples}" for step, samples in bad_steps)
        raise RuntimeError(
            "Single-chunk anchor evaluation expected one sample per denoise step, "
            f"but anchor_step={anchor_step} saw {bad_steps_str}."
        )


def run_native_async_anchor_chunk(
    *,
    task,
    initial_state: Any,
    model: torch.nn.Module,
    action_model: Optional[torch.nn.Module] = None,
    processor: FastWAMProcessor,
    cfg: DictConfig,
    action_horizon: int,
    input_w: int,
    input_h: int,
    render_gpu_device_id: int,
    gt_actions_env: np.ndarray,
    anchor_step: int,
    submit_selector: SubmissionSelector,
    seed: int | None,
    submit_anchor_if_aligned: bool = True,
    episode_obs_trace: Optional[list[dict[str, Any]]] = None,
    task_description_override: str | None = None,
) -> dict[str, Any]:
    print(
        f"[anchor-debug] run_native_async_anchor_chunk start anchor_step={int(anchor_step)} use_precollected_trace={episode_obs_trace is not None}",
        flush=True,
    )
    if not hasattr(model, "reset_streaming_state"):
        raise ValueError("Streaming evaluation requires a FastWAMStreaming-style model.")
    resolved_action_model = model if action_model is None else action_model
    if not hasattr(resolved_action_model, "reset_streaming_state"):
        raise ValueError("Streaming evaluation requires a FastWAMStreaming-style action model.")
    model.reset_streaming_state()
    if resolved_action_model is not model:
        resolved_action_model.reset_streaming_state()

    obs_stride_env_steps = int(cfg.EVALUATION.get("async_obs_stride_env_steps", 3))
    trigger_every_n_obs = int(cfg.EVALUATION.get("async_action_trigger_every_n_obs", 3))
    video_layers_per_chunk = int(cfg.EVALUATION.get("async_video_layers_per_chunk", 2))
    force_first_job = bool(cfg.EVALUATION.get("async_force_first_job", True))
    control_dt_ms = float(cfg.EVALUATION.get("async_control_dt_ms", 50.0))
    min_step_dt_s = 0.0
    warmup_action_jobs = int(cfg.EVALUATION.get("async_warmup_action_jobs", 0))
    if warmup_action_jobs < 0:
        raise ValueError(f"`async_warmup_action_jobs` must be >= 0, got {warmup_action_jobs}.")
    num_inference_steps = int(cfg.EVALUATION.get("num_inference_steps", cfg.get("eval_num_inference_steps", 10)))
    sigma_shift = None if cfg.EVALUATION.get("sigma_shift") is None else float(cfg.EVALUATION.get("sigma_shift"))
    rand_device = str(cfg.EVALUATION.get("rand_device", "cpu"))
    tiled = bool(cfg.EVALUATION.get("tiled", False))
    prompt_seed = None if seed is None else int(seed)

    env = None
    task_description = str(task_description_override) if task_description_override is not None else str(task.language)
    if episode_obs_trace is None:
        env, task_description = get_libero_env(
            task,
            LIBERO_ENV_RESOLUTION,
            cfg.get("seed"),
            render_gpu_device_id=render_gpu_device_id,
        )
    elif len(episode_obs_trace) <= int(anchor_step):
        raise ValueError(
            f"episode_obs_trace length {len(episode_obs_trace)} is too short for anchor_step={anchor_step}."
        )
    runtime: Optional[ProfiledRuntime] = None
    try:
        if env is not None:
            env.reset()
            obs = env.set_init_state(initial_state)
        else:
            obs = clone_obs_dict(episode_obs_trace[0])
        obs_history: list[dict[str, Any]] = [clone_obs_dict(obs)]

        prompt = DEFAULT_PROMPT.format(task=task_description)
        with torch.no_grad():
            video_context, video_context_mask = model.encode_prompt(prompt)
        video_context = video_context.to(device="cpu", dtype=model.torch_dtype)
        video_context_mask = video_context_mask.to(device="cpu", dtype=torch.bool)
        action_postprocess = lambda x: _postprocess_libero_action_chunk(x, processor=processor, cfg=cfg)
        runtime = ProfiledRuntime(
            video_model=model,
            action_model=resolved_action_model,
            video_context=video_context,
            video_context_mask=video_context_mask,
            action_context=video_context,
            action_context_mask=video_context_mask,
            action_postprocess=action_postprocess,
            action_horizon=int(action_horizon),
            num_inference_steps=int(num_inference_steps),
            sigma_shift=sigma_shift,
            rand_device=rand_device,
            tiled=tiled,
            action_trigger_every_n_obs=int(trigger_every_n_obs),
            video_layers_per_chunk=int(video_layers_per_chunk),
            seed=prompt_seed,
        )
        runtime.start()
        print(
            f"[anchor-debug] runtime started anchor_step={int(anchor_step)}",
            flush=True,
        )

        bootstrap_obs, bootstrap_meta = submit_selector(
            "bootstrap",
            0,
            clone_obs_dict(obs),
            obs_history,
            -1,
        )
        bootstrap_image, _, _ = encode_obs_cpu(
            bootstrap_obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
        )
        runtime.bootstrap_sync(
            input_image=bootstrap_image,
            obs_index=0,
            obs_timestamp_ms=0.0,
        )
        runtime.wait_until_idle()
        print(
            f"[anchor-debug] bootstrap complete anchor_step={int(anchor_step)}",
            flush=True,
        )
        runtime.reset_for_formal_phase(env_step=0)
        runner = AsyncStreamingRunner(
            runtime=runtime,
            obs_stride_env_steps=obs_stride_env_steps,
            control_dt_ms=control_dt_ms,
            force_first_job=force_first_job,
        )

        submission_trace: list[dict[str, Any]] = [
            {
                "submission_kind": "bootstrap",
                "obs_index": 0,
                "env_step": 0,
                **bootstrap_meta,
            }
        ]

        current_obs_index = 0
        last_submitted_obs_index = -1

        if warmup_action_jobs > 0:
            print(
                f"[anchor-debug] warmup begin anchor_step={int(anchor_step)} warmup_action_jobs={int(warmup_action_jobs)}",
                flush=True,
            )
            warmup_span = max(1, warmup_action_jobs) * max(1, action_horizon)
            warmup_start = -int(warmup_span)
            current_obs_index = runner.run_warmup(
                input_image=bootstrap_image,
                proprio=None,
                warmup_action_jobs=warmup_action_jobs,
                start_env_step=warmup_start,
                start_obs_index=warmup_start,
            )
            runtime.reset_for_formal_phase(env_step=0)
            print(
                f"[anchor-debug] warmup done anchor_step={int(anchor_step)} current_obs_index={int(current_obs_index)}",
                flush=True,
            )

        def _submit_selected(
            *,
            phase: str,
            env_step: int,
            current_obs: dict[str, Any],
        ) -> None:
            nonlocal current_obs_index, last_submitted_obs_index, submission_trace
            submit_obs, meta = submit_selector(
                phase,
                int(env_step),
                clone_obs_dict(current_obs),
                obs_history,
                int(current_obs_index),
            )
            image_cpu, _, _ = encode_obs_cpu(
                submit_obs,
                cfg=cfg,
                processor=processor,
                width=input_w,
                height=input_h,
            )
            runtime.submit_observation(
                input_image=image_cpu,
                proprio=None,
                env_step=int(env_step),
                obs_index=int(current_obs_index),
                obs_timestamp_ms=float(current_obs_index) * float(obs_stride_env_steps) * float(control_dt_ms),
                trigger_job=False,
            )
            submission_trace.append(
                {
                    "submission_kind": str(phase),
                    "obs_index": int(current_obs_index),
                    "env_step": int(env_step),
                    **meta,
                }
            )
            last_submitted_obs_index = int(current_obs_index)
            current_obs_index += 1

        for env_step in range(int(anchor_step)):
            if env_step % obs_stride_env_steps == 0:
                _submit_selected(
                    phase="scheduled_pre_trigger",
                    env_step=int(env_step),
                    current_obs=obs,
                )
            if env is not None:
                obs, _, done, _ = step_env_compat(
                    env,
                    np.asarray(gt_actions_env[env_step], dtype=np.float32).tolist(),
                    min_step_dt_s=min_step_dt_s,
                )
                if done:
                    raise RuntimeError(f"Environment terminated before reaching anchor step {anchor_step}.")
            else:
                obs = clone_obs_dict(episode_obs_trace[env_step + 1])
            obs_history.append(clone_obs_dict(obs))
        print(
            f"[anchor-debug] replay_to_anchor complete anchor_step={int(anchor_step)} submitted_obs={int(last_submitted_obs_index + 1)}",
            flush=True,
        )

        if bool(submit_anchor_if_aligned) and int(anchor_step) % obs_stride_env_steps == 0:
            _submit_selected(
                phase="anchor_submit",
                env_step=int(anchor_step),
                current_obs=obs,
            )
        print(
            f"[anchor-debug] anchor_submit complete anchor_step={int(anchor_step)} last_submitted_obs_index={int(last_submitted_obs_index)}",
            flush=True,
        )

        _, trigger_proprio, _ = encode_obs_cpu(
            obs,
            cfg=cfg,
            processor=processor,
            width=input_w,
            height=input_h,
        )
        runtime.submit_action_job(
            env_step=int(anchor_step),
            proprio=trigger_proprio,
            obs_index=int(last_submitted_obs_index),
        )
        print(
            f"[anchor-debug] action_job submitted anchor_step={int(anchor_step)} trigger_obs_index={int(last_submitted_obs_index)}",
            flush=True,
        )

        future_env_step = int(anchor_step)
        max_future_steps = (
            int(gt_actions_env.shape[0])
            if episode_obs_trace is None
            else min(int(gt_actions_env.shape[0]), int(len(episode_obs_trace) - 1))
        )
        while runtime.completed_jobs() < 1 and future_env_step < max_future_steps:
            if env is not None:
                obs, _, done, _ = step_env_compat(
                    env,
                    np.asarray(gt_actions_env[future_env_step], dtype=np.float32).tolist(),
                    min_step_dt_s=min_step_dt_s,
                )
            else:
                obs = clone_obs_dict(episode_obs_trace[future_env_step + 1])
                done = False
            future_env_step += 1
            obs_history.append(clone_obs_dict(obs))
            if done:
                break
            if future_env_step % obs_stride_env_steps == 0:
                _submit_selected(
                    phase="scheduled_post_trigger",
                    env_step=int(future_env_step),
                    current_obs=obs,
                )
            if future_env_step == int(anchor_step) + 1 or future_env_step % 50 == 0:
                print(
                    f"[anchor-debug] waiting_job anchor_step={int(anchor_step)} future_env_step={int(future_env_step)} completed_jobs={int(runtime.completed_jobs())}",
                    flush=True,
                )

        print(
            f"[anchor-debug] wait_until_idle begin anchor_step={int(anchor_step)} completed_jobs={int(runtime.completed_jobs())}",
            flush=True,
        )
        runtime.wait_until_idle()
        print(
            f"[anchor-debug] wait_until_idle done anchor_step={int(anchor_step)} completed_jobs={int(runtime.completed_jobs())}",
            flush=True,
        )
        runtime_stats = runtime.stats()
        _assert_single_formal_job_runtime_stats(runtime_stats=runtime_stats, anchor_step=int(anchor_step))

        action_rows: list[np.ndarray] = []
        for step in range(int(action_horizon)):
            action = runtime.get_action(int(anchor_step) + step, count_miss=False)
            if action is None:
                break
            action_rows.append(np.asarray(action, dtype=np.float32))
        if len(action_rows) == 0:
            raise RuntimeError(f"No action chunk became available for anchor step {anchor_step}.")
        action_chunk = np.stack(action_rows, axis=0)
        print(
            f"[anchor-debug] run_native_async_anchor_chunk done anchor_step={int(anchor_step)} action_chunk_len={int(action_chunk.shape[0])}",
            flush=True,
        )
        return {
            "task_description": task_description,
            "action_chunk": np.asarray(action_chunk, dtype=np.float32),
            "trigger_obs_index": int(last_submitted_obs_index),
            "submission_trace": submission_trace,
            "runtime_stats": runtime_stats,
        }
    finally:
        if runtime is not None:
            runtime.stop()
        if env is not None and hasattr(env, "close"):
            try:
                env.close()
            except Exception:
                pass
