from __future__ import annotations

import argparse
import json
import pathlib
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from rollout_libero_rtc_async_video import (
    ScheduledExecuteChunk,
    _active_chunk_action,
    _build_state_tensor,
    _drain_execute_packets,
    _has_active_chunk,
    _load_task_init_states,
    _save_video,
    _set_init_state,
    _summary,
    benchmark,
)
from model import Qwen3RTCVLAOnlinePipeline
from oat.oat.env.libero.env import LiberoEnv
from oat.oat.env.libero.env import task_name_to_suite_and_ids


def _resolve_task(task_name: str):
    if task_name not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {task_name}")
    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    return task_suite.get_task(task_id)


def _resolve_benchmark_task_names(benchmark_name: str) -> List[str]:
    benchmark_dict = benchmark.get_benchmark_dict()
    if benchmark_name not in benchmark_dict:
        raise ValueError(
            f"Unknown LIBERO benchmark: {benchmark_name}. "
            f"Available: {sorted(benchmark_dict.keys())}"
        )
    task_suite = benchmark_dict[benchmark_name]()
    return [str(x) for x in task_suite.get_task_names()]


def _resolve_eval_match_ranks(
    *,
    total_init_states: int,
    start_match_rank: int,
    n_eval: Optional[int],
) -> List[int]:
    if total_init_states <= 0:
        return []
    if start_match_rank < 0 or start_match_rank >= total_init_states:
        raise ValueError(
            f"start_match_rank out of range: {start_match_rank}, total_init_states={total_init_states}"
        )
    if n_eval is None or int(n_eval) <= 0:
        end_idx = total_init_states
    else:
        end_idx = min(total_init_states, int(start_match_rank) + int(n_eval))
    return list(range(int(start_match_rank), int(end_idx)))


def _build_video_path(video_root: Optional[str], task_name: str, match_rank: int) -> str:
    base_dir = pathlib.Path(video_root or (pathlib.Path.cwd() / "videos" / "rtc_async_eval"))
    return str(base_dir / task_name / f"match_{int(match_rank):03d}.mp4")


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_async_rollout_optional_video(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    cfg,
    task_name: str,
    match_rank: int,
    num_frames: int,
    source_dt_ms: int,
    observation_gap_steps: Optional[int],
    step_dt_min_ms: int,
    step_dt_max_ms: int,
    max_env_steps: int,
    save_video_path: Optional[str],
    max_wait_s: float,
) -> Dict[str, Any]:
    task = _resolve_task(task_name)
    init_states = _load_task_init_states(task)
    num_init_states = int(init_states.shape[0])
    if num_init_states <= 0:
        raise ValueError(f"Task has no init states: {task.name}")
    if match_rank < 0 or match_rank >= num_init_states:
        raise ValueError(f"match_rank out of range for init states: {match_rank}, total_init_states={num_init_states}")

    init_idx = int(match_rank)
    image_key = str(cfg.dataset.image_key)
    aux_image_key = str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    env = LiberoEnv(
        task_name=task_name,
        image_size=256,
        seed=int(cfg.training.seed),
        camera_names=[
            image_key.replace("_rgb", ""),
            *([aux_image_key.replace("_rgb", "")] if aux_image_key is not None else []),
        ],
        state_ports=state_keys,
        max_episode_steps=max_env_steps,
    )

    video_frames: List[np.ndarray] = []
    scheduled_chunks: deque[ScheduledExecuteChunk] = deque()
    packet_history: List[Dict[str, int]] = []
    sampled_gap_steps: List[int] = []
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            del obs
            obs = _set_init_state(env, init_states[init_idx])
            prompt = str(obs["prompt"])

            pipeline.reset(prompt=prompt)
            pipeline.set_runtime_timebase(source_dt_ms=int(source_dt_ms))
            pipeline.start_async_pipeline()
            if save_video_path is not None:
                video_frames.append(env.render())

            total_env_steps = 0
            done = bool(env.done)
            success = False
            next_obs_step = 0
            control_dt_s = float(source_dt_ms) / 1000.0
            rollout_start = time.perf_counter()
            rng = np.random.default_rng(int(cfg.training.seed))

            while total_env_steps < max_env_steps and not done:
                while total_env_steps >= next_obs_step:
                    state = _build_state_tensor(obs, state_keys, device=pipeline.device)
                    aux_frames = None
                    if aux_image_key is not None and aux_image_key in obs:
                        aux_frames = np.asarray(obs[aux_image_key], dtype=np.uint8)
                    pipeline.push_observation(
                        frames=np.asarray(obs[image_key], dtype=np.uint8),
                        aux_frames=aux_frames,
                        state=state,
                        ts_ms=total_env_steps * source_dt_ms,
                        num_frames=1,
                    )
                    if observation_gap_steps is None:
                        gap_ms = int(rng.integers(int(step_dt_min_ms), int(step_dt_max_ms) + 1))
                        gap_steps = max(1, int(round(float(gap_ms) / float(source_dt_ms))))
                    else:
                        gap_steps = int(observation_gap_steps)
                    sampled_gap_steps.append(int(gap_steps))
                    next_obs_step += int(gap_steps)

                _drain_execute_packets(
                    pipeline=pipeline,
                    scheduled_chunks=scheduled_chunks,
                    packet_history=packet_history,
                    source_dt_ms=source_dt_ms,
                )

                if not _has_active_chunk(scheduled_chunks, current_step=total_env_steps):
                    wait_deadline = time.perf_counter() + float(max_wait_s)
                    while time.perf_counter() < wait_deadline:
                        _drain_execute_packets(
                            pipeline=pipeline,
                            scheduled_chunks=scheduled_chunks,
                            packet_history=packet_history,
                            source_dt_ms=source_dt_ms,
                        )
                        if _has_active_chunk(scheduled_chunks, current_step=total_env_steps):
                            break
                        time.sleep(0.001)
                    if not _has_active_chunk(scheduled_chunks, current_step=total_env_steps):
                        raise RuntimeError(
                            f"No execute chunk became available for current_step={total_env_steps} "
                            f"within max_wait_s={max_wait_s}"
                        )

                target_time = rollout_start + (total_env_steps + 1) * control_dt_s
                _drain_execute_packets(
                    pipeline=pipeline,
                    scheduled_chunks=scheduled_chunks,
                    packet_history=packet_history,
                    source_dt_ms=source_dt_ms,
                )
                sleep_s = target_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)

                _drain_execute_packets(
                    pipeline=pipeline,
                    scheduled_chunks=scheduled_chunks,
                    packet_history=packet_history,
                    source_dt_ms=source_dt_ms,
                )

                action = _active_chunk_action(
                    scheduled_chunks,
                    current_step=total_env_steps,
                )
                obs, reward, done, _, _ = env.step(action)
                total_env_steps += 1
                if save_video_path is not None:
                    video_frames.append(env.render())
                if reward >= 1.0:
                    success = True

            if save_video_path is not None:
                _save_video(save_video_path, video_frames, fps=max(1, round(1000 / source_dt_ms)))

            return {
                "task_name": task.name,
                "task_prompt": task.language,
                "init_state_idx": int(init_idx),
                "source_dt_ms": int(source_dt_ms),
                "observation_gap_steps": (None if observation_gap_steps is None else int(observation_gap_steps)),
                "observation_gap_mode": ("random" if observation_gap_steps is None else "fixed"),
                "sampled_gap_steps_summary": _summary([float(x) for x in sampled_gap_steps]),
                "num_frames": 1,
                "observation_mode": "image",
                "success": bool(success),
                "done": bool(done),
                "total_env_steps": int(total_env_steps),
                "packets_received": int(len(packet_history)),
                "last_packet": (packet_history[-1] if packet_history else None),
                "video_path": (None if save_video_path is None else str(save_video_path)),
            }
    finally:
        pipeline.stop_async_pipeline()
        env.close()


def evaluate_task_success_rate(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    cfg,
    task_name: str,
    num_frames: int,
    source_dt_ms: int,
    observation_gap_steps: Optional[int],
    step_dt_min_ms: int,
    step_dt_max_ms: int,
    max_env_steps: int,
    max_wait_s: float,
    start_match_rank: int = 0,
    n_eval: Optional[int] = None,
    save_videos: bool = False,
    video_root: Optional[str] = None,
    include_episodes: bool = False,
) -> Dict[str, Any]:
    task = _resolve_task(task_name)
    init_states = _load_task_init_states(task)
    match_ranks = _resolve_eval_match_ranks(
        total_init_states=int(init_states.shape[0]),
        start_match_rank=int(start_match_rank),
        n_eval=n_eval,
    )
    if not match_ranks:
        raise ValueError(f"Task has no init states: {task_name}")

    episodes: List[Dict[str, Any]] = []
    ep_progress = tqdm(match_ranks, desc=f"Task {task_name}", dynamic_ncols=True, leave=False)
    for match_rank in ep_progress:
        episode_metrics = run_async_rollout_optional_video(
            pipeline=pipeline,
            cfg=cfg,
            task_name=str(task_name),
            match_rank=int(match_rank),
            num_frames=int(num_frames),
            source_dt_ms=int(source_dt_ms),
            observation_gap_steps=observation_gap_steps,
            step_dt_min_ms=int(step_dt_min_ms),
            step_dt_max_ms=int(step_dt_max_ms),
            max_env_steps=int(max_env_steps),
            save_video_path=(_build_video_path(video_root, str(task_name), int(match_rank)) if save_videos else None),
            max_wait_s=float(max_wait_s),
        )
        episodes.append(episode_metrics)
        success_rate = float(np.mean([float(x["success"]) for x in episodes]))
        ep_progress.set_postfix(success_rate=f"{success_rate:.3f}")

    success_flags = [float(x["success"]) for x in episodes]
    result = {
        "task_name": task.name,
        "task_prompt": task.language,
        "available_init_states": int(init_states.shape[0]),
        "start_match_rank": int(start_match_rank),
        "n_eval": int(len(episodes)),
        "successes": int(sum(success_flags)),
        "success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
        "mean_env_steps": float(np.mean([float(x["total_env_steps"]) for x in episodes])) if episodes else 0.0,
    }
    if include_episodes:
        result["episodes"] = episodes
    return result


def evaluate_benchmark_success_rate(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    cfg,
    benchmark_name: str,
    num_frames: int,
    source_dt_ms: int,
    observation_gap_steps: Optional[int],
    step_dt_min_ms: int,
    step_dt_max_ms: int,
    max_env_steps: int,
    max_wait_s: float,
    start_match_rank: int = 0,
    n_eval_per_task: Optional[int] = None,
    save_videos: bool = False,
    video_root: Optional[str] = None,
    include_episodes: bool = False,
) -> Dict[str, Any]:
    task_names = _resolve_benchmark_task_names(benchmark_name)
    task_results: List[Dict[str, Any]] = []
    total_successes = 0
    total_episodes = 0

    task_progress = tqdm(task_names, desc=f"Benchmark {benchmark_name}", dynamic_ncols=True)
    for task_name in task_progress:
        task_result = evaluate_task_success_rate(
            pipeline=pipeline,
            cfg=cfg,
            task_name=str(task_name),
            num_frames=int(num_frames),
            source_dt_ms=int(source_dt_ms),
            observation_gap_steps=observation_gap_steps,
            step_dt_min_ms=int(step_dt_min_ms),
            step_dt_max_ms=int(step_dt_max_ms),
            max_env_steps=int(max_env_steps),
            max_wait_s=float(max_wait_s),
            start_match_rank=int(start_match_rank),
            n_eval=n_eval_per_task,
            save_videos=save_videos,
            video_root=video_root,
            include_episodes=include_episodes,
        )
        task_results.append(task_result)
        total_successes += int(task_result["successes"])
        total_episodes += int(task_result["n_eval"])
        current_success_rate = float(total_successes / total_episodes) if total_episodes else 0.0
        task_progress.set_postfix(
            tasks_done=f"{len(task_results)}/{len(task_names)}",
            success_rate=f"{current_success_rate:.3f}",
        )

    task_success_rates = [float(x["success_rate"]) for x in task_results]
    return {
        "benchmark_name": str(benchmark_name),
        "num_tasks": int(len(task_results)),
        "start_match_rank": int(start_match_rank),
        "n_eval_per_task": (None if n_eval_per_task is None else int(n_eval_per_task)),
        "total_episodes": int(total_episodes),
        "total_successes": int(total_successes),
        "success_rate": float(total_successes / total_episodes) if total_episodes else 0.0,
        "mean_task_success_rate": float(np.mean(task_success_rates)) if task_success_rates else 0.0,
        "tasks": task_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate async RTC rollout success rate on LIBERO benchmarks.")
    parser.add_argument("--vla-checkpoint", type=str, required=True)
    parser.add_argument("--action-expert-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--benchmark", type=str, default="libero_90")
    parser.add_argument("--vlm-device", type=str, default=None)
    parser.add_argument("--dit-device", type=str, default=None)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument(
        "--n-eval-per-task",
        type=int,
        default=1,
        help="Number of init states to evaluate per task starting at --match-rank. "
        "Defaults to 1. If <= 0, evaluate all available init states for each task.",
    )
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=500)
    parser.add_argument("--observation-gap-steps", type=int, default=None)
    parser.add_argument("--step-dt-min-ms", type=int, default=None)
    parser.add_argument("--step-dt-max-ms", type=int, default=None)
    parser.add_argument("--max-wait-s", type=float, default=2.0)
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--video-root", type=str, default=None)
    parser.add_argument("--report-path", type=str, default=None)
    parser.add_argument("--include-episodes", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    pipeline = Qwen3RTCVLAOnlinePipeline(
        vla_config_path=str(cfg.model.vla_config_path),
        rtc_config_path=str(cfg.rtc_async.config_path),
        vlm_device=args.vlm_device,
        dit_device=args.dit_device,
    )
    pipeline.load_action_expert_checkpoint(str(args.vla_checkpoint), strict=False)
    pipeline.load_action_expert_checkpoint(str(args.action_expert_checkpoint), strict=False)

    metrics = evaluate_benchmark_success_rate(
        pipeline=pipeline,
        cfg=cfg,
        benchmark_name=str(args.benchmark),
        num_frames=int(cfg.model.get("num_frames", 1) if args.num_frames is None else args.num_frames),
        source_dt_ms=int(cfg.training.source_dt_ms),
        observation_gap_steps=(None if args.observation_gap_steps is None else int(args.observation_gap_steps)),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms if args.step_dt_min_ms is None else args.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms if args.step_dt_max_ms is None else args.step_dt_max_ms),
        max_env_steps=int(args.max_env_steps),
        max_wait_s=float(args.max_wait_s),
        start_match_rank=int(args.match_rank),
        n_eval_per_task=args.n_eval_per_task,
        save_videos=bool(args.save_videos),
        video_root=args.video_root,
        include_episodes=bool(args.include_episodes),
    )

    if args.report_path is not None:
        _write_json(str(args.report_path), metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
