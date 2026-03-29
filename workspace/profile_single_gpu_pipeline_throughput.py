from __future__ import annotations

import argparse
import json
import os
import pathlib
import random
import sys
import time
from collections import deque
from functools import wraps
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

LEGACY_ROOT = ROOT_DIR.parent / "streaming_vla"
if str(LEGACY_ROOT) not in sys.path:
    sys.path.append(str(LEGACY_ROOT))
OAT_ROOT = LEGACY_ROOT / "oat"
if str(OAT_ROOT) not in sys.path:
    sys.path.append(str(OAT_ROOT))
LIBERO_ROOT = OAT_ROOT / "third_party" / "LIBERO"
if str(LIBERO_ROOT) not in sys.path:
    sys.path.append(str(LIBERO_ROOT))


def _ensure_libero_config() -> None:
    libero_config_root = LEGACY_ROOT / ".libero"
    libero_config_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(libero_config_root))

    config_path = libero_config_root / "config.yaml"
    if config_path.exists():
        return

    benchmark_root = LIBERO_ROOT / "libero" / "libero"
    config = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(LEGACY_ROOT / "oat" / "data" / "libero"),
        "assets": str(benchmark_root / "assets"),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)


_ensure_libero_config()


from dataset.libero90_async_dataset import LiberoEpisodeDataset
from libero.libero import benchmark, get_libero_path
from model import Qwen3RTCVLAOnlinePipeline
from oat.oat.env.libero.env import LiberoEnv, task_name_to_suite_and_ids


def _build_state_tensor(obs: Dict[str, Any], state_keys: List[str], device: str) -> torch.Tensor:
    pieces = []
    for key in state_keys:
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        pieces.append(arr)
    state = np.concatenate(pieces, axis=0)
    return torch.from_numpy(state).to(device=device).unsqueeze(0)


def _raw_window_size(num_frames: int, frame_stride_steps: int) -> int:
    return 1 + max(0, int(num_frames) - 1) * int(frame_stride_steps)


def _initial_frame_window(frame: np.ndarray, num_frames: int, frame_stride_steps: int) -> deque[np.ndarray]:
    q: deque[np.ndarray] = deque(maxlen=_raw_window_size(num_frames, frame_stride_steps))
    for _ in range(q.maxlen):
        q.append(np.asarray(frame, dtype=np.uint8))
    return q


def _window_array(frame_window: deque[np.ndarray], *, num_frames: int, frame_stride_steps: int) -> np.ndarray:
    arr = list(frame_window)
    end = len(arr) - 1
    indices = [
        max(0, end - int(frame_stride_steps) * (int(num_frames) - 1 - i))
        for i in range(int(num_frames))
    ]
    return np.stack([np.asarray(arr[idx], dtype=np.uint8) for idx in indices], axis=0)


def _set_init_state(env: LiberoEnv, init_state: np.ndarray) -> Dict[str, Any]:
    raw_obs = env.env.set_init_state(init_state)
    env.done = False
    env.cur_step = 0
    return env._extract_obs(raw_obs)


def _load_task_init_states(task) -> torch.Tensor:
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)


def _find_matching_episode_indices(dataset: LiberoEpisodeDataset, prompt: str) -> List[int]:
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if str(dataset.get_prompt(ep_idx)) == prompt:
            out.append(int(ep_idx))
    return out


def _summary(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def _install_queue_stats(pipeline: Qwen3RTCVLAOnlinePipeline) -> Dict[str, Dict[str, int]]:
    queue_stats: Dict[str, Dict[str, int]] = {}
    for name in ("step_queue", "context_queue", "action_queue", "execute_queue"):
        queue = getattr(pipeline.queues, name)
        stats = {"put_count": 0, "pop_count": 0, "full_count": 0}
        queue_stats[name] = stats

        orig_put_latest = queue.put_latest
        orig_pop = queue.pop
        orig_full = queue.full

        @wraps(orig_put_latest)
        def put_latest(item, *, _orig=orig_put_latest, _full=orig_full, _stats=stats):
            if _full():
                _stats["full_count"] += 1
            _stats["put_count"] += 1
            return _orig(item)

        @wraps(orig_pop)
        def pop(*, _orig=orig_pop, _stats=stats):
            item = _orig()
            if item is not None:
                _stats["pop_count"] += 1
            return item

        queue.put_latest = put_latest
        queue.pop = pop
    return queue_stats


def _install_worker_stats(pipeline: Qwen3RTCVLAOnlinePipeline) -> Dict[str, Dict[str, List[float] | int]]:
    worker_stats: Dict[str, Dict[str, List[float] | int]] = {}
    drain_names = [
        "_drain_step_to_context",
        "_drain_context_to_action",
        "_drain_action_to_execute",
    ]
    for name in drain_names:
        stats: Dict[str, List[float] | int] = {"call_count": 0, "emit_count": 0, "durations_ms": []}
        worker_stats[name] = stats
        orig = getattr(pipeline, name)

        @wraps(orig)
        def wrapped(*args, _orig=orig, _stats=stats, **kwargs):
            t0 = time.perf_counter()
            out = _orig(*args, **kwargs)
            dt_ms = 1000.0 * (time.perf_counter() - t0)
            _stats["call_count"] += 1
            _stats["durations_ms"].append(dt_ms)
            if out is not None:
                _stats["emit_count"] += 1
            return out

        setattr(pipeline, name, wrapped)
    return worker_stats


def run_single_gpu_throughput_profile(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    cfg,
    task_name: str,
    match_rank: int,
    num_frames: int,
    duration_s: float,
    warmup_s: float,
    source_dt_ms: int,
    step_dt_min_ms: int,
    step_dt_max_ms: int,
) -> Dict[str, Any]:
    if task_name not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {task_name}")

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)

    dataset = LiberoEpisodeDataset(
        zarr_path=str(cfg.dataset.zarr_path),
        image_key=str(cfg.dataset.image_key),
        extra_image_keys=([str(cfg.dataset.aux_image_key)] if cfg.dataset.get("aux_image_key", None) else []),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        max_episodes=cfg.dataset.max_episodes,
    )
    matched = _find_matching_episode_indices(dataset, str(task.language))
    if not matched:
        raise ValueError(f"No dataset episodes matched task prompt: {task.language}")
    if match_rank < 0 or match_rank >= len(matched):
        raise ValueError(f"match_rank out of range: {match_rank}, total_matches={len(matched)}")

    init_idx = min(match_rank, int(init_states.shape[0]) - 1)
    image_key = str(cfg.dataset.image_key)
    aux_image_key = str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    frame_stride_steps = int(cfg.model.get("video_frame_stride_steps", 1))

    env = LiberoEnv(
        task_name=task_name,
        image_size=256,
        seed=int(cfg.training.seed),
        camera_names=[
            image_key.replace("_rgb", ""),
            *([aux_image_key.replace("_rgb", "")] if aux_image_key is not None else []),
        ],
        state_ports=state_keys,
        max_episode_steps=550,
    )

    rng = random.Random(int(cfg.training.seed))
    enqueue_times: Dict[int, float] = {}
    latencies_ms: List[float] = []
    step_delays: List[int] = []
    prefix_lens: List[int] = []
    first_ready_latency_ms: Optional[float] = None
    pushed_steps = 0
    produced_chunks = 0
    effective_actions = 0
    nominal_actions = 0
    first_step_id: Optional[int] = None

    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            del obs
            obs = _set_init_state(env, init_states[init_idx])
            prompt = str(obs["prompt"])

            pipeline.reset(prompt=prompt)
            pipeline.set_runtime_timebase(source_dt_ms=int(source_dt_ms))
            queue_stats = _install_queue_stats(pipeline)
            worker_stats = _install_worker_stats(pipeline)
            pipeline.start_async_pipeline()

            frame_window = _initial_frame_window(
                obs[image_key],
                num_frames=num_frames,
                frame_stride_steps=frame_stride_steps,
            )
            state = _build_state_tensor(obs, state_keys, device=pipeline.device)
            aux_frames = None
            if aux_image_key is not None and aux_image_key in obs:
                aux_frames = np.asarray(obs[aux_image_key], dtype=np.uint8)[None, ...]

            start_time = time.perf_counter()
            end_time = start_time + float(duration_s)
            next_push_time = start_time
            ts_ms = 0

            while time.perf_counter() < end_time:
                now = time.perf_counter()
                if now >= next_push_time:
                    inserted = pipeline.push_observation(
                        frames=_window_array(
                            frame_window,
                            num_frames=num_frames,
                            frame_stride_steps=frame_stride_steps,
                        ),
                        aux_frames=aux_frames,
                        state=state,
                        ts_ms=int(ts_ms),
                        num_frames=num_frames,
                    )
                    if inserted:
                        step_id = int(pipeline._next_step_id - 1)
                        enqueue_times[step_id] = now
                        pushed_steps += 1
                    gap_ms = rng.randint(int(step_dt_min_ms), int(step_dt_max_ms))
                    ts_ms += int(gap_ms)
                    next_push_time += float(gap_ms) / 1000.0

                out = pipeline.poll_execute_packet()
                if out is not None:
                    step_id = int(out["step_id"])
                    enqueue_time = enqueue_times.pop(step_id, None)
                    if enqueue_time is not None:
                        latency_ms = 1000.0 * (time.perf_counter() - enqueue_time)
                        latencies_ms.append(latency_ms)
                        if first_ready_latency_ms is None:
                            first_ready_latency_ms = latency_ms
                    produced_chunks += 1
                    step_delay_steps = int(out["step_delay_steps"])
                    prefix_len = int(out["prefix_len"])
                    step_delays.append(step_delay_steps)
                    prefix_lens.append(prefix_len)
                    nominal_actions += int(out["action_chunk"].shape[1])
                    if first_step_id is None:
                        first_step_id = step_id
                    else:
                        effective_actions += int(out["execute_chunk"].shape[1])

                time.sleep(0.0005)

            drain_deadline = time.perf_counter() + float(warmup_s)
            while time.perf_counter() < drain_deadline:
                out = pipeline.poll_execute_packet()
                if out is None:
                    time.sleep(0.0005)
                    continue
                step_id = int(out["step_id"])
                enqueue_time = enqueue_times.pop(step_id, None)
                if enqueue_time is not None:
                    latency_ms = 1000.0 * (time.perf_counter() - enqueue_time)
                    latencies_ms.append(latency_ms)
                    if first_ready_latency_ms is None:
                        first_ready_latency_ms = latency_ms
                produced_chunks += 1
                step_delay_steps = int(out["step_delay_steps"])
                prefix_len = int(out["prefix_len"])
                step_delays.append(step_delay_steps)
                prefix_lens.append(prefix_len)
                nominal_actions += int(out["action_chunk"].shape[1])
                if first_step_id is None:
                    first_step_id = step_id
                else:
                    effective_actions += int(out["execute_chunk"].shape[1])

            elapsed_s = max(time.perf_counter() - start_time, 1e-6)
            horizon = int(pipeline.action_expert.config.horizon)
            theoretical_max_action_hz = 1000.0 / float(source_dt_ms)
            worker_summary = {
                name: {
                    "call_count": int(stats["call_count"]),
                    "emit_count": int(stats["emit_count"]),
                    "duration_ms_summary": _summary(list(stats["durations_ms"])),
                }
                for name, stats in worker_stats.items()
            }

            return {
                "task_name": task.name,
                "task_prompt": task.language,
                "init_state_idx": int(init_idx),
                "duration_s": float(elapsed_s),
                "source_dt_ms": int(source_dt_ms),
                "step_dt_min_ms": int(step_dt_min_ms),
                "step_dt_max_ms": int(step_dt_max_ms),
                "num_frames": int(num_frames),
                "video_frame_stride_steps": int(frame_stride_steps),
                "horizon": int(horizon),
                "theoretical_max_action_hz": float(theoretical_max_action_hz),
                "pushed_steps": int(pushed_steps),
                "produced_chunks": int(produced_chunks),
                "chunk_rate_hz": float(produced_chunks / elapsed_s),
                "nominal_action_hz": float(nominal_actions / elapsed_s),
                "effective_action_hz": float(effective_actions / elapsed_s),
                "step_drop_ratio": float(max(pushed_steps - produced_chunks, 0) / max(pushed_steps, 1)),
                "first_ready_latency_ms": None if first_ready_latency_ms is None else float(first_ready_latency_ms),
                "step_delay_steps_summary": _summary([float(x) for x in step_delays]),
                "prefix_len_summary": _summary([float(x) for x in prefix_lens]),
                "ready_latency_ms_summary": _summary(latencies_ms),
                "queue_stats": queue_stats,
                "worker_stats": worker_summary,
            }
    finally:
        pipeline.stop_async_pipeline()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile single-GPU RTC threaded pipeline throughput.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--vlm-device", type=str, default=None)
    parser.add_argument("--dit-device", type=str, default=None)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--warmup-s", type=float, default=2.0)
    parser.add_argument("--step-dt-min-ms", type=int, default=None)
    parser.add_argument("--step-dt-max-ms", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    pipeline = Qwen3RTCVLAOnlinePipeline(
        vla_config_path=str(cfg.model.vla_config_path),
        rtc_config_path=str(cfg.rtc_async.config_path),
        vlm_device=args.vlm_device,
        dit_device=args.dit_device,
    )
    pipeline.load_action_expert_checkpoint(str(args.checkpoint), strict=False)

    metrics = run_single_gpu_throughput_profile(
        pipeline=pipeline,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        num_frames=int(cfg.model.get("num_frames", 1) if args.num_frames is None else args.num_frames),
        duration_s=float(args.duration_s),
        warmup_s=float(args.warmup_s),
        source_dt_ms=int(cfg.training.source_dt_ms),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms if args.step_dt_min_ms is None else args.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms if args.step_dt_max_ms is None else args.step_dt_max_ms),
    )

    output_text = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(output_text)
    if args.output is not None:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
