from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import os
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional

import imageio
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


from libero.libero import benchmark, get_libero_path
from model import Qwen3RTCVLAOnlinePipeline
from model.rtc_async.pipeline.pipeline_types import ExecutePacket
from oat.oat.env.libero.env import LiberoEnv, task_name_to_suite_and_ids


def _build_state_tensor(obs: Dict[str, Any], state_keys: List[str], device: str) -> torch.Tensor:
    pieces = []
    for key in state_keys:
        arr = np.asarray(obs[key], dtype=np.float32).reshape(-1)
        pieces.append(arr)
    state = np.concatenate(pieces, axis=0)
    return torch.from_numpy(state).to(device=device).unsqueeze(0)


def _set_init_state(env: LiberoEnv, init_state: np.ndarray) -> Dict[str, Any]:
    raw_obs = env.env.set_init_state(init_state)
    env.done = False
    env.cur_step = 0
    return env._extract_obs(raw_obs)


def _load_task_init_states(task) -> torch.Tensor:
    init_states_path = pathlib.Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file
    return torch.load(init_states_path, weights_only=False)


def _save_video(path: str, frames: List[np.ndarray], fps: int) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


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


@dataclass
class ScheduledExecuteChunk:
    step_id: int
    start_step: int
    end_step: int
    step_delay_steps: int
    prefix_len: int
    actions: np.ndarray


class PrefixScaledRTCOnlinePipeline(Qwen3RTCVLAOnlinePipeline):
    def __init__(self, *, prefix_scale: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.prefix_scale = float(prefix_scale)

    def _scaled_step_delay_steps(self, step_delay_steps: int) -> int:
        horizon = int(self.scheduler.horizon)
        natural_prefix_len = max(horizon - int(step_delay_steps), 0)
        scaled_prefix_len = int(round(float(natural_prefix_len) * float(self.prefix_scale)))
        scaled_prefix_len = max(0, min(horizon, scaled_prefix_len))
        return int(horizon - scaled_prefix_len)

    def _drain_context_to_action(
        self,
        *,
        kv_cache_key: Optional[tuple[Any, ...]] = None,
        generator: torch.Generator | None = None,
    ) -> Any:
        context_packet = self.queues.context_queue.pop()
        if context_packet is None:
            return None
        step_delay_steps = self._resolve_step_delay_steps(context_packet.ts_ms)
        effective_step_delay_steps = self._scaled_step_delay_steps(step_delay_steps)
        prefix_len = max(self.scheduler.horizon - int(effective_step_delay_steps), 0)
        known_action = None
        known_mask = None
        if prefix_len > 0:
            prefix_chunk = self.scheduler.get_prefix_chunk(
                batch_size=context_packet.state.shape[0],
                step_delay_steps=int(effective_step_delay_steps),
            )
            if prefix_chunk.shape[0] != context_packet.state.shape[0]:
                raise RuntimeError(
                    f"scheduler prefix_chunk batch mismatch: {prefix_chunk.shape[0]} vs {context_packet.state.shape[0]}"
                )
            known_action = prefix_chunk.to(device=self.dit_device, non_blocking=True)
            if self.normalizer is not None:
                known_action = self.normalizer.normalize_action(known_action)
            known_mask = torch.zeros(
                (known_action.shape[0], known_action.shape[1]),
                dtype=torch.bool,
                device=known_action.device,
            )
            known_mask[:, :prefix_len] = True
        packet = self.dit_stage(
            context_packet,
            step_delay_steps=effective_step_delay_steps,
            known_action=known_action,
            known_mask=known_mask,
            normalizer=self.normalizer,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        self.queues.action_queue.put_latest(packet)
        return packet

    def _drain_action_to_execute(self) -> Any:
        action_packet = self.queues.action_queue.pop()
        if action_packet is None:
            return None
        effective_step_delay_steps = self._scaled_step_delay_steps(action_packet.step_delay_steps)
        stitched_chunk, execute_chunk, _, prefix_len, _ = self.scheduler.schedule(
            next_chunk=action_packet.action_chunk,
            step_delay_steps=int(effective_step_delay_steps),
        )
        packet = ExecutePacket(
            step_id=action_packet.step_id,
            ts_ms=action_packet.ts_ms,
            step_delay_steps=int(effective_step_delay_steps),
            prefix_len=int(prefix_len),
            action_chunk=action_packet.action_chunk,
            stitched_chunk=stitched_chunk,
            execute_chunk=execute_chunk,
        )
        self.queues.execute_queue.put_latest(packet)
        return packet


def _ingest_execute_packet(
    *,
    scheduled_chunks: deque[ScheduledExecuteChunk],
    packet_history: List[Dict[str, int]],
    latest_out: Dict[str, torch.Tensor | int],
    source_dt_ms: int,
) -> None:
    stitched_chunk = latest_out["stitched_chunk"]
    if not isinstance(stitched_chunk, torch.Tensor):
        raise RuntimeError(f"stitched_chunk must be torch.Tensor, got {type(stitched_chunk)}")
    ts_ms = latest_out.get("ts_ms", None)
    if not isinstance(ts_ms, int):
        raise RuntimeError(f"execute packet must include integer ts_ms, got {type(ts_ms)}")
    start_step = max(int(round(float(ts_ms) / float(source_dt_ms))), 0)
    if scheduled_chunks and start_step <= int(scheduled_chunks[-1].start_step):
        raise RuntimeError(
            f"Execute chunk start_step must be strictly increasing, got {start_step} "
            f"after {scheduled_chunks[-1].start_step}"
        )
    horizon_steps = int(stitched_chunk.shape[1])
    execute_chunk = latest_out["execute_chunk"]
    if not isinstance(execute_chunk, torch.Tensor):
        raise RuntimeError(f"execute_chunk must be torch.Tensor, got {type(execute_chunk)}")
    scheduled_chunks.append(
        ScheduledExecuteChunk(
            step_id=int(latest_out["step_id"]),
            start_step=start_step,
            end_step=start_step + horizon_steps,
            step_delay_steps=int(latest_out["step_delay_steps"]),
            prefix_len=int(latest_out["prefix_len"]),
            actions=stitched_chunk[0].detach().to("cpu").numpy().astype(np.float32),
        )
    )
    packet_history.append(
        {
            "step_id": int(latest_out["step_id"]),
            "ts_ms": int(ts_ms),
            "start_step": int(start_step),
            "end_step": int(start_step + horizon_steps),
            "step_delay_steps": int(latest_out["step_delay_steps"]),
            "prefix_len": int(latest_out["prefix_len"]),
            "horizon_steps": int(horizon_steps),
            "execute_steps": int(execute_chunk.shape[1]),
        }
    )


def _advance_chunk_queue(
    scheduled_chunks: deque[ScheduledExecuteChunk],
    *,
    current_step: int,
) -> None:
    while len(scheduled_chunks) >= 2 and int(scheduled_chunks[1].start_step) <= int(current_step):
        scheduled_chunks.popleft()
    while scheduled_chunks and int(scheduled_chunks[0].end_step) <= int(current_step):
        scheduled_chunks.popleft()


def _active_chunk_action(
    scheduled_chunks: deque[ScheduledExecuteChunk],
    *,
    current_step: int,
) -> np.ndarray:
    _advance_chunk_queue(scheduled_chunks, current_step=current_step)
    if not scheduled_chunks:
        raise RuntimeError(f"No execute chunk covers current_step={current_step}")
    active = scheduled_chunks[0]
    if not (int(active.start_step) <= int(current_step) < int(active.end_step)):
        raise RuntimeError(f"No execute chunk covers current_step={current_step}")
    offset = int(current_step) - int(active.start_step)
    if offset < 0 or offset >= int(active.actions.shape[0]):
        raise RuntimeError(
            f"Active chunk offset out of range: current_step={current_step}, "
            f"start_step={active.start_step}, action_len={active.actions.shape[0]}"
        )
    return active.actions[offset]


def _drain_execute_packets(
    *,
    pipeline: Qwen3RTCVLAOnlinePipeline,
    scheduled_chunks: deque[ScheduledExecuteChunk],
    packet_history: List[Dict[str, int]],
    source_dt_ms: int,
) -> None:
    latest_out = pipeline.poll_execute_packet()
    while latest_out is not None:
        _ingest_execute_packet(
            scheduled_chunks=scheduled_chunks,
            packet_history=packet_history,
            latest_out=latest_out,
            source_dt_ms=source_dt_ms,
        )
        latest_out = pipeline.poll_execute_packet()


def _has_active_chunk(
    scheduled_chunks: deque[ScheduledExecuteChunk],
    *,
    current_step: int,
) -> bool:
    _advance_chunk_queue(scheduled_chunks, current_step=current_step)
    if not scheduled_chunks:
        return False
    head = scheduled_chunks[0]
    return int(head.start_step) <= int(current_step) < int(head.end_step)


def run_async_rollout(
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
    save_video_path: str,
    max_wait_s: float,
) -> Dict[str, Any]:
    if task_name not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {task_name}")

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
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
            video_frames.append(env.render())

            total_env_steps = 0
            done = bool(env.done)
            success = False
            next_obs_step = 0
            control_dt_s = float(source_dt_ms) / 1000.0
            rollout_start = time.perf_counter()
            rng = random.Random(int(cfg.training.seed))

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
                        gap_ms = rng.randint(int(step_dt_min_ms), int(step_dt_max_ms))
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
                video_frames.append(env.render())
                if reward >= 1.0:
                    success = True

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
                "video_path": str(save_video_path),
            }
    finally:
        pipeline.stop_async_pipeline()
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single async RTC rollout on LIBERO and save a video.")
    parser.add_argument("--vla-checkpoint", type=str, required=True)
    parser.add_argument("--action-expert-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--vlm-device", type=str, default=None)
    parser.add_argument("--dit-device", type=str, default=None)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=900)
    parser.add_argument("--observation-gap-steps", type=int, default=None)
    parser.add_argument("--step-dt-min-ms", type=int, default=None)
    parser.add_argument("--step-dt-max-ms", type=int, default=None)
    parser.add_argument("--prefix-scale", type=float, default=1.0)
    parser.add_argument("--max-wait-s", type=float, default=2.0)
    parser.add_argument("--video-path", type=str, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    pipeline = PrefixScaledRTCOnlinePipeline(
        vla_config_path=str(cfg.model.vla_config_path),
        rtc_config_path=str(cfg.rtc_async.config_path),
        vlm_device=args.vlm_device,
        dit_device=args.dit_device,
        prefix_scale=float(args.prefix_scale),
    )
    pipeline.load_action_expert_checkpoint(str(args.vla_checkpoint), strict=False)
    pipeline.load_action_expert_checkpoint(str(args.action_expert_checkpoint), strict=False)

    num_frames = int(cfg.model.get("num_frames", 1) if args.num_frames is None else args.num_frames)
    video_path = args.video_path
    if video_path is None:
        video_path = str(
            pathlib.Path.cwd() / "videos" / "rtc_async_rollout" / str(args.task) / f"match_{int(args.match_rank):03d}.mp4"
        )

    metrics = run_async_rollout(
        pipeline=pipeline,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        num_frames=num_frames,
        source_dt_ms=int(cfg.training.source_dt_ms),
        observation_gap_steps=(None if args.observation_gap_steps is None else int(args.observation_gap_steps)),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms if args.step_dt_min_ms is None else args.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms if args.step_dt_max_ms is None else args.step_dt_max_ms),
        max_env_steps=int(args.max_env_steps),
        save_video_path=str(video_path),
        max_wait_s=float(args.max_wait_s),
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
