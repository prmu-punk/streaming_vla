from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from model.vla_qwen3 import Qwen3VLA
from utils.stages import action_head_forward, backbone_forward, vision_forward
from workspace.libero_rollout import (
    LiberoEnv,
    _build_state_tensor,
    _initial_frame_window,
    _load_task_init_states,
    _set_init_state,
    _warmup_env,
    _window_array,
    benchmark,
    task_name_to_suite_and_ids,
)


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_call(device: str, fn, *args, **kwargs):
    sync_device(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync_device(device)
    dt = time.perf_counter() - t0
    return out, dt


def rollout_one_episode_pipeline(
    *,
    vla: Qwen3VLA,
    env: LiberoEnv,
    init_state: np.ndarray,
    image_key: str,
    state_keys: List[str],
    num_frames: int,
    fixed_action_tokens: int,
    source_dt_ms: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    warmup_steps: int = 5,
    execute_prefix_actions: Optional[int] = None,
) -> Dict[str, Any]:
    with torch.inference_mode():
        _, _ = timed_call(vla.device, env.reset)
        obs, _ = timed_call(vla.device, _set_init_state, env, init_state)
        obs, _ = timed_call(vla.device, _warmup_env, env, warmup_steps)

        runner = vla.new_runner()
        prompt = str(obs["prompt"])
        _, _ = timed_call(vla.device, vla.prefill, runner, prompt)

        pipeline = vla.new_pipeline_state()
        frame_window = _initial_frame_window(obs[image_key], num_frames=num_frames)
        step_idx = warmup_steps
        done = bool(env.done)

        vision_time = 0.0
        backbone_time = 0.0
        action_head_time = 0.0
        env_exec_time = 0.0
        decision_count = 0
        action_steps = 0
        token_growths: List[int] = []
        max_token_log_len = int(getattr(runner, "token_log", torch.empty(0)).shape[0])

        while not done:
            token_len_before = int(runner.token_log.shape[0])
            state = _build_state_tensor(obs, state_keys, device=vla.device)

            vla.push_step(
                pipeline,
                _window_array(frame_window),
                state=state,
                ts=step_idx * source_dt_ms,
                num_frames=num_frames,
            )

            if not pipeline.step_queue:
                raise RuntimeError("step_queue is empty right after push_step.")
            step_packet = pipeline.step_queue[0]
            inserted, dt = timed_call(vla.device, vision_forward, vla, runner, step_packet)
            vision_time += dt
            if not inserted:
                raise RuntimeError("vision_forward returned False.")
            pipeline.step_queue.popleft()

            token_len_after_insert = int(runner.token_log.shape[0])
            token_growths.append(token_len_after_insert - token_len_before)
            max_token_log_len = max(max_token_log_len, token_len_after_insert)

            token_packet, dt = timed_call(
                vla.device,
                backbone_forward,
                vla,
                runner,
                step_packet.step_id,
                fixed_action_tokens=fixed_action_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            backbone_time += dt
            pipeline.token_queue.append(token_packet)

            token_packet = pipeline.token_queue.popleft()
            chunk_packet, dt = timed_call(vla.device, action_head_forward, vla, token_packet)
            action_head_time += dt
            pipeline.chunk_queue.append(chunk_packet)

            chunk_packet = vla.pop_action_chunk(pipeline)
            if chunk_packet is None:
                raise RuntimeError("Failed to pop action chunk from pipeline.")

            action_chunk_np = chunk_packet.action_chunk[0].detach().to("cpu").numpy().astype(np.float32)
            if execute_prefix_actions is not None:
                action_chunk_np = action_chunk_np[: int(execute_prefix_actions)]
            decision_count += 1

            for action in action_chunk_np:
                step_out, dt = timed_call(vla.device, env.step, action)
                obs, _, done, _, _ = step_out
                env_exec_time += dt
                action_steps += 1
                step_idx += 1
                frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                if done:
                    break

        return {
            "decision_count": int(decision_count),
            "action_steps": int(action_steps),
            "vision_ms_per_step": 1000.0 * vision_time / max(decision_count, 1),
            "backbone_ms_per_step": 1000.0 * backbone_time / max(decision_count, 1),
            "decode_ms_per_step": 1000.0 * action_head_time / max(decision_count, 1),
            "env_exec_ms_per_action": 1000.0 * env_exec_time / max(action_steps, 1),
            "env_exec_ms_per_step": 1000.0 * env_exec_time / max(decision_count, 1),
            "mean_step_token_growth": float(np.mean(token_growths)) if token_growths else 0.0,
            "final_token_log_len": int(runner.token_log.shape[0]),
            "max_token_log_len": int(max_token_log_len),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fixed-action-tokens", type=int, default=None)
    parser.add_argument("--source-dt-ms", type=int, default=None)
    parser.add_argument("--execute-prefix-actions", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if args.task not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {args.task}")

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location=vla.device)
    vla.load_state_dict(payload["model"], strict=True)
    vla.eval()

    suite_name, task_id, _ = task_name_to_suite_and_ids[args.task]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)
    ep_idx = int(args.episode_idx)
    if ep_idx < 0 or ep_idx >= int(init_states.shape[0]):
        raise ValueError(f"episode-idx out of range: {ep_idx}, total={int(init_states.shape[0])}")

    image_key = str(cfg.dataset.image_key)
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    num_frames = int(args.num_frames) if args.num_frames is not None else int(cfg.model.num_frames)
    fixed_action_tokens = (
        int(args.fixed_action_tokens) if args.fixed_action_tokens is not None else int(cfg.model.fixed_action_tokens)
    )
    source_dt_ms = int(args.source_dt_ms) if args.source_dt_ms is not None else int(cfg.training.source_dt_ms)

    env = LiberoEnv(
        task_name=str(args.task),
        image_size=128,
        seed=int(cfg.training.seed),
        camera_names=[image_key.replace("_rgb", "")],
        state_ports=state_keys,
        max_episode_steps=550,
    )
    try:
        metrics = rollout_one_episode_pipeline(
            vla=vla,
            env=env,
            init_state=init_states[ep_idx],
            image_key=image_key,
            state_keys=state_keys,
            num_frames=num_frames,
            fixed_action_tokens=fixed_action_tokens,
            source_dt_ms=source_dt_ms,
            temperature=float(args.temperature),
            top_k=args.top_k,
            execute_prefix_actions=args.execute_prefix_actions,
        )
    finally:
        env.close()

    out: Dict[str, Any] = {
        "task_name": task.name,
        "task_prompt": task.language,
        "episode_idx": ep_idx,
        "num_frames": num_frames,
        "fixed_action_tokens": fixed_action_tokens,
        "source_dt_ms": source_dt_ms,
        "execute_prefix_actions": int(args.execute_prefix_actions) if args.execute_prefix_actions is not None else None,
        "metrics": metrics,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
