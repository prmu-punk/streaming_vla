from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque
from typing import Any, Dict, List, Optional

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from dataset.libero90_dataset import LiberoEpisodeDataset
from model.vla_qwen3 import Qwen3VLA
from workspace.libero_rollout import (
    LiberoEnv,
    _build_state_tensor,
    _initial_frame_window,
    _load_task_init_states,
    _set_init_state,
    _window_array,
    benchmark,
    infer_chunk_horizon,
    task_name_to_suite_and_ids,
)


def _find_matching_episode_indices(dataset: LiberoEpisodeDataset, prompt: str) -> List[int]:
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if str(dataset[ep_idx]["prompt"]) == prompt:
            out.append(int(ep_idx))
    return out


def _append_gt_action_chunk(vla: Qwen3VLA, runner, gt_chunk: torch.Tensor) -> None:
    gt_tokens = vla.action_tokens(gt_chunk.unsqueeze(0).to(vla.device))
    eos_id = vla.action_tokenizer.act_eos_hf_id
    eos_token = torch.tensor([[eos_id]], dtype=torch.long, device=vla.device)
    runner.append_text_tokens(input_ids=gt_tokens)
    runner.append_text_tokens(input_ids=eos_token)


def _save_video(path: str, frames: List[np.ndarray], fps: int) -> None:
    out_path = pathlib.Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)


def run_gt_prefix_rollout(
    *,
    vla: Qwen3VLA,
    cfg,
    task_name: str,
    match_rank: int,
    anchor_pos: int,
    history_steps: int,
    step_dt_ms: int,
    num_frames: int,
    save_video_path: Optional[str],
    temperature: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    source_dt_ms = int(cfg.training.source_dt_ms)
    if step_dt_ms % source_dt_ms != 0:
        raise ValueError(f"step_dt_ms must be divisible by source_dt_ms, got {step_dt_ms}/{source_dt_ms}")
    step_stride = int(step_dt_ms // source_dt_ms)
    if anchor_pos < history_steps:
        raise ValueError(f"anchor_pos must be >= history_steps, got {anchor_pos} < {history_steps}")

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)

    dataset = LiberoEpisodeDataset(
        zarr_path=str(cfg.dataset.zarr_path),
        image_key=str(cfg.dataset.image_key),
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

    dataset_ep_idx = matched[match_rank]
    ep = dataset[dataset_ep_idx]
    actions = ep["actions"]
    episode_len = int(actions.shape[0])
    chunk_horizon = infer_chunk_horizon(vla, fixed_action_tokens=int(cfg.model.fixed_action_tokens))

    anchor_t = anchor_pos * step_stride
    if anchor_t + chunk_horizon > episode_len:
        raise ValueError(
            f"anchor_t+chunk_horizon exceeds episode: anchor_t={anchor_t}, "
            f"chunk_horizon={chunk_horizon}, episode_len={episode_len}"
        )

    history_positions = list(range(anchor_pos - history_steps, anchor_pos))
    history_times = [p * step_stride for p in history_positions]

    init_idx = min(match_rank, int(init_states.shape[0]) - 1)
    image_key = str(cfg.dataset.image_key)
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    env = LiberoEnv(
        task_name=task_name,
        image_size=128,
        seed=int(cfg.training.seed),
        camera_names=[image_key.replace("_rgb", "")],
        state_ports=state_keys,
        max_episode_steps=550,
    )
    frames_to_save: List[np.ndarray] = []
    metrics: Dict[str, Any] = {}
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            del obs
            obs = _set_init_state(env, init_states[init_idx])

            runner = vla.new_runner()
            prompt = str(obs["prompt"])
            vla.prefill(runner, prompt)
            frame_window: deque[np.ndarray] = _initial_frame_window(obs[image_key], num_frames=num_frames)
            if save_video_path is not None:
                frames_to_save.append(env.render())

            current_t = 0
            for hist_t in history_times:
                if current_t != hist_t:
                    raise RuntimeError(f"Environment time drifted: current_t={current_t}, expected hist_t={hist_t}")

                state = _build_state_tensor(obs, state_keys, device=vla.device)
                inserted = vla.insert_step(
                    runner,
                    _window_array(frame_window),
                    state=state,
                    ts=hist_t * source_dt_ms,
                    num_frames=num_frames,
                    source_dt_ms=source_dt_ms,
                )
                if not inserted:
                    raise RuntimeError(f"Failed to insert GT history step at t={hist_t}")

                gt_chunk = actions[hist_t : hist_t + chunk_horizon]
                _append_gt_action_chunk(vla, runner, gt_chunk)

                gt_prefix = actions[hist_t : hist_t + step_stride]
                for action in gt_prefix.detach().cpu().numpy().astype(np.float32):
                    obs, _, done, _, _ = env.step(action)
                    if save_video_path is not None:
                        frames_to_save.append(env.render())
                    current_t += 1
                    frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                    if done:
                        raise RuntimeError("Environment ended during GT prefix rollout.")

            if current_t != anchor_t:
                raise RuntimeError(f"Anchor mismatch: current_t={current_t}, anchor_t={anchor_t}")

            state = _build_state_tensor(obs, state_keys, device=vla.device)
            inserted = vla.insert_step(
                runner,
                _window_array(frame_window),
                state=state,
                ts=anchor_t * source_dt_ms,
                num_frames=num_frames,
                source_dt_ms=source_dt_ms,
            )
            if not inserted:
                raise RuntimeError("Failed to insert anchor step.")

            gen = vla.generate_action_chunk(
                runner,
                fixed_action_tokens=int(cfg.model.fixed_action_tokens),
                temperature=float(temperature),
                top_k=top_k,
            )
            pred_chunk = gen["action_chunk"]
            if pred_chunk is None:
                raise RuntimeError("Model returned no action chunk at anchor.")

            pred_chunk_np = pred_chunk[0].detach().to("cpu").numpy().astype(np.float32)
            success = False
            done = bool(env.done)
            for action in pred_chunk_np:
                obs, reward, done, _, _ = env.step(action)
                if save_video_path is not None:
                    frames_to_save.append(env.render())
                current_t += 1
                frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                if reward >= 1.0:
                    success = True
                if done:
                    break

            if save_video_path is not None:
                _save_video(save_video_path, frames_to_save, fps=max(1, round(1000 / source_dt_ms)))

            metrics = {
                "task_name": task.name,
                "task_prompt": task.language,
                "dataset_episode_idx": int(dataset_ep_idx),
                "init_state_idx": int(init_idx),
                "history_steps": int(history_steps),
                "step_dt_ms": int(step_dt_ms),
                "num_frames": int(num_frames),
                "anchor_pos": int(anchor_pos),
                "anchor_t": int(anchor_t),
                "history_times": history_times,
                "chunk_horizon": int(chunk_horizon),
                "success": bool(success),
                "done": bool(done),
                "prompt": prompt,
            }
    finally:
        env.close()

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--anchor-pos", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--step-dt-ms", type=int, default=200)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location=vla.device)
    vla.load_state_dict(payload["model"], strict=True)
    vla.eval()

    video_path = None
    if args.save_video:
        video_path = str(
            pathlib.Path.cwd() / "videos" / "gt_prefix" / str(args.task) / f"match_{int(args.match_rank):03d}.mp4"
        )

    metrics = run_gt_prefix_rollout(
        vla=vla,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        anchor_pos=int(args.anchor_pos),
        history_steps=int(args.history_steps),
        step_dt_ms=int(args.step_dt_ms),
        num_frames=int(args.num_frames),
        save_video_path=video_path,
        temperature=float(args.temperature),
        top_k=args.top_k,
    )
    if video_path is not None:
        metrics["video_path"] = video_path
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
