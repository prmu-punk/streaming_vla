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
from utils.vla_utils import module_device


def _find_matching_episode_indices(dataset: LiberoEpisodeDataset, prompt: str) -> List[int]:
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if str(dataset[ep_idx]["prompt"]) == prompt:
            out.append(int(ep_idx))
    return out


def _append_gt_action_chunk(vla: Qwen3VLA, runner, gt_chunk: torch.Tensor) -> None:
    device = module_device(vla.model)
    gt_tokens = vla.action_tokens(gt_chunk.unsqueeze(0).to(device))
    eos_id = vla.action_tokenizer.act_eos_hf_id
    eos_token = torch.tensor([[eos_id]], dtype=torch.long, device=device)
    runner.append_text_tokens(input_ids=gt_tokens)
    runner.append_text_tokens(input_ids=eos_token)


def _compose_history_actions(
    *,
    vla: Qwen3VLA,
    actions: torch.Tensor,
    history_times: List[int],
    anchor_t: int,
    chunk_horizon: int,
) -> tuple[np.ndarray, Dict[str, float]]:
    device = module_device(vla.model)
    action_dim = int(actions.shape[-1])
    composed = np.zeros((anchor_t, action_dim), dtype=np.float32)
    written = np.zeros((anchor_t,), dtype=bool)

    for hist_t in history_times:
        gt_chunk = actions[hist_t : hist_t + chunk_horizon].unsqueeze(0).to(device)
        gt_tokens = vla.action_tokens(gt_chunk)
        detok = vla.action_tokenizer.detokenize(gt_tokens)[0].detach().to("cpu").numpy().astype(np.float32)
        end_t = min(anchor_t, hist_t + detok.shape[0])
        if end_t <= hist_t:
            continue
        composed[hist_t:end_t] = detok[: end_t - hist_t]
        written[hist_t:end_t] = True

    gt_prefix = actions[:anchor_t].detach().to("cpu").numpy().astype(np.float32)
    valid = written
    if valid.any():
        diff = composed[valid] - gt_prefix[valid]
        mse = float(np.mean(diff * diff))
        max_abs = float(np.max(np.abs(diff)))
        coverage = float(valid.mean())
    else:
        mse = 0.0
        max_abs = 0.0
        coverage = 0.0

    return composed, {
        "history_prefix_mse_vs_raw_gt": mse,
        "history_prefix_max_abs_vs_raw_gt": max_abs,
        "history_prefix_coverage": coverage,
    }


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
    save_video_path: Optional[str],
    temperature: float,
    top_k: Optional[int],
) -> Dict[str, Any]:
    device = module_device(vla.model)
    source_dt_ms = int(cfg.training.source_dt_ms)
    num_frames = int(cfg.model.num_frames)
    anchor_stride_steps = int(cfg.dataset.anchor_stride_steps or 1)
    step_dt_min_ms = int(cfg.training.step_dt_min_ms)
    step_dt_max_ms = int(cfg.training.step_dt_max_ms)
    max_context_len = int(float(cfg.model.max_context_len))
    fixed_action_tokens = int(cfg.model.fixed_action_tokens)

    stride_min = max(1, int(round(float(step_dt_min_ms) / float(source_dt_ms))))
    stride_max = max(1, int(round(float(step_dt_max_ms) / float(source_dt_ms))))
    step_strides = list(range(stride_min, stride_max + 1))

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)

    dataset = LiberoEpisodeDataset(
        zarr_path=str(cfg.dataset.zarr_path),
        image_key=str(cfg.dataset.image_key),
        extra_image_keys=(
            [str(cfg.dataset.aux_image_key)]
            if cfg.dataset.get("aux_image_key", None)
            else []
        ),
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
    chunk_horizon = infer_chunk_horizon(vla, fixed_action_tokens=fixed_action_tokens)

    anchor_candidates = list(range(0, episode_len - chunk_horizon + 1, anchor_stride_steps))
    if anchor_pos < 0 or anchor_pos >= len(anchor_candidates):
        raise ValueError(
            f"anchor_pos out of range: {anchor_pos}, valid=[0,{max(len(anchor_candidates) - 1, 0)}]"
        )
    anchor_t = int(anchor_candidates[anchor_pos])
    if anchor_t + chunk_horizon > episode_len:
        raise ValueError(
            f"anchor_t+chunk_horizon exceeds episode: anchor_t={anchor_t}, "
            f"chunk_horizon={chunk_horizon}, episode_len={episode_len}"
        )

    def make_rng(episode_idx: int, anchor_t_: int) -> np.random.Generator:
        seed = int((episode_idx + 1) * 1_000_003 + anchor_t_ * 97 + source_dt_ms * 17)
        return np.random.default_rng(seed)

    def history_step_times(anchor_t_: int, episode_idx: int) -> List[int]:
        rng = make_rng(episode_idx=episode_idx, anchor_t_=anchor_t_)
        times_rev: List[int] = []
        cursor = int(anchor_t_)
        while True:
            stride = int(rng.choice(step_strides))
            prev_t = cursor - stride
            if prev_t < 0:
                break
            times_rev.append(int(prev_t))
            cursor = int(prev_t)
        return list(reversed(times_rev))

    estimated_prompt_tokens = 64
    estimated_video_tokens_per_frame = 16
    estimated_user_overhead = 14
    estimated_assistant_overhead = 6
    estimated_state_tokens = 1

    def estimate_anchor_tokens() -> int:
        return (
            estimated_user_overhead
            + estimated_state_tokens
            + num_frames * estimated_video_tokens_per_frame
            + estimated_assistant_overhead
            + fixed_action_tokens
        )

    def estimate_history_step_tokens() -> int:
        return (
            estimated_user_overhead
            + estimated_state_tokens
            + num_frames * estimated_video_tokens_per_frame
            + estimated_assistant_overhead
            + fixed_action_tokens
            + 1
        )

    def truncate_history_by_budget(history_t_: List[int]) -> List[int]:
        used = estimated_prompt_tokens + estimate_anchor_tokens()
        keep_rev: List[int] = []
        hist_step_tokens = estimate_history_step_tokens()
        for t_idx in reversed(history_t_):
            if used + hist_step_tokens > max_context_len:
                break
            keep_rev.append(int(t_idx))
            used += hist_step_tokens
        return list(reversed(keep_rev))

    history_times = truncate_history_by_budget(history_step_times(anchor_t, dataset_ep_idx))

    init_idx = min(match_rank, int(init_states.shape[0]) - 1)
    image_key = str(cfg.dataset.image_key)
    aux_image_key = str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    env = LiberoEnv(
        task_name=task_name,
        image_size=128,
        seed=int(cfg.training.seed),
        camera_names=[
            image_key.replace("_rgb", ""),
            *([aux_image_key.replace("_rgb", "")] if aux_image_key is not None else []),
        ],
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

            composed_history_actions, history_compare = _compose_history_actions(
                vla=vla,
                actions=actions,
                history_times=history_times,
                anchor_t=anchor_t,
                chunk_horizon=chunk_horizon,
            )

            current_t = 0
            for hist_t in history_times:
                while current_t < hist_t:
                    action = composed_history_actions[current_t]
                    obs, _, done, _, _ = env.step(action)
                    if save_video_path is not None:
                        frames_to_save.append(env.render())
                    current_t += 1
                    frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                    if done:
                        raise RuntimeError("Environment ended while advancing to the next GT history step.")
                if current_t != hist_t:
                    raise RuntimeError(f"Environment time drifted: current_t={current_t}, expected hist_t={hist_t}")

                state = _build_state_tensor(obs, state_keys, device=device)
                aux_frames = None
                if aux_image_key is not None and aux_image_key in obs:
                    aux_frames = np.asarray(obs[aux_image_key], dtype=np.uint8)[None, ...]
                inserted = vla.insert_step(
                    runner,
                    _window_array(frame_window),
                    aux_frames=aux_frames,
                    state=state,
                    ts=hist_t * source_dt_ms,
                    num_frames=num_frames,
                    source_dt_ms=source_dt_ms,
                )
                if not inserted:
                    raise RuntimeError(f"Failed to insert GT history step at t={hist_t}")

                gt_chunk = actions[hist_t : hist_t + chunk_horizon]
                _append_gt_action_chunk(vla, runner, gt_chunk)

            if current_t != anchor_t:
                while current_t < anchor_t:
                    action = composed_history_actions[current_t]
                    obs, _, done, _, _ = env.step(action)
                    if save_video_path is not None:
                        frames_to_save.append(env.render())
                    current_t += 1
                    frame_window.append(np.asarray(obs[image_key], dtype=np.uint8))
                    if done:
                        raise RuntimeError("Environment ended while advancing to anchor.")
                if current_t != anchor_t:
                    raise RuntimeError(f"Anchor mismatch: current_t={current_t}, anchor_t={anchor_t}")

            state = _build_state_tensor(obs, state_keys, device=device)
            aux_frames = None
            if aux_image_key is not None and aux_image_key in obs:
                aux_frames = np.asarray(obs[aux_image_key], dtype=np.uint8)[None, ...]
            inserted = vla.insert_step(
                runner,
                _window_array(frame_window),
                aux_frames=aux_frames,
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
                "history_steps": int(len(history_times)),
                "step_dt_min_ms": int(step_dt_min_ms),
                "step_dt_max_ms": int(step_dt_max_ms),
                "anchor_stride_steps": int(anchor_stride_steps),
                "num_frames": int(num_frames),
                "anchor_pos": int(anchor_pos),
                "anchor_t": int(anchor_t),
                "history_times": history_times,
                "chunk_horizon": int(chunk_horizon),
                **history_compare,
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
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location="cpu")
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
        save_video_path=video_path,
        temperature=float(args.temperature),
        top_k=args.top_k,
    )
    if video_path is not None:
        metrics["video_path"] = video_path
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
