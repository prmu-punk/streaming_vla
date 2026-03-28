from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from dataset.libero90_async_dataset import LiberoEpisodeDataset
from model.rtc_async.action_expert.runner import ActionExpertRunner, ActionExpertRunnerConfig
from model.rtc_async.qwen3_stream.kv_export import export_selected_kv_cache
from model.template_qwen3_vla import build_prompt_prefill_text, build_step_user_prefix, build_video_text
from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder
from normalization import RTCNormalizer
from workspace.rollout_libero_rtc_async_video import (
    LiberoEnv,
    _build_state_tensor,
    _load_task_init_states,
    _save_video,
    _set_init_state,
    benchmark,
    task_name_to_suite_and_ids,
)


@dataclass
class SimRuntime:
    encoder: Qwen3RTCVLAEncoder
    action_expert: ActionExpertRunner
    selected_layers: List[int]
    normalizer: RTCNormalizer | None
    max_context_len: int


def _resolve_repo_path(path_str: str) -> pathlib.Path:
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def _find_matching_episode_indices(dataset: LiberoEpisodeDataset, prompt: str) -> List[int]:
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if str(dataset.get_prompt(ep_idx)) == prompt:
            out.append(int(ep_idx))
    return out


def _load_rtc_runtime(path: pathlib.Path) -> tuple[List[int], Dict[str, Any], int]:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    stream = raw.get("stream", {}) or {}
    selected_layers = [int(x) for x in stream.get("selected_layers", [])]
    if not selected_layers:
        raise ValueError("rtc_async.stream.selected_layers must be non-empty.")
    action_cfg = dict(raw.get("action_expert", {}) or {})
    max_context_len = int(float(stream.get("max_context_len", 10_000)))
    return selected_layers, action_cfg, max_context_len


def _build_action_expert(*, encoder: Qwen3RTCVLAEncoder, action_cfg: Dict[str, Any]) -> ActionExpertRunner:
    return ActionExpertRunner(
        ActionExpertRunnerConfig(
            state_dim=int(encoder.state_dim),
            action_dim=int(action_cfg["action_dim"]),
            horizon=int(action_cfg["horizon"]),
            cond_dim=int(encoder.kv_cache_dim),
            hidden_size=int(action_cfg.get("hidden_size", 512)),
            num_layers=int(action_cfg.get("num_layers", 8)),
            num_heads=int(action_cfg.get("num_heads", 8)),
            mlp_ratio=float(action_cfg.get("mlp_ratio", 4.0)),
            time_embed_dim=int(action_cfg.get("time_embed_dim", 256)),
            norm_eps=float(action_cfg.get("norm_eps", 1e-6)),
            ffn_multiple_of=int(action_cfg.get("ffn_multiple_of", 256)),
            ffn_dim_multiplier=action_cfg.get("ffn_dim_multiplier", None),
            num_inference_steps=int(action_cfg.get("num_inference_steps", 5)),
        )
    ).to(encoder.device)


def _build_runtime(*, cfg, checkpoint_path: pathlib.Path) -> SimRuntime:
    encoder = Qwen3RTCVLAEncoder(config_path=str(_resolve_repo_path(str(cfg.model.vla_config_path))))
    selected_layers, action_cfg, max_context_len = _load_rtc_runtime(
        _resolve_repo_path(str(cfg.rtc_async.config_path))
    )
    action_expert = _build_action_expert(encoder=encoder, action_cfg=action_cfg)

    payload = torch.load(str(checkpoint_path), map_location=encoder.device)
    vla_state = payload.get("vla", None)
    if vla_state is not None:
        encoder.load_state_dict(vla_state, strict=False)

    action_state = payload.get("action_expert", payload)
    action_expert.load_state_dict(action_state, strict=False)
    action_expert.eval()

    normalization_payload = payload.get("normalization", None)
    normalizer = RTCNormalizer.from_payload(normalization_payload) if normalization_payload is not None else None
    return SimRuntime(
        encoder=encoder,
        action_expert=action_expert,
        selected_layers=selected_layers,
        normalizer=normalizer,
        max_context_len=max_context_len,
    )


def _video_window_indices(t_idx: int, num_frames: int, frame_stride_steps: int) -> List[int]:
    end = int(t_idx)
    return [
        max(0, end - int(frame_stride_steps) * (int(num_frames) - 1 - i))
        for i in range(int(num_frames))
    ]


def _make_video_tensor(frames: np.ndarray | torch.Tensor, num_frames: int) -> torch.Tensor:
    if isinstance(frames, np.ndarray):
        frames_t = torch.from_numpy(frames)
    else:
        frames_t = frames
    if frames_t.dim() == 3:
        frames_t = frames_t.unsqueeze(0)
    if frames_t.shape[-1] == 3:
        frames_t = frames_t.permute(0, 3, 1, 2)
    if frames_t.shape[0] < num_frames:
        repeat = num_frames - frames_t.shape[0]
        frames_t = torch.cat([frames_t, frames_t[-1:].repeat(repeat, 1, 1, 1)], dim=0)
    elif frames_t.shape[0] > num_frames:
        frames_t = frames_t[:num_frames]
    return frames_t


def _build_step_text(*, ts_ms: int | None, video_token: str, has_aux: bool) -> str:
    return build_step_user_prefix(
        ts_ms=ts_ms,
        video_token=build_video_text(video_token=video_token, has_aux=has_aux),
    )


def _prompt_length(encoder: Qwen3RTCVLAEncoder, prompt: str) -> int:
    prompt_text = build_prompt_prefill_text(str(prompt))
    encoded = encoder.processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return int(len(encoded["input_ids"]))


def _step_length(
    *,
    encoder: Qwen3RTCVLAEncoder,
    images: torch.Tensor,
    aux_stack: torch.Tensor | None,
    source_dt_ms: int,
    num_frames: int,
    frame_stride_steps: int,
) -> int:
    has_aux = aux_stack is not None
    t_idx = max(0, int(images.shape[0] // 2))
    step_text = _build_step_text(
        ts_ms=int(t_idx * int(source_dt_ms)),
        video_token=encoder.processor.image_token,
        has_aux=has_aux,
    )
    images_for_step = [[images[int(t_idx)]]]
    if has_aux:
        images_for_step[0].append(aux_stack[int(t_idx)])
    proc = encoder.processor(
        text=[step_text],
        images=images_for_step,
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return int(proc["input_ids"].shape[1])


def run_gt_prefix_rollout(
    *,
    runtime: SimRuntime,
    cfg,
    task_name: str,
    match_rank: int,
    anchor_pos: int,
    save_video_path: Optional[str],
) -> Dict[str, Any]:
    if task_name not in task_name_to_suite_and_ids:
        raise ValueError(f"Unknown LIBERO task: {task_name}")

    encoder = runtime.encoder
    action_expert = runtime.action_expert
    selected_layers = runtime.selected_layers
    normalizer = runtime.normalizer
    max_context_len = runtime.max_context_len

    source_dt_ms = int(cfg.training.source_dt_ms)
    num_frames = int(cfg.model.num_frames)
    anchor_stride_steps = int(cfg.dataset.anchor_stride_steps or 1)
    step_dt_min_ms = int(cfg.training.step_dt_min_ms)
    step_dt_max_ms = int(cfg.training.step_dt_max_ms)

    stride_min = max(1, int(round(float(step_dt_min_ms) / float(source_dt_ms))))
    stride_max = max(1, int(round(float(step_dt_max_ms) / float(source_dt_ms))))
    step_strides = list(range(stride_min, stride_max + 1))

    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)

    dataset = LiberoEpisodeDataset(
        zarr_path=str(_resolve_repo_path(str(cfg.dataset.zarr_path))),
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

    dataset_ep_idx = matched[match_rank]
    ep = dataset[dataset_ep_idx]
    actions = ep["actions"]
    episode_len = int(actions.shape[0])
    chunk_horizon = int(action_expert.config.horizon)

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

    aux_stack = None
    if cfg.dataset.get("aux_image_key", None):
        aux_stack = ep.get("extra_images", {}).get(str(cfg.dataset.aux_image_key), None)
    prompt_len = _prompt_length(encoder, str(ep["prompt"]))
    step_len = _step_length(
        encoder=encoder,
        images=ep["images"],
        aux_stack=aux_stack,
        source_dt_ms=source_dt_ms,
        num_frames=num_frames,
        frame_stride_steps=1,
    )
    base_len = prompt_len + step_len
    available_len = max(max_context_len - base_len, 0)
    max_history_steps = available_len // max(step_len, 1)
    full_history_times = history_step_times(anchor_t, dataset_ep_idx)
    history_times = full_history_times[-max_history_steps:] if max_history_steps > 0 else []

    init_idx = min(match_rank, int(init_states.shape[0]) - 1)
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
        max_episode_steps=550,
    )

    frames_to_save: List[np.ndarray] = []
    metrics: Dict[str, Any] = {}
    try:
        with torch.inference_mode():
            obs, _ = env.reset()
            del obs
            obs = _set_init_state(env, init_states[init_idx])

            prompt = str(obs["prompt"])
            if save_video_path is not None:
                frames_to_save.append(env.render())

            context_videos: List[np.ndarray] = []
            context_aux_videos: List[np.ndarray] = []
            current_t = 0
            for hist_t in history_times:
                while current_t < hist_t:
                    action = actions[current_t].detach().to("cpu").numpy().astype(np.float32)
                    obs, _, done, _, _ = env.step(action)
                    if save_video_path is not None:
                        frames_to_save.append(env.render())
                    current_t += 1
                    if done:
                        raise RuntimeError("Environment ended while advancing to the next GT history step.")
                if current_t != hist_t:
                    raise RuntimeError(f"Environment time drifted: current_t={current_t}, expected hist_t={hist_t}")

                context_videos.append(
                    np.asarray(obs[image_key], dtype=np.uint8)[None, ...]
                )
                if aux_image_key is not None and aux_image_key in obs:
                    context_aux_videos.append(np.asarray(obs[aux_image_key], dtype=np.uint8)[None, ...])

            while current_t < anchor_t:
                action = actions[current_t].detach().to("cpu").numpy().astype(np.float32)
                obs, _, done, _, _ = env.step(action)
                if save_video_path is not None:
                    frames_to_save.append(env.render())
                current_t += 1
                if done:
                    raise RuntimeError("Environment ended while advancing to anchor.")
            if current_t != anchor_t:
                raise RuntimeError(f"Anchor mismatch: current_t={current_t}, anchor_t={anchor_t}")

            state = _build_state_tensor(obs, state_keys, device=encoder.device)
            anchor_video_np = np.asarray(obs[image_key], dtype=np.uint8)[None, ...]
            aux_frames = None
            if aux_image_key is not None and aux_image_key in obs:
                aux_frames = np.asarray(obs[aux_image_key], dtype=np.uint8)[None, ...]

            if context_videos:
                context_videos_t = torch.from_numpy(np.stack(context_videos, axis=0))
            else:
                context_videos_t = torch.empty((0, num_frames, *anchor_video_np.shape[1:]), dtype=torch.uint8)

            if context_aux_videos:
                context_aux_videos_t = torch.from_numpy(np.stack(context_aux_videos, axis=0))
            else:
                context_aux_videos_t = torch.empty((0, 1, *anchor_video_np.shape[1:]), dtype=torch.uint8)

            if aux_frames is not None:
                anchor_aux_video = torch.from_numpy(aux_frames)
            else:
                anchor_aux_video = torch.empty((0, *anchor_video_np.shape[1:]), dtype=torch.uint8)

            sample = {
                "prompt": prompt,
                "context_videos": context_videos_t,
                "context_aux_videos": context_aux_videos_t,
                "context_time_indices": torch.tensor(history_times, dtype=torch.long),
                "anchor_video": torch.from_numpy(anchor_video_np),
                "anchor_aux_video": anchor_aux_video,
                "anchor_time_idx": torch.tensor(anchor_t, dtype=torch.long),
                "target_chunk": torch.zeros((chunk_horizon, int(actions.shape[-1])), dtype=torch.float32),
            }
            encoded = encoder.forward_offline_context_batch(
                samples=[sample],
                num_frames=num_frames,
                source_dt_ms=source_dt_ms,
                return_condition_cache=True,
            )
            kv_cache = export_selected_kv_cache(
                past_key_values=encoded["past_key_values"],
                selected_layers=selected_layers,
                clone=False,
            )
            sample_state = state
            if normalizer is not None:
                sample_state = normalizer.normalize_state(sample_state)
            pred_chunk = action_expert.sample(
                state=sample_state,
                kv_cache=kv_cache,
                attention_mask=encoded["attention_mask"],
                prompt_mask=encoded["prompt_mask"],
                step_mask=encoded["step_mask"],
            )
            if normalizer is not None:
                pred_chunk = normalizer.unnormalize_action(pred_chunk)

            pred_chunk_np = pred_chunk[0].detach().to("cpu").numpy().astype(np.float32)
            success = False
            done = bool(env.done)
            for action in pred_chunk_np:
                obs, reward, done, _, _ = env.step(action)
                if save_video_path is not None:
                    frames_to_save.append(env.render())
                current_t += 1
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
                "frame_stride_steps": 1,
                "num_frames": 1,
                "observation_mode": "image",
                "anchor_pos": int(anchor_pos),
                "anchor_t": int(anchor_t),
                "history_times": history_times,
                "chunk_horizon": int(chunk_horizon),
                "selected_layers": [int(x) for x in selected_layers],
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
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--anchor-pos", type=int, default=0)
    parser.add_argument("--save-video", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(str(_resolve_repo_path(args.config)))
    OmegaConf.resolve(cfg)

    runtime = _build_runtime(cfg=cfg, checkpoint_path=_resolve_repo_path(args.checkpoint))

    video_path = None
    if args.save_video:
        video_path = str(
            pathlib.Path.cwd() / "videos" / "gt_prefix_sim" / str(args.task) / f"match_{int(args.match_rank):03d}.mp4"
        )

    metrics = run_gt_prefix_rollout(
        runtime=runtime,
        cfg=cfg,
        task_name=str(args.task),
        match_rank=int(args.match_rank),
        anchor_pos=int(args.anchor_pos),
        save_video_path=video_path,
    )
    if video_path is not None:
        metrics["video_path"] = video_path
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
