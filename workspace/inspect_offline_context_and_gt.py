from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf


ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from dataset.libero90_async_dataset import LiberoEpisodeDataset
from model.template_qwen3_vla import build_prompt_prefill_text, build_step_user_prefix, build_video_text
from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder
from workspace.rollout_libero_rtc_async_video import (
    LiberoEnv,
    _load_task_init_states,
    _save_video,
    _set_init_state,
    benchmark,
    task_name_to_suite_and_ids,
)


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


def _task_uid_for_episode(dataset: LiberoEpisodeDataset, episode_idx: int) -> int:
    ep_start, _ = dataset.get_episode_bounds(int(episode_idx))
    value = np.asarray(dataset.buffer["task_uid"][ep_start]).reshape(-1)[0]
    return int(value)


def _find_episode_indices_by_task_uid(dataset: LiberoEpisodeDataset, task_uid: int) -> List[int]:
    out: List[int] = []
    for ep_idx in range(len(dataset)):
        if _task_uid_for_episode(dataset, ep_idx) == int(task_uid):
            out.append(int(ep_idx))
    return out


def _find_task_name_by_uid(task_uid: int) -> str:
    for task_name, (_, _, uid) in task_name_to_suite_and_ids.items():
        if int(uid) == int(task_uid):
            return str(task_name)
    raise ValueError(f"Failed to map task_uid to LIBERO task: {task_uid}")


def _find_task_name_by_prompt(prompt: str) -> str:
    for task_name, (suite_name, task_id, _) in task_name_to_suite_and_ids.items():
        task_suite = benchmark.get_benchmark_dict()[suite_name]()
        task = task_suite.get_task(task_id)
        if str(task.language) == str(prompt):
            return str(task_name)
    raise ValueError(f"Failed to map prompt to LIBERO task: {prompt}")


def _build_sample_text(
    *,
    encoder: Qwen3RTCVLAEncoder,
    prompt: str,
    history_t: List[int],
    anchor_t: int,
    source_dt_ms: int,
    has_aux: bool,
) -> tuple[str, str, List[str]]:
    prompt_text = build_prompt_prefill_text(str(prompt))
    step_texts: List[str] = []
    for t_idx in history_t:
        ts_ms = int(int(t_idx) * int(source_dt_ms))
        step_texts.append(
            build_step_user_prefix(
                ts_ms=ts_ms,
                video_token=build_video_text(video_token=encoder.processor.image_token, has_aux=has_aux),
            )
        )
    anchor_text = build_step_user_prefix(
        ts_ms=int(int(anchor_t) * int(source_dt_ms)),
        video_token=build_video_text(video_token=encoder.processor.image_token, has_aux=has_aux),
    )
    return prompt_text + "".join(step_texts) + anchor_text, prompt_text, step_texts + [anchor_text]


def _make_rng(episode_idx: int, anchor_t: int, source_dt_ms: int) -> np.random.Generator:
    seed = int((episode_idx + 1) * 1_000_003 + anchor_t * 97 + source_dt_ms * 17)
    return np.random.default_rng(seed)


def _history_step_times(
    *,
    episode_idx: int,
    anchor_t: int,
    step_strides: List[int],
    source_dt_ms: int,
) -> List[int]:
    rng = _make_rng(episode_idx=episode_idx, anchor_t=anchor_t, source_dt_ms=source_dt_ms)
    times_rev: List[int] = []
    cursor = int(anchor_t)
    while True:
        stride = int(rng.choice(step_strides))
        prev_t = cursor - stride
        if prev_t < 0:
            break
        times_rev.append(int(prev_t))
        cursor = int(prev_t)
    return list(reversed(times_rev))


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
    image: torch.Tensor,
    aux_image: torch.Tensor | None,
    t_idx: int,
    source_dt_ms: int,
) -> int:
    has_aux = aux_image is not None
    step_text = build_step_user_prefix(
        ts_ms=int(t_idx) * int(source_dt_ms),
        video_token=build_video_text(video_token=encoder.processor.image_token, has_aux=has_aux),
    )
    images = [image]
    if aux_image is not None:
        images.append(aux_image)
    proc = encoder.processor(
        text=[step_text],
        images=[images],
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return int(proc["input_ids"].shape[1])


def _rollout_gt_episode(
    *,
    cfg,
    task_name: str,
    match_rank: int,
    episode_actions: torch.Tensor,
    save_video_path: str | None,
) -> Dict[str, Any]:
    suite_name, task_id, _ = task_name_to_suite_and_ids[task_name]
    task_suite = benchmark.get_benchmark_dict()[suite_name]()
    task = task_suite.get_task(task_id)
    init_states = _load_task_init_states(task)
    init_idx = min(int(match_rank), int(init_states.shape[0]) - 1)

    image_key = str(cfg.dataset.image_key)
    aux_image_key = str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None
    state_keys = [str(k) for k in cfg.dataset.state_keys]
    image_size = 256

    env = LiberoEnv(
        task_name=task_name,
        image_size=image_size,
        seed=int(cfg.training.seed),
        camera_names=[
            image_key.replace("_rgb", ""),
            *([aux_image_key.replace("_rgb", "")] if aux_image_key is not None else []),
        ],
        state_ports=state_keys,
        max_episode_steps=max(550, int(episode_actions.shape[0]) + 5),
    )

    frames: List[np.ndarray] = []
    rewards: List[float] = []
    success = False
    done = False
    try:
        obs, _ = env.reset()
        del obs
        obs = _set_init_state(env, init_states[init_idx])
        if save_video_path is not None:
            frames.append(env.render())

        for step_idx in range(int(episode_actions.shape[0])):
            action = episode_actions[step_idx].detach().cpu().numpy().astype(np.float32)
            obs, reward, done, _, _ = env.step(action)
            rewards.append(float(reward))
            if save_video_path is not None:
                frames.append(env.render())
            if reward >= 1.0:
                success = True
            if done:
                break
    finally:
        env.close()

    if save_video_path is not None:
        _save_video(save_video_path, frames, fps=max(1, round(1000 / int(cfg.training.source_dt_ms))))

    return {
        "gt_rollout_steps": int(len(rewards)),
        "gt_rollout_success": bool(success),
        "gt_rollout_done": bool(done),
        "gt_rollout_reward_max": float(max(rewards) if rewards else 0.0),
        "gt_rollout_reward_final": float(rewards[-1] if rewards else 0.0),
    }


def _save_rgb_image(path: pathlib.Path, image: torch.Tensor | np.ndarray) -> None:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().numpy()
    else:
        arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
        arr = np.transpose(arr, (1, 2, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, arr.astype(np.uint8))


def _extract_image_token_spans(
    *,
    encoder: Qwen3RTCVLAEncoder,
    step_text: str,
    step_images: List[torch.Tensor],
) -> Dict[str, Any]:
    proc = encoder.processor(
        text=[step_text],
        images=[step_images],
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
        return_mm_token_type_ids=True,
    )
    input_ids = proc["input_ids"][0]
    image_grid_thw = proc["image_grid_thw"]
    mm_token_type_ids = proc["mm_token_type_ids"][0]
    image_token_id = int(encoder.processor.image_token_id)
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=False).flatten().tolist()
    if not image_positions:
        return {
            "step_token_count": int(input_ids.shape[0]),
            "num_images": 0,
            "image_token_spans": [],
        }

    merge_size = int(encoder.processor.image_processor.merge_size)
    expected_counts = [
        int((int(grid[0]) * int(grid[1]) * int(grid[2])) // (merge_size**2))
        for grid in image_grid_thw.tolist()
    ]

    spans: List[Dict[str, int]] = []
    cursor = 0
    for image_idx, expected_count in enumerate(expected_counts):
        if cursor >= len(image_positions):
            break
        start = image_positions[cursor]
        end_cursor = cursor + expected_count - 1
        if end_cursor >= len(image_positions):
            raise ValueError(
                f"Image token count mismatch for image {image_idx}: expected {expected_count}, "
                f"available={len(image_positions) - cursor}"
            )
        end = image_positions[end_cursor]
        spans.append(
            {
                "image_index": int(image_idx),
                "token_start": int(start),
                "token_end_exclusive": int(end + 1),
                "token_count": int(expected_count),
            }
        )
        cursor += expected_count

    return {
        "step_token_count": int(input_ids.shape[0]),
        "mm_token_count": int(mm_token_type_ids.sum().item()),
        "num_images": int(len(expected_counts)),
        "image_token_spans": spans,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_libero90_async.yaml")
    parser.add_argument("--episode-idx", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--match-rank", type=int, default=0)
    parser.add_argument("--anchor-pos", type=int, default=0)
    parser.add_argument("--anchor-t", type=int, default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-step-images", action="store_true")
    args = parser.parse_args()

    cfg = OmegaConf.load(str(_resolve_repo_path(args.config)))
    OmegaConf.resolve(cfg)
    rtc_cfg = OmegaConf.load(str(_resolve_repo_path(str(cfg.rtc_async.config_path))))
    OmegaConf.resolve(rtc_cfg)

    encoder = Qwen3RTCVLAEncoder(config_path=str(_resolve_repo_path(str(cfg.model.vla_config_path))))
    episode_dataset = LiberoEpisodeDataset(
        zarr_path=str(_resolve_repo_path(str(cfg.dataset.zarr_path))),
        image_key=str(cfg.dataset.image_key),
        extra_image_keys=([str(cfg.dataset.aux_image_key)] if cfg.dataset.get("aux_image_key", None) else []),
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        max_episodes=cfg.dataset.max_episodes,
    )

    if args.task is not None:
        task_name = str(args.task)
        if task_name not in task_name_to_suite_and_ids:
            raise ValueError(f"Unknown LIBERO task: {task_name}")
        _, _, task_uid = task_name_to_suite_and_ids[task_name]
        matched = _find_episode_indices_by_task_uid(episode_dataset, int(task_uid))
        if not matched:
            raise ValueError(f"No dataset episodes matched task uid: task={task_name}, task_uid={task_uid}")
        if int(args.match_rank) < 0 or int(args.match_rank) >= len(matched):
            raise ValueError(f"match_rank out of range: {args.match_rank}, total_matches={len(matched)}")
        episode_idx = int(matched[int(args.match_rank)])
    elif args.episode_idx is not None:
        episode_idx = int(args.episode_idx)
    else:
        raise ValueError("Provide either --episode-idx or --task.")

    ep = episode_dataset[episode_idx]
    prompt = str(ep["prompt"])
    task_uid = _task_uid_for_episode(episode_dataset, episode_idx)
    matched = _find_episode_indices_by_task_uid(episode_dataset, task_uid)
    match_rank = matched.index(episode_idx)
    task_name = str(args.task) if args.task is not None else _find_task_name_by_uid(task_uid)

    chunk_horizon = int(rtc_cfg.action_expert.horizon)
    anchor_stride_steps = int(cfg.dataset.anchor_stride_steps)
    episode_len = int(ep["actions"].shape[0])
    anchor_candidates = list(range(0, episode_len - chunk_horizon + 1, anchor_stride_steps))
    if not anchor_candidates:
        raise ValueError(f"No valid anchors for episode_idx={episode_idx}, episode_len={episode_len}")
    if args.anchor_t is not None:
        anchor_t = int(args.anchor_t)
        if anchor_t not in anchor_candidates:
            raise ValueError(
                f"anchor_t={anchor_t} is invalid. It must be one of the anchor grid values; "
                f"first={anchor_candidates[0]}, last={anchor_candidates[-1]}, stride={anchor_stride_steps}"
            )
        anchor_pos = anchor_candidates.index(anchor_t)
    else:
        anchor_pos = int(args.anchor_pos)
        if anchor_pos < 0 or anchor_pos >= len(anchor_candidates):
            raise ValueError(f"anchor_pos out of range: {anchor_pos}, total={len(anchor_candidates)}")
        anchor_t = int(anchor_candidates[anchor_pos])

    source_dt_ms = int(cfg.training.source_dt_ms)
    stride_min = max(1, int(round(float(cfg.training.step_dt_min_ms) / float(source_dt_ms))))
    stride_max = max(1, int(round(float(cfg.training.step_dt_max_ms) / float(source_dt_ms))))
    step_strides = list(range(stride_min, stride_max + 1))
    full_history_t = _history_step_times(
        episode_idx=episode_idx,
        anchor_t=anchor_t,
        step_strides=step_strides,
        source_dt_ms=source_dt_ms,
    )

    aux_key = str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None
    aux_stack = ep.get("extra_images", {}).get(aux_key, None) if aux_key is not None else None
    prompt_len = _prompt_length(encoder, prompt)
    step_len = _step_length(
        encoder=encoder,
        image=ep["images"][anchor_t],
        aux_image=(aux_stack[anchor_t] if aux_stack is not None else None),
        t_idx=anchor_t,
        source_dt_ms=source_dt_ms,
    )
    base_len = prompt_len + step_len
    available_len = max(int(rtc_cfg.stream.max_context_len) - base_len, 0)
    max_history_steps = available_len // max(step_len, 1)
    history_t = full_history_t[-max_history_steps:] if max_history_steps > 0 else []

    has_aux = bool(cfg.dataset.get("aux_image_key", None))
    full_text, prompt_text, step_texts = _build_sample_text(
        encoder=encoder,
        prompt=prompt,
        history_t=history_t,
        anchor_t=anchor_t,
        source_dt_ms=int(cfg.training.source_dt_ms),
        has_aux=has_aux,
    )

    proc_images: List[torch.Tensor] = []
    for t_idx in history_t:
        proc_images.append(ep["images"][t_idx])
        if aux_stack is not None:
            proc_images.append(aux_stack[t_idx])
    proc_images.append(ep["images"][anchor_t])
    if aux_stack is not None:
        proc_images.append(aux_stack[anchor_t])

    proc = encoder.processor(
        text=[full_text],
        images=[proc_images],
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    token_count = int(proc["input_ids"].shape[1])
    target_chunk = ep["actions"][anchor_t : anchor_t + chunk_horizon]

    anchor_step_images = [ep["images"][anchor_t]]
    if aux_stack is not None:
        anchor_step_images.append(aux_stack[anchor_t])
    anchor_step_text = step_texts[-1]
    anchor_step_span_info = _extract_image_token_spans(
        encoder=encoder,
        step_text=anchor_step_text,
        step_images=anchor_step_images,
    )

    video_path = None
    if args.save_video:
        video_path = str(
            pathlib.Path.cwd() / "videos" / "inspect_offline_context_gt" / task_name / f"ep_{episode_idx:06d}_anchor_{anchor_t:04d}.mp4"
        )
    saved_step_images: Dict[str, str] = {}
    if args.save_step_images:
        step_dir = pathlib.Path.cwd() / "debug" / "inspect_step_images" / task_name / f"ep_{episode_idx:06d}_anchor_{anchor_t:04d}"
        main_path = step_dir / "anchor_main.png"
        _save_rgb_image(main_path, ep["images"][anchor_t])
        saved_step_images["anchor_main"] = str(main_path)
        if aux_stack is not None:
            wrist_path = step_dir / "anchor_wrist.png"
            _save_rgb_image(wrist_path, aux_stack[anchor_t])
            saved_step_images["anchor_wrist"] = str(wrist_path)

    rollout_metrics = _rollout_gt_episode(
        cfg=cfg,
        task_name=task_name,
        match_rank=match_rank,
        episode_actions=ep["actions"],
        save_video_path=video_path,
    )

    result = {
        "task_name": task_name,
        "prompt": prompt,
        "episode_idx": episode_idx,
        "match_rank": int(match_rank),
        "episode_len": episode_len,
        "anchor_pos": int(anchor_pos),
        "anchor_t": anchor_t,
        "step_len": int(step_len),
        "prompt_len": int(prompt_len),
        "history_t": history_t,
        "history_steps": len(history_t),
        "sample_length_estimate": int(base_len + len(history_t) * step_len),
        "max_context_len": int(rtc_cfg.stream.max_context_len),
        "actual_token_count": token_count,
        "prompt_text": prompt_text,
        "step_texts": step_texts,
        "full_context_text": full_text,
        "target_chunk_shape": list(target_chunk.shape),
        "anchor_step_text": anchor_step_text,
        "anchor_step_image_token_info": anchor_step_span_info,
    }
    result.update(rollout_metrics)
    if video_path is not None:
        result["video_path"] = video_path
    if saved_step_images:
        result["saved_step_images"] = saved_step_images

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
