from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np
import torch
import yaml

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from dataset.libero90_async_dataset import LiberoEpisodeDataset
from model.qwen3_vl import Qwen3VLProcessor
from model.template_qwen3_vla import IM_END, build_prompt_prefill_text, build_step_user_prefix, build_video_text


def resolve_workspace_path(path_str: str) -> str:
    path = pathlib.Path(path_str)
    if path.is_absolute():
        return str(path)
    return str(pathlib.Path(ROOT_DIR) / path)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def spans_from_mask(mask_1d: torch.Tensor) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for idx, flag in enumerate(mask_1d.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx))
            start = None
    if start is not None:
        spans.append((start, int(mask_1d.shape[0])))
    return spans


def format_spans(spans: Iterable[tuple[int, int]]) -> str:
    items = [f"[{start}, {end})" for start, end in spans]
    return ", ".join(items) if items else "<empty>"


def make_rng(episode_idx: int, anchor_t: int, source_dt_ms: int) -> np.random.Generator:
    seed = int((episode_idx + 1) * 1_000_003 + anchor_t * 97 + source_dt_ms * 17)
    return np.random.default_rng(seed)


def video_window_indices(t_idx: int, num_frames: int) -> list[int]:
    start = int(t_idx) - int(num_frames) + 1
    return [max(0, start + i) for i in range(int(num_frames))]


def make_video_tensor(frames: np.ndarray | torch.Tensor, num_frames: int) -> torch.Tensor:
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


def build_step_text(*, ts_ms: int | None, video_token: str, has_aux: bool) -> str:
    return (
        build_step_user_prefix(
            ts_ms=ts_ms,
            video_token=build_video_text(video_token=video_token, has_aux=has_aux),
            close_previous_assistant=False,
        )
        + IM_END
        + "\n"
    )


def prompt_length(processor: Qwen3VLProcessor, prompt: str) -> int:
    prompt_text = build_prompt_prefill_text(prompt)
    encoded = processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return int(len(encoded["input_ids"]))


def step_length(
    *,
    processor: Qwen3VLProcessor,
    video: torch.Tensor,
    aux_video: torch.Tensor | None,
    ts_ms: int,
    video_token: str,
) -> int:
    has_aux = aux_video is not None and int(aux_video.shape[0]) > 0
    step_text = build_step_text(ts_ms=ts_ms, video_token=video_token, has_aux=has_aux)
    videos = [video]
    if has_aux:
        assert aux_video is not None
        videos.append(aux_video)
    proc = processor(
        text=[step_text],
        videos=[videos],
        padding=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return int(proc["input_ids"].shape[1])


def history_step_times(
    *,
    episode_idx: int,
    anchor_t: int,
    source_dt_ms: int,
    step_dt_min_ms: int,
    step_dt_max_ms: int,
) -> list[int]:
    stride_min = max(1, int(round(float(step_dt_min_ms) / float(source_dt_ms))))
    stride_max = max(1, int(round(float(step_dt_max_ms) / float(source_dt_ms))))
    step_strides = list(range(stride_min, stride_max + 1))
    rng = make_rng(episode_idx=episode_idx, anchor_t=anchor_t, source_dt_ms=source_dt_ms)
    times_rev: list[int] = []
    cursor = int(anchor_t)
    while True:
        stride = int(rng.choice(step_strides))
        prev_t = cursor - stride
        if prev_t < 0:
            break
        times_rev.append(int(prev_t))
        cursor = int(prev_t)
    return list(reversed(times_rev))


def trim_history_to_context(
    *,
    processor: Qwen3VLProcessor,
    episode: dict,
    episode_idx: int,
    full_history_t: list[int],
    anchor_t: int,
    source_dt_ms: int,
    num_frames: int,
    max_context_len: int,
    aux_image_key: str | None,
) -> list[int]:
    video_token = processor.video_token
    prompt_len = prompt_length(processor, str(episode["prompt"]))

    anchor_frame_ids = video_window_indices(anchor_t, num_frames)
    anchor_video = make_video_tensor(episode["images"][torch.as_tensor(anchor_frame_ids, dtype=torch.long)], num_frames)
    anchor_aux_video = None
    if aux_image_key is not None:
        aux_stack = episode["extra_images"].get(aux_image_key, None)
        if aux_stack is not None:
            anchor_aux_video = make_video_tensor(aux_stack[anchor_t].unsqueeze(0), 1)
    anchor_len = step_length(
        processor=processor,
        video=anchor_video,
        aux_video=anchor_aux_video,
        ts_ms=int(anchor_t) * int(source_dt_ms),
        video_token=video_token,
    )

    kept_rev: list[int] = []
    total_len = prompt_len + anchor_len
    aux_stack = episode["extra_images"].get(aux_image_key, None) if aux_image_key is not None else None
    for t_idx in reversed(full_history_t):
        frame_ids = video_window_indices(int(t_idx), num_frames)
        hist_video = make_video_tensor(episode["images"][torch.as_tensor(frame_ids, dtype=torch.long)], num_frames)
        hist_aux = None
        if aux_stack is not None:
            hist_aux = make_video_tensor(aux_stack[int(t_idx)].unsqueeze(0), 1)
        candidate_len = total_len + step_length(
            processor=processor,
            video=hist_video,
            aux_video=hist_aux,
            ts_ms=int(t_idx) * int(source_dt_ms),
            video_token=video_token,
        )
        if candidate_len > max_context_len:
            break
        kept_rev.append(int(t_idx))
        total_len = int(candidate_len)
    return list(reversed(kept_rev))


def build_single_sample(
    *,
    processor: Qwen3VLProcessor,
    dataset: LiberoEpisodeDataset,
    episode_idx: int,
    anchor_t: int,
    source_dt_ms: int,
    step_dt_min_ms: int,
    step_dt_max_ms: int,
    num_frames: int,
    horizon: int,
    max_context_len: int,
    aux_image_key: str | None,
) -> dict:
    episode = dataset[int(episode_idx)]
    full_history_t = history_step_times(
        episode_idx=episode_idx,
        anchor_t=anchor_t,
        source_dt_ms=source_dt_ms,
        step_dt_min_ms=step_dt_min_ms,
        step_dt_max_ms=step_dt_max_ms,
    )
    history_t = trim_history_to_context(
        processor=processor,
        episode=episode,
        episode_idx=episode_idx,
        full_history_t=full_history_t,
        anchor_t=anchor_t,
        source_dt_ms=source_dt_ms,
        num_frames=num_frames,
        max_context_len=max_context_len,
        aux_image_key=aux_image_key,
    )

    context_videos = []
    context_aux_videos = []
    aux_stack = episode["extra_images"].get(aux_image_key, None) if aux_image_key is not None else None
    for t_idx in history_t:
        frame_ids = video_window_indices(int(t_idx), num_frames)
        context_videos.append(episode["images"][torch.as_tensor(frame_ids, dtype=torch.long)])
        if aux_stack is not None:
            context_aux_videos.append(aux_stack[int(t_idx)].unsqueeze(0))

    if context_videos:
        context_videos_t = torch.stack(context_videos, dim=0)
    else:
        context_videos_t = torch.empty((0, num_frames, *episode["images"].shape[1:]), dtype=episode["images"].dtype)

    if context_aux_videos:
        context_aux_videos_t = torch.stack(context_aux_videos, dim=0)
    else:
        context_aux_videos_t = torch.empty((0, 1, *episode["images"].shape[1:]), dtype=episode["images"].dtype)

    anchor_frame_ids = video_window_indices(anchor_t, num_frames)
    anchor_video = episode["images"][torch.as_tensor(anchor_frame_ids, dtype=torch.long)]
    if aux_stack is not None:
        anchor_aux_video = aux_stack[anchor_t].unsqueeze(0)
    else:
        anchor_aux_video = torch.empty((0, *episode["images"].shape[1:]), dtype=episode["images"].dtype)
    target_t = list(range(anchor_t, anchor_t + horizon))
    target_chunk = episode["actions"][torch.as_tensor(target_t, dtype=torch.long)]

    return {
        "prompt": episode["prompt"],
        "context_videos": context_videos_t,
        "context_aux_videos": context_aux_videos_t,
        "context_time_indices": torch.tensor(history_t, dtype=torch.long),
        "anchor_video": anchor_video,
        "anchor_aux_video": anchor_aux_video,
        "anchor_time_idx": torch.tensor(anchor_t, dtype=torch.long),
        "target_chunk": target_chunk,
        "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
    }


def build_masks_for_sample(
    *,
    processor: Qwen3VLProcessor,
    sample: dict,
    num_frames: int,
    source_dt_ms: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    video_token = processor.video_token
    parts: list[str] = []
    videos: list[torch.Tensor] = []

    prompt = sample.get("prompt", None)
    if prompt is not None:
        parts.append(build_prompt_prefill_text(str(prompt)))

    context_videos = sample["context_videos"]
    context_aux_videos = sample.get("context_aux_videos", None)
    context_time_indices = sample["context_time_indices"]
    n_context = int(context_videos.shape[0])

    for i in range(n_context):
        ts_ms = int(context_time_indices[i].item()) * int(source_dt_ms)
        has_aux = context_aux_videos is not None and int(context_aux_videos.shape[0]) > i and int(context_aux_videos[i].shape[0]) > 0
        parts.append(build_step_text(ts_ms=ts_ms, video_token=video_token, has_aux=has_aux))
        videos.append(make_video_tensor(context_videos[i], num_frames))
        if has_aux:
            videos.append(make_video_tensor(context_aux_videos[i], 1))

    anchor_ts_ms = int(sample["anchor_time_idx"].item()) * int(source_dt_ms)
    anchor_aux_video = sample.get("anchor_aux_video", None)
    has_anchor_aux = anchor_aux_video is not None and int(anchor_aux_video.shape[0]) > 0
    anchor_text = build_step_text(ts_ms=anchor_ts_ms, video_token=video_token, has_aux=has_anchor_aux)
    parts.append(anchor_text)
    anchor_video = make_video_tensor(sample["anchor_video"], num_frames)
    videos.append(anchor_video)
    step_videos = [anchor_video]
    if has_anchor_aux:
        anchor_aux = make_video_tensor(anchor_aux_video, 1)
        videos.append(anchor_aux)
        step_videos.append(anchor_aux)

    batch_text = "".join(parts)
    proc = processor(
        text=[batch_text],
        videos=[videos],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    attention_mask = proc["attention_mask"][0].bool()

    prompt_len = 0
    if prompt is not None:
        prompt_len = prompt_length(processor, str(prompt))
    step_len = step_length(
        processor=processor,
        video=anchor_video,
        aux_video=(step_videos[1] if len(step_videos) > 1 else None),
        ts_ms=anchor_ts_ms,
        video_token=video_token,
    )

    prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    step_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
    valid_positions = attention_mask.nonzero(as_tuple=False).flatten()
    if prompt_len > 0:
        prompt_mask[valid_positions[:prompt_len]] = True
    if step_len > 0:
        step_mask[valid_positions[-step_len:]] = True
    return attention_mask, prompt_mask, step_mask


def run_full_kv_check(
    *,
    sample: dict,
    quick_attention_mask: torch.Tensor,
    quick_prompt_mask: torch.Tensor,
    quick_step_mask: torch.Tensor,
    train_cfg: dict,
    vla_config_path: str,
    rtc_cfg: dict,
    device: str | None,
) -> None:
    from model.rtc_async.action_expert.model import ActionExpertBackbone
    from model.rtc_async.qwen3_stream.kv_export import export_selected_kv_cache
    from model.vla_qwen3_rtc import Qwen3RTCVLAEncoder

    encoder = Qwen3RTCVLAEncoder(config_path=vla_config_path)
    if device is not None and device != encoder.device:
        encoder.device = str(device)
        encoder.model.to(encoder.device)

    with torch.inference_mode():
        out = encoder(
            samples=[sample],
            num_frames=int(train_cfg["model"]["num_frames"]),
            source_dt_ms=int(train_cfg["training"]["source_dt_ms"]),
            return_condition_cache=True,
        )

    full_attention_mask = out["attention_mask"][0].detach().cpu().bool()
    full_prompt_mask = out["prompt_mask"][0].detach().cpu().bool()
    full_step_mask = out["step_mask"][0].detach().cpu().bool()
    print("=== Full KV Check ===")
    print(f"attention_mask_match: {bool(torch.equal(quick_attention_mask, full_attention_mask))}")
    print(f"prompt_mask_match: {bool(torch.equal(quick_prompt_mask, full_prompt_mask))}")
    print(f"step_mask_match: {bool(torch.equal(quick_step_mask, full_step_mask))}")

    selected_layers = [int(x) for x in rtc_cfg["stream"]["selected_layers"]]
    kv_cache = export_selected_kv_cache(
        past_key_values=out["past_key_values"],
        selected_layers=selected_layers,
        clone=False,
    )
    valid_len = int(full_attention_mask.sum().item())
    for layer_idx, (layer_id, (k, v)) in enumerate(zip(selected_layers, kv_cache)):
        k_tokens = ActionExpertBackbone._kv_to_tokens(k.detach().cpu())
        v_tokens = ActionExpertBackbone._kv_to_tokens(v.detach().cpu())
        k_seq = int(k_tokens.shape[1])
        v_seq = int(v_tokens.shape[1])
        print(
            f"layer[{layer_idx}]={layer_id}: "
            f"k_tokens={tuple(k_tokens.shape)} v_tokens={tuple(v_tokens.shape)} "
            f"seq_matches_valid_len={bool(k_seq == valid_len == v_seq)}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast inspection of RTC prompt/step masks on one sample.")
    parser.add_argument("--train-config", default="configs/train_libero90_async.yaml", help="Path to train config yaml.")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to inspect.")
    parser.add_argument("--anchor-t", type=int, default=None, help="Anchor timestep inside the episode.")
    parser.add_argument("--full-kv-check", action="store_true", help="Load full encoder and verify KV/mask alignment.")
    parser.add_argument("--device", default=None, help="Optional device override for full KV check, e.g. cuda:0 or cpu.")
    args = parser.parse_args()

    train_cfg = load_yaml(resolve_workspace_path(args.train_config))
    vla_cfg = load_yaml(resolve_workspace_path(str(train_cfg["model"]["vla_config_path"])))
    rtc_cfg = load_yaml(resolve_workspace_path(str(train_cfg["rtc_async"]["config_path"])))

    processor = Qwen3VLProcessor.from_pretrained(str(vla_cfg["model_name_or_path"]), trust_remote_code=False)
    dataset = LiberoEpisodeDataset(
        zarr_path=str(train_cfg["dataset"]["zarr_path"]),
        image_key=str(train_cfg["dataset"]["image_key"]),
        extra_image_keys=([train_cfg["dataset"]["aux_image_key"]] if train_cfg["dataset"].get("aux_image_key", None) else []),
        action_key=str(train_cfg["dataset"]["action_key"]),
        state_keys=tuple(train_cfg["dataset"]["state_keys"]),
        prompt_key=str(train_cfg["dataset"]["prompt_key"]),
        max_episodes=train_cfg["dataset"].get("max_episodes", None),
    )

    episode = dataset[int(args.episode_idx)]
    episode_len = int(episode["episode_len"].item())
    horizon = int(rtc_cfg["action_expert"]["horizon"])
    max_anchor_t = max(0, episode_len - horizon)
    if args.anchor_t is None:
        anchor_t = min(max_anchor_t, max(0, episode_len // 2))
    else:
        anchor_t = int(args.anchor_t)
    if anchor_t < 0 or anchor_t > max_anchor_t:
        raise ValueError(f"anchor_t out of range: {anchor_t}, valid=[0, {max_anchor_t}]")

    sample = build_single_sample(
        processor=processor,
        dataset=dataset,
        episode_idx=int(args.episode_idx),
        anchor_t=anchor_t,
        source_dt_ms=int(train_cfg["training"]["source_dt_ms"]),
        step_dt_min_ms=int(train_cfg["training"]["step_dt_min_ms"]),
        step_dt_max_ms=int(train_cfg["training"]["step_dt_max_ms"]),
        num_frames=int(train_cfg["model"]["num_frames"]),
        horizon=horizon,
        max_context_len=int(rtc_cfg["stream"]["max_context_len"]),
        aux_image_key=train_cfg["dataset"].get("aux_image_key", None),
    )
    attention_mask, prompt_mask, step_mask = build_masks_for_sample(
        processor=processor,
        sample=sample,
        num_frames=int(train_cfg["model"]["num_frames"]),
        source_dt_ms=int(train_cfg["training"]["source_dt_ms"]),
    )

    history_t = sample["context_time_indices"].tolist()
    source_dt_ms = int(train_cfg["training"]["source_dt_ms"])
    prompt_spans = spans_from_mask(prompt_mask)
    step_spans = spans_from_mask(step_mask)

    print("=== Sample Meta ===")
    print(f"episode_idx: {int(sample['episode_idx'].item())}")
    print(f"episode_len: {episode_len}")
    print(f"history_t: {history_t}")
    print(f"anchor_t: {int(sample['anchor_time_idx'].item())}")
    print(f"target_chunk_steps: [{anchor_t}, {anchor_t + horizon})")
    print(f"history_ms: {[int(t) * source_dt_ms for t in history_t]}")
    print(f"anchor_ms: {anchor_t * source_dt_ms}")
    print()

    print("=== Training Semantics ===")
    print("Current sample still means: use prompt + history before anchor_t + anchor observation to predict action chunk starting at anchor_t.")
    print(f"All history steps < anchor_t: {all(int(t) < anchor_t for t in history_t)}")
    print()

    print("=== Mask Coverage ===")
    print(f"valid_seq_len: {int(attention_mask.sum().item())}")
    print(f"prompt_true_count: {int(prompt_mask.sum().item())}")
    print(f"step_true_count: {int(step_mask.sum().item())}")
    print(f"prompt_spans: {format_spans(prompt_spans)}")
    print(f"step_spans: {format_spans(step_spans)}")
    print(f"prompt_step_overlap: {bool((prompt_mask & step_mask).any().item())}")
    print()

    print("=== Expected Condition Window ===")
    print("Cross-attn should only see the union of prompt_mask and step_mask.")
    print("Prompt covers the prefill prompt prefix; step covers the final anchor step tokens.")
    print("All intermediate history step tokens remain in the full sequence, but are excluded from the condition mask.")

    if args.full_kv_check:
        print()
        run_full_kv_check(
            sample=sample,
            quick_attention_mask=attention_mask,
            quick_prompt_mask=prompt_mask,
            quick_step_mask=step_mask,
            train_cfg=train_cfg,
            vla_config_path=resolve_workspace_path(str(train_cfg["model"]["vla_config_path"])),
            rtc_cfg=rtc_cfg,
            device=args.device,
        )


if __name__ == "__main__":
    main()
