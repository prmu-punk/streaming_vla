from __future__ import annotations

import argparse
import json
import pathlib
import sys

from omegaconf import OmegaConf
import torch

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from dataset.libero90_offline_context_dataset import LiberoOfflineContextDataset
from model.action_tokenizers import OATActionTokenizer
from model.qwen3_vl import Qwen3VLProcessor
from model.template_qwen3_vla import (
    build_prompt_prefill_text,
    build_step_assistant_prefix,
    build_video_text,
    build_step_user_prefix,
)


def infer_chunk_horizon(tokenizer: OATActionTokenizer) -> int:
    probe = torch.zeros((1, tokenizer.tokens_per_step), dtype=torch.long)
    tokenizer.oat_tokenizer.to(probe.device)
    with torch.no_grad():
        chunk = tokenizer.oat_tokenizer.detokenize(probe)
    return int(chunk.shape[1])


def load_vla_model_name(vla_config_path: str) -> str:
    raw = OmegaConf.load(vla_config_path)
    return str(raw.model_name_or_path)


def oat_chunk_to_text(tokenizer: OATActionTokenizer, chunk: torch.Tensor) -> str:
    if chunk.dim() == 2:
        chunk = chunk.unsqueeze(0)
    tokenizer.oat_tokenizer.to(chunk.device)
    with torch.no_grad():
        oat_ids = tokenizer.oat_tokenizer.tokenize(chunk.to(torch.float32))[0]
    return "".join(f"<act_oat_{int(tok)}>" for tok in oat_ids.tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--print-text", action="store_true")
    parser.add_argument("--max-text-chars", type=int, default=1200)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    action_tokenizer = OATActionTokenizer(
        checkpoint=str(OmegaConf.load(str(cfg.model.vla_config_path)).oat_tokenizer_checkpoint)
    )
    processor = Qwen3VLProcessor.from_pretrained(
        load_vla_model_name(str(cfg.model.vla_config_path)),
        trust_remote_code=False,
    )
    chunk_horizon = infer_chunk_horizon(action_tokenizer)

    dataset = LiberoOfflineContextDataset(
        zarr_path=str(cfg.dataset.zarr_path),
        image_key=str(cfg.dataset.image_key),
        aux_image_key=str(cfg.dataset.aux_image_key) if cfg.dataset.get("aux_image_key", None) else None,
        action_key=str(cfg.dataset.action_key),
        state_keys=[str(k) for k in cfg.dataset.state_keys],
        prompt_key=str(cfg.dataset.prompt_key),
        source_dt_ms=int(cfg.training.source_dt_ms),
        step_dt_min_ms=int(cfg.training.step_dt_min_ms),
        step_dt_max_ms=int(cfg.training.step_dt_max_ms),
        num_frames=int(cfg.model.num_frames),
        chunk_horizon=int(chunk_horizon),
        anchor_stride_steps=int(cfg.dataset.anchor_stride_steps or 1),
        max_context_len=int(float(cfg.model.max_context_len)),
        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
        max_episodes=cfg.dataset.max_episodes,
        processor=processor,
        action_tokenizer=action_tokenizer,
        state_placeholder_token="<state_token>",
    )

    sample = dataset[int(args.sample_idx)]
    history_t = sample["context_time_indices"].tolist()
    anchor_t = int(sample["anchor_time_idx"].item())
    history_gaps_ms = []
    if history_t:
        history_gaps_ms.append(anchor_t - history_t[-1])
        for i in range(len(history_t) - 1, 0, -1):
            history_gaps_ms.append(history_t[i] - history_t[i - 1])
        history_gaps_ms = [int(gap * int(cfg.training.source_dt_ms)) for gap in history_gaps_ms]

    out = {
        "dataset_len": len(dataset),
        "sample_idx": int(args.sample_idx),
        "prompt_head": str(sample["prompt"])[:120],
        "episode_idx": int(sample["episode_idx"].item()),
        "history_steps": int(sample["context_time_indices"].numel()),
        "history_time_indices": history_t[:20],
        "history_gaps_ms_from_newest_backward": history_gaps_ms[:20],
        "anchor_time_idx": anchor_t,
        "target_time_indices": sample["target_time_indices"].tolist(),
        "context_videos_shape": list(sample["context_videos"].shape),
        "context_states_shape": list(sample["context_states"].shape),
        "context_action_chunks_shape": list(sample["context_action_chunks"].shape),
        "anchor_video_shape": list(sample["anchor_video"].shape),
        "anchor_state_shape": list(sample["anchor_state"].shape),
        "target_chunk_shape": list(sample["target_chunk"].shape),
    }

    if args.print_text:
        token_strings = [f"<act_oat_{i}>" for i in range(action_tokenizer.codebook_size)]
        extra_tokens = token_strings + ["<act_eos>", "<state_token>"]
        vocab = processor.tokenizer.get_vocab()
        new_tokens = [t for t in extra_tokens if t not in vocab]
        if new_tokens:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

        video_token = processor.video_token
        act_eos_text = processor.tokenizer.decode(
            [processor.tokenizer.convert_tokens_to_ids("<act_eos>")],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        parts: list[str] = []
        if sample.get("prompt", None) is not None:
            parts.append(build_prompt_prefill_text(str(sample["prompt"])))

        videos = []
        context_aux_videos = sample.get("context_aux_videos", None)
        context_action_chunks = sample["context_action_chunks"]
        for i in range(int(context_action_chunks.shape[0])):
            hist_chunk = context_action_chunks[i].unsqueeze(0)
            hist_text = oat_chunk_to_text(action_tokenizer, hist_chunk)
            has_aux = (
                context_aux_videos is not None
                and int(context_aux_videos.shape[0]) > i
                and int(context_aux_videos[i].shape[0]) > 0
            )
            parts.append(
                build_step_user_prefix(
                    ts_ms=int(history_t[i]) * int(cfg.training.source_dt_ms),
                    video_token=build_video_text(video_token=video_token, has_aux=has_aux),
                    close_previous_assistant=(i > 0),
                )
                + "<state_token>"
                + build_step_assistant_prefix()
                + hist_text
                + act_eos_text
            )
            videos.append(sample["context_videos"][i])
            if has_aux:
                videos.append(sample["context_aux_videos"][i])

        tgt_text = oat_chunk_to_text(action_tokenizer, sample["target_chunk"].unsqueeze(0))
        anchor_aux_video = sample.get("anchor_aux_video", None)
        has_anchor_aux = anchor_aux_video is not None and int(anchor_aux_video.shape[0]) > 0
        parts.append(
            build_step_user_prefix(
                ts_ms=anchor_t * int(cfg.training.source_dt_ms),
                video_token=build_video_text(video_token=video_token, has_aux=has_anchor_aux),
                close_previous_assistant=(len(history_t) > 0),
            )
            + "<state_token>"
            + build_step_assistant_prefix()
            + tgt_text
        )
        videos.append(sample["anchor_video"])
        if has_anchor_aux:
            videos.append(anchor_aux_video)
        text = "".join(parts)
        encoded = processor(
            text=[text],
            videos=[videos],
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        decoded = processor.tokenizer.decode(
            encoded["input_ids"][0].tolist(),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        out["template_text_head"] = text[: int(args.max_text_chars)]
        out["expanded_text_head"] = decoded[: int(args.max_text_chars)]
        out["input_token_len"] = int(encoded["input_ids"].shape[1])

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
