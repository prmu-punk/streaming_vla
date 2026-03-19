from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf


ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)


from dataset.libero90_offline_context_dataset import LiberoOfflineContextDataset
from model.vla_qwen3 import Qwen3VLA
from utils.stages import action_head_forward, backbone_forward, vision_forward
from utils.vla_utils import module_device


def sync_device(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def timed_call(device: str, fn, *args, **kwargs):
    sync_device(device)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    sync_device(device)
    return out, time.perf_counter() - t0


def infer_chunk_horizon(vla: Qwen3VLA, fixed_action_tokens: int) -> int:
    allowed = vla.action_tokenizer.allowed_hf_token_ids(device=module_device(vla.model), include_eos=False)
    if allowed.numel() == 0:
        raise ValueError("No allowed action token ids found.")
    probe = allowed[0].view(1, 1).repeat(1, fixed_action_tokens)
    with torch.no_grad():
        chunk = vla.action_tokenizer.detokenize(probe)
    return int(chunk.shape[1])


def profile_one_sample(
    *,
    vla: Qwen3VLA,
    sample: Dict[str, Any],
    num_frames: int,
    fixed_action_tokens: int,
    source_dt_ms: int,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
) -> Dict[str, Any]:
    device = module_device(vla.model)
    with torch.inference_mode():
        runner = vla.new_runner()
        prompt = str(sample["prompt"])
        _, prefill_dt = timed_call(device, vla.prefill, runner, prompt)

        pipeline = vla.new_pipeline_state()
        vision_time = 0.0
        backbone_time = 0.0
        action_head_time = 0.0
        token_growths = []
        max_token_log_len = int(getattr(runner, "token_log", torch.empty((1, 0))).shape[1])

        history_times = sample["context_time_indices"].tolist()
        context_videos = sample["context_videos"]
        context_aux_videos = sample.get("context_aux_videos", None)
        context_states = sample["context_states"]
        context_action_chunks = sample["context_action_chunks"]

        for i, hist_t in enumerate(history_times):
            token_len_before = int(runner.token_log.shape[1])
            aux_frames = None
            if context_aux_videos is not None and int(context_aux_videos.shape[0]) > i and int(context_aux_videos[i].shape[0]) > 0:
                aux_frames = context_aux_videos[i].detach().to("cpu").numpy()

            vla.push_step(
                pipeline,
                context_videos[i].detach().to("cpu").numpy(),
                aux_frames=aux_frames,
                state=context_states[i].unsqueeze(0).to(device),
                ts=int(hist_t) * source_dt_ms,
                num_frames=num_frames,
            )

            step_packet = pipeline.step_queue.popleft()
            encoded_step, dt = timed_call(device, vision_forward, vla, runner, step_packet)
            vision_time += dt
            pipeline.encoded_step_queue.append(encoded_step)

            encoded_step = pipeline.encoded_step_queue.popleft()
            token_packet, dt = timed_call(
                device,
                backbone_forward,
                vla,
                runner,
                encoded_step,
                fixed_action_tokens=fixed_action_tokens,
                temperature=temperature,
                top_k=top_k,
            )
            backbone_time += dt
            pipeline.token_queue.append(token_packet)

            token_packet = pipeline.token_queue.popleft()
            chunk_packet, dt = timed_call(device, action_head_forward, vla, token_packet)
            action_head_time += dt
            _ = chunk_packet

            # Keep runner history aligned with training semantics by appending GT action tokens.
            gt_chunk = context_action_chunks[i].unsqueeze(0).to(device)
            gt_tokens = vla.action_tokens(gt_chunk)
            eos_id = vla.action_tokenizer.act_eos_hf_id
            eos_token = torch.tensor([[eos_id]], dtype=torch.long, device=device)
            runner.append_text_tokens(input_ids=gt_tokens)
            runner.append_text_tokens(input_ids=eos_token)

            token_len_after = int(runner.token_log.shape[1])
            token_growths.append(token_len_after - token_len_before)
            max_token_log_len = max(max_token_log_len, token_len_after)

        token_len_before = int(runner.token_log.shape[1])
        anchor_t = int(sample["anchor_time_idx"].item())
        anchor_aux = sample.get("anchor_aux_video", None)
        aux_frames = None
        if anchor_aux is not None and int(anchor_aux.shape[0]) > 0:
            aux_frames = anchor_aux.detach().to("cpu").numpy()

        vla.push_step(
            pipeline,
            sample["anchor_video"].detach().to("cpu").numpy(),
            aux_frames=aux_frames,
            state=sample["anchor_state"].unsqueeze(0).to(device),
            ts=anchor_t * source_dt_ms,
            num_frames=num_frames,
        )

        step_packet = pipeline.step_queue.popleft()
        encoded_step, dt = timed_call(device, vision_forward, vla, runner, step_packet)
        vision_time += dt
        pipeline.encoded_step_queue.append(encoded_step)

        encoded_step = pipeline.encoded_step_queue.popleft()
        token_packet, dt = timed_call(
            device,
            backbone_forward,
            vla,
            runner,
            encoded_step,
            fixed_action_tokens=fixed_action_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        backbone_time += dt
        pipeline.token_queue.append(token_packet)

        token_packet = pipeline.token_queue.popleft()
        chunk_packet, dt = timed_call(device, action_head_forward, vla, token_packet)
        action_head_time += dt

        token_len_after = int(runner.token_log.shape[1])
        token_growths.append(token_len_after - token_len_before)
        max_token_log_len = max(max_token_log_len, token_len_after)

        pred_chunk = chunk_packet.action_chunk[0].detach().to("cpu")
        gt_chunk = sample["target_chunk"].detach().to("cpu")
        chunk_mse = float(torch.nn.functional.mse_loss(pred_chunk, gt_chunk).item())

        decision_count = int(len(history_times) + 1)
        return {
            "history_steps": int(len(history_times)),
            "prefill_ms": 1000.0 * prefill_dt,
            "vision_ms_per_step": 1000.0 * vision_time / max(decision_count, 1),
            "backbone_ms_per_step": 1000.0 * backbone_time / max(decision_count, 1),
            "decode_ms_per_step": 1000.0 * action_head_time / max(decision_count, 1),
            "mean_step_token_growth": float(sum(token_growths) / max(len(token_growths), 1)),
            "final_token_log_len": int(runner.token_log.shape[1]),
            "max_token_log_len": int(max_token_log_len),
            "anchor_chunk_mse": chunk_mse,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/train_libero90_sync.yaml")
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    vla = Qwen3VLA(config_path=str(cfg.model.vla_config_path))
    payload = torch.load(args.checkpoint, map_location="cpu")
    vla.load_state_dict(payload["model"], strict=True)
    vla.eval()

    chunk_horizon = infer_chunk_horizon(vla, fixed_action_tokens=int(cfg.model.fixed_action_tokens))
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
        processor=vla.processor,
        action_tokenizer=vla.action_tokenizer,
        state_placeholder_token=vla.state_placeholder_token,
    )

    sample = dataset[int(args.sample_idx)]
    metrics = profile_one_sample(
        vla=vla,
        sample=sample,
        num_frames=int(cfg.model.num_frames),
        fixed_action_tokens=int(cfg.model.fixed_action_tokens),
        source_dt_ms=int(cfg.training.source_dt_ms),
        temperature=float(args.temperature),
        top_k=args.top_k,
    )

    out = {
        "sample_idx": int(args.sample_idx),
        "num_frames": int(cfg.model.num_frames),
        "fixed_action_tokens": int(cfg.model.fixed_action_tokens),
        "source_dt_ms": int(cfg.training.source_dt_ms),
        "max_context_len": int(float(cfg.model.max_context_len)),
        "metrics": metrics,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
