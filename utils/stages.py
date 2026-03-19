from __future__ import annotations

from typing import Optional, Union

import torch

from model.template_qwen3_vla import build_step_assistant_prefix, build_step_user_prefix, build_video_text
from .vla_utils import make_video_tensor, module_device
from .pipeline_types import ChunkPacket, EncodedStepPacket, StepPacket, TokenPacket


def vision_forward(vla, runner, step_packet: StepPacket) -> EncodedStepPacket:
    device = module_device(vla.model)
    video = make_video_tensor(step_packet.frames, step_packet.num_frames)
    aux_video = (
        make_video_tensor(step_packet.aux_frames, 1)
        if step_packet.aux_frames is not None
        else None
    )
    video_token = getattr(vla.processor, "video_token", "<|video_pad|>")
    has_aux = aux_video is not None
    prefix_text = build_step_user_prefix(
        ts_ms=int(step_packet.ts) if step_packet.ts is not None else None,
        video_token=build_video_text(video_token=video_token, has_aux=has_aux),
        close_previous_assistant=(len(runner.step_spans) > 0),
    )
    videos = [video]
    if aux_video is not None:
        videos.append(aux_video)
    proc = vla.processor(
        text=[prefix_text],
        videos=[videos],
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    input_ids = proc["input_ids"].to(device)
    attention_mask = proc["attention_mask"].to(device)
    pixel_values_videos = proc["pixel_values_videos"].to(device)
    video_grid_thw = proc["video_grid_thw"].to(device)
    seq_len = int(attention_mask[0].sum().item())
    precomputed_video_outputs = vla.model.get_video_features(
        pixel_values_videos=pixel_values_videos,
        video_grid_thw=video_grid_thw,
        return_dict=True,
    )
    return EncodedStepPacket(
        step_id=step_packet.step_id,
        input_ids=input_ids[:, :seq_len],
        attention_mask=attention_mask[:, :seq_len],
        pixel_values_videos=None,
        video_grid_thw=video_grid_thw,
        state_tokens=step_packet.state if step_packet.state.device == device else step_packet.state.to(device),
        precomputed_video_outputs=precomputed_video_outputs,
    )


def _consume_encoded_step(vla, runner, encoded_step: EncodedStepPacket) -> None:
    inserted = runner.append_vision_tokens(
        input_ids=encoded_step.input_ids,
        pixel_values_videos=encoded_step.pixel_values_videos,
        precomputed_video_outputs=encoded_step.precomputed_video_outputs,
        video_grid_thw=encoded_step.video_grid_thw,
        attention_mask=encoded_step.attention_mask,
    )
    if not inserted:
        raise RuntimeError(f"append_vision_tokens returned False for step_id={encoded_step.step_id}")
    runner.append_state_tokens(state_tokens=encoded_step.state_tokens)
    suffix_ids = vla.processor.tokenizer(
        build_step_assistant_prefix(),
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(module_device(vla.model))
    runner.append_text_tokens(input_ids=suffix_ids)
    step_end = runner.token_log.shape[1]
    if runner.step_spans:
        step_start = runner.step_spans[-1][1]
    else:
        step_start = runner.prefill_len
    runner.step_spans.append((step_start, step_end))
    runner._evict_steps_if_needed(step_end - step_start)


def backbone_forward(
    vla,
    runner,
    encoded_step: Union[EncodedStepPacket, int],
    *,
    fixed_action_tokens: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> TokenPacket:
    if isinstance(encoded_step, EncodedStepPacket):
        _consume_encoded_step(vla, runner, encoded_step)
        step_id = encoded_step.step_id
    else:
        step_id = int(encoded_step)
    logits = runner.get_last_logits()
    if logits.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported in action generation, got {logits.shape[0]}.")
    if fixed_action_tokens <= 0:
        raise ValueError(f"fixed_action_tokens must be positive, got {fixed_action_tokens}.")

    allowed = vla.action_tokenizer.allowed_hf_token_ids(
        device=logits.device,
        include_eos=False,
    )
    eos_id = vla.action_tokenizer.act_eos_hf_id

    generated = []
    for _ in range(fixed_action_tokens):
        next_token = vla._sample_masked_next_token(
            logits,
            allowed_token_ids=allowed,
            temperature=temperature,
            top_k=top_k,
        )
        generated.append(next_token)
        logits = runner.generate_next(next_token)

    eos_token = torch.tensor([[eos_id]], dtype=torch.long, device=logits.device)
    runner.append_text_tokens(input_ids=eos_token)

    return TokenPacket(
        step_id=step_id,
        action_token_ids=torch.cat(generated, dim=1),
        ended_by_eos=True,
    )


def action_head_forward(vla, token_packet: TokenPacket) -> ChunkPacket:
    action_chunk = vla.action_tokenizer.detokenize(token_packet.action_token_ids)
    return ChunkPacket(
        step_id=token_packet.step_id,
        action_token_ids=token_packet.action_token_ids,
        action_chunk=action_chunk,
    )
