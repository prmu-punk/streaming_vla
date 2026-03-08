from __future__ import annotations

from typing import Optional

import torch

from .pipeline_types import ChunkPacket, StepPacket, TokenPacket


def vision_forward(vla, runner, step_packet: StepPacket) -> bool:
    video = vla._make_video_tensor(step_packet.frames, step_packet.num_frames)
    return runner.insert_step(
        processor=vla.processor,
        video=video,
        state_tokens=step_packet.state.to(vla.device),
        ts=str(step_packet.ts) if step_packet.ts is not None else None,
    )


def backbone_forward(
    vla,
    runner,
    step_id: int,
    *,
    fixed_action_tokens: int = 5,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> TokenPacket:
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
