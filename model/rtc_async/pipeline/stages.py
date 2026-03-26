from __future__ import annotations

from typing import Any, Optional, cast

import torch
import torch.nn as nn

from model.qwen3_vl.stream_runner import Qwen3VLStreamRunner
from model.template_qwen3_vla import build_step_assistant_prefix, build_step_user_prefix, build_video_text
from normalization import RTCNormalizer

from ..action_expert.runner import ActionExpertRunner
from ..qwen3_stream.kv_export import export_compact_selected_kv_cache
from .pipeline_types import ActionPacket, ContextPacket, ExecutePacket, StepPacket
from .scheduler import RTCChunkScheduler


def _kv_cache_to_device(kv_cache, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    if kv_cache is None:
        return None
    out = []
    for k, v in kv_cache:
        out.append((k.to(device=device, non_blocking=True), v.to(device=device, non_blocking=True)))
    return out


class RTCVLMStage(nn.Module):
    def __init__(
        self,
        *,
        encoder: Any,
        runner: Qwen3VLStreamRunner,
        processor: Any,
        selected_layers: list[int],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.runner = runner
        self.processor = processor
        self.selected_layers = list(selected_layers)
        self._assistant_prefix_ids = cast(
            torch.LongTensor,
            processor.tokenizer(
                build_step_assistant_prefix(),
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"],
        )

    def forward(self, step_packet: StepPacket) -> ContextPacket:
        processor = self.encoder.processor
        video = self.encoder._make_video_tensor(step_packet.frames, step_packet.num_frames)
        aux_video = (
            self.encoder._make_video_tensor(step_packet.aux_frames, 1)
            if step_packet.aux_frames is not None
            else None
        )
        has_aux = aux_video is not None
        prefix_text = build_step_user_prefix(
            ts_ms=step_packet.ts_ms,
            video_token=build_video_text(video_token=processor.video_token, has_aux=has_aux),
            close_previous_assistant=bool(step_packet.close_previous_assistant),
        )
        videos = [video]
        if aux_video is not None:
            videos.append(aux_video)
        proc = processor(
            text=[prefix_text],
            videos=[videos],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        device = self.encoder.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        video_grid_thw = proc["video_grid_thw"].to(device)
        pixel_values_videos = proc["pixel_values_videos"].to(device)
        seq_len = int(attention_mask[0].sum().item())
        input_ids = input_ids[:, :seq_len]
        attention_mask = attention_mask[:, :seq_len]
        video_features = self.encoder.model.get_video_features(
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
        )

        inserted = self.runner.append_step_tokens(
            input_ids=input_ids,
            suffix_ids=self._assistant_prefix_ids.to(self.runner.model.device),
            pixel_values_videos=None,
            precomputed_video_outputs=video_features,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        if not inserted:
            raise RuntimeError(f"Step packet for step_id={step_packet.step_id} was rejected by runner gating.")

        attention_mask = self.runner.state.attention_mask
        if attention_mask is None:
            raise RuntimeError("runner attention_mask is unavailable after append.")
        prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        step_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        prompt_mask[:, : self.runner.prefill_len] = True
        latest_step_start, latest_step_end = self.runner.step_spans[-1]
        assistant_len = int(self._assistant_prefix_ids.shape[1])
        latest_step_end = max(latest_step_start, latest_step_end - assistant_len)
        step_mask[:, latest_step_start:latest_step_end] = True

        kv_cache, compact_attention_mask, compact_prompt_mask, compact_step_mask = export_compact_selected_kv_cache(
            past_key_values=self.runner.state.past_key_values,
            selected_layers=self.selected_layers,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
            clone=True,
        )
        return ContextPacket(
            step_id=step_packet.step_id,
            state=step_packet.state,
            ts_ms=step_packet.ts_ms,
            kv_cache=kv_cache,
            attention_mask=compact_attention_mask,
            prompt_mask=compact_prompt_mask,
            step_mask=compact_step_mask,
        )


class RTCDiTStage(nn.Module):
    def __init__(self, *, action_expert: ActionExpertRunner) -> None:
        super().__init__()
        self.action_expert = action_expert

    def forward(
        self,
        context_packet: ContextPacket,
        *,
        normalizer: Optional[RTCNormalizer] = None,
        kv_cache_key: object | None = None,
        generator: torch.Generator | None = None,
    ) -> ActionPacket:
        dit_device = next(self.action_expert.parameters()).device
        state = context_packet.state.to(device=dit_device, non_blocking=True)
        if normalizer is not None:
            state = normalizer.normalize_state(state)
        kv_cache = _kv_cache_to_device(context_packet.kv_cache, dit_device)
        attention_mask = context_packet.attention_mask.to(device=dit_device, non_blocking=True)
        prompt_mask = context_packet.prompt_mask.to(device=dit_device, non_blocking=True)
        step_mask = context_packet.step_mask.to(device=dit_device, non_blocking=True)
        action_chunk = self.action_expert.sample(
            state=state,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
        if normalizer is not None:
            action_chunk = normalizer.unnormalize_action(action_chunk)
        return ActionPacket(
            step_id=context_packet.step_id,
            ts_ms=context_packet.ts_ms,
            action_chunk=action_chunk,
        )


class RTCExecutionStage(nn.Module):
    def __init__(self, *, scheduler: RTCChunkScheduler) -> None:
        super().__init__()
        self.scheduler = scheduler

    def forward(
        self,
        action_packet: ActionPacket,
        *,
        step_delay_steps: int,
    ) -> ExecutePacket:
        stitched_chunk, execute_chunk, _, prefix_len, _ = self.scheduler.schedule(
            next_chunk=action_packet.action_chunk,
            step_delay_steps=int(step_delay_steps),
        )
        return ExecutePacket(
            step_id=action_packet.step_id,
            step_delay_steps=int(step_delay_steps),
            prefix_len=int(prefix_len),
            action_chunk=action_packet.action_chunk,
            stitched_chunk=stitched_chunk,
            execute_chunk=execute_chunk,
        )
