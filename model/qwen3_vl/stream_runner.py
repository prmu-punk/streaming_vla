# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional

import torch

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from ..template_qwen3_vla import build_step_user_prefix, build_video_text


@dataclass
class StreamState:
    past_key_values: Optional[object] = None
    attention_mask: Optional[torch.Tensor] = None


class Qwen3VLStreamRunner:
    def __init__(
        self,
        model: Qwen3VLForConditionalGeneration,
        max_context_len: Optional[int] = None,
        use_step_eviction: bool = True,
        tokenizer=None,
    ) -> None:
        self.model = model
        self.state = StreamState()
        self.token_log: Optional[torch.LongTensor] = None
        self.image_grid_log: Optional[torch.LongTensor] = None
        self.video_grid_log: Optional[torch.LongTensor] = None
        self.prefill_len = 0
        self.max_context_len = int(max_context_len) if max_context_len is not None else None
        self.use_step_eviction = bool(use_step_eviction)
        self.step_spans: list[tuple[int, int]] = []
        self.step_mm_counts: list[tuple[int, int]] = []
        self.last_logits: Optional[torch.Tensor] = None

    def _get_rope_model(self) -> Any:
        base_model = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        rope_model = getattr(base_model, "model", None)
        if rope_model is None or not hasattr(rope_model, "get_rope_index"):
            raise AttributeError("unable to resolve model object with `get_rope_index`")
        return rope_model

    def reset(self) -> None:
        self.state = StreamState()
        self.token_log = None
        self.image_grid_log = None
        self.video_grid_log = None
        self.prefill_len = 0
        self.step_spans = []
        self.step_mm_counts = []
        self._get_rope_model().rope_deltas = None
        self.last_logits = None

    def _select_cache_mask(self, keep_mask: torch.BoolTensor) -> None:
        if self.state.past_key_values is None:
            return
        select_fn = getattr(self.state.past_key_values, "select_mask", None)
        if select_fn is not None:
            select_fn(keep_mask)
            return

        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        layers = getattr(self.state.past_key_values, "layers", None)
        if layers is None:
            raise ValueError("past_key_values does not support masked selection")

        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None or keys.numel() == 0:
                continue
            layer.keys = keys.index_select(-2, keep_idx.to(keys.device))
            layer.values = values.index_select(-2, keep_idx.to(values.device))
            if hasattr(layer, "cumulative_length"):
                layer.cumulative_length = int(layer.keys.shape[-2])

    def _evict_steps_if_needed(self, incoming_step_len: int) -> None:
        if self.max_context_len is None:
            return
        if self.token_log is None:
            return
        while self.token_log.shape[1] > self.max_context_len:
            if not self.step_spans:
                available_len = self.max_context_len - self.token_log.shape[1]
                raise ValueError(
                    "Context length exceeds max_context_len, but no step segments exist to evict. "
                    f"incoming_len={incoming_step_len} available_len={available_len}"
                )
            start, end = self.step_spans[0]
            if start < self.prefill_len:
                raise ValueError("Step span overlaps prefill region; cannot evict prefill.")
            current_len = self.token_log.shape[1]
            keep_mask = torch.ones(current_len, dtype=torch.bool, device=self.token_log.device)
            keep_mask[start:end] = False
            keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()

            self.token_log = self.token_log.index_select(1, keep_idx)
            if self.state.attention_mask is not None:
                self.state.attention_mask = self.state.attention_mask.index_select(1, keep_idx)

            if self.state.past_key_values is not None:
                self._select_cache_mask(keep_mask)

            image_count, video_count = self.step_mm_counts[0]
            if self.image_grid_log is not None and image_count > 0:
                self.image_grid_log = self.image_grid_log[image_count:]
            if self.video_grid_log is not None and video_count > 0:
                self.video_grid_log = self.video_grid_log[video_count:]

            removed_len = end - start
            self.step_spans = [
                (s - removed_len, e - removed_len) for (s, e) in self.step_spans[1:]
            ]
            self.step_mm_counts = self.step_mm_counts[1:]

        if self.token_log.shape[1] > self.max_context_len:
            available_len = self.max_context_len - self.prefill_len
            raise ValueError(
                "Context length still exceeds max_context_len after evicting all steps. "
                f"incoming_len={incoming_step_len} available_len={available_len}"
            )

    def _append_attention_mask(self, local_mask: torch.Tensor) -> torch.Tensor:
        if self.state.attention_mask is None:
            self.state.attention_mask = local_mask
        else:
            self.state.attention_mask = torch.cat([self.state.attention_mask, local_mask], dim=1)
        return self.state.attention_mask

    def _append_token_log(self, input_ids: torch.LongTensor) -> None:
        if self.token_log is None:
            self.token_log = input_ids
        else:
            self.token_log = torch.cat([self.token_log, input_ids], dim=1)

    def _append_image_grid(self, image_grid_thw: Optional[torch.LongTensor]) -> None:
        if image_grid_thw is None:
            return
        if self.image_grid_log is None:
            self.image_grid_log = image_grid_thw
        else:
            self.image_grid_log = torch.cat([self.image_grid_log, image_grid_thw], dim=0)

    def _append_video_grid(self, video_grid_thw: Optional[torch.LongTensor]) -> None:
        if video_grid_thw is None:
            return
        if self.video_grid_log is None:
            self.video_grid_log = video_grid_thw
        else:
            self.video_grid_log = torch.cat([self.video_grid_log, video_grid_thw], dim=0)

    def _ensure_special_token(self, tokenizer, token: str) -> None:
        if token in tokenizer.get_vocab():
            return
        tokenizer.add_special_tokens({"additional_special_tokens": [token]})
        self.model.resize_token_embeddings(len(tokenizer))

    def _compute_full_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.token_log is None:
            raise ValueError("token_log is empty; prefill_text must be called first.")
        if self.state.attention_mask is None:
            raise ValueError("attention_mask is empty; prefill_text must be called first.")

        total_len = self.token_log.shape[1]
        device = self.token_log.device
        text_positions = torch.arange(total_len, device=device).view(1, 1, -1).expand(
            1, self.token_log.shape[0], -1
        )
        rope_model = self._get_rope_model()
        vision_positions, rope_deltas = rope_model.get_rope_index(
            self.token_log,
            image_grid_thw=self.image_grid_log,
            video_grid_thw=self.video_grid_log,
            attention_mask=self.state.attention_mask,
        )
        position_ids = torch.cat([text_positions, vision_positions], dim=0)
        return position_ids, rope_deltas

    def _forward_append(
        self,
        *,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        precomputed_video_outputs=None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        return_output: bool = False,
    ) -> None:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            if input_ids.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError("input_ids and inputs_embeds must have matching batch/seq shapes.")

        if inputs_embeds is not None:
            batch_size, seq_len, _ = inputs_embeds.shape
            device = inputs_embeds.device
        else:
            batch_size, seq_len = input_ids.shape
            device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=device, dtype=torch.long)

        if input_ids is not None:
            self._append_token_log(input_ids)
        else:
            dummy_ids = torch.zeros((batch_size, seq_len), device=device, dtype=torch.long)
            self._append_token_log(dummy_ids)
        self._append_image_grid(image_grid_thw)
        self._append_video_grid(video_grid_thw)
        self._append_attention_mask(attention_mask)
        full_attention_mask = self.state.attention_mask

        full_position_ids, rope_deltas = self._compute_full_positions()
        position_ids = full_position_ids[..., -seq_len:]

        past_len = self.state.past_key_values.get_seq_length() if self.state.past_key_values is not None else 0
        cache_position = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)

        self._get_rope_model().rope_deltas = rope_deltas
        out = self.model(
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            past_key_values=self.state.past_key_values,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            precomputed_video_outputs=precomputed_video_outputs,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=True,
        )
        self.state.past_key_values = out.past_key_values
        self.last_logits = out.logits[:, -1, :].detach()
        if return_output:
            return out
        return None

    def get_last_logits(self) -> torch.Tensor:
        if self.last_logits is None:
            raise ValueError("No logits cached yet. Call prefill/insert_step/append first.")
        return self.last_logits

    
    def prefill_text(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)
        if self.token_log is not None:
            raise ValueError("prefill_text should only be called once per stream; call reset() to start over.")
        self.prefill_len = input_ids.shape[1]
        if os.getenv("QWEN3VL_DEBUG"):
            seq_len = input_ids.shape[1]
            print(
                f"[debug prefill_text] past_len=0 seq_len={seq_len} "
                f"mask_shape={tuple(attention_mask.shape)} mask_sum={int(attention_mask.sum())} "
                f"cache_pos=(0->{seq_len - 1})"
            )
        self._forward_append(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )


    def append_text_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        batch_size, seq_len = input_ids.shape
        local_mask = attention_mask
        if local_mask is None:
            local_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        if os.getenv("QWEN3VL_DEBUG"):
            past_len = self.state.past_key_values.get_seq_length() if self.state.past_key_values is not None else 0
            print(
                f"[debug append_text] past_len={past_len} seq_len={seq_len} "
                f"mask_shape={tuple(local_mask.shape)} mask_sum={int(local_mask.sum())} "
                f"cache_pos=({past_len}->{past_len + seq_len - 1})"
            )
        self._forward_append(
            input_ids=input_ids,
            attention_mask=local_mask,
        )

    def append_text_tokens_with_logits(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        local_mask = attention_mask
        if local_mask is None:
            local_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        out = self._forward_append(
            input_ids=input_ids,
            attention_mask=local_mask,
            return_output=True,
        )
        return out.logits[:, -seq_len:, :]

    def append_step_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        precomputed_video_outputs=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> bool:
        if pixel_values is None and pixel_values_videos is None and precomputed_video_outputs is None:
            raise ValueError("append_step_tokens requires image or video inputs.")

        step_start = 0 if self.token_log is None else int(self.token_log.shape[1])
        batch_size, seq_len = input_ids.shape
        local_mask = attention_mask
        if local_mask is None:
            local_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        self._forward_append(
            input_ids=input_ids,
            attention_mask=local_mask,
            pixel_values=pixel_values,
            pixel_values_videos=None if precomputed_video_outputs is not None else pixel_values_videos,
            precomputed_video_outputs=precomputed_video_outputs,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        step_end = int(self.token_log.shape[1])
        self.step_spans.append((step_start, step_end))
        self.step_mm_counts.append(
            (
                0 if image_grid_thw is None else int(image_grid_thw.shape[0]),
                0 if video_grid_thw is None else int(video_grid_thw.shape[0]),
            )
        )
        self._evict_steps_if_needed(step_end - step_start)
        return True

    def insert_step(
        self,
        *,
        processor,
        video: Optional[object] = None,
        aux_video: Optional[object] = None,
        video_path: Optional[str] = None,
        ts: Optional[str] = None,
    ) -> bool:
        if (video is None) == (video_path is None):
            raise ValueError("Provide exactly one of video or video_path.")
        if self.token_log is None:
            raise ValueError("prefill_text must be called before insert_step.")
        step_start = self.token_log.shape[1]
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("processor.tokenizer is required for insert_step.")

        video_token = getattr(processor, "image_token", "<|image_pad|>")
        has_aux = aux_video is not None
        prefix_text = build_step_user_prefix(
            ts_ms=0,
            video_token=build_video_text(video_token=video_token, has_aux=has_aux),
        )
        video_payload = video if video is not None else video_path
        proc = processor(
            text=[prefix_text],
            images=[[video_payload] + ([aux_video] if aux_video is not None else [])],
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = proc["input_ids"].to(self.model.device)
        attention_mask = proc["attention_mask"].to(self.model.device)
        pixel_values = proc["pixel_values"].to(self.model.device)
        image_grid_thw = proc["image_grid_thw"].to(self.model.device)
        seq_len = int(attention_mask[0].sum().item())
        input_ids = input_ids[:, :seq_len]
        attention_mask = attention_mask[:, :seq_len]
        did_append = self.append_step_tokens(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
        if not did_append:
            return False
        return True
    
    def append_vision_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values_videos: Optional[torch.FloatTensor],
        video_grid_thw: torch.LongTensor,
        precomputed_video_outputs=None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> bool:
        if pixel_values_videos is None and precomputed_video_outputs is None:
            raise ValueError("append_vision_tokens requires pixel_values_videos or precomputed_video_outputs.")
        batch_size, seq_len = input_ids.shape
        local_mask = attention_mask
        if local_mask is None:
            local_mask = torch.ones((batch_size, seq_len), device=input_ids.device, dtype=torch.long)
        if os.getenv("QWEN3VL_DEBUG"):
            past_len = self.state.past_key_values.get_seq_length() if self.state.past_key_values is not None else 0
            print(
                f"[debug insert_vision] past_len={past_len} seq_len={seq_len} "
                f"mask_shape={tuple(local_mask.shape)} mask_sum={int(local_mask.sum())} "
                f"cache_pos=({past_len}->{past_len + seq_len - 1})"
            )
        self._forward_append(
            input_ids=input_ids,
            attention_mask=local_mask,
            pixel_values_videos=None if precomputed_video_outputs is not None else pixel_values_videos,
            precomputed_video_outputs=precomputed_video_outputs,
            video_grid_thw=video_grid_thw,
        )
        return True

    
    def generate_next(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Generate logits for next token given current KV cache."""
        device = input_ids.device
        ones = torch.ones_like(input_ids, device=device)
        self._forward_append(
            input_ids=input_ids,
            attention_mask=ones,
        )
        return self.last_logits
