# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Callable

import torch
import torch.nn.functional as F

from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration


@dataclass
class StreamState:
    past_key_values: Optional[object] = None
    attention_mask: Optional[torch.Tensor] = None
    last_state_time: float = 0.0
    last_vision_time: float = 0.0


class Qwen3VLStreamRunner:
    def __init__(
        self,
        model: Qwen3VLForConditionalGeneration,
        state_interval_s: float,
        vision_interval_s: float,
        state_encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        state_token_id: Optional[int] = None,
        max_context_len: Optional[int] = None,
        use_step_eviction: bool = True,
        tokenizer=None,
        obs_same_token: str = "<obs_same>",
        vision_sim_std_weight: float = 0.5,
    ) -> None:
        self.model = model
        self.state_interval_s = float(state_interval_s)
        self.vision_interval_s = float(vision_interval_s)
        # Optional encoder to map raw robot state -> token embeddings.
        self.state_encoder = state_encoder
        self.state = StreamState()
        self.token_log: Optional[torch.LongTensor] = None
        self.image_grid_log: Optional[torch.LongTensor] = None
        self.video_grid_log: Optional[torch.LongTensor] = None
        self.prefill_len = 0
        self.max_context_len = int(max_context_len) if max_context_len is not None else None
        self.use_step_eviction = bool(use_step_eviction)
        self.step_spans: list[tuple[int, int]] = []
        self.last_logits: Optional[torch.Tensor] = None
        if state_token_id is None:
            state_token_id = 0
        self.state_token_id = int(state_token_id)
        self.latest_vision: Optional[torch.Tensor] = None
        self.vision_sim_history: list[float] = []
        self.vision_sim_window = 5
        self.vision_sim_std_weight = float(vision_sim_std_weight)
        self.obs_same_token = obs_same_token
        if tokenizer is not None:
            self._ensure_special_token(tokenizer, obs_same_token)

    def reset(self) -> None:
        self.state = StreamState()
        self.token_log = None
        self.image_grid_log = None
        self.video_grid_log = None
        self.prefill_len = 0
        self.step_spans = []
        self.model.model.rope_deltas = None
        self.latest_vision = None
        self.vision_sim_history = []
        self.last_logits = None

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
                select_fn = getattr(self.state.past_key_values, "select_mask", None)
                if select_fn is None:
                    raise ValueError("past_key_values does not support select_mask")
                select_fn(keep_mask)

            if self.video_grid_log is not None and self.video_grid_log.shape[0] > 0:
                self.video_grid_log = self.video_grid_log[1:]

            removed_len = end - start
            self.step_spans = [
                (s - removed_len, e - removed_len) for (s, e) in self.step_spans[1:]
            ]

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

    def _append_vision_sim(self, sim: float) -> None:
        self.vision_sim_history.append(float(sim))
        if len(self.vision_sim_history) > self.vision_sim_window:
            self.vision_sim_history = self.vision_sim_history[-self.vision_sim_window :]

    def _compute_vision_embedding(
        self,
        *,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            video_outputs = self.model.model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            )
            pooled = video_outputs.pooler_output
            if isinstance(pooled, (list, tuple)):
                pooled = torch.cat(pooled, dim=0)
            embed = pooled.mean(dim=0)
        embed = F.normalize(embed.float(), dim=0)
        return embed.detach().cpu()

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
        vision_positions, rope_deltas = self.model.model.get_rope_index(
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
            dummy_ids = torch.full(
                (batch_size, seq_len), self.state_token_id, device=device, dtype=torch.long
            )
            self._append_token_log(dummy_ids)
        self._append_image_grid(image_grid_thw)
        self._append_video_grid(video_grid_thw)
        self._append_attention_mask(attention_mask)
        full_attention_mask = self.state.attention_mask

        full_position_ids, rope_deltas = self._compute_full_positions()
        position_ids = full_position_ids[..., -seq_len:]

        past_len = self.state.past_key_values.get_seq_length() if self.state.past_key_values is not None else 0
        cache_position = torch.arange(past_len, past_len + seq_len, device=device, dtype=torch.long)

        self.model.model.rope_deltas = rope_deltas
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

    def append_state_tokens(
        self,
        *,
        state_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        now: Optional[float] = None,
    ) -> bool:
        if self.state_encoder is None:
            raise ValueError("state_encoder must be provided to append state tokens.")
        now = time.monotonic() if now is None else now
        if now - self.state.last_state_time < self.state_interval_s:
            return False

        state_tokens = state_tokens.to(self.model.device)
        inputs_embeds = self.state_encoder(state_tokens)
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        batch_size, seq_len, _ = inputs_embeds.shape
        local_mask = attention_mask
        if local_mask is None:
            local_mask = torch.ones((batch_size, seq_len), device=inputs_embeds.device, dtype=torch.long)
        if os.getenv("QWEN3VL_DEBUG"):
            past_len = self.state.past_key_values.get_seq_length() if self.state.past_key_values is not None else 0
            print(
                f"[debug append_state] past_len={past_len} seq_len={seq_len} "
                f"mask_shape={tuple(local_mask.shape)} mask_sum={int(local_mask.sum())} "
                f"cache_pos=({past_len}->{past_len + seq_len - 1})"
            )
        self._forward_append(
            inputs_embeds=inputs_embeds,
            attention_mask=local_mask,
        )
        self.state.last_state_time = now
        return True
    def insert_step(
        self,
        *,
        processor,
        video: Optional[object] = None,
        video_path: Optional[str] = None,
        state_tokens: torch.Tensor,
        ts: Optional[str] = None,
        act_bos: str = "<act_bos>",
        now: Optional[float] = None,
    ) -> bool:
        if (video is None) == (video_path is None):
            raise ValueError("Provide exactly one of video or video_path.")
        if self.token_log is None:
            raise ValueError("prefill_text must be called before insert_step.")
        step_start = self.token_log.shape[1]
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError("processor.tokenizer is required for insert_step.")

        def _encode(text: str) -> torch.LongTensor:
            encoded = tokenizer(text, add_special_tokens=False, return_tensors="pt")
            return encoded["input_ids"].to(self.model.device)

        video_token = getattr(processor, "video_token", "<|video_pad|>")

        parts = ["<step>"]
        if ts is not None:
            parts.append(f"<ts>{int(ts)}</ts>")
        parts.append(f"<obs>{video_token}</obs>")
        parts.append("<state>")
        prefix_text = "".join(parts)
        prefix_text_same = prefix_text.replace(video_token, self.obs_same_token, 1)

        video_payload = video if video is not None else video_path
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prefix_text},
                    {"type": "video", "video": video_payload},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(self.model.device)
        pixel_values_videos = inputs["pixel_values_videos"].to(self.model.device)
        video_grid_thw = inputs["video_grid_thw"].to(self.model.device)

        with torch.no_grad():
            video_outputs = self.model.model.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            )
        pooled = video_outputs.pooler_output
        if isinstance(pooled, (list, tuple)):
            pooled = torch.cat(pooled, dim=0)
        current_vision = F.normalize(pooled.mean(dim=0).float(), dim=0).detach().cpu()

        skip_vision = False
        if self.latest_vision is not None:
            sim = float(torch.dot(current_vision, self.latest_vision).item())
            if self.vision_sim_history:
                mean = sum(self.vision_sim_history) / len(self.vision_sim_history)
                var = sum((v - mean) ** 2 for v in self.vision_sim_history) / len(self.vision_sim_history)
                std = var**0.5
                required = mean - self.vision_sim_std_weight * std
            else:
                required = None
            if required is not None and sim >= required:
                skip_vision = True
            self._append_vision_sim(sim)
            if os.getenv("QWEN3VL_DEBUG"):
                print(f"[debug vision_sim] sim={sim:.4f} required={required}")
        self.latest_vision = current_vision
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is required to close assistant.")
        prefix = torch.tensor([[eos_id]], device=input_ids.device, dtype=input_ids.dtype)

        if skip_vision:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prefix_text_same},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_ids = inputs["input_ids"].to(self.model.device)
            input_ids = torch.cat([prefix, input_ids], dim=1)
            self.append_text_tokens(input_ids=input_ids)
        else:
            input_ids = torch.cat([prefix, input_ids], dim=1)
            did_append = self.append_vision_tokens(
                input_ids=input_ids,
                pixel_values_videos=pixel_values_videos,
                precomputed_video_outputs=video_outputs,
                video_grid_thw=video_grid_thw,
                now=now,
            )
            if not did_append:
                return False
        self.append_state_tokens(state_tokens=state_tokens, now=now)

        suffix_ids = _encode("</state>" + act_bos)
        self.append_text_tokens(input_ids=suffix_ids)
        step_end = self.token_log.shape[1]
        self.step_spans.append((step_start, step_end))
        self._evict_steps_if_needed(step_end - step_start)
        return True
    
    def append_vision_tokens(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values_videos: torch.FloatTensor,
        video_grid_thw: torch.LongTensor,
        precomputed_video_outputs=None,
        attention_mask: Optional[torch.Tensor] = None,
        now: Optional[float] = None,
    ) -> bool:
        now = time.monotonic() if now is None else now
        if now - self.state.last_vision_time < self.vision_interval_s:
            return False
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
        self.state.last_vision_time = now
        return True

    
    def generate_next(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Generate logits for next token given current KV cache."""
        device = input_ids.device
        ones = torch.ones_like(input_ids, device=device)
        self._append_attention_mask(ones)
        self._append_token_log(input_ids)
        out = self.model(
            input_ids=input_ids,
            attention_mask=self.state.attention_mask,
            past_key_values=self.state.past_key_values,
            use_cache=True,
        )
        self.state.past_key_values = out.past_key_values
        self.last_logits = out.logits[:, -1, :].detach()
        return self.last_logits
