from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from .action_tokenizers import OATActionTokenizer
from .template_qwen3_vla import (
    build_video_text,
    build_prompt_prefill_text,
    build_step_assistant_prefix,
    build_step_user_prefix,
)
from .qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .qwen3_vl.stream_runner import Qwen3VLStreamRunner
from utils.pipeline_queue import PipelineState
from utils.pipeline_types import StepPacket
from utils.stages import action_head_forward, backbone_forward
from utils.vla_utils import decode_token_ids, make_video_tensor, module_device


@dataclass
class StreamConfig:
    state_interval_s: float = 0.0
    vision_interval_s: float = 0.0
    max_context_len: Optional[int] = None


@dataclass
class VLAConfig:
    model_name_or_path: str
    state_dim: int
    oat_tokenizer_checkpoint: str
    device: Optional[str] = None
    stream: StreamConfig = field(default_factory=StreamConfig)


def _load_vla_config(config_path: str) -> VLAConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    stream_raw = raw.get("stream", {}) or {}
    max_context_len = stream_raw.get("max_context_len", None)
    if max_context_len is not None:
        max_context_len = int(float(max_context_len))
    stream_cfg = StreamConfig(
        state_interval_s=float(stream_raw.get("state_interval_s", 0.0)),
        vision_interval_s=float(stream_raw.get("vision_interval_s", 0.0)),
        max_context_len=max_context_len,
    )

    return VLAConfig(
        model_name_or_path=str(raw["model_name_or_path"]),
        state_dim=int(raw["state_dim"]),
        oat_tokenizer_checkpoint=str(raw["oat_tokenizer_checkpoint"]),
        device=raw.get("device", None),
        stream=stream_cfg,
    )


class Qwen3VLA(nn.Module):
    def __init__(
        self,
        config_path: Optional[str] = None,
        *,
        action_tokenizer: Optional[OATActionTokenizer] = None,
    ) -> None:
        super().__init__()
        if config_path is None:
            config_path = str(pathlib.Path(__file__).resolve().parent.parent / "configs" / "vla_qwen3.yaml")
        cfg = _load_vla_config(config_path)

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = Qwen3VLProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        self.model.to(device)
        hidden_size = self.model.config.text_config.hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.state_encoder.to(device)

        if action_tokenizer is None:
            action_tokenizer = OATActionTokenizer(checkpoint=cfg.oat_tokenizer_checkpoint)
        self.action_tokenizer = action_tokenizer
        self.action_tokenizer.add_tokens(self.processor.tokenizer, self.model)
        self.state_placeholder_token = "<state_token>"
        if self.state_placeholder_token not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.state_placeholder_token]}
            )
            self.model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
        self.state_placeholder_token_id = int(
            self.processor.tokenizer.convert_tokens_to_ids(self.state_placeholder_token)
        )

        self.stream_cfg = cfg.stream

    @property
    def device(self) -> torch.device:
        return module_device(self.model)

    def new_runner(self) -> Qwen3VLStreamRunner:
        return Qwen3VLStreamRunner(
            model=self.model,
            state_interval_s=self.stream_cfg.state_interval_s,
            vision_interval_s=self.stream_cfg.vision_interval_s,
            state_encoder=self.state_encoder,
            state_token_id=self.state_placeholder_token_id,
            max_context_len=self.stream_cfg.max_context_len,
            tokenizer=self.processor.tokenizer,
        )

    def prefill(self, runner: Qwen3VLStreamRunner, prompt: Optional[str] = None) -> None:
        device = module_device(self.model)
        if prompt is None:
            eos_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_id is None:
                raise ValueError("tokenizer.eos_token_id is required for prefill.")
            input_ids = torch.tensor([[eos_id]], device=device, dtype=torch.long)
        else:
            prefill_text = build_prompt_prefill_text(str(prompt))
            encoded = self.processor.tokenizer(prefill_text, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
        runner.prefill_text(input_ids=input_ids)

    def new_pipeline_state(self) -> PipelineState:
        return PipelineState()

    def push_step(
        self,
        pipeline: PipelineState,
        frames: np.ndarray | torch.Tensor,
        *,
        aux_frames: Optional[np.ndarray | torch.Tensor] = None,
        state: torch.Tensor,
        ts: Optional[int] = None,
        num_frames: int = 4,
    ) -> int:
        step_id = pipeline.next_id()
        pipeline.step_queue.append(
            StepPacket(
                step_id=step_id,
                frames=frames,
                aux_frames=aux_frames,
                state=state,
                ts=ts,
                num_frames=num_frames,
            )
        )
        return step_id

    def insert_step(
        self,
        runner: Qwen3VLStreamRunner,
        frames: np.ndarray | torch.Tensor,
        *,
        aux_frames: Optional[np.ndarray | torch.Tensor] = None,
        state: torch.Tensor,
        ts: Optional[int] = None,
        num_frames: int = 4,
        source_dt_ms: int = 50,
    ) -> bool:
        del source_dt_ms
        device = module_device(self.model)
        video = make_video_tensor(frames, num_frames)
        aux_video = make_video_tensor(aux_frames, 1) if aux_frames is not None else None
        state_tokens = state if state.device == device else state.to(device)
        return runner.insert_step(
            processor=self.processor,
            video=video,
            aux_video=aux_video,
            state_tokens=state_tokens,
            ts=str(ts) if ts is not None else None,
        )

    def action_tokens(self, actions: torch.Tensor) -> torch.LongTensor:
        return self.action_tokenizer.tokenize(actions)

    def _sample_masked_next_token(
        self,
        logits: torch.Tensor,
        *,
        allowed_token_ids: torch.LongTensor,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        masked = torch.full_like(logits, -float("inf"))
        masked[:, allowed_token_ids] = logits[:, allowed_token_ids]

        if temperature <= 0:
            return torch.argmax(masked, dim=-1, keepdim=True)

        masked = masked / max(temperature, 1e-6)
        if top_k is not None and top_k > 0:
            k = min(int(top_k), masked.shape[-1])
            v, _ = torch.topk(masked, k=k, dim=-1)
            cutoff = v[:, [-1]]
            masked = torch.where(masked < cutoff, torch.full_like(masked, -float("inf")), masked)

        probs = torch.softmax(masked, dim=-1)
        return torch.multinomial(probs, num_samples=1)

    def generate_action_chunk(
        self,
        runner: Qwen3VLStreamRunner,
        *,
        fixed_action_tokens: int = 5,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Generate exactly `fixed_action_tokens` action tokens after <act_bos>,
        then append <act_eos>. The generated action tokens are detokenized by OAT.
        Requires runner cache to already end with <act_bos>.
        """
        token_packet = backbone_forward(
            self,
            runner,
            -1,
            fixed_action_tokens=fixed_action_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        chunk_packet = action_head_forward(self, token_packet)

        return {
            "action_token_ids": chunk_packet.action_token_ids,
            "action_chunk": chunk_packet.action_chunk,
            "ended_by_eos": torch.tensor([token_packet.ended_by_eos], device=module_device(self.model)),
        }

    def forward(
        self,
        *,
        samples: List[Dict[str, Any]],
        fixed_action_tokens: int,
        num_frames: int,
        source_dt_ms: int = 50,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return self.forward_offline_context_batch(
            samples=samples,
            fixed_action_tokens=fixed_action_tokens,
            num_frames=num_frames,
            source_dt_ms=source_dt_ms,
        )

    def forward_offline_context_batch(
        self,
        *,
        samples: List[Dict[str, Any]],
        fixed_action_tokens: int,
        num_frames: int,
        source_dt_ms: int = 50,
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = module_device(self.model)
        batch_texts: List[str] = []
        batch_videos: List[List[torch.Tensor]] = []
        batch_state_embeds: List[torch.Tensor] = []
        batch_target_tokens: List[torch.Tensor] = []
        batch_target_chunks: List[torch.Tensor] = []

        video_token = self.processor.video_token
        act_eos_text = self.processor.tokenizer.decode(
            [self.action_tokenizer.act_eos_hf_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        for sample in samples:
            prompt = sample.get("prompt", None)
            parts: List[str] = []
            if prompt is not None:
                parts.append(build_prompt_prefill_text(str(prompt)))

            videos: List[torch.Tensor] = []
            state_vectors: List[torch.Tensor] = []

            context_videos = sample["context_videos"]
            context_aux_videos = sample.get("context_aux_videos", None)
            context_states = sample["context_states"]
            context_action_chunks = sample["context_action_chunks"]
            context_time_indices = sample["context_time_indices"]
            n_context = int(context_videos.shape[0])

            for i in range(n_context):
                ts_ms = int(context_time_indices[i].item()) * int(source_dt_ms)
                hist_chunk = context_action_chunks[i].unsqueeze(0)
                hist_tokens = self.action_tokens(hist_chunk.to(device))[0]
                if int(hist_tokens.shape[0]) > int(fixed_action_tokens):
                    raise ValueError(
                        f"history token length {hist_tokens.shape[0]} exceeds "
                        f"fixed_action_tokens={fixed_action_tokens}."
                    )
                hist_text = decode_token_ids(self.processor.tokenizer, hist_tokens)
                has_aux = (
                    context_aux_videos is not None
                    and int(context_aux_videos.shape[0]) > i
                    and int(context_aux_videos[i].shape[0]) > 0
                )
                parts.append(
                    build_step_user_prefix(
                        ts_ms=ts_ms,
                        video_token=build_video_text(video_token=video_token, has_aux=has_aux),
                        close_previous_assistant=(i > 0),
                    )
                    + self.state_placeholder_token
                    + build_step_assistant_prefix()
                    + hist_text
                    + act_eos_text
                )
                videos.append(make_video_tensor(context_videos[i], num_frames))
                if has_aux:
                    videos.append(make_video_tensor(context_aux_videos[i], 1))
                state_vectors.append(context_states[i])

            anchor_ts_ms = int(sample["anchor_time_idx"].item()) * int(source_dt_ms)
            tgt_chunk = sample["target_chunk"].unsqueeze(0)
            tgt_tokens = self.action_tokens(tgt_chunk.to(device))[0]
            if int(tgt_tokens.shape[0]) > int(fixed_action_tokens):
                raise ValueError(
                    f"target token length {tgt_tokens.shape[0]} exceeds "
                    f"fixed_action_tokens={fixed_action_tokens}."
                )
            tgt_text = decode_token_ids(self.processor.tokenizer, tgt_tokens)
            anchor_aux_video = sample.get("anchor_aux_video", None)
            has_anchor_aux = anchor_aux_video is not None and int(anchor_aux_video.shape[0]) > 0
            parts.append(
                build_step_user_prefix(
                    ts_ms=anchor_ts_ms,
                    video_token=build_video_text(video_token=video_token, has_aux=has_anchor_aux),
                    close_previous_assistant=(n_context > 0),
                )
                + self.state_placeholder_token
                + build_step_assistant_prefix()
                + tgt_text
            )
            videos.append(make_video_tensor(sample["anchor_video"], num_frames))
            if has_anchor_aux:
                videos.append(make_video_tensor(anchor_aux_video, 1))
            state_vectors.append(sample["anchor_state"])

            batch_texts.append("".join(parts))
            batch_videos.append(videos)
            batch_state_embeds.append(self.state_encoder(torch.stack(state_vectors, dim=0).to(device)))
            batch_target_tokens.append(tgt_tokens)
            batch_target_chunks.append(tgt_chunk[0])

        proc = self.processor(
            text=batch_texts,
            videos=batch_videos,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        pixel_values_videos = proc["pixel_values_videos"].to(device)
        video_grid_thw = proc["video_grid_thw"].to(device)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        model_dtype = inputs_embeds.dtype
        state_placeholder_id = self.state_placeholder_token_id

        for batch_idx, state_embeds in enumerate(batch_state_embeds):
            positions = (input_ids[batch_idx] == state_placeholder_id).nonzero(as_tuple=False).flatten()
            if int(positions.numel()) != int(state_embeds.shape[0]):
                raise ValueError(
                    f"state placeholder count mismatch for sample {batch_idx}: "
                    f"text has {positions.numel()}, states have {state_embeds.shape[0]}."
                )
            inputs_embeds[batch_idx, positions] = state_embeds.to(dtype=model_dtype)

        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            return_dict=True,
        )
        logits = outputs.logits[:, :-1, :]

        labels = torch.full_like(input_ids, fill_value=-100)
        for batch_idx, tgt_tokens in enumerate(batch_target_tokens):
            seq_len = int(attention_mask[batch_idx].sum().item())
            tgt_len = int(tgt_tokens.shape[0])
            labels[batch_idx, seq_len - tgt_len : seq_len] = input_ids[batch_idx, seq_len - tgt_len : seq_len]
        shifted_labels = labels[:, 1:]

        return {
            "logits": logits,
            "target_tokens": torch.nn.utils.rnn.pad_sequence(
                batch_target_tokens, batch_first=True, padding_value=-100
            ),
            "target_chunk": torch.stack(batch_target_chunks, dim=0).to(device),
            "token_mask": shifted_labels != -100,
            "n_valid_tokens": (shifted_labels != -100).sum().to(torch.float32),
            "labels": shifted_labels,
        }
