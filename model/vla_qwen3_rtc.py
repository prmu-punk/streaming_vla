from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import yaml

from .qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .template_qwen3_vla import build_prompt_prefill_text, build_step_user_prefix, build_video_text

@dataclass
class StreamConfig:
    max_context_len: Optional[int] = None

@dataclass
class LoRAConfig:
    enabled: bool = False
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    layers_end: int | None = None
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    modules_to_save: list[str] = field(default_factory=list)

@dataclass
class RTCVLAConfig:
    model_name_or_path: str
    state_dim: int
    device: Optional[str] = None
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    stream: StreamConfig = field(default_factory=StreamConfig)

class OfflineContextSample(TypedDict, total=False):
    prompt: str
    context_videos: torch.Tensor | np.ndarray
    context_aux_videos: torch.Tensor | np.ndarray
    context_time_indices: torch.Tensor
    anchor_time_idx: torch.Tensor
    anchor_video: torch.Tensor | np.ndarray
    anchor_aux_video: torch.Tensor | np.ndarray
    target_chunk: torch.Tensor

class OfflineContextBatchOutput(TypedDict, total=False):
    target_chunk: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    step_mask: torch.Tensor

def _load_rtc_vla_config(config_path: str) -> RTCVLAConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    stream_raw = raw.get("stream", {}) or {}
    max_context_len = stream_raw.get("max_context_len", None)
    if max_context_len is not None:
        max_context_len = int(float(max_context_len))
    stream_cfg = StreamConfig(
        max_context_len=max_context_len,
    )
    lora_raw = raw.get("lora", {}) or {}
    lora_cfg = LoRAConfig(
        enabled=bool(lora_raw.get("enabled", False)),
        r=int(lora_raw.get("r", 16)),
        alpha=int(lora_raw.get("alpha", 32)),
        dropout=float(lora_raw.get("dropout", 0.05)),
        layers_end=(None if lora_raw.get("layers_end", None) is None else int(lora_raw["layers_end"])),
        target_modules=[str(x) for x in lora_raw.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])],
        modules_to_save=[str(x) for x in lora_raw.get("modules_to_save", [])],
    )

    return RTCVLAConfig(
        model_name_or_path=str(raw["model_name_or_path"]),
        state_dim=int(raw["state_dim"]),
        device=raw.get("device", None),
        lora=lora_cfg,
        stream=stream_cfg,
    )

class Qwen3RTCVLAEncoder(nn.Module):
    def __init__(self, config_path: Optional[str] = None) -> None:
        super().__init__()
        if config_path is None:
            config_path = str(pathlib.Path(__file__).resolve().parent.parent / "configs" / "vla_qwen3_rtc.yaml")
        cfg = _load_rtc_vla_config(config_path)

        device = cfg.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = Qwen3VLProcessor.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(cfg.model_name_or_path, trust_remote_code=False)
        if cfg.lora.enabled:
            try:
                from peft import LoraConfig as PeftLoraConfig, TaskType, get_peft_model
            except ImportError as exc:
                raise ImportError("LoRA is enabled but `peft` is not installed. Install project dependencies again.") from exc
            text_layers = int(self.model.config.text_config.num_hidden_layers)
            layer_end = text_layers - 1 if cfg.lora.layers_end is None else min(int(cfg.lora.layers_end), text_layers - 1)
            lora_cfg = PeftLoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=int(cfg.lora.r),
                lora_alpha=int(cfg.lora.alpha),
                lora_dropout=float(cfg.lora.dropout),
                bias="none",
                target_modules=list(cfg.lora.target_modules),
                modules_to_save=list(cfg.lora.modules_to_save),
                layers_to_transform=list(range(layer_end + 1)),
                layers_pattern=["layers"],
            )
            self.model = get_peft_model(self.model, lora_cfg)
        self.model.to(self.device)
        self.state_dim = cfg.state_dim
        text_cfg = self.model.config.text_config
        head_dim = getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads)
        self.kv_cache_dim = int(text_cfg.num_key_value_heads * head_dim)

    def _make_image_tensor(self, frames: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            frames_t = torch.from_numpy(frames)
        else:
            frames_t = frames
        if frames_t.dim() == 4:
            frames_t = frames_t[-1]
        if frames_t.shape[-1] == 3:
            frames_t = frames_t.permute(2, 0, 1)
        return frames_t

    def _build_step_text(self, *, ts_ms: int | None, video_token: str, has_aux: bool) -> str:
        return build_step_user_prefix(
            ts_ms=ts_ms,
            video_token=build_video_text(video_token=video_token, has_aux=has_aux),
        )

    def forward(
        self,
        *,
        samples: List[OfflineContextSample],
        source_dt_ms: int = 50,
        return_condition_cache: bool = True,
    ) -> OfflineContextBatchOutput:
        return self.forward_offline_context_batch(
            samples=samples,
            source_dt_ms=source_dt_ms,
            return_condition_cache=return_condition_cache,
        )

    def forward_offline_context_batch(
        self,
        *,
        samples: List[OfflineContextSample],
        source_dt_ms: int = 50,
        return_condition_cache: bool = True,
    ) -> OfflineContextBatchOutput:
        batch_texts: List[str] = []
        batch_images: List[List[torch.Tensor]] = []
        batch_target_chunks: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        step_lengths: List[int] = []

        image_token = self.processor.image_token

        for sample in samples:
            prompt = sample.get("prompt", None)
            parts: List[str] = []
            if prompt is not None:
                prompt_text = build_prompt_prefill_text(str(prompt))
                parts.append(prompt_text)
                prompt_lengths.append(
                    int(
                        len(
                            self.processor.tokenizer(
                                prompt_text,
                                add_special_tokens=False,
                                return_attention_mask=False,
                                return_token_type_ids=False,
                            )["input_ids"]
                        )
                    )
                )
            else:
                prompt_lengths.append(0)

            images: List[torch.Tensor] = []

            context_videos = sample["context_videos"]
            context_aux_videos = sample.get("context_aux_videos", None)
            context_time_indices = sample["context_time_indices"]
            anchor_t_idx = int(sample["anchor_time_idx"].item())
            n_context = int(context_videos.shape[0])

            for i in range(n_context):
                context_t_idx = int(context_time_indices[i].item())
                ts_ms = int(context_t_idx * int(source_dt_ms))
                has_aux = (
                    context_aux_videos is not None
                    and int(context_aux_videos.shape[0]) > i
                    and int(context_aux_videos[i].shape[0]) > 0
                )
                parts.append(self._build_step_text(ts_ms=ts_ms, video_token=image_token, has_aux=has_aux))
                images.append(self._make_image_tensor(context_videos[i]))
                if has_aux:
                    images.append(self._make_image_tensor(context_aux_videos[i]))

            anchor_aux_video = sample.get("anchor_aux_video", None)
            has_anchor_aux = anchor_aux_video is not None and int(anchor_aux_video.shape[0]) > 0
            anchor_text = self._build_step_text(
                ts_ms=int(anchor_t_idx * int(source_dt_ms)),
                video_token=image_token,
                has_aux=has_anchor_aux,
            )
            parts.append(anchor_text)
            anchor_image = self._make_image_tensor(sample["anchor_video"])
            images.append(anchor_image)
            step_images = [anchor_image]
            if has_anchor_aux:
                anchor_aux = self._make_image_tensor(anchor_aux_video)
                images.append(anchor_aux)
                step_images.append(anchor_aux)

            batch_texts.append("".join(parts))
            batch_images.append(images)
            batch_target_chunks.append(sample["target_chunk"].to(self.device))
            step_proc = self.processor(
                text=[anchor_text],
                images=[step_images],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            step_lengths.append(int(step_proc["input_ids"].shape[1]))

        proc = self.processor(
            text=batch_texts,
            images=batch_images,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc["attention_mask"].to(self.device)
        pixel_values = proc["pixel_values"].to(self.device)
        image_grid_thw = proc["image_grid_thw"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            use_cache=return_condition_cache,
            return_dict=True,
        )

        result: OfflineContextBatchOutput = {
            "target_chunk": torch.stack(batch_target_chunks, dim=0),
        }
        if return_condition_cache:
            prompt_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            step_mask = torch.zeros_like(attention_mask, dtype=torch.bool)
            for batch_idx, (prompt_len, step_len) in enumerate(zip(prompt_lengths, step_lengths)):
                valid_positions = attention_mask[batch_idx].nonzero(as_tuple=False).flatten()
                if prompt_len > 0:
                    prompt_mask[batch_idx, valid_positions[:prompt_len]] = True
                if step_len > 0:
                    step_mask[batch_idx, valid_positions[-step_len:]] = True
            result["past_key_values"] = outputs.past_key_values
            result["attention_mask"] = attention_mask
            result["prompt_mask"] = prompt_mask
            result["step_mask"] = step_mask
        return result
