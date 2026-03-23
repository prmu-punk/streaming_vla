from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import yaml

from .qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .template_qwen3_vla import IM_END, build_prompt_prefill_text, build_step_user_prefix, build_video_text


@dataclass
class StreamConfig:
    vision_interval_s: float = 0.0
    max_context_len: Optional[int] = None


@dataclass
class RTCVLAConfig:
    model_name_or_path: str
    state_dim: int
    device: Optional[str] = None
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
    """加载 RTC-VLA 编码配置，并映射到 `RTCVLAConfig` 接口。

    该函数负责把 YAML 文件中的 `model_name_or_path`、`state_dim`、`device`
    以及 `stream` 子配置解析为强类型配置对象，供 `Qwen3RTCVLAEncoder.__init__`
    的构造流程直接消费。

    参数:
        config_path: RTC-VLA YAML 配置文件路径。

    返回:
        `RTCVLAConfig` 实例，字段与编码器初始化接口一一对应。
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    stream_raw = raw.get("stream", {}) or {}
    max_context_len = stream_raw.get("max_context_len", None)
    if max_context_len is not None:
        max_context_len = int(float(max_context_len))
    stream_cfg = StreamConfig(
        vision_interval_s=float(stream_raw.get("vision_interval_s", 0.0)),
        max_context_len=max_context_len,
    )

    return RTCVLAConfig(
        model_name_or_path=str(raw["model_name_or_path"]),
        state_dim=int(raw["state_dim"]),
        device=raw.get("device", None),
        stream=stream_cfg,
    )


class Qwen3RTCVLAEncoder(nn.Module):
    """RTC 异步训练专用条件编码器：仅输出 KV/attention，不依赖 OAT 和动作 token 监督。"""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """构建 RTC 异步训练用 VLA 条件编码器。

        接口对应关系:
        - 输入接口: 通过 `forward_offline_context_batch` 接收离线上下文样本。
        - 输出接口: 产出 `target_chunk` 以及可选的 `past_key_values/attention_mask`
          供动作专家训练路径使用。

        参数:
            config_path: 编码器配置路径；为空时使用仓库默认配置。
        """
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
        self.model.to(self.device)
        self.state_dim = cfg.state_dim
        text_cfg = self.model.config.text_config
        head_dim = getattr(text_cfg, "head_dim", None) or (text_cfg.hidden_size // text_cfg.num_attention_heads)
        self.kv_cache_dim = int(text_cfg.num_key_value_heads * head_dim)

    def _make_video_tensor(self, frames: np.ndarray | torch.Tensor, num_frames: int) -> torch.Tensor:
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

    def _build_step_text(self, *, ts_ms: int | None, video_token: str, has_aux: bool) -> str:
        return (
            build_step_user_prefix(
                ts_ms=ts_ms,
                video_token=build_video_text(video_token=video_token, has_aux=has_aux),
                close_previous_assistant=False,
            )
            + IM_END
            + "\n"
        )

    def forward(
        self,
        *,
        samples: List[OfflineContextSample],
        num_frames: int,
        source_dt_ms: int = 50,
        return_condition_cache: bool = True,
    ) -> OfflineContextBatchOutput:
        return self.forward_offline_context_batch(
            samples=samples,
            num_frames=num_frames,
            source_dt_ms=source_dt_ms,
            return_condition_cache=return_condition_cache,
        )

    def forward_offline_context_batch(
        self,
        *,
        samples: List[OfflineContextSample],
        num_frames: int,
        source_dt_ms: int = 50,
        return_condition_cache: bool = True,
    ) -> OfflineContextBatchOutput:
        """编码离线 context 批次，并按动作专家接口返回训练条件。

        接口对应关系:
        - 输入接口 `samples` 需包含:
          `context_videos/context_time_indices/anchor_* /target_chunk`，
          可选 `prompt`。
        - 输出接口包含:
          - `target_chunk`: 对齐动作监督目标，供 RTC loss 使用。
          - `past_key_values` 与 `attention_mask`(可选): 供动作专家注入 VLM 条件。

        参数:
            samples: 离线 context 样本列表，每个元素遵循 `OfflineContextSample`。
            num_frames: 每个 step 使用的视频帧数。
            source_dt_ms: 索引时间步到毫秒时间戳的换算尺度。
            return_condition_cache: 是否返回 `past_key_values` 与 `attention_mask`。

        返回:
            `OfflineContextBatchOutput`，键集合由 `return_condition_cache` 控制。
        """
        batch_texts: List[str] = []
        batch_videos: List[List[torch.Tensor]] = []
        batch_target_chunks: List[torch.Tensor] = []
        prompt_lengths: List[int] = []
        step_lengths: List[int] = []

        video_token = self.processor.video_token

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

            videos: List[torch.Tensor] = []

            context_videos = sample["context_videos"]
            context_aux_videos = sample.get("context_aux_videos", None)
            context_time_indices = sample["context_time_indices"]
            n_context = int(context_videos.shape[0])

            for i in range(n_context):
                ts_ms = int(context_time_indices[i].item()) * int(source_dt_ms)
                has_aux = (
                    context_aux_videos is not None
                    and int(context_aux_videos.shape[0]) > i
                    and int(context_aux_videos[i].shape[0]) > 0
                )
                parts.append(self._build_step_text(ts_ms=ts_ms, video_token=video_token, has_aux=has_aux))
                videos.append(self._make_video_tensor(context_videos[i], num_frames))
                if has_aux:
                    videos.append(self._make_video_tensor(context_aux_videos[i], 1))

            anchor_ts_ms = int(sample["anchor_time_idx"].item()) * int(source_dt_ms)
            anchor_aux_video = sample.get("anchor_aux_video", None)
            has_anchor_aux = anchor_aux_video is not None and int(anchor_aux_video.shape[0]) > 0
            anchor_text = self._build_step_text(ts_ms=anchor_ts_ms, video_token=video_token, has_aux=has_anchor_aux)
            parts.append(anchor_text)
            anchor_video = self._make_video_tensor(sample["anchor_video"], num_frames)
            videos.append(anchor_video)
            step_videos = [anchor_video]
            if has_anchor_aux:
                anchor_aux = self._make_video_tensor(anchor_aux_video, 1)
                videos.append(anchor_aux)
                step_videos.append(anchor_aux)

            batch_texts.append("".join(parts))
            batch_videos.append(videos)
            batch_target_chunks.append(sample["target_chunk"].to(self.device))
            step_proc = self.processor(
                text=[anchor_text],
                videos=[step_videos],
                padding=False,
                return_tensors="pt",
                add_special_tokens=False,
            )
            step_lengths.append(int(step_proc["input_ids"].shape[1]))

        proc = self.processor(
            text=batch_texts,
            videos=batch_videos,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = proc["input_ids"].to(self.device)
        attention_mask = proc["attention_mask"].to(self.device)
        pixel_values_videos = proc["pixel_values_videos"].to(self.device)
        video_grid_thw = proc["video_grid_thw"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
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
