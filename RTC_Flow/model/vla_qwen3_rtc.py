from __future__ import annotations

from dataclasses import dataclass, field
import pathlib
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import yaml

from .qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from .template_qwen3_vla import IM_END, IM_START, build_prompt_prefill_text


@dataclass
class StreamConfig:
    state_interval_s: float = 0.0
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
    context_states: torch.Tensor
    context_time_indices: torch.Tensor
    anchor_time_idx: torch.Tensor
    anchor_video: torch.Tensor | np.ndarray
    anchor_state: torch.Tensor
    target_chunk: torch.Tensor


class OfflineContextBatchOutput(TypedDict, total=False):
    target_chunk: torch.Tensor
    past_key_values: Any
    attention_mask: torch.Tensor


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
        state_interval_s=float(stream_raw.get("state_interval_s", 0.0)),
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

        hidden_size = self.model.config.text_config.hidden_size
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.state_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.state_encoder.to(self.device)

        self.state_placeholder_token = "<state_token>"
        if self.state_placeholder_token not in self.processor.tokenizer.get_vocab():
            self.processor.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.state_placeholder_token]}
            )
            self.model.resize_token_embeddings(len(self.processor.tokenizer), mean_resizing=False)
        self.state_placeholder_token_id = int(
            self.processor.tokenizer.convert_tokens_to_ids(self.state_placeholder_token)
        )

    def _make_video_tensor(self, frames: np.ndarray | torch.Tensor, num_frames: int) -> torch.Tensor:
        """将单步视频帧规整为模型接口要求的 `[T, C, H, W]` 张量。

        接口对应关系:
        - 输入接口: 接收样本中的 `context_videos[i]` 或 `anchor_video`。
        - 输出接口: 输出固定长度 `num_frames` 的视频张量，供 processor 的
          `videos` 入参使用。

        参数:
            frames: 原始帧序列，支持 `np.ndarray` 或 `torch.Tensor`。
            num_frames: 目标时间长度，不足补最后一帧，超出则截断。

        返回:
            形状对齐后的视频张量。
        """
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

    def _build_step_text(self, *, ts_ms: int | None, video_token: str) -> str:
        """构造单个 step 的文本片段，匹配多模态输入模板接口。

        接口对应关系:
        - 输入接口: 时间戳 `ts_ms` 与 processor 暴露的 `video_token`。
        - 输出接口: 返回可直接拼接到批量 prompt 的 `<step>...</step>` 文本。

        参数:
            ts_ms: 当前 step 的毫秒级时间戳；为空时省略时间标签。
            video_token: 视频占位 token，由 processor 提供。

        返回:
            单个 step 的模板化文本。
        """
        ts_text = f"<ts>{int(ts_ms)}</ts>" if ts_ms is not None else ""
        return (
            f"{IM_START}user\n"
            f"<step>{ts_text}{video_token}<state>{self.state_placeholder_token}</state>{IM_END}\n"
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
          `context_videos/context_states/context_time_indices/anchor_* /target_chunk`，
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
        batch_state_embeds: List[torch.Tensor] = []
        batch_target_chunks: List[torch.Tensor] = []

        video_token = self.processor.video_token

        for sample in samples:
            prompt = sample.get("prompt", None)
            parts: List[str] = []
            if prompt is not None:
                parts.append(build_prompt_prefill_text(str(prompt)))

            videos: List[torch.Tensor] = []
            state_vectors: List[torch.Tensor] = []

            context_videos = sample["context_videos"]
            context_states = sample["context_states"]
            context_time_indices = sample["context_time_indices"]
            n_context = int(context_videos.shape[0])

            for i in range(n_context):
                _ = i
                ts_ms = int(context_time_indices[i].item()) * int(source_dt_ms)
                parts.append(self._build_step_text(ts_ms=ts_ms, video_token=video_token))
                videos.append(self._make_video_tensor(context_videos[i], num_frames))
                state_vectors.append(context_states[i].to(self.device))

            anchor_ts_ms = int(sample["anchor_time_idx"].item()) * int(source_dt_ms)
            parts.append(self._build_step_text(ts_ms=anchor_ts_ms, video_token=video_token))
            videos.append(self._make_video_tensor(sample["anchor_video"], num_frames))
            state_vectors.append(sample["anchor_state"].to(self.device))

            batch_texts.append("".join(parts))
            batch_videos.append(videos)
            batch_state_embeds.append(self.state_encoder(torch.stack(state_vectors, dim=0).to(self.device)))
            batch_target_chunks.append(sample["target_chunk"].to(self.device))

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

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        model_dtype = inputs_embeds.dtype

        for batch_idx, state_embeds in enumerate(batch_state_embeds):
            positions = (input_ids[batch_idx] == self.state_placeholder_token_id).nonzero(as_tuple=False).flatten()
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
            use_cache=return_condition_cache,
            return_dict=True,
        )

        result: OfflineContextBatchOutput = {
            "target_chunk": torch.stack(batch_target_chunks, dim=0),
        }
        if return_condition_cache:
            result["past_key_values"] = outputs.past_key_values
            result["attention_mask"] = attention_mask
        return result
