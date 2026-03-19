from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from model.rtc_async.pipeline import RTCChunkScheduler
from model.rtc_async.qwen3_stream import Qwen3VLStreamRunnerSnapshot
from model.rtc_async.training import RTCInpaintingBatch, build_rtc_inpainting_batch


@dataclass
class RTCVLAEntry:
    """rtc_async 路径统一构建入口。"""

    config_path: str

    def build(self) -> dict[str, Any]:
        """返回最小运行元信息。"""

        return {"config_path": self.config_path, "mode": "rtc_async"}

    def build_scheduler(
        self,
        *,
        horizon: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> RTCChunkScheduler:
        """构建推理期 RTC chunk 调度器。"""

        return RTCChunkScheduler(
            horizon=horizon,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )

    def build_training_rtc_batch(
        self,
        *,
        action_chunk: torch.Tensor,
        simulated_delay: int | None,
    ) -> RTCInpaintingBatch:
        """构建训练期 RTC inpainting 批次。"""

        return build_rtc_inpainting_batch(
            action=action_chunk,
            simulated_delay=simulated_delay,
        )

    def build_stream_runner_snapshot(
        self,
        *,
        model: Any,
        state_interval_s: float,
        vision_interval_s: float,
        state_encoder: Any = None,
        state_token_id: int | None = None,
        max_context_len: int | None = None,
        use_step_eviction: bool = True,
        tokenizer: Any = None,
    ) -> Qwen3VLStreamRunnerSnapshot:
        """构建与原始实现同构的 stream runner 快照。"""

        return Qwen3VLStreamRunnerSnapshot(
            model=model,
            state_interval_s=state_interval_s,
            vision_interval_s=vision_interval_s,
            state_encoder=state_encoder,
            state_token_id=state_token_id,
            max_context_len=max_context_len,
            use_step_eviction=use_step_eviction,
            tokenizer=tokenizer,
        )
