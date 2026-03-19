from __future__ import annotations

from dataclasses import dataclass

import torch


def validate_rtc_params(
    *,
    horizon: int,
    inference_delay: int,
    execute_horizon: int,
) -> None:
    """校验 RTC 调度超参数，确保拼接与执行窗口定义一致。"""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if inference_delay < 0:
        raise ValueError(f"inference_delay must be non-negative, got {inference_delay}")
    if execute_horizon <= 0:
        raise ValueError(f"execute_horizon must be positive, got {execute_horizon}")
    if execute_horizon > horizon:
        raise ValueError(f"execute_horizon must be <= horizon ({horizon}), got {execute_horizon}")
    if inference_delay > execute_horizon:
        raise ValueError(
            f"inference_delay must be <= execute_horizon ({execute_horizon}), got {inference_delay}"
        )


def stitch_action_for_execution(
    *,
    prev_chunk: torch.Tensor,
    next_chunk: torch.Tensor,
    inference_delay: int,
    execute_horizon: int,
) -> torch.Tensor:
    """按 delay/horizon 将历史前缀与新预测拼接成可立即执行的动作段。"""

    if prev_chunk.shape != next_chunk.shape:
        raise ValueError(
            f"prev_chunk and next_chunk must share shape, got {tuple(prev_chunk.shape)} vs {tuple(next_chunk.shape)}"
        )
    horizon = int(next_chunk.shape[1])
    validate_rtc_params(
        horizon=horizon,
        inference_delay=inference_delay,
        execute_horizon=execute_horizon,
    )
    left = prev_chunk[:, :inference_delay]
    right = next_chunk[:, inference_delay:execute_horizon]
    return torch.cat([left, right], dim=1)


def roll_chunk_after_execution(
    *,
    next_chunk: torch.Tensor,
    execute_horizon: int,
) -> torch.Tensor:
    """在执行后滚动缓存 chunk，保留未执行尾部并在末尾补零。"""

    horizon = int(next_chunk.shape[1])
    if execute_horizon < 0 or execute_horizon > horizon:
        raise ValueError(f"execute_horizon must be in [0, {horizon}], got {execute_horizon}")
    tail = next_chunk[:, execute_horizon:]
    pad = torch.zeros(
        (next_chunk.shape[0], execute_horizon, next_chunk.shape[2]),
        device=next_chunk.device,
        dtype=next_chunk.dtype,
    )
    return torch.cat([tail, pad], dim=1)


@dataclass
class RTCChunkScheduler:
    """状态化的 RTC chunk 调度器，管理跨 step 的 prev_chunk 记忆。"""

    horizon: int
    action_dim: int
    device: torch.device
    dtype: torch.dtype = torch.float32
    prev_chunk: torch.Tensor | None = None

    def reset(self, batch_size: int) -> None:
        """按批次大小重置 prev_chunk 缓存。"""

        self.prev_chunk = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )

    def schedule(
        self,
        *,
        next_chunk: torch.Tensor,
        inference_delay: int,
        execute_horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成可执行 chunk，并更新下一步所需的 prev_chunk。"""

        if next_chunk.dim() != 3:
            raise ValueError(f"next_chunk must be [B, H, D], got {tuple(next_chunk.shape)}")
        if next_chunk.shape[1] != self.horizon or next_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"next_chunk shape must be [B, {self.horizon}, {self.action_dim}], got {tuple(next_chunk.shape)}"
            )
        if self.prev_chunk is None or self.prev_chunk.shape[0] != next_chunk.shape[0]:
            self.reset(next_chunk.shape[0])
        execute_chunk = stitch_action_for_execution(
            prev_chunk=self.prev_chunk,
            next_chunk=next_chunk,
            inference_delay=inference_delay,
            execute_horizon=execute_horizon,
        )
        self.prev_chunk = roll_chunk_after_execution(
            next_chunk=next_chunk,
            execute_horizon=execute_horizon,
        )
        return execute_chunk, self.prev_chunk
