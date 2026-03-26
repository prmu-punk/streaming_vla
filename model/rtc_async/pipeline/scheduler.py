from __future__ import annotations

from dataclasses import dataclass

import torch


def validate_rtc_params(*, horizon: int, step_delay_steps: int) -> None:
    """校验基于真实 step delay 的 RTC 调度参数。"""

    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if step_delay_steps < 0:
        raise ValueError(f"step_delay_steps must be non-negative, got {step_delay_steps}")


def stitch_action_for_execution(
    *,
    prev_chunk: torch.Tensor,
    next_chunk: torch.Tensor,
    step_delay_steps: int,
) -> tuple[torch.Tensor, int]:
    """按真实 step delay 将历史重叠前缀与新预测拼接成当前时刻对齐的 chunk。"""

    if prev_chunk.shape != next_chunk.shape:
        raise ValueError(
            f"prev_chunk and next_chunk must share shape, got {tuple(prev_chunk.shape)} vs {tuple(next_chunk.shape)}"
        )
    horizon = int(next_chunk.shape[1])
    validate_rtc_params(horizon=horizon, step_delay_steps=step_delay_steps)
    prefix_len = max(horizon - int(step_delay_steps), 0)
    left = prev_chunk[:, :prefix_len]
    right = next_chunk[:, prefix_len:]
    return torch.cat([left, right], dim=1), int(prefix_len)


def roll_chunk_after_execution(
    *,
    stitched_chunk: torch.Tensor,
    executed_steps: int,
) -> torch.Tensor:
    """在执行后滚动缓存 chunk，保留未执行尾部并在末尾补零。"""

    horizon = int(stitched_chunk.shape[1])
    if executed_steps < 0 or executed_steps > horizon:
        raise ValueError(f"executed_steps must be in [0, {horizon}], got {executed_steps}")
    tail = stitched_chunk[:, executed_steps:]
    pad = torch.zeros(
        (stitched_chunk.shape[0], executed_steps, stitched_chunk.shape[2]),
        device=stitched_chunk.device,
        dtype=stitched_chunk.dtype,
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
        step_delay_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """生成当前时刻对齐的 stitched chunk、可执行片段，并更新下一步缓存。"""

        if next_chunk.dim() != 3:
            raise ValueError(f"next_chunk must be [B, H, D], got {tuple(next_chunk.shape)}")
        if next_chunk.shape[1] != self.horizon or next_chunk.shape[2] != self.action_dim:
            raise ValueError(
                f"next_chunk shape must be [B, {self.horizon}, {self.action_dim}], got {tuple(next_chunk.shape)}"
            )
        if self.prev_chunk is None or self.prev_chunk.shape[0] != next_chunk.shape[0]:
            self.reset(next_chunk.shape[0])
        stitched_chunk, prefix_len = stitch_action_for_execution(
            prev_chunk=self.prev_chunk,
            next_chunk=next_chunk,
            step_delay_steps=int(step_delay_steps),
        )
        execute_steps = min(max(int(step_delay_steps), 0), self.horizon)
        execute_chunk = stitched_chunk[:, :execute_steps]
        self.prev_chunk = roll_chunk_after_execution(
            stitched_chunk=stitched_chunk,
            executed_steps=execute_steps,
        )
        return stitched_chunk, execute_chunk, self.prev_chunk, int(prefix_len), int(execute_steps)
