from __future__ import annotations

import torch

from .scheduler import RTCChunkScheduler
from .types import RTCChunkPacket


def schedule_rtc_chunk(
    *,
    scheduler: RTCChunkScheduler,
    step_id: int,
    next_chunk: torch.Tensor,
    inference_delay: int,
    execute_horizon: int,
) -> RTCChunkPacket:
    """对单个 step 应用调度器并封装成可下游消费的 chunk 包。"""

    execute_chunk, _ = scheduler.schedule(
        next_chunk=next_chunk,
        inference_delay=inference_delay,
        execute_horizon=execute_horizon,
    )
    return RTCChunkPacket(
        step_id=step_id,
        action_chunk=next_chunk,
        execute_chunk=execute_chunk,
        inference_delay=inference_delay,
        execute_horizon=execute_horizon,
    )
