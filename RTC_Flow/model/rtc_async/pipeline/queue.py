from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from .types import RTCChunkPacket, RTCStepPacket


@dataclass
class RTCPipelineState:
    """维护 RTC 推理流水线中的输入队列、输出队列与步号游标。"""

    step_queue: Deque[RTCStepPacket] = field(default_factory=deque)
    chunk_queue: Deque[RTCChunkPacket] = field(default_factory=deque)
    next_step_id: int = 0

    def next_id(self) -> int:
        """分配并返回新的全局 step_id。"""

        sid = self.next_step_id
        self.next_step_id += 1
        return sid

    def pop_action_chunk(self) -> Optional[RTCChunkPacket]:
        """弹出最早生成的动作 chunk；若队列为空返回 None。"""

        if not self.chunk_queue:
            return None
        return self.chunk_queue.popleft()
