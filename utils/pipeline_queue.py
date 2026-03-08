from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

from .pipeline_types import ChunkPacket, StepPacket, TokenPacket


@dataclass
class PipelineState:
    step_queue: Deque[StepPacket] = field(default_factory=deque)
    token_queue: Deque[TokenPacket] = field(default_factory=deque)
    chunk_queue: Deque[ChunkPacket] = field(default_factory=deque)
    next_step_id: int = 0

    def next_id(self) -> int:
        sid = self.next_step_id
        self.next_step_id += 1
        return sid

    def pop_action_chunk(self) -> Optional[ChunkPacket]:
        if not self.chunk_queue:
            return None
        return self.chunk_queue.popleft()
