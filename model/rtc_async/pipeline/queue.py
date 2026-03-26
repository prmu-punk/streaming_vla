from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Generic, Optional, TypeVar

from .pipeline_types import ActionPacket, ContextPacket, ExecutePacket, StepPacket


T = TypeVar("T")


class LatestPacketQueue(Generic[T]):
    """Latest-only queue used to avoid backlog between stages."""

    def __init__(self) -> None:
        self._items: Deque[T] = deque(maxlen=1)
        self._lock = Lock()

    def put_latest(self, item: T) -> None:
        with self._lock:
            self._items.clear()
            self._items.append(item)

    def pop(self) -> Optional[T]:
        with self._lock:
            if not self._items:
                return None
            return self._items.popleft()

    def peek(self) -> Optional[T]:
        with self._lock:
            if not self._items:
                return None
            return self._items[0]

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def empty(self) -> bool:
        with self._lock:
            return not self._items


@dataclass
class RTCPipelineQueues:
    step_queue: LatestPacketQueue[StepPacket] = field(default_factory=LatestPacketQueue)
    context_queue: LatestPacketQueue[ContextPacket] = field(default_factory=LatestPacketQueue)
    action_queue: LatestPacketQueue[ActionPacket] = field(default_factory=LatestPacketQueue)
    execute_queue: LatestPacketQueue[ExecutePacket] = field(default_factory=LatestPacketQueue)

    def clear(self) -> None:
        self.step_queue.clear()
        self.context_queue.clear()
        self.action_queue.clear()
        self.execute_queue.clear()
