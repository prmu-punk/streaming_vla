from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Generic, Optional, TypeVar

from .pipeline_types import ActionPacket, ContextPacket, ExecutePacket, StepPacket


T = TypeVar("T")


class LatestPacketQueue(Generic[T]):
    """Bounded FIFO queue used between RTC pipeline stages."""

    def __init__(self, *, capacity: int = 2) -> None:
        if int(capacity) <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self._items: Deque[T] = deque()
        self._capacity = int(capacity)
        self._lock = Lock()

    def put_latest(self, item: T) -> None:
        with self._lock:
            if len(self._items) >= self._capacity:
                raise RuntimeError(f"RTC stage queue is full (capacity={self._capacity})")
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

    def full(self) -> bool:
        with self._lock:
            return len(self._items) >= self._capacity

    def size(self) -> int:
        with self._lock:
            return len(self._items)


@dataclass
class RTCPipelineQueues:
    step_queue: LatestPacketQueue[StepPacket] = field(default_factory=lambda: LatestPacketQueue(capacity=2))
    context_queue: LatestPacketQueue[ContextPacket] = field(default_factory=lambda: LatestPacketQueue(capacity=2))
    action_queue: LatestPacketQueue[ActionPacket] = field(default_factory=lambda: LatestPacketQueue(capacity=2))
    execute_queue: LatestPacketQueue[ExecutePacket] = field(default_factory=lambda: LatestPacketQueue(capacity=2))

    def clear(self) -> None:
        self.step_queue.clear()
        self.context_queue.clear()
        self.action_queue.clear()
        self.execute_queue.clear()
