from __future__ import annotations

import time
from dataclasses import dataclass
from threading import Event, Thread
from typing import Callable, Optional

from .pipeline_types import ActionPacket, ContextPacket, ExecutePacket
from .queue import RTCPipelineQueues


@dataclass
class RTCThreadedPipelineRunner:
    """Latest-only threaded wrapper for the staged RTC pipeline on a single device."""

    queues: RTCPipelineQueues
    step_to_context: Callable[[], Optional[ContextPacket]]
    context_to_action: Callable[[], Optional[ActionPacket]]
    action_to_execute: Callable[[], Optional[ExecutePacket]]
    poll_interval_s: float = 0.001

    def __post_init__(self) -> None:
        self._stop_event = Event()
        self._threads: list[Thread] = []
        self._exception: Exception | None = None
        self._exception_worker: str | None = None

    def start(self) -> None:
        if self._threads:
            raise RuntimeError("RTCThreadedPipelineRunner is already running.")
        self._stop_event.clear()
        self._exception = None
        self._exception_worker = None
        self._threads = [
            Thread(target=self._worker_step_to_context, name="rtc-step-to-context", daemon=True),
            Thread(target=self._worker_context_to_action, name="rtc-context-to-action", daemon=True),
            Thread(target=self._worker_action_to_execute, name="rtc-action-to-execute", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def stop(self, *, join_timeout_s: float = 1.0) -> None:
        if not self._threads:
            return
        self._stop_event.set()
        for thread in self._threads:
            thread.join(timeout=join_timeout_s)
        self._threads = []
        self._stop_event.clear()

    def running(self) -> bool:
        return bool(self._threads) and not self._stop_event.is_set()

    def check_error(self) -> None:
        if self._exception is not None:
            worker = "unknown" if self._exception_worker is None else self._exception_worker
            raise RuntimeError(f"RTC async worker failed in {worker}") from self._exception

    def _idle(self) -> None:
        time.sleep(self.poll_interval_s)

    def _record_exception(self, worker_name: str, exc: Exception) -> None:
        if self._exception is None:
            self._exception = exc
            self._exception_worker = worker_name
        self._stop_event.set()

    def _worker_step_to_context(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.queues.step_queue.empty():
                    self._idle()
                    continue
                packet = self.step_to_context()
                if packet is None:
                    self._idle()
        except Exception as exc:
            self._record_exception("step_to_context", exc)

    def _worker_context_to_action(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.queues.context_queue.empty():
                    self._idle()
                    continue
                packet = self.context_to_action()
                if packet is None:
                    self._idle()
        except Exception as exc:
            self._record_exception("context_to_action", exc)

    def _worker_action_to_execute(self) -> None:
        try:
            while not self._stop_event.is_set():
                if self.queues.action_queue.empty():
                    self._idle()
                    continue
                packet = self.action_to_execute()
                if packet is None:
                    self._idle()
        except Exception as exc:
            self._record_exception("action_to_execute", exc)
