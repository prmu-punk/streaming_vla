from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import threading
from typing import Any, Optional

import torch


@dataclass
class VideoCacheVersion:
    version: int
    obs_timestamp_ms: float
    video_seq_len: int
    tokens_per_frame: int
    cache_layers: list[Optional[dict[str, torch.Tensor]]]
    context: torch.Tensor
    context_mask: torch.Tensor
    obs_index: int = -1


@dataclass
class CacheSnapshot:
    version: int
    obs_timestamp_ms: float
    frontier: int
    video_seq_len: int
    tokens_per_frame: int
    cache_layers: list[dict[str, torch.Tensor]]
    context: torch.Tensor
    context_mask: torch.Tensor
    obs_index: int = -1
    layer_version_ids: list[int] = field(default_factory=list)
    layer_obs_indices: list[int] = field(default_factory=list)
    layer_obs_timestamps_ms: list[float] = field(default_factory=list)
    layer_ready_events: list[Optional[Any]] = field(default_factory=list)


@dataclass
class StreamingActionJob:
    timesteps: torch.Tensor
    deltas: torch.Tensor
    latents_action: torch.Tensor
    context: torch.Tensor
    context_mask: torch.Tensor
    proprio: Optional[torch.Tensor] = None
    current_step_idx: int = 0
    snapshot_history: list[CacheSnapshot] = field(default_factory=list)

    @property
    def done(self) -> bool:
        return self.current_step_idx >= int(self.timesteps.shape[0])


class StreamingCacheState:
    def __init__(self, num_layers: int):
        self.num_layers = int(num_layers)
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self.live_cache_layers: Optional[list[dict[str, torch.Tensor]]] = None
        self.live_context: Optional[torch.Tensor] = None
        self.live_context_mask: Optional[torch.Tensor] = None
        self.live_video_seq_len: Optional[int] = None
        self.live_tokens_per_frame: Optional[int] = None
        self.live_layer_version_ids: list[int] = []
        self.live_layer_obs_indices: list[int] = []
        self.live_layer_obs_timestamps_ms: list[float] = []
        self.live_layer_ready_events: list[Optional[Any]] = []
        self.pending_versions: deque[VideoCacheVersion] = deque()
        self.current_update: Optional[VideoCacheVersion] = None
        self.current_frontier: int = 0
        self.latest_version_id: int = -1

    def bootstrap(
        self,
        version: VideoCacheVersion,
        layer_ready_events: Optional[list[Optional[Any]]] = None,
    ) -> None:
        self.latest_version_id = max(self.latest_version_id, int(version.version))
        with self._lock:
            if any(layer is None for layer in version.cache_layers):
                raise ValueError("Cannot bootstrap live cache from a partially built version.")
            self.live_cache_layers = clone_cache_layers(version.cache_layers)  # type: ignore[arg-type]
            self.live_context = version.context
            self.live_context_mask = version.context_mask
            self.live_video_seq_len = int(version.video_seq_len)
            self.live_tokens_per_frame = int(version.tokens_per_frame)
            self.live_layer_version_ids = [int(version.version)] * self.num_layers
            self.live_layer_obs_indices = [int(version.obs_index)] * self.num_layers
            self.live_layer_obs_timestamps_ms = [float(version.obs_timestamp_ms)] * self.num_layers
            if layer_ready_events is None:
                self.live_layer_ready_events = [None] * self.num_layers
            else:
                if len(layer_ready_events) != self.num_layers:
                    raise ValueError(
                        f"`layer_ready_events` must contain {self.num_layers} entries, got {len(layer_ready_events)}."
                    )
                self.live_layer_ready_events = list(layer_ready_events)
            self.pending_versions.clear()
            self.current_update = None
            self.current_frontier = 0

    def has_live_cache(self) -> bool:
        with self._lock:
            return self.live_cache_layers is not None

    def has_pending_update(self) -> bool:
        with self._lock:
            return self.current_update is not None or bool(self.pending_versions)

    def apply_layer_update(
        self,
        version: VideoCacheVersion,
        layer_idx: int,
        ready_event: Optional[Any] = None,
    ) -> None:
        layer_idx = int(layer_idx)
        if layer_idx < 0 or layer_idx >= self.num_layers:
            raise ValueError(f"`layer_idx` must be in [0, {self.num_layers}), got {layer_idx}.")
        layer_cache = version.cache_layers[layer_idx]
        if layer_cache is None:
            raise ValueError(
                f"Cannot publish layer {layer_idx} from version {version.version}: layer cache is not ready."
            )
        with self._lock:
            if self.live_cache_layers is None:
                raise ValueError("No live video cache is available. Bootstrap a version before layer updates.")
            self.latest_version_id = max(self.latest_version_id, int(version.version))
            # Single-layer pointer swap: snapshots only need an immutable copy of this ref list.
            self.live_cache_layers[layer_idx] = {
                "k": layer_cache["k"],
                "v": layer_cache["v"],
            }
            self.live_context = version.context
            self.live_context_mask = version.context_mask
            self.live_video_seq_len = int(version.video_seq_len)
            self.live_tokens_per_frame = int(version.tokens_per_frame)
            self.live_layer_version_ids[layer_idx] = int(version.version)
            self.live_layer_obs_indices[layer_idx] = int(version.obs_index)
            self.live_layer_obs_timestamps_ms[layer_idx] = float(version.obs_timestamp_ms)
            self.live_layer_ready_events[layer_idx] = ready_event

    def _activate_next_pending(self) -> None:
        with self._lock:
            if self.current_update is None and self.pending_versions:
                self.current_update = self.pending_versions.popleft()
                self.current_frontier = 0

    def register_pending(self, version: VideoCacheVersion) -> None:
        if self.live_cache_layers is None:
            self.bootstrap(version)
            return
        with self._lock:
            self.latest_version_id = max(self.latest_version_id, int(version.version))
            self.pending_versions.append(version)
        self._activate_next_pending()

    def advance_frontier(self, max_layers: int = 1) -> int:
        if not self.has_live_cache():
            raise ValueError("No live video cache is available. Submit or bootstrap an observation first.")
        requested = max(0, int(max_layers))
        advanced = 0
        while advanced < requested:
            self._activate_next_pending()
            with self._lock:
                if self.current_update is None:
                    break
                current_update = self.current_update
                current_frontier = self.current_frontier
            self.apply_layer_update(current_update, current_frontier)
            with self._lock:
                self.current_frontier += 1
                current_frontier = self.current_frontier
            advanced += 1
            with self._lock:
                if current_frontier >= self.num_layers:
                    self.current_update = None
                    self.current_frontier = 0
        return advanced

    def _snapshot_header(self) -> tuple[int, int, float, int]:
        if not self.live_layer_version_ids:
            raise ValueError("No live video cache version is available for snapshot.")
        latest_version = max(self.live_layer_version_ids)
        frontier = 0
        while frontier < self.num_layers and self.live_layer_version_ids[frontier] == latest_version:
            frontier += 1
        max_idx = max(range(self.num_layers), key=lambda i: self.live_layer_version_ids[i])
        return (
            int(self.live_layer_version_ids[max_idx]),
            int(self.live_layer_obs_indices[max_idx]),
            float(self.live_layer_obs_timestamps_ms[max_idx]),
            int(self.num_layers if frontier == 0 else frontier),
        )

    def snapshot(self) -> CacheSnapshot:
        with self._lock:
            if (
                self.live_cache_layers is None
                or self.live_context is None
                or self.live_context_mask is None
                or self.live_video_seq_len is None
                or self.live_tokens_per_frame is None
            ):
                raise ValueError("No live video cache version is available for snapshot.")
            version, obs_index, obs_timestamp_ms, frontier = self._snapshot_header()
            return CacheSnapshot(
                version=version,
                obs_index=obs_index,
                obs_timestamp_ms=obs_timestamp_ms,
                frontier=int(frontier),
                video_seq_len=int(self.live_video_seq_len),
                tokens_per_frame=int(self.live_tokens_per_frame),
                cache_layers=clone_cache_layers(self.live_cache_layers),
                context=self.live_context,
                context_mask=self.live_context_mask,
                layer_version_ids=list(self.live_layer_version_ids),
                layer_obs_indices=list(self.live_layer_obs_indices),
                layer_obs_timestamps_ms=list(self.live_layer_obs_timestamps_ms),
                layer_ready_events=list(self.live_layer_ready_events),
            )


def clone_cache_layers(cache_layers: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    out: list[dict[str, torch.Tensor]] = []
    for layer in cache_layers:
        out.append({"k": layer["k"], "v": layer["v"]})
    return out


def stitch_prefix_cache(
    cache_new: list[dict[str, torch.Tensor]],
    cache_old: list[dict[str, torch.Tensor]],
    split_point: int,
) -> list[dict[str, torch.Tensor]]:
    if len(cache_new) != len(cache_old):
        raise ValueError(
            f"`cache_new` and `cache_old` must have same layer count, got {len(cache_new)} and {len(cache_old)}."
        )
    split_point = int(split_point)
    if split_point < 0 or split_point > len(cache_new):
        raise ValueError(f"`split_point` must be in [0, {len(cache_new)}], got {split_point}.")

    stitched: list[dict[str, torch.Tensor]] = []
    for idx in range(len(cache_new)):
        src = cache_new if idx < split_point else cache_old
        stitched.append(
            {
                "k": src[idx]["k"],
                "v": src[idx]["v"],
            }
        )
    return stitched
