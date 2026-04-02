from __future__ import annotations

import queue
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

from experiments.libero.action_ensembler import ActionEnsembler
from fastwam.models.wan22.streaming_cache import CacheSnapshot, VideoCacheVersion
from fastwam.utils.logging_config import get_logger

logger = get_logger(__name__)


def _summarize_ms(samples_ms: list[float]) -> dict[str, float | int | None]:
    if len(samples_ms) == 0:
        return {
            "count": 0,
            "avg_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "max_ms": None,
        }
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(np.max(arr)),
    }


class ActionSnapshotTransport:
    """Prepare action-side cache snapshots.

    Current A-path behavior:
    - same device: return the original snapshot
    - cross device: block on source ready events, copy snapshot payload to action device,
      and report copy latency separately

    This keeps a stable interface for a later B-path mirror-cache implementation.
    """

    def __init__(self, *, video_model, action_model) -> None:
        self.video_model = video_model
        self.action_model = action_model
        self.same_device = self.video_model.device == self.action_model.device
        self._tensor_cache: dict[int, torch.Tensor] = {}

    def _copy_tensor_cached(self, tensor: torch.Tensor) -> torch.Tensor:
        cache_key = int(id(tensor))
        cached = self._tensor_cache.get(cache_key)
        if cached is not None:
            return cached
        copied = tensor.to(device=self.action_model.device, non_blocking=False)
        self._tensor_cache[cache_key] = copied
        return copied

    def _wait_source_ready(self, snapshot: CacheSnapshot) -> None:
        if self.same_device:
            return
        for ready_event in snapshot.layer_ready_events:
            if ready_event is not None:
                ready_event.synchronize()

    def prepare(self, snapshot: CacheSnapshot) -> tuple[CacheSnapshot, float]:
        if self.same_device:
            return snapshot, 0.0

        self._wait_source_ready(snapshot)
        copy_t0 = time.perf_counter()
        copied_snapshot = CacheSnapshot(
            version=int(snapshot.version),
            obs_timestamp_ms=float(snapshot.obs_timestamp_ms),
            frontier=int(snapshot.frontier),
            video_seq_len=int(snapshot.video_seq_len),
            tokens_per_frame=int(snapshot.tokens_per_frame),
            cache_layers=[
                {
                    "k": self._copy_tensor_cached(layer_cache["k"]),
                    "v": self._copy_tensor_cached(layer_cache["v"]),
                }
                for layer_cache in snapshot.cache_layers
            ],
            context=self._copy_tensor_cached(snapshot.context),
            context_mask=self._copy_tensor_cached(snapshot.context_mask),
            obs_index=int(snapshot.obs_index),
            layer_version_ids=list(snapshot.layer_version_ids),
            layer_obs_indices=list(snapshot.layer_obs_indices),
            layer_obs_timestamps_ms=list(snapshot.layer_obs_timestamps_ms),
            layer_ready_events=[None] * len(snapshot.layer_ready_events),
        )
        if self.action_model.device.type == "cuda":
            torch.cuda.synchronize(self.action_model.device)
        return copied_snapshot, (time.perf_counter() - copy_t0) * 1000.0


class AsyncStreamingActionRuntime:
    def __init__(
        self,
        *,
        video_model,
        action_model,
        video_context: torch.Tensor,
        video_context_mask: torch.Tensor,
        action_context: torch.Tensor,
        action_context_mask: torch.Tensor,
        action_postprocess: Callable[[torch.Tensor], np.ndarray],
        action_horizon: int,
        num_inference_steps: int,
        sigma_shift: Optional[float],
        rand_device: str,
        tiled: bool,
        action_trigger_every_n_obs: int,
        video_layers_per_chunk: int,
        overlap_mode: str = "timestamp_average",
        seed: Optional[int] = None,
    ) -> None:
        if overlap_mode != "timestamp_average":
            raise ValueError(f"Unsupported overlap_mode={overlap_mode}.")
        self.video_model = video_model
        self.action_model = action_model
        self.video_context = video_context
        self.video_context_mask = video_context_mask
        self.action_context = action_context
        self.action_context_mask = action_context_mask
        self.action_postprocess = action_postprocess
        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = sigma_shift
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.action_trigger_every_n_obs = int(action_trigger_every_n_obs)
        self.video_layers_per_chunk = int(video_layers_per_chunk)
        self.overlap_mode = overlap_mode
        self.seed = seed

        if self.action_trigger_every_n_obs <= 0:
            raise ValueError("`action_trigger_every_n_obs` must be positive.")
        if self.video_layers_per_chunk <= 0:
            raise ValueError("`video_layers_per_chunk` must be positive.")

        self._obs_queue: queue.Queue[tuple[int, int, float, torch.Tensor] | None] = queue.Queue()
        self._job_queue: queue.Queue[tuple[int, int, float, Optional[torch.Tensor]] | None] = queue.Queue()
        self._error_queue: queue.Queue[BaseException] = queue.Queue()
        self._stop_event = threading.Event()
        self._action_lock = threading.Lock()
        self._ensembler = ActionEnsembler()
        self._video_thread: Optional[threading.Thread] = None
        self._action_thread: Optional[threading.Thread] = None
        self._obs_count = 0
        self._current_env_step = 0
        self._current_obs_index = -1
        self._job_seed_counter = 0
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._actions_served = 0
        self._actions_missed = 0
        self._dropped_prefix_actions = 0
        self._submitted_obs = 0
        self._video_refresh_samples_ms: list[float] = []
        self._action_job_samples_ms: list[float] = []
        self._action_step_samples_ms: list[float] = []
        self._snapshot_copy_samples_ms: list[float] = []
        self._action_job_samples_raw_ms: list[float] = []
        self._video_refresh_event_pairs: list[tuple[Any, Any]] = []
        self._snapshot_transport = ActionSnapshotTransport(
            video_model=self.video_model,
            action_model=self.action_model,
        )

        self._video_stream = (
            torch.cuda.Stream(device=self.video_model.device, priority=0)
            if self.video_model.device.type == "cuda"
            else None
        )
        self._action_stream = (
            torch.cuda.Stream(device=self.action_model.device, priority=-1)
            if self.action_model.device.type == "cuda"
            else None
        )

    def _record_error(self, exc: BaseException) -> None:
        self._stop_event.set()
        self._error_queue.put(exc)

    def _raise_if_error(self) -> None:
        if not self._error_queue.empty():
            raise self._error_queue.get()

    def start(self) -> None:
        self.video_model.eval()
        self.action_model.eval()
        self.video_model.reset_streaming_state()
        self.action_model.reset_streaming_state()
        self._ensembler.reset()
        self._stop_event.clear()
        self._video_thread = threading.Thread(target=self._video_worker, name="fastwam-video-runtime", daemon=True)
        self._action_thread = threading.Thread(target=self._action_worker, name="fastwam-action-runtime", daemon=True)
        self._video_thread.start()
        self._action_thread.start()

    def stop(self) -> None:
        self._obs_queue.put(None)
        self._job_queue.put(None)
        if self._video_thread is not None:
            self._video_thread.join()
        if self._action_thread is not None:
            self._action_thread.join()
        self._synchronize_cuda_devices()
        self._raise_if_error()

    def wait_until_idle(self) -> None:
        self._obs_queue.join()
        self._job_queue.join()
        self._synchronize_cuda_devices()
        self._raise_if_error()

    def _synchronize_cuda_devices(self) -> None:
        seen_devices: set[tuple[str, int | None]] = set()
        for device in (self.video_model.device, self.action_model.device):
            if device.type != "cuda":
                continue
            key = (device.type, device.index)
            if key in seen_devices:
                continue
            seen_devices.add(key)
            torch.cuda.synchronize(device)

    def warmup_action_job(self, *, proprio: Optional[torch.Tensor]) -> None:
        self._raise_if_error()
        if proprio is not None:
            proprio = proprio.detach().to(device=self.action_model.device, dtype=self.action_model.torch_dtype)
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)

        if self._action_stream is None:
            job = self.action_model.start_action_job(
                action_horizon=self.action_horizon,
                context=self.action_context,
                context_mask=self.action_context_mask,
                proprio=proprio,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=(None if self.seed is None else int(self.seed) - 1),
                rand_device=self.rand_device,
            )
            while not job.done:
                snapshot = self.video_model.snapshot_cache_for_action_step()
                snapshot, _ = self._snapshot_transport.prepare(snapshot)
                self.action_model.step_action_job(job, snapshot=snapshot)
            return

        with torch.cuda.stream(self._action_stream):
            job = self.action_model.start_action_job(
                action_horizon=self.action_horizon,
                context=self.action_context,
                context_mask=self.action_context_mask,
                proprio=proprio,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=(None if self.seed is None else int(self.seed) - 1),
                rand_device=self.rand_device,
            )
            while not job.done:
                snapshot = self.video_model.snapshot_cache_for_action_step()
                if self._snapshot_transport.same_device:
                    self.video_model._wait_for_snapshot_ready(snapshot, stream=self._action_stream)
                snapshot, _ = self._snapshot_transport.prepare(snapshot)
                self.action_model.step_action_job(job, snapshot=snapshot)
        self._action_stream.synchronize()
        self._raise_if_error()

    def bootstrap_sync(
        self,
        *,
        input_image: torch.Tensor,
        obs_index: int,
        obs_timestamp_ms: float,
    ) -> VideoCacheVersion:
        self.video_model.reset_streaming_state()
        self.action_model.reset_streaming_state()
        return self.video_model.bootstrap_observation(
            input_image=input_image,
            context=self.video_context,
            context_mask=self.video_context_mask,
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            tiled=self.tiled,
        )

    def reset_for_formal_phase(self, *, env_step: int = 0, obs_index: int = -1) -> None:
        with self._action_lock:
            self._ensembler.reset()
            self._current_env_step = int(env_step)
            self._current_obs_index = int(obs_index)
        self._obs_count = 0
        self._job_seed_counter = 0
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._actions_served = 0
        self._actions_missed = 0
        self._dropped_prefix_actions = 0
        self._submitted_obs = 0
        self._video_refresh_samples_ms = []
        self._action_job_samples_ms = []
        self._action_step_samples_ms = []
        self._snapshot_copy_samples_ms = []
        self._action_job_samples_raw_ms = []
        self._video_refresh_event_pairs = []

    def submit_observation(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        env_step: int,
        obs_index: int,
        obs_timestamp_ms: float,
        trigger_job: bool,
        latest_only_job: bool = False,
    ) -> None:
        self._raise_if_error()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        image = input_image.detach().to(device=self.video_model.device, dtype=self.video_model.torch_dtype)
        prop = None
        if proprio is not None:
            prop = proprio.detach().to(device=self.action_model.device, dtype=self.action_model.torch_dtype)
            if prop.ndim == 1:
                prop = prop.unsqueeze(0)
        self._obs_queue.put((int(env_step), int(obs_index), float(obs_timestamp_ms), image))
        self._submitted_obs += 1
        self._obs_count += 1
        if trigger_job:
            if latest_only_job:
                while True:
                    try:
                        dropped = self._job_queue.get_nowait()
                    except queue.Empty:
                        break
                    if dropped is None:
                        self._job_queue.task_done()
                        self._job_queue.put(None)
                        break
                    self._job_queue.task_done()
            self._job_queue.put((int(env_step), int(obs_index), float(obs_timestamp_ms), prop))
            self._submitted_jobs += 1

    def should_trigger_on_obs(self, obs_count: Optional[int] = None) -> bool:
        count = self._obs_count if obs_count is None else int(obs_count)
        return count > 0 and count % self.action_trigger_every_n_obs == 0

    def completed_jobs(self) -> int:
        self._raise_if_error()
        return int(self._completed_jobs)

    def get_action(self, env_step: int) -> Optional[np.ndarray]:
        self._raise_if_error()
        with self._action_lock:
            self._current_env_step = max(self._current_env_step, int(env_step))
            self._ensembler._cleanup(int(env_step))
            if int(env_step) not in self._ensembler.action_cache:
                self._actions_missed += 1
                return None
            action = np.asarray(self._ensembler.get_action(int(env_step)), dtype=np.float32)
            del self._ensembler.action_cache[int(env_step)]
            self._actions_served += 1
            return action

    def stats(self) -> dict[str, object]:
        self._synchronize_cuda_devices()
        video_samples_ms = list(self._video_refresh_samples_ms)
        video_samples_ms.extend(float(start.elapsed_time(end)) for start, end in self._video_refresh_event_pairs)
        payload = {
            "submitted_obs": int(self._submitted_obs),
            "submitted_jobs": int(self._submitted_jobs),
            "completed_jobs": int(self._completed_jobs),
            "actions_served": int(self._actions_served),
            "actions_missed": int(self._actions_missed),
            "dropped_prefix_actions": int(self._dropped_prefix_actions),
            "timing_ms": {
                "video_refresh": _summarize_ms(video_samples_ms),
                "action_job": _summarize_ms(self._action_job_samples_ms),
                "action_step": _summarize_ms(self._action_step_samples_ms),
                "snapshot_copy": _summarize_ms(self._snapshot_copy_samples_ms),
            },
            "timing_samples_ms": {
                "action_job": [float(v) for v in self._action_job_samples_raw_ms],
            },
        }
        return payload

    def _publish_action_chunk(self, action_chunk: np.ndarray, *, trigger_env_step: int) -> int:
        dropped = 0
        with self._action_lock:
            current_env_step = int(self._current_env_step)
            for i in range(int(action_chunk.shape[0])):
                target_step = int(trigger_env_step) + i
                if target_step < current_env_step:
                    dropped += 1
                    continue
                self._ensembler.action_cache[target_step].append(np.asarray(action_chunk[i], dtype=np.float32))
        self._dropped_prefix_actions += dropped
        return dropped

    def _video_worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._obs_queue.get()
                if item is None:
                    self._obs_queue.task_done()
                    break
                try:
                    _, obs_index, obs_timestamp_ms, input_image = item
                    if not self.video_model.streaming_cache_state.has_live_cache():
                        self.video_model.bootstrap_observation(
                            input_image=input_image,
                            context=self.video_context,
                            context_mask=self.video_context_mask,
                            obs_index=obs_index,
                            obs_timestamp_ms=obs_timestamp_ms,
                            tiled=self.tiled,
                        )
                        continue

                    video_t0 = None
                    video_start_event = None
                    video_end_event = None
                    if self._video_stream is None:
                        video_t0 = time.perf_counter()
                    else:
                        video_start_event = torch.cuda.Event(enable_timing=True)
                        video_end_event = torch.cuda.Event(enable_timing=True)
                        with torch.cuda.stream(self._video_stream):
                            video_start_event.record(self._video_stream)

                    if self._video_stream is None:
                        version, video_pre, video_attention_mask = self.video_model._prepare_streaming_video_version(
                            input_image=input_image,
                            context=self.video_context,
                            context_mask=self.video_context_mask,
                            obs_index=obs_index,
                            obs_timestamp_ms=obs_timestamp_ms,
                            tiled=self.tiled,
                        )
                    else:
                        with torch.cuda.stream(self._video_stream):
                            version, video_pre, video_attention_mask = self.video_model._prepare_streaming_video_version(
                                input_image=input_image,
                                context=self.video_context,
                                context_mask=self.video_context_mask,
                                obs_index=obs_index,
                                obs_timestamp_ms=obs_timestamp_ms,
                                tiled=self.tiled,
                            )

                    prefill_state = self.video_model.mot.init_video_prefill_state(
                        video_tokens=video_pre["tokens"],
                        video_freqs=video_pre["freqs"],
                        video_t_mod=video_pre["t_mod"],
                        video_context_payload={
                            "context": video_pre["context"],
                            "mask": video_pre["context_mask"],
                        },
                        video_attention_mask=video_attention_mask,
                    )

                    def _publish_layer(layer_idx: int, layer_cache: dict[str, torch.Tensor]) -> None:
                        version.cache_layers[layer_idx] = {"k": layer_cache["k"], "v": layer_cache["v"]}
                        ready_event = None
                        if self._video_stream is not None:
                            ready_event = torch.cuda.Event()
                            ready_event.record(self._video_stream)
                        self.video_model.streaming_cache_state.apply_layer_update(
                            version,
                            layer_idx,
                            ready_event=ready_event,
                        )

                    while prefill_state.next_layer_idx < self.video_model.mot.num_layers and not self._stop_event.is_set():
                        if self._video_stream is None:
                            self.video_model.mot.advance_video_prefill_state(
                                state=prefill_state,
                                max_layers=self.video_layers_per_chunk,
                                layer_callback=_publish_layer,
                            )
                        else:
                            with torch.cuda.stream(self._video_stream):
                                self.video_model.mot.advance_video_prefill_state(
                                    state=prefill_state,
                                    max_layers=self.video_layers_per_chunk,
                                    layer_callback=_publish_layer,
                                )
                        time.sleep(0)
                    if video_t0 is not None:
                        self._video_refresh_samples_ms.append((time.perf_counter() - video_t0) * 1000.0)
                    elif video_start_event is not None and video_end_event is not None:
                        with torch.cuda.stream(self._video_stream):
                            video_end_event.record(self._video_stream)
                        self._video_refresh_event_pairs.append((video_start_event, video_end_event))
                finally:
                    self._obs_queue.task_done()
        except BaseException as exc:
            self._record_error(exc)

    def _action_worker(self) -> None:
        try:
            while not self._stop_event.is_set():
                item = self._job_queue.get()
                if item is None:
                    self._job_queue.task_done()
                    break
                try:
                    trigger_env_step, trigger_obs_index, trigger_obs_timestamp_ms, proprio = item
                    if self._action_stream is None:
                        job = self.action_model.start_action_job(
                            action_horizon=self.action_horizon,
                            context=self.action_context,
                            context_mask=self.action_context_mask,
                            proprio=proprio,
                            num_inference_steps=self.num_inference_steps,
                            sigma_shift=self.sigma_shift,
                            seed=(None if self.seed is None else int(self.seed) + self._job_seed_counter),
                            rand_device=self.rand_device,
                        )
                    else:
                        with torch.cuda.stream(self._action_stream):
                            job = self.action_model.start_action_job(
                                action_horizon=self.action_horizon,
                                context=self.action_context,
                                context_mask=self.action_context_mask,
                                proprio=proprio,
                                num_inference_steps=self.num_inference_steps,
                                sigma_shift=self.sigma_shift,
                                seed=(None if self.seed is None else int(self.seed) + self._job_seed_counter),
                                rand_device=self.rand_device,
                            )
                    self._job_seed_counter += 1

                    del trigger_obs_index, trigger_obs_timestamp_ms
                    job_step_samples_ms: list[float] = []
                    job_step_event_pairs: list[tuple[Any, Any]] = []
                    while not job.done and not self._stop_event.is_set():
                        while not self.video_model.streaming_cache_state.has_live_cache() and not self._stop_event.is_set():
                            time.sleep(0.001)
                        snapshot = self.video_model.snapshot_cache_for_action_step()
                        if self._snapshot_transport.same_device and self._action_stream is not None:
                            self.video_model._wait_for_snapshot_ready(snapshot, stream=self._action_stream)
                        snapshot, copy_ms = self._snapshot_transport.prepare(snapshot)
                        if not self._snapshot_transport.same_device:
                            self._snapshot_copy_samples_ms.append(float(copy_ms))
                        if self._action_stream is None:
                            step_t0 = time.perf_counter()
                            self.action_model.step_action_job(job, snapshot=snapshot)
                            job_step_samples_ms.append((time.perf_counter() - step_t0) * 1000.0)
                        else:
                            step_start_event = torch.cuda.Event(enable_timing=True)
                            step_end_event = torch.cuda.Event(enable_timing=True)
                            with torch.cuda.stream(self._action_stream):
                                step_start_event.record(self._action_stream)
                                self.action_model.step_action_job(job, snapshot=snapshot)
                                step_end_event.record(self._action_stream)
                            job_step_event_pairs.append((step_start_event, step_end_event))
                    if self._stop_event.is_set():
                        return
                    if self._action_stream is not None:
                        self._action_stream.synchronize()
                        job_step_samples_ms.extend(
                            float(step_start_event.elapsed_time(step_end_event))
                            for step_start_event, step_end_event in job_step_event_pairs
                        )
                    self._action_step_samples_ms.extend(job_step_samples_ms)
                    job_duration_ms = float(sum(job_step_samples_ms))
                    self._action_job_samples_ms.append(job_duration_ms)
                    self._action_job_samples_raw_ms.append(job_duration_ms)
                    action_chunk = self.action_postprocess(job.latents_action.detach())
                    dropped = self._publish_action_chunk(action_chunk, trigger_env_step=trigger_env_step)
                    self._completed_jobs += 1
                    del dropped
                finally:
                    self._job_queue.task_done()
        except BaseException as exc:
            self._record_error(exc)
