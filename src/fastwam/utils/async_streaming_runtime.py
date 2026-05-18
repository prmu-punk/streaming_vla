from __future__ import annotations

from collections import defaultdict, deque
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

from fastwam.utils.action_ensembler import ActionEnsembler


def _summarize_ms(samples_ms: list[float]) -> dict[str, float | int | None]:
    if len(samples_ms) == 0:
        return {"count": 0, "avg_ms": None, "p50_ms": None, "p90_ms": None, "max_ms": None}
    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "max_ms": float(np.max(arr)),
    }


def _summarize_int_samples(samples: list[int]) -> dict[str, float | int | None]:
    if len(samples) == 0:
        return {"count": 0, "avg": None, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "count": int(arr.shape[0]),
        "avg": float(np.mean(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _offset_to_label(offset: int) -> str:
    if offset == 0:
        return "cur"
    if offset == 1:
        return "next"
    if offset > 1:
        return f"next{offset}"
    if offset == -1:
        return "prev"
    return f"prev{abs(offset)}"


def _offset_to_full_mode(offset: int) -> str:
    return f"full_{_offset_to_label(offset)}"


def _classify_layer_sources(
    *,
    layer_env_steps: list[int],
    trigger_env_step: int,
) -> tuple[str, int, dict[str, int], int, Optional[int]]:
    if len(layer_env_steps) == 0:
        return "full_cur", 0, {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}, 0, None

    latest_env_step = max(int(v) for v in layer_env_steps)
    latest_frontier = 0
    while latest_frontier < len(layer_env_steps) and int(layer_env_steps[latest_frontier]) == latest_env_step:
        latest_frontier += 1
    if latest_frontier == 0:
        latest_frontier = len(layer_env_steps)

    older_env_step = None
    for env_step in layer_env_steps[latest_frontier:]:
        if int(env_step) != latest_env_step:
            older_env_step = int(env_step)
            break

    latest_offset = int(latest_env_step - int(trigger_env_step))
    mode = _offset_to_full_mode(latest_offset)
    older_offset: Optional[int] = None
    if older_env_step is not None:
        older_offset = int(older_env_step - int(trigger_env_step))
        mode = f"{_offset_to_label(older_offset)}_to_{_offset_to_label(latest_offset)}"

    age_hist = {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}
    for env_step in layer_env_steps:
        age = int(latest_env_step - int(env_step))
        if age <= 0:
            age_hist["age0"] += 1
        elif age == 1:
            age_hist["age1"] += 1
        elif age == 2:
            age_hist["age2"] += 1
        else:
            age_hist["age3p"] += 1
    return mode, int(latest_frontier), age_hist, int(latest_offset), older_offset


class StreamingRuntime:
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
        seed: Optional[int] = None,
        profile: bool = False,
        collect_layer_source_stats: bool = False,
        collect_detailed_records: bool = False,
        collect_timing_samples: bool = False,
        collect_full_trace: bool = False,
    ) -> None:
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
        self.seed = seed
        self.profile = bool(profile)
        self.collect_layer_source_stats = bool(collect_layer_source_stats)
        self.collect_detailed_records = bool(collect_detailed_records)
        self.collect_timing_samples = bool(collect_timing_samples)
        self.collect_full_trace = bool(collect_full_trace)

        self.device = torch.device(self.video_model.device)
        if torch.device(self.action_model.device) != self.device:
            raise ValueError(
                "Single-process streaming runtime requires `video_model` and `action_model` "
                f"on the same device, got {self.video_model.device} and {self.action_model.device}."
            )

        self._started = False
        self._video_stream: Optional[torch.cuda.Stream] = None
        self._action_stream: Optional[torch.cuda.Stream] = None
        self._completion_lock = threading.RLock()
        self._ready_action_cv = threading.Condition(self._completion_lock)
        self._completion_thread: Optional[threading.Thread] = None
        self._completion_stop_event = threading.Event()
        self._completion_wakeup_event = threading.Event()
        self._completion_poll_interval_s = 0.0005
        self._ensembler = ActionEnsembler()
        self._obs_count = 0
        self._current_env_step = 0
        self._phase_min_env_step = 0
        self._phase_id = 0
        self._submitted_jobs = 0
        self._completed_steps = 0
        self._actions_served = 0
        self._actions_missed = 0
        self._dropped_prefix_actions = 0
        self._submitted_obs = 0
        self._video_refresh_samples_ms: list[float] = []
        self._action_step_loop_samples_ms: list[float] = []
        self._action_step_loop_wall_samples_ms: list[float] = []
        self._action_step_samples_ms: list[float] = []
        self._snapshot_copy_samples_ms: list[float] = []
        self._released_action_trace: list[dict[str, Any]] = []
        self._served_action_trace: list[dict[str, Any]] = []
        self._pending_action_trace_meta: dict[int, dict[str, Any]] = {}

        self._active_job = None
        self._last_snapshot = None
        self._completed_snapshot = None
        self._pending_video_refreshes: deque[dict[str, Any]] = deque()
        self._inflight_action_step: Optional[dict[str, Any]] = None
        self._profile_trace_max_windows = 8
        self._reset_profile_trace_samples()
        self._reset_layer_source_stats()

    def _notify_completion_worker(self) -> None:
        self._completion_wakeup_event.set()

    def _completion_loop(self) -> None:
        while not self._completion_stop_event.is_set():
            progressed = False
            with self._completion_lock:
                if self._started:
                    self._collect_completed_video_refreshes(force=False)
                    progressed = self._harvest_action_step(force=False)
            if progressed:
                continue
            self._completion_wakeup_event.wait(timeout=self._completion_poll_interval_s)
            self._completion_wakeup_event.clear()

    def _reset_layer_source_stats(self) -> None:
        self._layer_source_step_samples: dict[int, int] = defaultdict(int)
        self._layer_source_mode_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_frontier_samples: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._layer_source_latest_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_older_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_age_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}
        )

    def _reset_profile_trace_samples(self) -> None:
        self._profile_trace_window_order: list[int] = []
        self._profile_trace_by_window: dict[int, list[dict[str, Any]]] = {}

    def _should_sample_profile_window(self, window_start_env_step: int) -> bool:
        window_t = int(window_start_env_step)
        if window_t in self._profile_trace_by_window:
            return True
        if len(self._profile_trace_window_order) >= int(self._profile_trace_max_windows):
            return False
        self._profile_trace_window_order.append(window_t)
        self._profile_trace_by_window[window_t] = []
        return True

    def _build_profile_trace_step(self, snapshot, job) -> Optional[dict[str, Any]]:
        if not self.profile:
            return None
        window_start_env_step = int(job.window_start_env_step)
        if not self._should_sample_profile_window(window_start_env_step):
            return None
        if job.token_denoise_counts is None or job.token_env_steps is None:
            return None

        max_denoise_steps = int(job.timesteps.shape[0])
        counts_before = job.token_denoise_counts.detach().to(device="cpu", dtype=torch.int64)[0].tolist()
        token_env_steps = job.token_env_steps.detach().to(device="cpu", dtype=torch.int64)[0].tolist()
        return {
            "sample_t": int(window_start_env_step),
            "obs_t": int(snapshot.env_step),
            "token_ts": [int(v) for v in token_env_steps],
            "token_remaining_steps_before": [int(max_denoise_steps - int(v)) for v in counts_before],
        }

    def _finalize_profile_trace_step(self, trace_step: Optional[dict[str, Any]], job) -> Optional[dict[str, Any]]:
        if trace_step is None:
            return None
        if job.token_denoise_counts is None:
            return trace_step
        counts_after = job.token_denoise_counts.detach().to(device="cpu", dtype=torch.int64)[0].tolist()
        trace_step["token_remaining_steps_after"] = [
            int(job.timesteps.shape[0] - int(v)) for v in counts_after
        ]
        released_tokens: list[dict[str, int]] = []
        if (
            job.just_released_mask is not None
            and job.token_env_steps is not None
            and bool(torch.any(job.just_released_mask).item())
        ):
            released_env_steps = [
                int(v)
                for v in job.token_env_steps[job.just_released_mask].detach().to(device="cpu", dtype=torch.int64).tolist()
            ]
            obs_t = int(trace_step["obs_t"])
            released_tokens = [
                {
                    "token_t": int(token_t),
                    "delta_from_obs_t": int(token_t - obs_t),
                }
                for token_t in released_env_steps
            ]
        trace_step["released_tokens"] = released_tokens
        return trace_step

    def _record_profile_trace_step(self, trace_step: Optional[dict[str, Any]]) -> None:
        if trace_step is None:
            return
        window_t = int(trace_step["sample_t"])
        self._profile_trace_by_window.setdefault(window_t, []).append(trace_step)

    def _build_profile_trace_payload(self) -> list[dict[str, Any]]:
        payload: list[dict[str, Any]] = []
        for window_t in self._profile_trace_window_order:
            payload.append({"sample_t": int(window_t), "trace": list(self._profile_trace_by_window.get(int(window_t), []))})
        return payload

    def _accumulate_layer_source_steps(self, layer_source_steps: list[dict[str, Any]]) -> None:
        for step in layer_source_steps:
            denoise_step = int(step.get("denoise_step", -1))
            mode = str(step.get("mode", "full_cur"))
            frontier = int(step.get("frontier", 0))
            age_hist = dict(step.get("age_hist", {}))
            latest_offset = step.get("latest_offset", None)
            older_offset = step.get("older_offset", None)
            self._layer_source_step_samples[denoise_step] += 1
            self._layer_source_mode_counts[denoise_step][mode] += 1
            if "_to_" in mode:
                self._layer_source_frontier_samples[denoise_step][mode].append(frontier)
            if latest_offset is not None:
                self._layer_source_latest_offset_counts[denoise_step][int(latest_offset)] += 1
            if older_offset is not None:
                self._layer_source_older_offset_counts[denoise_step][int(older_offset)] += 1
            self._layer_source_age_counts[denoise_step]["age0"] += int(age_hist.get("age0", 0))
            self._layer_source_age_counts[denoise_step]["age1"] += int(age_hist.get("age1", 0))
            self._layer_source_age_counts[denoise_step]["age2"] += int(age_hist.get("age2", 0))
            self._layer_source_age_counts[denoise_step]["age3p"] += int(age_hist.get("age3p", 0))

    def _build_layer_source_stats(self) -> dict[str, Any]:
        per_step: list[dict[str, Any]] = []
        for denoise_step in sorted(self._layer_source_step_samples.keys()):
            mode_counts = dict(self._layer_source_mode_counts[denoise_step])
            samples = int(sum(mode_counts.values()))
            if samples <= 0:
                continue
            frontier_samples_by_mode = {
                mode: [int(v) for v in values]
                for mode, values in self._layer_source_frontier_samples[denoise_step].items()
            }
            per_step.append(
                {
                    "denoise_step": int(denoise_step),
                    "samples": int(samples),
                    "mode_counts": mode_counts,
                    "mode_probs": {mode: float(count) / float(samples) for mode, count in mode_counts.items()},
                    "frontier_samples_by_mode": frontier_samples_by_mode,
                    "frontier_stats_by_mode": {
                        mode: _summarize_int_samples(values)
                        for mode, values in frontier_samples_by_mode.items()
                    },
                    "latest_offset_counts": {
                        str(key): int(value)
                        for key, value in sorted(dict(self._layer_source_latest_offset_counts[denoise_step]).items())
                    },
                    "older_offset_counts": {
                        str(key): int(value)
                        for key, value in sorted(dict(self._layer_source_older_offset_counts[denoise_step]).items())
                    },
                    "age_counts": dict(self._layer_source_age_counts[denoise_step]),
                }
            )
        for step in per_step:
            samples = int(step["samples"])
            latest_counts = step["latest_offset_counts"]
            step["latest_offset_probs"] = {
                key: float(value) / float(samples) for key, value in latest_counts.items()
            }
            older_counts = step["older_offset_counts"]
            older_total = int(sum(older_counts.values()))
            step["older_offset_probs"] = (
                {key: float(value) / float(older_total) for key, value in older_counts.items()}
                if older_total > 0
                else {}
            )
            age_counts = step["age_counts"]
            age_total = int(sum(age_counts.values()))
            step["age_probs"] = (
                {key: float(value) / float(age_total) for key, value in age_counts.items()}
                if age_total > 0
                else {key: 0.0 for key in age_counts.keys()}
            )
        return {
            "enabled": bool(self.profile),
            "num_layers": int(self.action_model.mot.num_layers),
            "per_step": per_step,
        }

    def _publish_released_actions(
        self,
        released_actions: np.ndarray,
        *,
        released_env_steps: list[int],
        source_env_step: int,
        release_obs_index: int,
        release_obs_env_step: int,
        job_id: int,
        phase_id: int,
    ) -> int:
        dropped = 0
        current_env_step = int(self._current_env_step)
        if released_actions.ndim == 1:
            released_actions = released_actions[None, :]
        if int(released_actions.shape[0]) != len(released_env_steps):
            raise ValueError(
                f"Released action count mismatch: actions={int(released_actions.shape[0])}, "
                f"env_steps={len(released_env_steps)}"
            )
        for i, target_step in enumerate(released_env_steps):
            release_event = {
                "phase_id": int(phase_id),
                "job_id": int(job_id),
                "target_env_step": int(target_step),
                "source_env_step": int(source_env_step),
                "release_obs_index": int(release_obs_index),
                "release_obs_env_step": int(release_obs_env_step),
                "release_obs_gap": int(target_step - int(release_obs_env_step)),
                "dropped_prefix": bool(target_step < current_env_step),
            }
            if self.collect_full_trace:
                self._released_action_trace.append(release_event)
            if target_step < current_env_step:
                dropped += 1
                continue
            self._ensembler.action_cache[int(target_step)] = [np.asarray(released_actions[i], dtype=np.float32)]
            self._pending_action_trace_meta[int(target_step)] = release_event
        self._dropped_prefix_actions += dropped
        if len(released_env_steps) > dropped:
            self._ready_action_cv.notify_all()
        return dropped

    def _cleanup_pending_action_trace_meta(self, current_env_step: int) -> None:
        stale_steps = [int(step) for step in self._pending_action_trace_meta.keys() if int(step) < int(current_env_step)]
        for step in stale_steps:
            del self._pending_action_trace_meta[int(step)]

    def _should_accept_job_result(self, msg: dict[str, Any]) -> bool:
        if int(msg.get("phase_id", -1)) != int(self._phase_id):
            return False
        window_start_env_step = int(msg.get("window_start_env_step", -1))
        if window_start_env_step < int(self._phase_min_env_step):
            return False
        return True

    def _build_step_record(self, msg: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "phase_id": int(msg.get("phase_id", -1)),
            "job_id": int(msg.get("job_id", -1)),
            "step_id": int(msg.get("job_id", -1)),
            "trigger_obs_index": int(msg.get("trigger_obs_index", -1)),
            "trigger_env_step": int(msg.get("trigger_env_step", -1)),
            "window_start_env_step": int(msg.get("window_start_env_step", -1)),
            "num_released_actions": int(len(msg.get("released_env_steps", []))),
            "step_loop_duration_ms": float(msg.get("job_duration_ms", 0.0)),
            "step_loop_wall_ms": float(msg.get("job_wall_ms", 0.0)),
            "num_step_samples": int(len(msg.get("job_step_samples_ms", []))),
            "num_snapshot_copy_samples": int(len(msg.get("job_snapshot_copy_samples_ms", []))),
        }
        layer_source_steps = msg.get("job_layer_source_steps")
        if layer_source_steps is not None:
            record["job_layer_source_steps"] = layer_source_steps
        profile_trace_step = msg.get("job_profile_trace_step")
        if profile_trace_step is not None:
            record["job_profile_trace_step"] = profile_trace_step
        return record

    def _collect_completed_video_refreshes(self, *, force: bool = False) -> None:
        if self.device.type != "cuda":
            return
        while self._pending_video_refreshes:
            refresh = self._pending_video_refreshes[0]
            end_event = refresh["end_event"]
            if not force and not end_event.query():
                break
            if force:
                end_event.synchronize()
            start_event = refresh["start_event"]
            try:
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except RuntimeError:
                elapsed_ms = float((time.perf_counter() - float(refresh["wall_t0"])) * 1000.0)
            self._video_refresh_samples_ms.append(elapsed_ms)
            self._completed_snapshot = self.video_model.snapshot_cache_for_action_step()
            self._pending_video_refreshes.popleft()

    def _latest_snapshot_env_step(self, snapshot) -> int:
        if len(snapshot.layer_env_steps) == 0:
            return int(snapshot.env_step)
        return max(int(v) for v in snapshot.layer_env_steps)

    def _has_live_cache(self) -> bool:
        return bool(self.video_model.streaming_cache_state.has_live_cache())

    def _snapshot_for_action(self):
        if self._completed_snapshot is not None:
            snapshot = self._completed_snapshot
            self._last_snapshot = snapshot
            return snapshot
        if self._pending_video_refreshes:
            return None
        if not self._has_live_cache():
            return None
        snapshot = self.video_model.snapshot_cache_for_action_step()
        self._completed_snapshot = snapshot
        self._last_snapshot = snapshot
        return snapshot

    def _can_launch_action_step(self, snapshot) -> bool:
        if snapshot is None:
            return False
        if self._active_job is None:
            return True
        job = self._active_job
        if job.token_denoise_counts is None:
            return True
        max_steps = int(job.timesteps.shape[0])
        has_active = bool(torch.any(job.token_denoise_counts < max_steps).item())
        if has_active:
            return True
        return int(snapshot.env_step) > int(job.window_start_env_step)

    def _build_layer_source_step(self, snapshot, job) -> list[dict[str, Any]]:
        if not self.profile or not self.collect_layer_source_stats:
            return []
        mode, mode_frontier, age_hist, latest_offset, older_offset = _classify_layer_sources(
            layer_env_steps=snapshot.layer_env_steps,
            trigger_env_step=int(job.window_start_env_step),
        )
        return [
            {
                "denoise_step": int(job.current_step_idx),
                "layer_obs_indices": [int(v) for v in snapshot.layer_obs_indices],
                "mode": str(mode),
                "frontier": int(mode_frontier),
                "age_hist": {
                    "age0": int(age_hist["age0"]),
                    "age1": int(age_hist["age1"]),
                    "age2": int(age_hist["age2"]),
                    "age3p": int(age_hist["age3p"]),
                },
                "latest_offset": int(latest_offset),
                "older_offset": None if older_offset is None else int(older_offset),
            }
        ]

    def _launch_action_step(self) -> bool:
        if self._inflight_action_step is not None:
            return False
        snapshot = self._snapshot_for_action()
        if not self._can_launch_action_step(snapshot):
            return False
        assert snapshot is not None

        if self._active_job is None:
            self._active_job = self.action_model.start_action_job(
                action_horizon=int(self.action_horizon),
                context=self.action_context,
                context_mask=self.action_context_mask,
                trigger_obs_index=int(snapshot.obs_index),
                trigger_env_step=int(snapshot.env_step),
                num_inference_steps=int(self.num_inference_steps),
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device=self.rand_device,
                persistent=True,
            )

        job = self._active_job
        wall_t0 = time.perf_counter()
        profile_trace_step = self._build_profile_trace_step(snapshot, job)

        if self.device.type == "cuda":
            assert self._action_stream is not None
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(self._action_stream):
                start_event.record(self._action_stream)
                self.action_model.step_action_job(job, snapshot=snapshot)
                end_event.record(self._action_stream)
            self._inflight_action_step = {
                "job_id": int(self._submitted_jobs),
                "phase_id": int(self._phase_id),
                "job": job,
                "start_event": start_event,
                "end_event": end_event,
                "wall_t0": float(wall_t0),
                "snapshot_obs_index": int(snapshot.obs_index),
                "snapshot_env_step": int(snapshot.env_step),
                "job_profile_trace_step": profile_trace_step,
            }
        else:
            step_t0 = time.perf_counter()
            self.action_model.step_action_job(job, snapshot=snapshot)
            self._inflight_action_step = {
                "job_id": int(self._submitted_jobs),
                "phase_id": int(self._phase_id),
                "job": job,
                "job_step_samples_ms": [float((time.perf_counter() - step_t0) * 1000.0)],
                "wall_t0": float(wall_t0),
                "snapshot_obs_index": int(snapshot.obs_index),
                "snapshot_env_step": int(snapshot.env_step),
                "job_profile_trace_step": profile_trace_step,
            }
        self._submitted_jobs += 1
        self._notify_completion_worker()
        return True

    def _finalize_action_step(self, payload: dict[str, Any]) -> None:
        if not self._should_accept_job_result(payload):
            return
        self._completed_steps += 1
        step_loop_duration_ms = float(payload["job_duration_ms"])
        step_loop_wall_ms = float(payload["job_wall_ms"])
        self._action_step_loop_samples_ms.append(step_loop_duration_ms)
        self._action_step_loop_wall_samples_ms.append(step_loop_wall_ms)
        self._action_step_samples_ms.extend([float(v) for v in payload["job_step_samples_ms"]])
        self._snapshot_copy_samples_ms.extend([float(v) for v in payload["job_snapshot_copy_samples_ms"]])
        self._record_profile_trace_step(payload.get("job_profile_trace_step"))
        released_actions_cpu = payload.get("released_actions_cpu")
        released_env_steps = [int(v) for v in payload.get("released_env_steps", [])]
        if released_actions_cpu is not None and len(released_env_steps) > 0:
            released_actions = self.action_postprocess(released_actions_cpu)
            self._publish_released_actions(
                released_actions=np.asarray(released_actions, dtype=np.float32),
                released_env_steps=released_env_steps,
                source_env_step=int(payload.get("window_start_env_step", -1)),
                release_obs_index=int(payload.get("trigger_obs_index", -1)),
                release_obs_env_step=int(payload.get("trigger_env_step", -1)),
                job_id=int(payload.get("job_id", -1)),
                phase_id=int(payload.get("phase_id", -1)),
            )

    def _harvest_action_step(self, *, force: bool = False) -> bool:
        if self._inflight_action_step is None:
            return False
        inflight = self._inflight_action_step
        if self.device.type == "cuda":
            end_event = inflight["end_event"]
            if not force and not end_event.query():
                return False
            end_event.synchronize()
            start_event = inflight["start_event"]
            try:
                job_step_samples_ms = [float(start_event.elapsed_time(end_event))]
            except RuntimeError:
                job_step_samples_ms = [float((time.perf_counter() - float(inflight["wall_t0"])) * 1000.0)]
        else:
            job_step_samples_ms = [float(v) for v in inflight["job_step_samples_ms"]]

        job = inflight["job"]
        released_mask = job.just_released_mask
        released_actions_cpu = torch.empty((0, int(job.latents_action.shape[-1])), dtype=torch.float32)
        released_env_steps: list[int] = []
        if released_mask is not None and bool(torch.any(released_mask).item()):
            released_actions_cpu = job.latents_action[released_mask].detach().to(device="cpu", dtype=torch.float32)
            if job.token_env_steps is None:
                raise ValueError("`job.token_env_steps` must be initialized when releasing actions.")
            released_env_steps = [
                int(v) for v in job.token_env_steps[released_mask].detach().to(device="cpu").tolist()
            ]

        payload = {
            "type": "job_done",
            "phase_id": int(inflight["phase_id"]),
            "job_id": int(inflight["job_id"]),
            "trigger_obs_index": int(inflight.get("snapshot_obs_index", job.trigger_obs_index)),
            "trigger_env_step": int(inflight.get("snapshot_env_step", job.trigger_env_step)),
            "window_start_env_step": int(job.window_start_env_step),
            "released_actions_cpu": released_actions_cpu,
            "released_env_steps": released_env_steps,
            "job_step_samples_ms": job_step_samples_ms,
            "job_snapshot_copy_samples_ms": [0.0],
            "job_duration_ms": float(sum(job_step_samples_ms)),
            "job_wall_ms": float((time.perf_counter() - float(inflight["wall_t0"])) * 1000.0),
        }
        payload["job_profile_trace_step"] = self._finalize_profile_trace_step(
            inflight.get("job_profile_trace_step"), job
        )
        self._inflight_action_step = None
        self._finalize_action_step(payload)
        return True

    def _drain_once(self, *, force: bool = False) -> bool:
        self._collect_completed_video_refreshes(force=force)
        harvested = self._harvest_action_step(force=force)
        launched = False
        if self._inflight_action_step is None:
            launched = self._launch_action_step()
        return bool(harvested or launched)

    def start(self) -> None:
        with self._completion_lock:
            if self._started:
                return
            self.video_model = self.video_model.to(self.video_model.device).eval()
            if self.action_model is not self.video_model:
                self.action_model = self.action_model.to(self.action_model.device).eval()
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
                self._video_stream = torch.cuda.Stream(device=self.device)
                self._action_stream = torch.cuda.Stream(device=self.device)
            self.video_model.reset_streaming_state()
            if self.action_model is not self.video_model:
                self.action_model.reset_streaming_state()
            self._completion_stop_event.clear()
            self._completion_wakeup_event.clear()
            self._started = True
            self._completion_thread = threading.Thread(
                target=self._completion_loop,
                name="fastwam-action-completion",
                daemon=True,
            )
            self._completion_thread.start()

    def stop(self) -> None:
        with self._completion_lock:
            if not self._started:
                return
            self._started = False
            self._completion_stop_event.set()
            self._completion_wakeup_event.set()
        if self._completion_thread is not None and self._completion_thread.is_alive():
            self._completion_thread.join(timeout=1.0)
        with self._completion_lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._collect_completed_video_refreshes(force=True)
            self._harvest_action_step(force=True)
            self._video_stream = None
            self._action_stream = None
            self._pending_video_refreshes.clear()
            self._inflight_action_step = None
            self._completed_snapshot = None
            self._completion_thread = None
            self._ready_action_cv.notify_all()

    def poll(self) -> None:
        with self._completion_lock:
            self._drain_once(force=False)

    def wait_until_idle(self) -> None:
        max_iters = max(4, int(self.num_inference_steps) * 2 + int(self.action_horizon))
        target_step = int(self._current_env_step)
        for _ in range(max_iters):
            with self._completion_lock:
                progressed = self._drain_once(force=True)
                if target_step in self._ensembler.action_cache:
                    return
                if not progressed and self._inflight_action_step is None:
                    return

    def bootstrap_sync(self, *, input_image: torch.Tensor, obs_index: int, obs_timestamp_ms: float) -> None:
        self.submit_observation(
            input_image=input_image,
            proprio=None,
            env_step=-1,
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            trigger_job=False,
        )
        self.wait_until_idle()

    def reset_for_formal_phase(self, *, env_step: int = 0, preserve_streaming_state: bool = False) -> None:
        with self._completion_lock:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self._collect_completed_video_refreshes(force=True)
            self._harvest_action_step(force=True)
            if bool(preserve_streaming_state):
                self._ensembler._cleanup(int(env_step))
            else:
                self._ensembler.reset()
            self._current_env_step = int(env_step)
            self._phase_min_env_step = int(env_step)
            self._phase_id += 1
            self._submitted_jobs = 0
            self._completed_steps = 0
            self._actions_served = 0
            self._actions_missed = 0
            self._dropped_prefix_actions = 0
            self._submitted_obs = 0
            self._obs_count = 0
            self._video_refresh_samples_ms = []
            self._action_step_loop_samples_ms = []
            self._action_step_loop_wall_samples_ms = []
            self._action_step_samples_ms = []
            self._snapshot_copy_samples_ms = []
            self._released_action_trace = []
            self._served_action_trace = []
            self._pending_action_trace_meta = {}
            self._reset_profile_trace_samples()
            self._reset_layer_source_stats()
            self._completed_snapshot = None
            if not bool(preserve_streaming_state):
                self._pending_video_refreshes.clear()
                self._inflight_action_step = None
                self._active_job = None
                self._last_snapshot = None
                self.video_model.reset_streaming_state()
                if self.action_model is not self.video_model:
                    self.action_model.reset_streaming_state()
            self._ready_action_cv.notify_all()

    def submit_observation(
        self,
        *,
        input_image: torch.Tensor,
        proprio: Optional[torch.Tensor],
        env_step: int,
        obs_index: int,
        obs_timestamp_ms: float,
        trigger_job: bool,
    ) -> None:
        del trigger_job
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)

        with self._completion_lock:
            if self.device.type == "cuda":
                assert self._video_stream is not None
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                wall_t0 = time.perf_counter()
                with torch.cuda.stream(self._video_stream):
                    start_event.record(self._video_stream)
                    if not self.video_model.streaming_cache_state.has_live_cache():
                        self.video_model.bootstrap_observation(
                            input_image=input_image,
                            context=self.video_context,
                            context_mask=self.video_context_mask,
                            proprio=proprio,
                            obs_index=int(obs_index),
                            env_step=int(env_step),
                            obs_timestamp_ms=float(obs_timestamp_ms),
                            tiled=self.tiled,
                        )
                    else:
                        self.video_model.submit_observation(
                            input_image=input_image,
                            context=self.video_context,
                            context_mask=self.video_context_mask,
                            proprio=proprio,
                            obs_index=int(obs_index),
                            env_step=int(env_step),
                            obs_timestamp_ms=float(obs_timestamp_ms),
                            tiled=self.tiled,
                        )
                    end_event.record(self._video_stream)
                self._pending_video_refreshes.append(
                    {
                        "start_event": start_event,
                        "end_event": end_event,
                        "wall_t0": float(wall_t0),
                    }
                )
            else:
                wall_t0 = time.perf_counter()
                if not self.video_model.streaming_cache_state.has_live_cache():
                    self.video_model.bootstrap_observation(
                        input_image=input_image,
                        context=self.video_context,
                        context_mask=self.video_context_mask,
                        proprio=proprio,
                        obs_index=int(obs_index),
                        env_step=int(env_step),
                        obs_timestamp_ms=float(obs_timestamp_ms),
                        tiled=self.tiled,
                    )
                else:
                    self.video_model.submit_observation(
                        input_image=input_image,
                        context=self.video_context,
                        context_mask=self.video_context_mask,
                        proprio=proprio,
                        obs_index=int(obs_index),
                        env_step=int(env_step),
                        obs_timestamp_ms=float(obs_timestamp_ms),
                        tiled=self.tiled,
                    )
                self._video_refresh_samples_ms.append(float((time.perf_counter() - wall_t0) * 1000.0))
                self._completed_snapshot = self.video_model.snapshot_cache_for_action_step()

            self._submitted_obs += 1
            self._obs_count += 1
            self._drain_once(force=False)
            self._notify_completion_worker()

    def completed_jobs(self) -> int:
        self.poll()
        with self._completion_lock:
            return int(self._completed_steps)

    def pending_jobs(self) -> int:
        self.poll()
        with self._completion_lock:
            return int(self._inflight_action_step is not None)

    def _pop_action_locked(self, env_step: int, *, count_miss: bool) -> Optional[np.ndarray]:
        self._current_env_step = max(self._current_env_step, int(env_step))
        self._ensembler._cleanup(int(env_step))
        self._cleanup_pending_action_trace_meta(int(env_step))
        if int(env_step) not in self._ensembler.action_cache:
            if count_miss:
                self._actions_missed += 1
            return None
        action = np.asarray(self._ensembler.get_action(int(env_step)), dtype=np.float32)
        trace_meta = self._pending_action_trace_meta.pop(int(env_step), None)
        del self._ensembler.action_cache[int(env_step)]
        self._actions_served += 1
        if self.collect_full_trace:
            served_event: dict[str, Any] = {"env_step": int(env_step)}
            if trace_meta is not None:
                served_event.update(
                    {
                        "source_env_step": int(trace_meta["source_env_step"]),
                        "release_obs_index": int(trace_meta["release_obs_index"]),
                        "release_obs_env_step": int(trace_meta["release_obs_env_step"]),
                        "release_obs_gap": int(trace_meta["release_obs_gap"]),
                        "job_id": int(trace_meta["job_id"]),
                        "phase_id": int(trace_meta["phase_id"]),
                        "served_env_gap": int(env_step - int(trace_meta["source_env_step"])),
                    }
                )
            self._served_action_trace.append(served_event)
        return action

    def get_action(self, env_step: int, *, count_miss: bool = True) -> Optional[np.ndarray]:
        self.poll()
        with self._completion_lock:
            return self._pop_action_locked(int(env_step), count_miss=count_miss)

    def wait_for_action_available(
        self,
        env_step: int,
        *,
        timeout_s: Optional[float] = None,
        count_miss: bool = False,
    ) -> Optional[np.ndarray]:
        deadline = None if timeout_s is None else time.perf_counter() + float(timeout_s)
        with self._completion_lock:
            while True:
                self._drain_once(force=False)
                action = self._pop_action_locked(int(env_step), count_miss=False)
                if action is not None:
                    return action
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0.0:
                        if count_miss:
                            self._actions_missed += 1
                        return None
                    self._ready_action_cv.wait(timeout=min(0.01, remaining))
                else:
                    self._ready_action_cv.wait(timeout=0.01)

    def stats(self) -> dict[str, object]:
        with self._completion_lock:
            self._collect_completed_video_refreshes(force=True)
            self._harvest_action_step(force=True)
            payload: dict[str, object] = {
                "submitted_obs": int(self._submitted_obs),
                "submitted_jobs": int(self._submitted_jobs),
                "completed_jobs": int(self._completed_steps),
                "completed_steps": int(self._completed_steps),
                "actions_served": int(self._actions_served),
                "actions_missed": int(self._actions_missed),
                "dropped_prefix_actions": int(self._dropped_prefix_actions),
                "timing_ms": {
                    "video_refresh": _summarize_ms(self._video_refresh_samples_ms),
                    "action_step_loop": _summarize_ms(self._action_step_loop_samples_ms),
                    "action_step_loop_wall": _summarize_ms(self._action_step_loop_wall_samples_ms),
                    "action_step": _summarize_ms(self._action_step_samples_ms),
                    "snapshot_copy": _summarize_ms(self._snapshot_copy_samples_ms),
                },
            }
            if self.collect_full_trace:
                payload["full_action_trace"] = {
                    "released": list(self._released_action_trace),
                    "served": list(self._served_action_trace),
                }
            if self.profile:
                payload["sampled_denoise_traces"] = self._build_profile_trace_payload()
            return payload


class ProfiledRuntime(StreamingRuntime):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["profile"] = True
        super().__init__(*args, **kwargs)


__all__ = [
    "StreamingRuntime",
    "ProfiledRuntime",
]
