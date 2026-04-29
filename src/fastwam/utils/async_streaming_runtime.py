from __future__ import annotations

from collections import defaultdict
import queue
from typing import Any, Callable, Optional

import numpy as np
import torch
import torch.multiprocessing as mp

from fastwam.utils.action_ensembler import ActionEnsembler
from fastwam.utils.async_streaming_workers import (
    _action_worker_loop,
    _action_worker_loop_profiled,
    _action_worker_loop_spawn_init_profiled,
    _video_worker_loop,
)


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
        action_trigger_every_n_obs: int,
        video_layers_per_chunk: int,
        seed: Optional[int] = None,
        profile: bool = False,
    ) -> None:
        if int(action_trigger_every_n_obs) <= 0:
            raise ValueError("`action_trigger_every_n_obs` must be positive.")
        if int(video_layers_per_chunk) <= 0:
            raise ValueError("`video_layers_per_chunk` must be positive.")

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
        self.seed = seed
        self.profile = bool(profile)

        self._ctx = mp.get_context("spawn")
        self._obs_queue = None
        self._job_queue = None
        self._layer_queue = None
        self._result_queue = None
        self._control_queue = None
        self._video_process = None
        self._action_process = None
        self._started = False
        self._pending_worker_error: Optional[str] = None
        self._pending_flush_acks: set[tuple[str, int]] = set()

        self._ensembler = ActionEnsembler()
        self._obs_count = 0
        self._current_env_step = 0
        self._phase_min_env_step = 0
        self._phase_id = 0
        self._job_seed_counter = 0
        self._job_id_counter = 0
        self._flush_counter = 0
        self._submitted_jobs = 0
        self._completed_jobs = 0
        self._actions_served = 0
        self._actions_missed = 0
        self._dropped_prefix_actions = 0
        self._submitted_obs = 0
        self._video_refresh_samples_ms: list[float] = []
        self._action_job_samples_ms: list[float] = []
        self._action_job_wall_samples_ms: list[float] = []
        self._action_step_samples_ms: list[float] = []
        self._snapshot_copy_samples_ms: list[float] = []
        self._action_job_samples_raw_ms: list[float] = []
        self._job_records: list[dict[str, Any]] = []
        self._reset_layer_source_stats()

    @staticmethod
    def _close_mp_queue(q) -> None:
        if q is None:
            return
        try:
            q.close()
        except Exception:
            pass
        try:
            q.join_thread()
        except Exception:
            pass

    def _raise_if_error(self) -> None:
        if self._pending_worker_error is not None:
            raise RuntimeError(self._pending_worker_error)

    def _poll_control_queue(self) -> None:
        if self._control_queue is None:
            return
        while True:
            try:
                msg = self._control_queue.get_nowait()
            except queue.Empty:
                break
            msg_type = str(msg.get("type"))
            if msg_type == "worker_error":
                self._pending_worker_error = str(msg.get("traceback", ""))
            elif msg_type == "flush_ack":
                self._pending_flush_acks.add((str(msg.get("worker")), int(msg.get("flush_id", -1))))
            elif msg_type == "worker_stats" and str(msg.get("worker")) == "video":
                self._video_refresh_samples_ms = list(map(float, msg.get("video_refresh_samples_ms", [])))

    def _put_queue(self, q, msg: dict[str, Any]) -> None:
        while True:
            self._poll_control_queue()
            self._raise_if_error()
            try:
                q.put(msg, timeout=0.1)
                return
            except queue.Full:
                self._drain_action_results()

    def _wait_flush_ack(self, *, worker: str, flush_id: int) -> None:
        if self._control_queue is None:
            return
        while True:
            self._drain_action_results()
            self._poll_control_queue()
            self._raise_if_error()
            ack_key = (str(worker), int(flush_id))
            if ack_key in self._pending_flush_acks:
                self._pending_flush_acks.remove(ack_key)
                return
            try:
                msg = self._control_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if (
                str(msg.get("type")) == "flush_ack"
                and str(msg.get("worker")) == str(worker)
                and int(msg.get("flush_id", -1)) == int(flush_id)
            ):
                return
            if str(msg.get("type")) == "flush_ack":
                self._pending_flush_acks.add((str(msg.get("worker")), int(msg.get("flush_id", -1))))
                continue
            if str(msg.get("type")) == "worker_error":
                self._pending_worker_error = str(msg.get("traceback", ""))
                self._raise_if_error()
            elif str(msg.get("type")) == "worker_stats" and str(msg.get("worker")) == "video":
                self._video_refresh_samples_ms = list(map(float, msg.get("video_refresh_samples_ms", [])))

    def _drain_action_results(self) -> None:
        if self._result_queue is None:
            return
        while True:
            try:
                msg = self._result_queue.get_nowait()
            except queue.Empty:
                break
            except (FileNotFoundError, EOFError, OSError, ConnectionError):
                break
            if str(msg.get("type")) != "job_done":
                continue
            if not self._should_accept_job_result(msg):
                continue
            self._completed_jobs += 1
            job_duration_ms = float(msg["job_duration_ms"])
            job_wall_ms = float(msg["job_wall_ms"])
            self._action_job_samples_ms.append(job_duration_ms)
            self._action_job_wall_samples_ms.append(job_wall_ms)
            self._action_job_samples_raw_ms.append(job_duration_ms)
            self._action_step_samples_ms.extend([float(v) for v in msg["job_step_samples_ms"]])
            self._snapshot_copy_samples_ms.extend([float(v) for v in msg["job_snapshot_copy_samples_ms"]])
            if self.profile:
                self._accumulate_layer_source_steps(msg.get("job_layer_source_steps", []))
            self._job_records.append(self._build_job_record(msg))
            action_chunk = self.action_postprocess(msg["latents_action_cpu"])
            self._publish_action_chunk(
                action_chunk=np.asarray(action_chunk, dtype=np.float32),
                trigger_env_step=int(msg["trigger_env_step"]),
            )

    def _sync_main_state(self) -> None:
        self._drain_action_results()
        self._poll_control_queue()
        self._raise_if_error()

    def start(self) -> None:
        if self._started:
            return
        self._obs_queue = self._ctx.Queue(maxsize=1)
        self._job_queue = self._ctx.Queue(maxsize=1)
        self._layer_queue = self._ctx.Queue(maxsize=128)
        self._result_queue = self._ctx.Queue(maxsize=16)
        self._control_queue = self._ctx.Queue(maxsize=32)

        self._video_process = self._ctx.Process(
            target=_video_worker_loop,
            kwargs={
                "video_model": self.video_model,
                "video_context": self.video_context,
                "video_context_mask": self.video_context_mask,
                "tiled": self.tiled,
                "video_layers_per_chunk": self.video_layers_per_chunk,
                "obs_queue": self._obs_queue,
                "layer_queue": self._layer_queue,
                "control_queue": self._control_queue,
            },
            daemon=True,
        )
        action_worker = _action_worker_loop_profiled if self.profile else _action_worker_loop
        self._action_process = self._ctx.Process(
            target=action_worker,
            kwargs={
                "action_model": self.action_model,
                "action_context": self.action_context,
                "action_context_mask": self.action_context_mask,
                "action_horizon": self.action_horizon,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "rand_device": self.rand_device,
                "seed": self.seed,
                "layer_queue": self._layer_queue,
                "job_queue": self._job_queue,
                "result_queue": self._result_queue,
                "control_queue": self._control_queue,
            },
            daemon=True,
        )
        self._video_process.start()
        self._action_process.start()
        self._started = True

    def wait_until_idle(self) -> None:
        if not self._started:
            return
        self._raise_if_error()
        self._flush_counter += 1
        video_flush_id = int(self._flush_counter)
        self._put_queue(self._obs_queue, {"type": "flush", "flush_id": video_flush_id})
        self._wait_flush_ack(worker="video", flush_id=video_flush_id)

        self._flush_counter += 1
        action_flush_id = int(self._flush_counter)
        self._put_queue(self._job_queue, {"type": "flush", "flush_id": action_flush_id})
        self._wait_flush_ack(worker="action", flush_id=action_flush_id)
        self._drain_action_results()
        self._poll_control_queue()
        self._raise_if_error()

    def stop(self) -> None:
        if not self._started:
            return
        try:
            self.wait_until_idle()
        except Exception:
            pass
        if self._job_queue is not None:
            self._put_queue(self._job_queue, {"type": "stop"})
        if self._action_process is not None:
            self._action_process.join(timeout=30.0)
            if self._action_process.is_alive():
                self._action_process.terminate()
                self._action_process.join(timeout=5.0)
        if self._obs_queue is not None:
            self._put_queue(self._obs_queue, {"type": "stop"})
        if self._video_process is not None:
            self._video_process.join(timeout=30.0)
            if self._video_process.is_alive():
                self._video_process.terminate()
                self._video_process.join(timeout=5.0)

        self._poll_control_queue()
        self._drain_action_results()
        self._started = False
        self._close_mp_queue(self._obs_queue)
        self._close_mp_queue(self._job_queue)
        self._close_mp_queue(self._layer_queue)
        self._close_mp_queue(self._result_queue)
        self._close_mp_queue(self._control_queue)
        self._obs_queue = None
        self._job_queue = None
        self._layer_queue = None
        self._result_queue = None
        self._control_queue = None
        self._video_process = None
        self._action_process = None
        self._raise_if_error()

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

    def reset_for_formal_phase(self, *, env_step: int = 0) -> None:
        self._drain_action_results()
        self._ensembler.reset()
        self._current_env_step = int(env_step)
        self._phase_min_env_step = int(env_step)
        self._phase_id += 1
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
        self._action_job_wall_samples_ms = []
        self._action_step_samples_ms = []
        self._snapshot_copy_samples_ms = []
        self._action_job_samples_raw_ms = []
        self._job_records = []
        self._reset_layer_source_stats()

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
        self._poll_control_queue()
        self._raise_if_error()
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        image_cpu = input_image.detach().to(device="cpu", dtype=torch.float32)
        self._put_queue(
            self._obs_queue,
            {
                "type": "obs",
                "obs_index": int(obs_index),
                "obs_timestamp_ms": float(obs_timestamp_ms),
                "input_image": image_cpu,
            },
        )
        self._submitted_obs += 1
        self._obs_count += 1
        if trigger_job:
            self.submit_action_job(env_step=env_step, proprio=proprio, obs_index=obs_index)

    def submit_action_job(
        self,
        *,
        env_step: int,
        proprio: Optional[torch.Tensor],
        obs_index: int = -1,
    ) -> None:
        self._poll_control_queue()
        self._raise_if_error()
        prop_cpu = None
        if proprio is not None:
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            prop_cpu = proprio.detach().to(device="cpu", dtype=torch.float32)
        self._put_queue(
            self._job_queue,
            {
                "type": "job",
                "phase_id": int(self._phase_id),
                "job_id": int(self._job_id_counter),
                "trigger_env_step": int(env_step),
                "obs_index": int(obs_index),
                "proprio": prop_cpu,
                "job_seed_offset": int(self._job_seed_counter),
            },
        )
        self._job_id_counter += 1
        self._job_seed_counter += 1
        self._submitted_jobs += 1

    def should_trigger_on_obs(self, obs_count: Optional[int] = None) -> bool:
        count = self._obs_count if obs_count is None else int(obs_count)
        return count > 0 and count % self.action_trigger_every_n_obs == 0

    def completed_jobs(self) -> int:
        self._sync_main_state()
        return int(self._completed_jobs)

    def pending_jobs(self) -> int:
        self._sync_main_state()
        return max(0, int(self._submitted_jobs - self._completed_jobs))

    def _publish_action_chunk(self, action_chunk: np.ndarray, *, trigger_env_step: int) -> int:
        dropped = 0
        current_env_step = int(self._current_env_step)
        for i in range(int(action_chunk.shape[0])):
            target_step = int(trigger_env_step) + i
            if target_step < current_env_step:
                dropped += 1
                continue
            self._ensembler.action_cache[target_step].append(np.asarray(action_chunk[i], dtype=np.float32))
        self._dropped_prefix_actions += dropped
        return dropped

    def get_action(self, env_step: int, *, count_miss: bool = True) -> Optional[np.ndarray]:
        self._sync_main_state()
        self._current_env_step = max(self._current_env_step, int(env_step))
        self._ensembler._cleanup(int(env_step))
        if int(env_step) not in self._ensembler.action_cache:
            if count_miss:
                self._actions_missed += 1
            return None
        action = np.asarray(self._ensembler.get_action(int(env_step)), dtype=np.float32)
        del self._ensembler.action_cache[int(env_step)]
        self._actions_served += 1
        return action

    def _should_accept_job_result(self, msg: dict[str, Any]) -> bool:
        if int(msg.get("phase_id", -1)) != int(self._phase_id):
            return False
        trigger_env_step = int(msg.get("trigger_env_step", -1))
        if trigger_env_step < int(self._phase_min_env_step):
            return False
        return True

    def _build_job_record(self, msg: dict[str, Any]) -> dict[str, Any]:
        record: dict[str, Any] = {
            "phase_id": int(msg.get("phase_id", -1)),
            "job_id": int(msg.get("job_id", -1)),
            "trigger_env_step": int(msg.get("trigger_env_step", -1)),
            "trigger_obs_index": int(msg.get("trigger_obs_index", -1)),
            "job_duration_ms": float(msg.get("job_duration_ms", 0.0)),
            "job_wall_ms": float(msg.get("job_wall_ms", 0.0)),
            "num_step_samples": int(len(msg.get("job_step_samples_ms", []))),
            "num_snapshot_copy_samples": int(len(msg.get("job_snapshot_copy_samples_ms", []))),
        }
        layer_source_steps = msg.get("job_layer_source_steps")
        if layer_source_steps is not None:
            record["job_layer_source_steps"] = layer_source_steps
        return record

    def _reset_layer_source_stats(self) -> None:
        self._layer_source_step_samples: dict[int, int] = defaultdict(int)
        self._layer_source_mode_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_frontier_samples: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._layer_source_latest_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_older_offset_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_age_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}
        )

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
                        for key, value in sorted(
                            dict(self._layer_source_latest_offset_counts[denoise_step]).items()
                        )
                    },
                    "older_offset_counts": {
                        str(key): int(value)
                        for key, value in sorted(
                            dict(self._layer_source_older_offset_counts[denoise_step]).items()
                        )
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

    def stats(self) -> dict[str, object]:
        self._sync_main_state()
        payload: dict[str, object] = {
            "phase_id": int(self._phase_id),
            "submitted_obs": int(self._submitted_obs),
            "submitted_jobs": int(self._submitted_jobs),
            "completed_jobs": int(self._completed_jobs),
            "actions_served": int(self._actions_served),
            "actions_missed": int(self._actions_missed),
            "dropped_prefix_actions": int(self._dropped_prefix_actions),
            "timing_ms": {
                "video_refresh": _summarize_ms(self._video_refresh_samples_ms),
                "action_job": _summarize_ms(self._action_job_samples_ms),
                "action_job_wall": _summarize_ms(self._action_job_wall_samples_ms),
                "action_step": _summarize_ms(self._action_step_samples_ms),
                "snapshot_copy": _summarize_ms(self._snapshot_copy_samples_ms),
            },
            "timing_samples_ms": {
                "action_job": [float(v) for v in self._action_job_samples_raw_ms],
                "action_job_wall": [float(v) for v in self._action_job_wall_samples_ms],
            },
            "job_records": list(self._job_records),
        }
        if self.profile:
            payload["layer_source_stats"] = self._build_layer_source_stats()
        return payload


class ProfiledRuntime(StreamingRuntime):
    def __init__(self, *args, **kwargs) -> None:
        kwargs["profile"] = True
        super().__init__(*args, **kwargs)


class SpawnInitRuntime(ProfiledRuntime):
    def __init__(
        self,
        *args,
        action_model_spec: dict[str, Any],
        debug_process_start: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._action_model_spec = dict(action_model_spec)
        self._debug_process_start = bool(debug_process_start)

    def start(self) -> None:
        if self._started:
            return
        self._obs_queue = self._ctx.Queue(maxsize=1)
        self._job_queue = self._ctx.Queue(maxsize=1)
        self._layer_queue = self._ctx.Queue(maxsize=128)
        self._result_queue = self._ctx.Queue(maxsize=16)
        self._control_queue = self._ctx.Queue(maxsize=32)
        if self._debug_process_start:
            print("[robotwin-debug] runtime.start queues created", flush=True)

        self._video_process = self._ctx.Process(
            target=_video_worker_loop,
            kwargs={
                "video_model": self.video_model,
                "video_context": self.video_context,
                "video_context_mask": self.video_context_mask,
                "tiled": self.tiled,
                "video_layers_per_chunk": self.video_layers_per_chunk,
                "obs_queue": self._obs_queue,
                "layer_queue": self._layer_queue,
                "control_queue": self._control_queue,
            },
            daemon=True,
        )
        self._action_process = self._ctx.Process(
            target=_action_worker_loop_spawn_init_profiled,
            kwargs={
                "action_model_spec": dict(self._action_model_spec),
                "action_context": self.action_context,
                "action_context_mask": self.action_context_mask,
                "action_horizon": self.action_horizon,
                "num_inference_steps": self.num_inference_steps,
                "sigma_shift": self.sigma_shift,
                "rand_device": self.rand_device,
                "seed": self.seed,
                "layer_queue": self._layer_queue,
                "job_queue": self._job_queue,
                "result_queue": self._result_queue,
                "control_queue": self._control_queue,
            },
            daemon=True,
        )
        if self._debug_process_start:
            print("[robotwin-debug] runtime.start process objects created", flush=True)
            print("[robotwin-debug] runtime.start video_process.start begin", flush=True)
        self._video_process.start()
        if self._debug_process_start:
            print(f"[robotwin-debug] runtime.start video_process.start done pid={self._video_process.pid}", flush=True)
            print("[robotwin-debug] runtime.start action_process.start begin", flush=True)
        self._action_process.start()
        if self._debug_process_start:
            print(f"[robotwin-debug] runtime.start action_process.start done pid={self._action_process.pid}", flush=True)
            print("[robotwin-debug] runtime.start done", flush=True)
        self._started = True


__all__ = [
    "StreamingRuntime",
    "ProfiledRuntime",
    "SpawnInitRuntime",
]
