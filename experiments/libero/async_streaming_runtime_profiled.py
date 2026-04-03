from __future__ import annotations

from collections import defaultdict
import queue
from typing import Any, Optional

import numpy as np
import torch

from experiments.libero.async_streaming_runtime import AsyncStreamingActionRuntime
from experiments.libero.async_streaming_workers_profiled import _action_worker_loop_profiled, _video_worker_loop


def _summarize_int_samples(samples: list[int]) -> dict[str, float | int | None]:
    if len(samples) == 0:
        return {
            "count": 0,
            "avg": None,
            "p10": None,
            "p50": None,
            "p90": None,
            "min": None,
            "max": None,
        }
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


class AsyncStreamingActionRuntimeProfiled(AsyncStreamingActionRuntime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reset_layer_source_stats()

    def _reset_layer_source_stats(self) -> None:
        self._layer_source_step_samples: dict[int, int] = defaultdict(int)
        self._layer_source_mode_counts: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._layer_source_frontier_samples: dict[int, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._layer_source_age_counts: dict[int, dict[str, int]] = defaultdict(
            lambda: {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}
        )

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
        self._action_process = self._ctx.Process(
            target=_action_worker_loop_profiled,
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
            prop_cpu = None
            if proprio is not None:
                if proprio.ndim == 1:
                    proprio = proprio.unsqueeze(0)
                prop_cpu = proprio.detach().to(device="cpu", dtype=torch.float32)
            self._put_queue(
                self._job_queue,
                {
                    "type": "job",
                    "trigger_env_step": int(env_step),
                    "obs_index": int(obs_index),
                    "proprio": prop_cpu,
                    "job_seed_offset": int(self._job_seed_counter),
                },
            )
            self._job_seed_counter += 1
            self._submitted_jobs += 1

    def reset_for_formal_phase(self, *, env_step: int = 0) -> None:
        super().reset_for_formal_phase(env_step=env_step)
        self._reset_layer_source_stats()

    def _drain_action_results(self) -> None:
        if self._result_queue is None:
            return
        while True:
            try:
                msg = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if str(msg.get("type")) != "job_done":
                continue
            self._completed_jobs += 1
            job_duration_ms = float(msg["job_duration_ms"])
            job_wall_ms = float(msg["job_wall_ms"])
            self._action_job_samples_ms.append(job_duration_ms)
            self._action_job_wall_samples_ms.append(job_wall_ms)
            self._action_job_samples_raw_ms.append(job_duration_ms)
            self._action_step_samples_ms.extend([float(v) for v in msg["job_step_samples_ms"]])
            self._snapshot_copy_samples_ms.extend([float(v) for v in msg["job_snapshot_copy_samples_ms"]])
            self._accumulate_layer_source_steps(msg.get("job_layer_source_steps", []))

            action_chunk = self.action_postprocess(msg["latents_action_cpu"])
            self._publish_action_chunk(
                action_chunk=np.asarray(action_chunk, dtype=np.float32),
                trigger_env_step=int(msg["trigger_env_step"]),
            )

    def _accumulate_layer_source_steps(self, layer_source_steps: list[dict[str, Any]]) -> None:
        for step in layer_source_steps:
            denoise_step = int(step.get("denoise_step", -1))
            mode = str(step.get("mode", "full_cur"))
            frontier = int(step.get("frontier", 0))
            age_hist = dict(step.get("age_hist", {}))
            self._layer_source_step_samples[denoise_step] += 1
            self._layer_source_mode_counts[denoise_step][mode] += 1
            if "_to_" in mode:
                self._layer_source_frontier_samples[denoise_step][mode].append(frontier)
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
            mode_probs = {mode: float(count) / float(samples) for mode, count in mode_counts.items()}
            frontier_samples_by_mode = {
                mode: [int(v) for v in values]
                for mode, values in self._layer_source_frontier_samples[denoise_step].items()
            }
            frontier_stats_by_mode = {
                mode: _summarize_int_samples(values)
                for mode, values in frontier_samples_by_mode.items()
            }
            age_counts = dict(self._layer_source_age_counts[denoise_step])
            age_total = int(sum(age_counts.values()))
            if age_total > 0:
                age_probs = {key: float(value) / float(age_total) for key, value in age_counts.items()}
            else:
                age_probs = {key: 0.0 for key in age_counts.keys()}
            per_step.append(
                {
                    "denoise_step": int(denoise_step),
                    "samples": int(samples),
                    "mode_counts": mode_counts,
                    "mode_probs": mode_probs,
                    "frontier_samples_by_mode": frontier_samples_by_mode,
                    "frontier_stats_by_mode": frontier_stats_by_mode,
                    "age_counts": age_counts,
                    "age_probs": age_probs,
                }
            )
        return {
            "enabled": True,
            "num_layers": int(self.action_model.mot.num_layers),
            "per_step": per_step,
        }

    def stats(self) -> dict[str, object]:
        payload = super().stats()
        payload["layer_source_stats"] = self._build_layer_source_stats()
        return payload
