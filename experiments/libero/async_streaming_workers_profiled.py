from __future__ import annotations

import queue
import time
import traceback
from typing import Any, Optional

import torch

from fastwam.models.wan22.streaming_cache import CacheSnapshot

from experiments.libero.async_streaming_workers import _snapshot_header, _video_worker_loop


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
    layer_obs_indices: list[int],
    trigger_obs_index: int,
) -> tuple[str, int, dict[str, int], int, Optional[int]]:
    if len(layer_obs_indices) == 0:
        return "full_cur", 0, {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}, 0, None

    latest_obs_index = max(int(v) for v in layer_obs_indices)
    latest_frontier = 0
    while (
        latest_frontier < len(layer_obs_indices)
        and int(layer_obs_indices[latest_frontier]) == latest_obs_index
    ):
        latest_frontier += 1
    if latest_frontier == 0:
        latest_frontier = len(layer_obs_indices)

    older_obs_index = None
    for obs_idx in layer_obs_indices[latest_frontier:]:
        if int(obs_idx) != latest_obs_index:
            older_obs_index = int(obs_idx)
            break

    latest_offset = int(latest_obs_index - int(trigger_obs_index))
    mode = _offset_to_full_mode(latest_offset)
    older_offset: Optional[int] = None
    if older_obs_index is not None:
        older_offset = int(older_obs_index - int(trigger_obs_index))
        mode = f"{_offset_to_label(older_offset)}_to_{_offset_to_label(latest_offset)}"

    age_hist = {"age0": 0, "age1": 0, "age2": 0, "age3p": 0}
    for obs_idx in layer_obs_indices:
        age = int(latest_obs_index - int(obs_idx))
        if age <= 0:
            age_hist["age0"] += 1
        elif age == 1:
            age_hist["age1"] += 1
        elif age == 2:
            age_hist["age2"] += 1
        else:
            age_hist["age3p"] += 1
    return mode, int(latest_frontier), age_hist, int(latest_offset), older_offset


def _action_worker_loop_profiled(
    *,
    action_model,
    action_context: torch.Tensor,
    action_context_mask: torch.Tensor,
    action_horizon: int,
    num_inference_steps: int,
    sigma_shift: Optional[float],
    rand_device: str,
    seed: Optional[int],
    layer_queue,
    job_queue,
    result_queue,
    control_queue,
) -> None:
    def _drain_layer_updates(
        *,
        block: bool,
        timeout_s: float = 0.0,
    ) -> tuple[int, float]:
        pending_by_layer: dict[int, dict[str, Any]] = {}
        copy_ms = 0.0
        while True:
            try:
                if block and len(pending_by_layer) == 0:
                    msg = layer_queue.get(timeout=timeout_s)
                else:
                    msg = layer_queue.get_nowait()
            except queue.Empty:
                break
            if str(msg.get("type")) != "layer_update":
                continue
            layer_idx = int(msg["layer_idx"])
            pending_by_layer[layer_idx] = msg

        for layer_idx, msg in pending_by_layer.items():
            t_copy0 = time.perf_counter()
            layer_cache_on_action[layer_idx] = {
                "k": msg["k"].to(device=action_model.device, non_blocking=True),
                "v": msg["v"].to(device=action_model.device, non_blocking=True),
            }
            copy_ms += (time.perf_counter() - t_copy0) * 1000.0
            layer_version_ids[layer_idx] = int(msg["version"])
            layer_obs_indices[layer_idx] = int(msg["obs_index"])
            layer_obs_timestamps_ms[layer_idx] = float(msg["obs_timestamp_ms"])
            live_video_seq_len[0] = int(msg["video_seq_len"])
            live_tokens_per_frame[0] = int(msg["tokens_per_frame"])
        return len(pending_by_layer), copy_ms

    try:
        action_model = action_model.to(action_model.device).eval()
        if action_model.device.type == "cuda":
            torch.cuda.set_device(action_model.device)
        action_model.reset_streaming_state()
        action_context = action_context.to(
            device=action_model.device,
            dtype=action_model.torch_dtype,
            non_blocking=True,
        )
        action_context_mask = action_context_mask.to(
            device=action_model.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        resolved_rand_device = str(rand_device)
        if resolved_rand_device == "cuda" and action_model.device.type == "cuda":
            resolved_rand_device = str(action_model.device)

        num_layers = int(action_model.mot.num_layers)
        layer_cache_on_action: list[Optional[dict[str, torch.Tensor]]] = [None] * num_layers
        layer_version_ids: list[int] = [-1] * num_layers
        layer_obs_indices: list[int] = [-1] * num_layers
        layer_obs_timestamps_ms: list[float] = [0.0] * num_layers
        live_video_seq_len = [-1]
        live_tokens_per_frame = [-1]

        while True:
            _drain_layer_updates(block=False)
            try:
                msg = job_queue.get(timeout=0.01)
            except queue.Empty:
                continue
            msg_type = str(msg.get("type"))
            if msg_type == "stop":
                _drain_layer_updates(block=False)
                break
            if msg_type == "flush":
                _drain_layer_updates(block=False)
                control_queue.put(
                    {
                        "type": "flush_ack",
                        "worker": "action",
                        "flush_id": int(msg["flush_id"]),
                    }
                )
                continue
            if msg_type != "job":
                continue

            trigger_env_step = int(msg["trigger_env_step"])
            trigger_obs_index = int(msg.get("obs_index", -1))
            seed_offset = int(msg["job_seed_offset"])
            proprio_cpu = msg.get("proprio")
            proprio = None
            if proprio_cpu is not None:
                proprio = proprio_cpu.to(device=action_model.device, dtype=action_model.torch_dtype, non_blocking=True)

            job_seed = None
            if seed is not None:
                job_seed = int(seed) + seed_offset
            job = action_model.start_action_job(
                action_horizon=int(action_horizon),
                context=action_context,
                context_mask=action_context_mask,
                proprio=proprio,
                num_inference_steps=int(num_inference_steps),
                sigma_shift=sigma_shift,
                seed=job_seed,
                rand_device=resolved_rand_device,
            )

            job_step_samples_ms: list[float] = []
            job_step_event_pairs: list[tuple[Any, Any]] = []
            job_step_wall_samples_ms: list[float] = []
            job_snapshot_copy_samples_ms: list[float] = []
            job_layer_source_steps: list[dict[str, Any]] = []
            job_wall_t0 = time.perf_counter()
            while not job.done:
                copy_ms_step = 0.0
                _, drained_copy_ms = _drain_layer_updates(block=False)
                copy_ms_step += drained_copy_ms
                while (
                    any(layer is None for layer in layer_cache_on_action)
                    or live_video_seq_len[0] <= 0
                    or live_tokens_per_frame[0] <= 0
                ):
                    _, drained_copy_ms = _drain_layer_updates(block=True, timeout_s=0.01)
                    copy_ms_step += drained_copy_ms

                version, obs_index, obs_timestamp_ms, frontier = _snapshot_header(
                    layer_version_ids=layer_version_ids,
                    layer_obs_indices=layer_obs_indices,
                    layer_obs_timestamps_ms=layer_obs_timestamps_ms,
                )
                snapshot = CacheSnapshot(
                    version=version,
                    obs_index=obs_index,
                    obs_timestamp_ms=obs_timestamp_ms,
                    frontier=frontier,
                    video_seq_len=int(live_video_seq_len[0]),
                    tokens_per_frame=int(live_tokens_per_frame[0]),
                    cache_layers=[
                        {"k": layer["k"], "v": layer["v"]}  # type: ignore[index]
                        for layer in layer_cache_on_action
                    ],
                    context=action_context,
                    context_mask=action_context_mask,
                    layer_version_ids=list(layer_version_ids),
                    layer_obs_indices=list(layer_obs_indices),
                    layer_obs_timestamps_ms=list(layer_obs_timestamps_ms),
                    layer_ready_events=[None] * num_layers,
                )
                mode, mode_frontier, age_hist, latest_offset, older_offset = _classify_layer_sources(
                    layer_obs_indices=snapshot.layer_obs_indices,
                    trigger_obs_index=trigger_obs_index,
                )
                job_layer_source_steps.append(
                    {
                        "denoise_step": int(job.current_step_idx),
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
                )
                if action_model.device.type == "cuda":
                    step_wall_t0 = time.perf_counter()
                    step_start_event = torch.cuda.Event(enable_timing=True)
                    step_end_event = torch.cuda.Event(enable_timing=True)
                    current_stream = torch.cuda.current_stream(device=action_model.device)
                    step_start_event.record(current_stream)
                    action_model.step_action_job(job, snapshot=snapshot)
                    step_end_event.record(current_stream)
                    job_step_event_pairs.append((step_start_event, step_end_event))
                    job_step_wall_samples_ms.append((time.perf_counter() - step_wall_t0) * 1000.0)
                else:
                    step_t0 = time.perf_counter()
                    action_model.step_action_job(job, snapshot=snapshot)
                    job_step_samples_ms.append((time.perf_counter() - step_t0) * 1000.0)
                job_snapshot_copy_samples_ms.append(float(copy_ms_step))

            if action_model.device.type == "cuda":
                torch.cuda.synchronize(action_model.device)
                try:
                    job_step_samples_ms.extend(
                        float(step_start_event.elapsed_time(step_end_event))
                        for step_start_event, step_end_event in job_step_event_pairs
                    )
                except RuntimeError:
                    job_step_samples_ms.extend(float(v) for v in job_step_wall_samples_ms)
            result_queue.put(
                {
                    "type": "job_done",
                    "trigger_env_step": int(trigger_env_step),
                    "latents_action_cpu": job.latents_action.detach().to(device="cpu", dtype=torch.float32),
                    "job_step_samples_ms": [float(v) for v in job_step_samples_ms],
                    "job_snapshot_copy_samples_ms": [float(v) for v in job_snapshot_copy_samples_ms],
                    "job_duration_ms": float(sum(job_step_samples_ms)),
                    "job_wall_ms": float((time.perf_counter() - job_wall_t0) * 1000.0),
                    "job_layer_source_steps": job_layer_source_steps,
                }
            )
    except BaseException:
        control_queue.put(
            {
                "type": "worker_error",
                "worker": "action",
                "traceback": traceback.format_exc(),
            }
        )
