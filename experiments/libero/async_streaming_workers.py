from __future__ import annotations

import queue
import time
import traceback
from typing import Any, Optional

import torch

from fastwam.models.wan22.streaming_cache import CacheSnapshot


def _snapshot_header(
    layer_version_ids: list[int],
    layer_obs_indices: list[int],
    layer_obs_timestamps_ms: list[float],
) -> tuple[int, int, float, int]:
    if len(layer_version_ids) == 0:
        raise ValueError("No mirrored cache layers are available for snapshot.")
    latest_version = max(layer_version_ids)
    frontier = 0
    while frontier < len(layer_version_ids) and layer_version_ids[frontier] == latest_version:
        frontier += 1
    max_idx = max(range(len(layer_version_ids)), key=lambda i: layer_version_ids[i])
    return (
        int(layer_version_ids[max_idx]),
        int(layer_obs_indices[max_idx]),
        float(layer_obs_timestamps_ms[max_idx]),
        int(len(layer_version_ids) if frontier == 0 else frontier),
    )


def _video_worker_loop(
    *,
    video_model,
    video_context: torch.Tensor,
    video_context_mask: torch.Tensor,
    tiled: bool,
    video_layers_per_chunk: int,
    obs_queue,
    layer_queue,
    control_queue,
) -> None:
    video_refresh_samples_ms: list[float] = []
    try:
        video_model = video_model.to(video_model.device).eval()
        if video_model.device.type == "cuda":
            torch.cuda.set_device(video_model.device)
        video_model.reset_streaming_state()
        video_context = video_context.to(
            device=video_model.device,
            dtype=video_model.torch_dtype,
            non_blocking=True,
        )
        video_context_mask = video_context_mask.to(
            device=video_model.device,
            dtype=torch.bool,
            non_blocking=True,
        )
        while True:
            msg = obs_queue.get()
            msg_type = str(msg.get("type"))
            if msg_type == "stop":
                break
            if msg_type == "flush":
                control_queue.put(
                    {
                        "type": "flush_ack",
                        "worker": "video",
                        "flush_id": int(msg["flush_id"]),
                    }
                )
                continue
            if msg_type != "obs":
                continue

            obs_index = int(msg["obs_index"])
            obs_timestamp_ms = float(msg["obs_timestamp_ms"])
            input_image_cpu = msg["input_image"]
            if input_image_cpu.ndim == 3:
                input_image_cpu = input_image_cpu.unsqueeze(0)
            input_image = input_image_cpu.to(
                device=video_model.device,
                dtype=video_model.torch_dtype,
                non_blocking=True,
            )

            t0 = time.perf_counter()
            if not video_model.streaming_cache_state.has_live_cache():
                version = video_model.bootstrap_observation(
                    input_image=input_image,
                    context=video_context,
                    context_mask=video_context_mask,
                    obs_index=obs_index,
                    obs_timestamp_ms=obs_timestamp_ms,
                    tiled=tiled,
                )
                for layer_idx, layer_cache in enumerate(version.cache_layers):
                    if layer_cache is None:
                        raise RuntimeError(f"Bootstrap layer {layer_idx} is None.")
                    layer_queue.put(
                        {
                            "type": "layer_update",
                            "layer_idx": int(layer_idx),
                            "version": int(version.version),
                            "obs_index": int(version.obs_index),
                            "obs_timestamp_ms": float(version.obs_timestamp_ms),
                            "video_seq_len": int(version.video_seq_len),
                            "tokens_per_frame": int(version.tokens_per_frame),
                            "k": layer_cache["k"].detach(),
                            "v": layer_cache["v"].detach(),
                        }
                    )
                video_refresh_samples_ms.append((time.perf_counter() - t0) * 1000.0)
                continue

            version, video_pre, video_attention_mask = video_model._prepare_streaming_video_version(
                input_image=input_image,
                context=video_context,
                context_mask=video_context_mask,
                obs_index=obs_index,
                obs_timestamp_ms=obs_timestamp_ms,
                tiled=tiled,
            )

            def _publish_layer(layer_idx: int, layer_cache: dict[str, torch.Tensor]) -> None:
                version.cache_layers[layer_idx] = {"k": layer_cache["k"], "v": layer_cache["v"]}
                video_model.streaming_cache_state.apply_layer_update(version, layer_idx, ready_event=None)
                layer_queue.put(
                    {
                        "type": "layer_update",
                        "layer_idx": int(layer_idx),
                        "version": int(version.version),
                        "obs_index": int(version.obs_index),
                        "obs_timestamp_ms": float(version.obs_timestamp_ms),
                        "video_seq_len": int(version.video_seq_len),
                        "tokens_per_frame": int(version.tokens_per_frame),
                        "k": layer_cache["k"].detach(),
                        "v": layer_cache["v"].detach(),
                    }
                )

            prefill_state = video_model.mot.init_video_prefill_state(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                video_attention_mask=video_attention_mask,
            )
            while prefill_state.next_layer_idx < video_model.mot.num_layers:
                video_model.mot.advance_video_prefill_state(
                    state=prefill_state,
                    max_layers=video_layers_per_chunk,
                    layer_callback=_publish_layer,
                )
            video_refresh_samples_ms.append((time.perf_counter() - t0) * 1000.0)

        control_queue.put(
            {
                "type": "worker_stats",
                "worker": "video",
                "video_refresh_samples_ms": [float(v) for v in video_refresh_samples_ms],
            }
        )
    except BaseException:
        control_queue.put(
            {
                "type": "worker_error",
                "worker": "video",
                "traceback": traceback.format_exc(),
            }
        )


def _action_worker_loop(
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
                    # Fall back to host wall-time timing if CUDA event timing is unavailable.
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
