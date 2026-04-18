from __future__ import annotations

import queue
import time
import traceback
from typing import Any, Optional

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
                    {"type": "flush_ack", "worker": "video", "flush_id": int(msg["flush_id"])}
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
                            "source_delta": int(layer_cache.get("source_delta", 0)),
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
                version.cache_layers[layer_idx] = {
                    "k": layer_cache["k"],
                    "v": layer_cache["v"],
                    "source_delta": int(layer_cache.get("source_delta", 0)),
                }
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
                        "source_delta": int(layer_cache.get("source_delta", 0)),
                    }
                )

            prefill_state = video_model.mot.init_video_prefill_state(
                video_tokens=video_pre["tokens"],
                video_freqs=video_pre["freqs"],
                video_t_mod=video_pre["t_mod"],
                video_context_payload={"context": video_pre["context"], "mask": video_pre["context_mask"]},
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


def _run_action_worker_loop(
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
    profiled: bool,
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
                "source_delta": int(msg.get("source_delta", 0)),
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
                    {"type": "flush_ack", "worker": "action", "flush_id": int(msg["flush_id"])}
                )
                continue
            if msg_type != "job":
                continue

            trigger_env_step = int(msg["trigger_env_step"])
            trigger_obs_index = int(msg.get("obs_index", -1))
            phase_id = int(msg.get("phase_id", -1))
            job_id = int(msg.get("job_id", -1))
            seed_offset = int(msg["job_seed_offset"])
            proprio_cpu = msg.get("proprio")
            proprio = None
            if proprio_cpu is not None:
                proprio = proprio_cpu.to(
                    device=action_model.device,
                    dtype=action_model.torch_dtype,
                    non_blocking=True,
                )

            job_seed = None
            if seed is not None:
                job_seed = int(seed) + seed_offset
            job = action_model.start_action_job(
                action_horizon=int(action_horizon),
                context=action_context,
                context_mask=action_context_mask,
                proprio=proprio,
                trigger_obs_index=int(trigger_obs_index),
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
                        {
                            "k": layer["k"],
                            "v": layer["v"],
                            "source_delta": (
                                0
                                if int(trigger_obs_index) < 0
                                else int(layer_obs_indices[layer_idx]) - int(trigger_obs_index)
                            ),
                        }  # type: ignore[index]
                        for layer_idx, layer in enumerate(layer_cache_on_action)
                    ],
                    context=action_context,
                    context_mask=action_context_mask,
                    layer_version_ids=list(layer_version_ids),
                    layer_obs_indices=list(layer_obs_indices),
                    layer_obs_timestamps_ms=list(layer_obs_timestamps_ms),
                    layer_ready_events=[None] * num_layers,
                )
                if profiled:
                    mode, mode_frontier, age_hist, latest_offset, older_offset = _classify_layer_sources(
                        layer_obs_indices=snapshot.layer_obs_indices,
                        trigger_obs_index=trigger_obs_index,
                    )
                    job_layer_source_steps.append(
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

            result_payload = {
                "type": "job_done",
                "phase_id": int(phase_id),
                "job_id": int(job_id),
                "trigger_env_step": int(trigger_env_step),
                "trigger_obs_index": int(trigger_obs_index),
                "latents_action_cpu": job.latents_action.detach().to(device="cpu", dtype=torch.float32),
                "job_step_samples_ms": [float(v) for v in job_step_samples_ms],
                "job_snapshot_copy_samples_ms": [float(v) for v in job_snapshot_copy_samples_ms],
                "job_duration_ms": float(sum(job_step_samples_ms)),
                "job_wall_ms": float((time.perf_counter() - job_wall_t0) * 1000.0),
            }
            if profiled:
                result_payload["job_layer_source_steps"] = job_layer_source_steps
            result_queue.put(result_payload)
    except BaseException:
        control_queue.put(
            {
                "type": "worker_error",
                "worker": "action",
                "traceback": traceback.format_exc(),
            }
        )


def _action_worker_loop(**kwargs: Any) -> None:
    _run_action_worker_loop(profiled=False, **kwargs)


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
    while latest_frontier < len(layer_obs_indices) and int(layer_obs_indices[latest_frontier]) == latest_obs_index:
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


def _action_worker_loop_profiled(**kwargs: Any) -> None:
    _run_action_worker_loop(profiled=True, **kwargs)


def _torch_dtype_from_name(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).strip().lower()
    if key in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if key in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if key in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype name: {dtype_name}")


def torch_dtype_to_name(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float32"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.bfloat16:
        return "bfloat16"
    raise ValueError(f"Unsupported torch dtype: {dtype}")


def _build_action_model_from_spec(
    *,
    model_cfg: dict[str, Any],
    checkpoint_path: str,
    device: str,
    model_dtype_name: str,
    move_text_encoder_to_cpu: bool,
):
    model = instantiate(
        OmegaConf.create(model_cfg),
        model_dtype=_torch_dtype_from_name(model_dtype_name),
        device=str(device),
    )
    model.load_checkpoint(str(checkpoint_path))
    model = model.to(str(device)).eval()
    if move_text_encoder_to_cpu and getattr(model, "text_encoder", None) is not None:
        try:
            model.text_encoder.to("cpu")
        except Exception:
            pass
    return model


def _action_worker_loop_spawn_init_profiled(
    *,
    action_model_spec: dict[str, Any],
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
    try:
        action_model = _build_action_model_from_spec(**action_model_spec)
    except BaseException:
        control_queue.put(
            {
                "type": "worker_error",
                "worker": "action",
                "traceback": traceback.format_exc(),
            }
        )
        return
    _run_action_worker_loop(
        action_model=action_model,
        action_context=action_context,
        action_context_mask=action_context_mask,
        action_horizon=action_horizon,
        num_inference_steps=num_inference_steps,
        sigma_shift=sigma_shift,
        rand_device=rand_device,
        seed=seed,
        layer_queue=layer_queue,
        job_queue=job_queue,
        result_queue=result_queue,
        control_queue=control_queue,
        profiled=True,
    )


__all__ = [
    "_action_worker_loop",
    "_action_worker_loop_profiled",
    "_action_worker_loop_spawn_init_profiled",
    "_build_action_model_from_spec",
    "_classify_layer_sources",
    "_snapshot_header",
    "_video_worker_loop",
    "torch_dtype_to_name",
]
