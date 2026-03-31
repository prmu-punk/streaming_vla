from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional

import torch
import torch.nn.functional as F

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .streaming_cache import (
    CacheSnapshot,
    StreamingActionJob,
    StreamingCacheState,
    VideoCacheVersion,
    stitch_prefix_cache,
)

logger = get_logger(__name__)


class FastWAMStreaming(FastWAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_cache_state = StreamingCacheState(num_layers=self.mot.num_layers)
        self.streaming_version_counter = 0
        self.streaming_cfg: dict[str, Any] = {}
        self.streaming_train_cfg: dict[str, Any] = {"enabled": False}
        self.freeze_video_expert = True
        self.streaming_proprio_to_action_only = True

    def configure_streaming(self, streaming: Optional[dict[str, Any]] = None) -> "FastWAMStreaming":
        cfg = {} if streaming is None else dict(streaming)
        self.streaming_cfg = cfg
        self.streaming_train_cfg = dict(cfg.get("streaming_train", {}))
        self.freeze_video_expert = bool(cfg.get("freeze_video_expert", True))
        self.streaming_proprio_to_action_only = bool(cfg.get("proprio_to_action_only", True))
        if self.freeze_video_expert:
            for module in (self.video_expert, self.vae):
                for param in module.parameters():
                    param.requires_grad_(False)
        return self

    def reset_streaming_state(self) -> None:
        self.streaming_cache_state.reset()
        self.streaming_version_counter = 0

    def training_loss_base(self, sample, tiled: bool = False):
        return super().training_loss(sample, tiled=tiled)

    def training_loss(self, sample, tiled: bool = False):
        if bool(self.streaming_train_cfg.get("enabled", False)):
            return self.training_loss_streaming_action_ft(sample, tiled=tiled)
        return super().training_loss(sample, tiled=tiled)

    def _resolve_streaming_condition_inputs(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)

        action_proprio = None
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None`.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2:
                raise ValueError(f"`proprio` must be [B,D] or [D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)
            if self.streaming_proprio_to_action_only:
                action_proprio = proprio
            else:
                context, context_mask = self._append_proprio_to_context(
                    context=context,
                    context_mask=context_mask,
                    proprio=proprio,
                )
        return context, context_mask, action_proprio

    @torch.no_grad()
    def _build_cache_version(
        self,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        obs_timestamp_ms: float,
        obs_index: int = -1,
        tiled: bool = False,
    ) -> VideoCacheVersion:
        payload = self.build_streaming_video_cache_from_input_image(
            input_image=input_image,
            context=context,
            context_mask=context_mask,
            tiled=tiled,
        )
        version = VideoCacheVersion(
            version=int(self.streaming_version_counter),
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            video_seq_len=int(payload["video_seq_len"]),
            tokens_per_frame=int(payload["video_pre"]["meta"]["tokens_per_frame"]),
            cache_layers=payload["video_kv_cache"],
            context=context,
            context_mask=context_mask,
        )
        self.streaming_version_counter += 1
        return version

    @torch.no_grad()
    def _prepare_streaming_video_version(
        self,
        input_image: torch.Tensor,
        *,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        obs_timestamp_ms: float,
        obs_index: int = -1,
        tiled: bool = False,
    ) -> tuple[VideoCacheVersion, dict[str, Any], torch.Tensor]:
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [B,3,H,W], got {tuple(input_image.shape)}"
            )
        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_batch(input_image=input_image, tiled=tiled)
        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self.build_joint_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=1,
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        version = VideoCacheVersion(
            version=int(self.streaming_version_counter),
            obs_index=int(obs_index),
            obs_timestamp_ms=float(obs_timestamp_ms),
            video_seq_len=video_seq_len,
            tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            cache_layers=[None] * self.mot.num_layers,
            context=context,
            context_mask=context_mask,
        )
        self.streaming_version_counter += 1
        return version, video_pre, attention_mask[:video_seq_len, :video_seq_len]

    @torch.no_grad()
    def submit_observation(
        self,
        input_image: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        obs_index: int = -1,
        obs_timestamp_ms: float = 0.0,
        tiled: bool = False,
    ) -> VideoCacheVersion:
        resolved_context, resolved_context_mask, _ = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=None,
        )
        version = self._build_cache_version(
            input_image=input_image,
            context=resolved_context,
            context_mask=resolved_context_mask,
            obs_index=int(obs_index),
            obs_timestamp_ms=obs_timestamp_ms,
            tiled=tiled,
        )
        self.streaming_cache_state.register_pending(version)
        return version

    @torch.no_grad()
    def bootstrap_observation(
        self,
        input_image: torch.Tensor,
        *,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        obs_index: int = -1,
        obs_timestamp_ms: float = 0.0,
        tiled: bool = False,
    ) -> VideoCacheVersion:
        resolved_context, resolved_context_mask, _ = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=None,
        )
        version = self._build_cache_version(
            input_image=input_image,
            context=resolved_context,
            context_mask=resolved_context_mask,
            obs_index=int(obs_index),
            obs_timestamp_ms=obs_timestamp_ms,
            tiled=tiled,
        )
        self.streaming_cache_state.bootstrap(version)
        return version

    def overwrite_video_cache_layer(self, version: VideoCacheVersion, layer_idx: int) -> None:
        self.streaming_cache_state.apply_layer_update(version, layer_idx)

    def _wait_for_snapshot_ready(
        self,
        snapshot: CacheSnapshot,
        *,
        stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if self.device.type != "cuda":
            return
        wait_stream = stream if stream is not None else torch.cuda.current_stream(device=self.device)
        for ready_event in snapshot.layer_ready_events:
            if ready_event is not None:
                wait_stream.wait_event(ready_event)

    def advance_video_cache_frontier(self, max_layers: int = 1) -> int:
        return self.streaming_cache_state.advance_frontier(max_layers=max_layers)

    def snapshot_cache_for_action_step(self) -> CacheSnapshot:
        return self.streaming_cache_state.snapshot()

    def start_action_job(
        self,
        *,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
    ) -> StreamingActionJob:
        resolved_context, resolved_context_mask, action_proprio = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (resolved_context.shape[0], action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        return StreamingActionJob(
            timesteps=timesteps,
            deltas=deltas,
            latents_action=latents_action,
            context=resolved_context,
            context_mask=resolved_context_mask,
            proprio=action_proprio,
        )

    @torch.no_grad()
    def step_action_job(
        self,
        job: StreamingActionJob,
        snapshot: Optional[CacheSnapshot] = None,
    ) -> torch.Tensor:
        if job.done:
            return job.latents_action
        if snapshot is None:
            snapshot = self.snapshot_cache_for_action_step()

        step_t = job.timesteps[job.current_step_idx].unsqueeze(0).to(
            device=self.device,
            dtype=job.latents_action.dtype,
        )
        step_delta = job.deltas[job.current_step_idx]
        attention_mask = self.build_joint_attention_mask(
            video_seq_len=snapshot.video_seq_len,
            action_seq_len=job.latents_action.shape[1],
            video_tokens_per_frame=snapshot.tokens_per_frame,
            device=job.latents_action.device,
        )
        pred_action = self._predict_action_noise_with_cache(
            latents_action=job.latents_action,
            timestep_action=step_t,
            context=job.context,
            context_mask=job.context_mask,
            video_kv_cache=snapshot.cache_layers,
            attention_mask=attention_mask,
            video_seq_len=snapshot.video_seq_len,
            proprio=job.proprio,
        )
        job.latents_action = self.infer_action_scheduler.step(pred_action, step_delta, job.latents_action)
        job.snapshot_history.append(snapshot)
        job.current_step_idx += 1
        return job.latents_action

    @torch.no_grad()
    def infer_action_streaming(
        self,
        *,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        frontier_schedule: Optional[list[int]] = None,
        obs_timestamp_ms: float = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        self.submit_observation(
            input_image=input_image,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            obs_timestamp_ms=obs_timestamp_ms,
            tiled=tiled,
        )
        if frontier_schedule is None:
            frontier_schedule = [self.mot.num_layers] * num_inference_steps
        job = self.start_action_job(
            action_horizon=action_horizon,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
        for max_layers in frontier_schedule:
            self.advance_video_cache_frontier(max_layers=max_layers)
            self.step_action_job(job)
            if job.done:
                break
        while not job.done:
            self.step_action_job(job)
        return {
            "action": job.latents_action[0].detach().to(device="cpu", dtype=torch.float32),
            "num_snapshots": len(job.snapshot_history),
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        del negative_prompt, text_cfg_scale
        frontier_schedule = self.streaming_cfg.get("infer_frontier_schedule")
        return self.infer_action_streaming(
            prompt=prompt,
            input_image=input_image,
            action_horizon=action_horizon,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
            frontier_schedule=frontier_schedule,
        )

    @torch.no_grad()
    def simulate_async_runtime_trace(
        self,
        *,
        observation_images: torch.Tensor,
        action_horizon: int,
        action_trigger_every_n_obs: int,
        obs_dt_ms: float,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        proprio_seq: Optional[torch.Tensor] = None,
        obs_indices: Optional[list[int]] = None,
        num_inference_steps: int = 10,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        warmup_video_bootstrap: bool = True,
        warmup_action_job: bool = True,
    ) -> dict[str, Any]:
        if observation_images.ndim != 4 or observation_images.shape[1] != 3:
            raise ValueError(
                f"`observation_images` must be [T,3,H,W], got {tuple(observation_images.shape)}"
            )
        num_obs = int(observation_images.shape[0])
        if num_obs <= 0:
            raise ValueError("`observation_images` must contain at least one observation.")
        if obs_indices is None:
            obs_indices = list(range(num_obs))
        if len(obs_indices) != num_obs:
            raise ValueError(f"`obs_indices` length must match num_obs ({num_obs}), got {len(obs_indices)}.")
        if int(action_trigger_every_n_obs) <= 0:
            raise ValueError(
                f"`action_trigger_every_n_obs` must be positive, got {action_trigger_every_n_obs}."
            )
        if float(obs_dt_ms) < 0.0:
            raise ValueError(f"`obs_dt_ms` must be non-negative, got {obs_dt_ms}.")

        proprio_steps = None
        if proprio_seq is not None:
            if proprio_seq.ndim == 1:
                proprio_seq = proprio_seq.unsqueeze(0)
            if proprio_seq.ndim != 2 or proprio_seq.shape[0] != num_obs:
                raise ValueError(
                    f"`proprio_seq` must be [T,D] with T={num_obs}, got {tuple(proprio_seq.shape)}."
                )
            proprio_steps = proprio_seq.to(device=self.device, dtype=self.torch_dtype)

        self.eval()
        self.reset_streaming_state()
        resolved_context, resolved_context_mask, _ = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=None,
        )
        obs_timestamps_ms = [float(idx) * float(obs_dt_ms) for idx in obs_indices]
        obs_queue: queue.Queue[Optional[tuple[int, int, float]]] = queue.Queue()
        job_queue: queue.Queue[Optional[tuple[int, int, float]]] = queue.Queue()
        error_queue: queue.Queue[BaseException] = queue.Queue()
        stop_event = threading.Event()
        completed_jobs: list[dict[str, Any]] = []
        use_cuda_streams = self.device.type == "cuda"
        # Keep action denoising on a higher-priority stream so video cache refreshes
        # are less likely to starve action steps under contention.
        video_stream = torch.cuda.Stream(device=self.device, priority=0) if use_cuda_streams else None
        action_stream = torch.cuda.Stream(device=self.device, priority=-1) if use_cuda_streams else None
        default_stream = torch.cuda.default_stream(device=self.device) if use_cuda_streams else None
        stream_start_local_idx = 0

        if warmup_video_bootstrap:
            self.bootstrap_observation(
                input_image=observation_images[0],
                context=resolved_context,
                context_mask=resolved_context_mask,
                obs_index=int(obs_indices[0]),
                obs_timestamp_ms=float(obs_timestamps_ms[0]),
                tiled=tiled,
            )
            stream_start_local_idx = 1

        if warmup_action_job and self.streaming_cache_state.has_live_cache():
            warmup_proprio = None
            if proprio_steps is not None:
                warmup_proprio = proprio_steps[0]
            warmup_job = self.start_action_job(
                action_horizon=action_horizon,
                prompt=prompt,
                context=context,
                context_mask=context_mask,
                proprio=warmup_proprio,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=(None if seed is None else int(seed) - 1),
                rand_device=rand_device,
            )
            while not warmup_job.done:
                warmup_snapshot = self.snapshot_cache_for_action_step()
                self.step_action_job(warmup_job, snapshot=warmup_snapshot)
            if use_cuda_streams:
                torch.cuda.synchronize(self.device)

        def _record_error(exc: BaseException) -> None:
            stop_event.set()
            error_queue.put(exc)

        def _video_worker() -> None:
            try:
                if video_stream is not None and default_stream is not None:
                    video_stream.wait_stream(default_stream)
                while not stop_event.is_set():
                    item = obs_queue.get()
                    if item is None:
                        break
                    local_idx, obs_index, obs_timestamp_ms = item
                    if video_stream is None:
                        version, video_pre, video_attention_mask = self._prepare_streaming_video_version(
                            input_image=observation_images[local_idx],
                            context=resolved_context,
                            context_mask=resolved_context_mask,
                            obs_index=obs_index,
                            obs_timestamp_ms=obs_timestamp_ms,
                            tiled=tiled,
                        )
                    else:
                        with torch.cuda.stream(video_stream):
                            version, video_pre, video_attention_mask = self._prepare_streaming_video_version(
                                input_image=observation_images[local_idx],
                                context=resolved_context,
                                context_mask=resolved_context_mask,
                                obs_index=obs_index,
                                obs_timestamp_ms=obs_timestamp_ms,
                                tiled=tiled,
                            )
                    if not self.streaming_cache_state.has_live_cache():
                        if video_stream is None:
                            full_cache = self.mot.prefill_video_cache(
                                video_tokens=video_pre["tokens"],
                                video_freqs=video_pre["freqs"],
                                video_t_mod=video_pre["t_mod"],
                                video_context_payload={
                                    "context": video_pre["context"],
                                    "mask": video_pre["context_mask"],
                                },
                                video_attention_mask=video_attention_mask,
                            )
                            bootstrap_events = None
                        else:
                            with torch.cuda.stream(video_stream):
                                full_cache = self.mot.prefill_video_cache(
                                    video_tokens=video_pre["tokens"],
                                    video_freqs=video_pre["freqs"],
                                    video_t_mod=video_pre["t_mod"],
                                    video_context_payload={
                                        "context": video_pre["context"],
                                        "mask": video_pre["context_mask"],
                                    },
                                    video_attention_mask=video_attention_mask,
                                )
                                ready_event = torch.cuda.Event()
                                ready_event.record(video_stream)
                            bootstrap_events = [ready_event] * self.mot.num_layers
                        version.cache_layers = full_cache
                        self.streaming_cache_state.bootstrap(version, layer_ready_events=bootstrap_events)
                        continue

                    def _publish_layer(layer_idx: int, layer_cache: dict[str, torch.Tensor]) -> None:
                        version.cache_layers[layer_idx] = {
                            "k": layer_cache["k"],
                            "v": layer_cache["v"],
                        }
                        if video_stream is None:
                            ready_event = None
                        else:
                            ready_event = torch.cuda.Event()
                            ready_event.record(video_stream)
                        self.streaming_cache_state.apply_layer_update(
                            version,
                            layer_idx,
                            ready_event=ready_event,
                        )

                    if video_stream is None:
                        self.mot.prefill_video_cache(
                            video_tokens=video_pre["tokens"],
                            video_freqs=video_pre["freqs"],
                            video_t_mod=video_pre["t_mod"],
                            video_context_payload={
                                "context": video_pre["context"],
                                "mask": video_pre["context_mask"],
                            },
                            video_attention_mask=video_attention_mask,
                            layer_callback=_publish_layer,
                        )
                    else:
                        with torch.cuda.stream(video_stream):
                            self.mot.prefill_video_cache(
                                video_tokens=video_pre["tokens"],
                                video_freqs=video_pre["freqs"],
                                video_t_mod=video_pre["t_mod"],
                                video_context_payload={
                                    "context": video_pre["context"],
                                    "mask": video_pre["context_mask"],
                                },
                                video_attention_mask=video_attention_mask,
                                layer_callback=_publish_layer,
                            )
            except BaseException as exc:
                _record_error(exc)

        def _action_worker() -> None:
            try:
                job_counter = 0
                if action_stream is not None and default_stream is not None:
                    action_stream.wait_stream(default_stream)
                while not stop_event.is_set():
                    item = job_queue.get()
                    if item is None:
                        break
                    local_idx, obs_index, trigger_timestamp_ms = item
                    proprio = None
                    if proprio_steps is not None:
                        proprio = proprio_steps[local_idx]
                    if action_stream is None:
                        job = self.start_action_job(
                            action_horizon=action_horizon,
                            prompt=prompt,
                            context=context,
                            context_mask=context_mask,
                            proprio=proprio,
                            num_inference_steps=num_inference_steps,
                            sigma_shift=sigma_shift,
                            seed=(None if seed is None else int(seed) + job_counter),
                            rand_device=rand_device,
                        )
                    else:
                        with torch.cuda.stream(action_stream):
                            job = self.start_action_job(
                                action_horizon=action_horizon,
                                prompt=prompt,
                                context=context,
                                context_mask=context_mask,
                                proprio=proprio,
                                num_inference_steps=num_inference_steps,
                            sigma_shift=sigma_shift,
                            seed=(None if seed is None else int(seed) + job_counter),
                            rand_device=rand_device,
                        )
                    job_wall_start_ms = (time.perf_counter() - t0) * 1000.0
                    job_trace = {
                        "job_index": int(job_counter),
                        "trigger_obs_index": int(obs_index),
                        "trigger_obs_timestamp_ms": float(trigger_timestamp_ms),
                        "job_wall_start_ms": float(job_wall_start_ms),
                        "steps": [],
                    }
                    job_counter += 1
                    while not job.done:
                        while not self.streaming_cache_state.has_live_cache():
                            if stop_event.is_set():
                                return
                            time.sleep(0.001)
                        step_wall_start_ms = (time.perf_counter() - t0) * 1000.0
                        snapshot = self.snapshot_cache_for_action_step()
                        if action_stream is not None:
                            self._wait_for_snapshot_ready(snapshot, stream=action_stream)
                        step_record = {
                            "denoise_step": int(job.current_step_idx),
                            "step_wall_start_ms": float(step_wall_start_ms),
                            "snapshot_version": int(snapshot.version),
                            "snapshot_obs_index": int(snapshot.obs_index),
                            "snapshot_obs_timestamp_ms": float(snapshot.obs_timestamp_ms),
                            "frontier": int(snapshot.frontier),
                            "layer_version_ids": list(snapshot.layer_version_ids),
                            "layer_obs_indices": list(snapshot.layer_obs_indices),
                            "layer_obs_timestamps_ms": list(snapshot.layer_obs_timestamps_ms),
                        }
                        if action_stream is None:
                            self.step_action_job(job, snapshot=snapshot)
                        else:
                            with torch.cuda.stream(action_stream):
                                self.step_action_job(job, snapshot=snapshot)
                            action_stream.synchronize()
                        step_record["step_wall_end_ms"] = float((time.perf_counter() - t0) * 1000.0)
                        step_record["step_wall_duration_ms"] = (
                            step_record["step_wall_end_ms"] - step_record["step_wall_start_ms"]
                        )
                        job_trace["steps"].append(step_record)
                    job_trace["job_wall_end_ms"] = float((time.perf_counter() - t0) * 1000.0)
                    job_trace["job_wall_duration_ms"] = (
                        job_trace["job_wall_end_ms"] - job_trace["job_wall_start_ms"]
                    )
                    completed_jobs.append(job_trace)
            except BaseException as exc:
                _record_error(exc)

        video_thread = threading.Thread(target=_video_worker, name="fastwam-video-trace", daemon=True)
        action_thread = threading.Thread(target=_action_worker, name="fastwam-action-trace", daemon=True)
        video_thread.start()
        action_thread.start()

        t0 = time.perf_counter()
        try:
            for local_idx in range(stream_start_local_idx, num_obs):
                if stop_event.is_set():
                    break
                target_time = t0 + float(local_idx) * float(obs_dt_ms) / 1000.0
                while True:
                    remaining = target_time - time.perf_counter()
                    if remaining <= 0.0 or stop_event.is_set():
                        break
                    time.sleep(min(remaining, 0.001))
                obs_queue.put((local_idx, int(obs_indices[local_idx]), float(obs_timestamps_ms[local_idx])))
                if (local_idx + 1) % int(action_trigger_every_n_obs) == 0:
                    job_queue.put((local_idx, int(obs_indices[local_idx]), float(obs_timestamps_ms[local_idx])))
        finally:
            obs_queue.put(None)
            job_queue.put(None)
            video_thread.join()
            action_thread.join()
            if use_cuda_streams:
                torch.cuda.synchronize(self.device)

        if not error_queue.empty():
            raise error_queue.get()

        return {
            "obs_indices": [int(v) for v in obs_indices],
            "obs_timestamps_ms": [float(v) for v in obs_timestamps_ms],
            "obs_dt_ms": float(obs_dt_ms),
            "action_trigger_every_n_obs": int(action_trigger_every_n_obs),
            "warmup_video_bootstrap": bool(warmup_video_bootstrap),
            "warmup_action_job": bool(warmup_action_job),
            "num_inference_steps": int(num_inference_steps),
            "jobs": completed_jobs,
        }

    def _sample_streaming_batch_pair(self, sample) -> dict[str, Any]:
        video = sample["video"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        action = sample["action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context = sample["context"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = sample["context_mask"].to(device=self.device, dtype=torch.bool, non_blocking=True)
        batch_size, _, num_frames, _, _ = video.shape
        if num_frames <= 1:
            raise ValueError("Streaming action fine-tuning requires at least 2 video frames.")
        if action.shape[1] % (num_frames - 1) != 0:
            raise ValueError(
                f"`action` temporal dim must be divisible by num_transitions ({num_frames - 1}), got {action.shape[1]}."
            )

        obs_gap_choices = list(self.streaming_train_cfg.get("obs_gap_choices", [1]))
        obs_new_frame_choices = list(
            self.streaming_train_cfg.get(
                "obs_new_frame_choices",
                [1],
            )
        )
        if not obs_gap_choices:
            raise ValueError("`streaming_train.obs_gap_choices` cannot be empty.")
        obs_gap = int(obs_gap_choices[torch.randint(0, len(obs_gap_choices), (1,)).item()])
        valid_new_choices = [int(v) for v in obs_new_frame_choices if 0 < int(v) < (num_frames - 1)]
        if not valid_new_choices:
            raise ValueError(
                "`streaming_train.obs_new_frame_choices` must contain frame indices in [1, num_frames-2]."
            )
        obs_new_idx = valid_new_choices[torch.randint(0, len(valid_new_choices), (1,)).item()]
        obs_old_idx = max(0, obs_new_idx - obs_gap)

        actions_per_transition = action.shape[1] // (num_frames - 1)
        action_start = obs_new_idx * actions_per_transition
        if action_start >= action.shape[1]:
            raise ValueError(
                f"Selected `obs_new_idx={obs_new_idx}` leaves no future action tokens."
            )

        target_action = action[:, action_start:]
        action_is_pad = sample.get("action_is_pad")
        if action_is_pad is not None:
            action_is_pad = action_is_pad[:, action_start:].to(device=self.device, dtype=torch.bool, non_blocking=True)

        proprio_new = None
        if self.proprio_dim is not None and sample.get("proprio") is not None:
            proprio = sample["proprio"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            proprio_new = proprio[:, obs_new_idx, :]
            if proprio_new.shape != (batch_size, self.proprio_dim):
                raise ValueError(
                    f"`proprio_new` shape mismatch, got {tuple(proprio_new.shape)} expected ({batch_size}, {self.proprio_dim})."
                )

        return {
            "obs_old_img": video[:, :, obs_old_idx],
            "obs_new_img": video[:, :, obs_new_idx],
            "target_action": target_action,
            "action_is_pad": action_is_pad,
            "context": context,
            "context_mask": context_mask,
            "proprio_new": proprio_new,
        }

    def _sample_split_point(self, timestep_action: torch.Tensor) -> int:
        frontier_cfg = dict(self.streaming_train_cfg.get("frontier", {}))
        min_layers = int(frontier_cfg.get("min_layers", 0))
        max_layers = int(frontier_cfg.get("max_layers", self.mot.num_layers))
        jitter = int(frontier_cfg.get("random_jitter", 0))
        max_train_t = float(self.train_action_scheduler.num_train_timesteps)
        progress = 1.0 - float(timestep_action.detach().float().mean().item()) / max(max_train_t, 1.0)
        progress = max(0.0, min(1.0, progress))
        split = min_layers + int(round(progress * max(0, max_layers - min_layers)))
        if jitter > 0:
            split = split + int(torch.randint(-jitter, jitter + 1, (1,)).item())
        return max(0, min(self.mot.num_layers, split))

    def training_loss_streaming_action_ft(self, sample, tiled: bool = False):
        del tiled
        pair = self._sample_streaming_batch_pair(sample)
        target_action = pair["target_action"]
        batch_size = target_action.shape[0]

        noise_action = torch.randn_like(target_action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=target_action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(target_action, noise_action, timestep_action)
        target_noise = self.train_action_scheduler.training_target(target_action, noise_action, timestep_action)

        with torch.no_grad():
            cache_old_payload = self.build_streaming_video_cache_from_input_image(
                input_image=pair["obs_old_img"],
                context=pair["context"],
                context_mask=pair["context_mask"],
            )
            cache_new_payload = self.build_streaming_video_cache_from_input_image(
                input_image=pair["obs_new_img"],
                context=pair["context"],
                context_mask=pair["context_mask"],
            )

        split_point = self._sample_split_point(timestep_action)
        stitched_cache = stitch_prefix_cache(
            cache_new=cache_new_payload["video_kv_cache"],
            cache_old=cache_old_payload["video_kv_cache"],
            split_point=split_point,
        )
        attention_mask = self.build_joint_attention_mask(
            video_seq_len=int(cache_new_payload["video_seq_len"]),
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=int(cache_new_payload["video_pre"]["meta"]["tokens_per_frame"]),
            device=noisy_action.device,
        )

        pred_action = self._predict_action_noise_with_cache(
            latents_action=noisy_action,
            timestep_action=timestep_action,
            context=pair["context"],
            context_mask=pair["context_mask"],
            video_kv_cache=stitched_cache,
            attention_mask=attention_mask,
            video_seq_len=int(cache_new_payload["video_seq_len"]),
            proprio=pair["proprio_new"] if self.streaming_proprio_to_action_only else None,
        )

        action_loss_token = F.mse_loss(pred_action.float(), target_noise.float(), reduction="none").mean(dim=2)
        if pair["action_is_pad"] is not None:
            valid = (~pair["action_is_pad"]).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            loss_stream = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            loss_stream = action_loss_token.mean(dim=1)
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            loss_stream.device,
            dtype=loss_stream.dtype,
        )
        loss_stream = (loss_stream * action_weight).mean()

        if bool(self.streaming_train_cfg.get("mix_with_base_loss", False)):
            loss_base, loss_dict_base = self.training_loss_base(sample)
            total_loss = (
                float(self.streaming_train_cfg.get("lambda_streaming_action", 1.0)) * loss_stream
                + float(self.streaming_train_cfg.get("lambda_base", 1.0)) * loss_base
            )
            loss_dict = {
                "loss_streaming_action": float(loss_stream.detach().item()),
                "split_point": float(split_point),
                **loss_dict_base,
            }
            return total_loss, loss_dict

        return loss_stream, {
            "loss_streaming_action": float(loss_stream.detach().item()),
            "split_point": float(split_point),
        }
