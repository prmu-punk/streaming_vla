from __future__ import annotations

from typing import Any, Optional

import torch

from fastwam.utils.logging_config import get_logger

from .fastwam import FastWAM
from .streaming_backbone import StreamingBackbone
from .streaming_cache import (
    CacheSnapshot,
    StreamingActionJob,
    StreamingCacheState,
    VideoCacheVersion,
)

logger = get_logger(__name__)


class FastWAMStreaming(StreamingBackbone, FastWAM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming_cache_state = StreamingCacheState(num_layers=self.mot.num_layers)
        self.streaming_version_counter = 0
        self.streaming_cfg: dict[str, Any] = {}
        self.streaming_train_cfg: dict[str, Any] = {"enabled": False}
        self.freeze_video_expert = True
        self.streaming_proprio_to_action_only = True
        self._streaming_infer_timestep_cache: dict[int, torch.Tensor] = {}
        self._action_attn_mask: dict[tuple[int, int, int, str, int | None], torch.Tensor] = {}

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
        self._action_attn_mask.clear()

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
        attention_mask = self._get_action_attn_mask(
            video_seq_len=int(snapshot.video_seq_len),
            action_seq_len=int(job.latents_action.shape[1]),
            video_tokens_per_frame=int(snapshot.tokens_per_frame),
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

    def _get_action_attn_mask(
        self,
        *,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        key = (
            int(video_seq_len),
            int(action_seq_len),
            int(video_tokens_per_frame),
            str(device.type),
            device.index,
        )
        cached = self._action_attn_mask.get(key)
        if cached is not None:
            return cached
        mask = self.build_joint_attention_mask(
            video_seq_len=int(video_seq_len),
            action_seq_len=int(action_seq_len),
            video_tokens_per_frame=int(video_tokens_per_frame),
            device=device,
        )
        self._action_attn_mask[key] = mask
        return mask

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
