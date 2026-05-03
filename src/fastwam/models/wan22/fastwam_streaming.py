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
        self._streaming_infer_timestep_cache: dict[int, torch.Tensor] = {}
        self._action_attn_mask: dict[tuple[int, int, int, str, int | None], torch.Tensor] = {}

    def configure_streaming(self, streaming: Optional[dict[str, Any]] = None) -> "FastWAMStreaming":
        cfg = {} if streaming is None else dict(streaming)
        self.streaming_cfg = cfg
        self.streaming_train_cfg = dict(cfg.get("streaming_train", {}))
        self.freeze_video_expert = bool(cfg.get("freeze_video_expert", True))
        self.mot.use_cache_time_embedding = bool(cfg.get("use_cache_time_embedding", True))
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

    @torch.no_grad()
    def evaluate_streaming_action_mse(self, sample) -> dict[str, float]:
        loss_stream, _ = self._compute_streaming_action_ft_loss(sample)
        return {"val_loss": float(loss_stream.detach().item())}

    def _resolve_streaming_condition_inputs(
        self,
        prompt: Optional[str],
        context: Optional[torch.Tensor],
        context_mask: Optional[torch.Tensor],
        proprio: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )
        return context, context_mask

    @torch.no_grad()
    def _build_cache_version(
        self,
        input_image: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        obs_timestamp_ms: float,
        obs_index: int = -1,
        env_step: int = -1,
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
            env_step=int(env_step),
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
        env_step: int = -1,
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
            env_step=int(env_step),
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
        proprio: Optional[torch.Tensor] = None,
        obs_index: int = -1,
        env_step: int = -1,
        obs_timestamp_ms: float = 0.0,
        tiled: bool = False,
    ) -> VideoCacheVersion:
        resolved_context, resolved_context_mask = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        version = self._build_cache_version(
            input_image=input_image,
            context=resolved_context,
            context_mask=resolved_context_mask,
            obs_index=int(obs_index),
            env_step=int(env_step),
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
        proprio: Optional[torch.Tensor] = None,
        obs_index: int = -1,
        env_step: int = -1,
        obs_timestamp_ms: float = 0.0,
        tiled: bool = False,
    ) -> VideoCacheVersion:
        resolved_context, resolved_context_mask = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
        )
        version = self._build_cache_version(
            input_image=input_image,
            context=resolved_context,
            context_mask=resolved_context_mask,
            obs_index=int(obs_index),
            env_step=int(env_step),
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

    def _sample_action_latent_noise(
        self,
        *,
        batch_size: int,
        action_steps: int,
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        return torch.randn(
            (batch_size, action_steps, self.action_expert.action_dim),
            generator=generator,
            device=self.device,
            dtype=self.torch_dtype,
        )

    def _shift_persistent_action_window(
        self,
        job: StreamingActionJob,
        *,
        shift_steps: int,
    ) -> None:
        shift_steps = int(shift_steps)
        if shift_steps <= 0:
            return
        batch_size, action_horizon, _ = job.latents_action.shape
        shift_steps = min(shift_steps, int(action_horizon))
        keep = max(0, int(action_horizon) - shift_steps)

        new_latents = self._sample_action_latent_noise(
            batch_size=batch_size,
            action_steps=action_horizon,
            generator=job.generator,
        )
        new_counts = torch.zeros(
            (batch_size, action_horizon),
            device=self.device,
            dtype=torch.int64,
        )
        new_env_steps = torch.zeros(
            (batch_size, action_horizon),
            device=self.device,
            dtype=torch.int64,
        )

        if keep > 0:
            new_latents[:, :keep] = job.latents_action[:, shift_steps:]
            new_counts[:, :keep] = job.token_denoise_counts[:, shift_steps:]
            new_env_steps[:, :keep] = job.token_env_steps[:, shift_steps:]

        base_env_step = int(job.window_start_env_step) + shift_steps
        append_steps = torch.arange(
            base_env_step + keep,
            base_env_step + action_horizon,
            device=self.device,
            dtype=torch.int64,
        )
        if append_steps.numel() > 0:
            new_env_steps[:, keep:] = append_steps.unsqueeze(0).expand(batch_size, -1)

        job.latents_action = new_latents
        job.token_denoise_counts = new_counts
        job.token_env_steps = new_env_steps
        job.just_released_mask = torch.zeros(
            (batch_size, action_horizon),
            device=self.device,
            dtype=torch.bool,
        )
        job.window_start_env_step = base_env_step
        job.applied_shift_steps = int(job.window_start_env_step) - int(job.trigger_env_step)

    def _latest_snapshot_env_step(self, snapshot: CacheSnapshot) -> int:
        if len(snapshot.layer_env_steps) == 0:
            return int(snapshot.env_step)
        return max(int(v) for v in snapshot.layer_env_steps)

    def start_action_job(
        self,
        *,
        action_horizon: int,
        prompt: Optional[str] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        trigger_obs_index: int = -1,
        trigger_env_step: int = -1,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        persistent: bool = False,
    ) -> StreamingActionJob:
        resolved_context, resolved_context_mask = self._resolve_streaming_condition_inputs(
            prompt=prompt,
            context=context,
            context_mask=context_mask,
        )
        generator_device = self.device
        if str(rand_device) == "cpu":
            generator_device = torch.device("cpu")
        generator = None if seed is None else torch.Generator(device=generator_device).manual_seed(seed)
        latents_action = self._sample_action_latent_noise(
            batch_size=int(resolved_context.shape[0]),
            action_steps=int(action_horizon),
            generator=generator,
        )
        timesteps, deltas = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        token_env_steps = torch.arange(
            int(trigger_env_step),
            int(trigger_env_step) + int(action_horizon),
            device=self.device,
            dtype=torch.int64,
        ).unsqueeze(0).expand(int(resolved_context.shape[0]), -1).clone()
        return StreamingActionJob(
            timesteps=timesteps,
            deltas=deltas,
            latents_action=latents_action,
            trigger_obs_index=int(trigger_obs_index),
            trigger_env_step=int(trigger_env_step),
            window_start_env_step=int(trigger_env_step),
            action_is_pad=None,
            token_env_steps=token_env_steps,
            token_denoise_counts=torch.zeros(
                (resolved_context.shape[0], action_horizon),
                device=self.device,
                dtype=torch.int64,
            ),
            just_released_mask=torch.zeros(
                (resolved_context.shape[0], action_horizon),
                device=self.device,
                dtype=torch.bool,
            ),
            persistent=bool(persistent),
            generator=generator,
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

        if job.token_denoise_counts is None:
            raise ValueError("`job.token_denoise_counts` must be initialized.")
        max_denoise_steps = int(job.timesteps.shape[0])
        if job.persistent and int(job.window_start_env_step) >= 0:
            latest_env_step = self._latest_snapshot_env_step(snapshot)
            shift_steps = max(0, int(latest_env_step) - int(job.window_start_env_step))
            if shift_steps > 0:
                self._shift_persistent_action_window(job, shift_steps=shift_steps)

        active_mask = job.token_denoise_counts < max_denoise_steps
        timestep_indices = job.token_denoise_counts.clamp(max=max_denoise_steps - 1)
        step_t = job.timesteps[timestep_indices].to(
            device=self.device,
            dtype=job.latents_action.dtype,
        )
        step_delta = job.deltas[timestep_indices].to(
            device=self.device,
            dtype=job.latents_action.dtype,
        )
        step_delta = torch.where(
            active_mask,
            step_delta,
            torch.zeros_like(step_delta),
        )
        attention_mask = self._get_action_attn_mask(
            video_seq_len=int(snapshot.video_seq_len),
            action_seq_len=int(job.latents_action.shape[1]),
            video_tokens_per_frame=int(snapshot.tokens_per_frame),
            device=job.latents_action.device,
            action_is_pad=job.action_is_pad,
        )
        pred_action = self._predict_action_noise_with_cache(
            latents_action=job.latents_action,
            timestep_action=step_t,
            context=snapshot.context,
            context_mask=snapshot.context_mask,
            video_kv_cache=[
                {
                    "k": layer["k"],
                    "v": layer["v"],
                    "source_delta": 0,
                }
                for layer_idx, layer in enumerate(snapshot.cache_layers)
            ],
            attention_mask=attention_mask,
            video_seq_len=snapshot.video_seq_len,
        )
        pred_action = pred_action.masked_fill((~active_mask).unsqueeze(-1), 0.0)
        job.latents_action = self.infer_action_scheduler.step(pred_action, step_delta, job.latents_action)
        prev_counts = job.token_denoise_counts
        next_counts = torch.where(active_mask, prev_counts + 1, prev_counts)
        next_counts = next_counts.clamp(max=max_denoise_steps)
        job.just_released_mask = (prev_counts < max_denoise_steps) & (next_counts >= max_denoise_steps)
        job.token_denoise_counts = next_counts
        if not job.persistent:
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
        action_is_pad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if action_is_pad is not None:
            return self.build_joint_attention_mask(
                video_seq_len=int(video_seq_len),
                action_seq_len=int(action_seq_len),
                video_tokens_per_frame=int(video_tokens_per_frame),
                device=device,
                action_is_pad=action_is_pad,
            )
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
        obs_timestamp_ms: float = 0.0,
    ) -> dict[str, Any]:
        self.eval()
        self.submit_observation(
            input_image=input_image,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            proprio=proprio,
            obs_timestamp_ms=obs_timestamp_ms,
            tiled=tiled,
        )
        job = self.start_action_job(
            action_horizon=action_horizon,
            prompt=prompt,
            context=context,
            context_mask=context_mask,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
        )
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
        )
