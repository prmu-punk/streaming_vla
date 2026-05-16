from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F


class StreamingBackbone:
    def _get_obs_stride(self) -> float:
        cfg_value = self.streaming_cfg.get("obs_stride", None)
        stride = float(cfg_value)
        if stride <= 0.0:
            raise ValueError(f"`obs_stride` must be positive, got {stride}.")
        return stride

    def _sample_noisy_triplet(
        self,
        target_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        noise_action = torch.randn_like(target_action)
        timestep_action = self._sample_tokenwise_training_t(target_action=target_action)
        noisy_action = self.train_action_scheduler.add_noise(target_action, noise_action, timestep_action)
        target_noise = self.train_action_scheduler.training_target(target_action, noise_action, timestep_action)
        return timestep_action, noisy_action, target_noise

    def _sample_tokenwise_training_t(self, target_action: torch.Tensor) -> torch.Tensor:
        batch_size, action_horizon = target_action.shape[:2]
        sampled = self.train_action_scheduler.sample_training_t(
            batch_size=int(batch_size * action_horizon),
            device=self.device,
            dtype=target_action.dtype,
        ).view(batch_size, action_horizon)
        pattern = str(self.streaming_train_cfg.get("token_noise_pattern", "front_low_high")).strip().lower()
        if pattern == "random_all":
            return sampled
        if pattern == "front_low_high":
            return torch.sort(sampled, dim=1, descending=False).values
        if pattern == "position_low_high":
            steps = float(self.train_action_scheduler.num_train_timesteps)
            bias_strength = float(self.streaming_train_cfg.get("token_noise_position_bias_strength", 0.5))
            bias_strength = min(max(bias_strength, 0.0), 1.0)
            margin = float(self.streaming_train_cfg.get("token_noise_position_margin", 0.1))
            margin = min(max(margin, 0.0), 0.49)
            position_ratio = torch.linspace(
                margin,
                1.0 - margin,
                action_horizon,
                device=self.device,
                dtype=target_action.dtype,
            ).unsqueeze(0)
            position_target = position_ratio.expand(batch_size, -1) * steps
            return torch.lerp(sampled, position_target, bias_strength)
        raise ValueError(
            f"Unsupported `streaming_train.token_noise_pattern`: {pattern}. "
            "Expected one of: ['random_all', 'front_low_high', 'position_low_high']."
        )

    def _offset_to_shift_steps_tensor(self, offset: torch.Tensor, *, action_horizon: int) -> torch.Tensor:
        if offset.ndim != 1:
            raise ValueError(f"`offset` must be [B], got shape {tuple(offset.shape)}.")
        stride = float(self._get_obs_stride())
        shift_steps = torch.round(offset * stride).to(dtype=torch.int64)
        return shift_steps.clamp(min=-int(action_horizon), max=int(action_horizon))

    def _shift_action_window_by_steps(
        self,
        action_tensor: torch.Tensor,
        shift_steps: torch.Tensor,
        *,
        pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, action_horizon, action_dim = action_tensor.shape
        shifted = torch.full_like(action_tensor, fill_value=pad_value)
        action_is_pad = torch.ones(
            (batch_size, action_horizon),
            device=action_tensor.device,
            dtype=torch.bool,
        )
        for batch_idx in range(batch_size):
            shift = int(shift_steps[batch_idx].item())
            if shift >= 0:
                src_start = shift
                dst_start = 0
                valid_len = max(0, action_horizon - shift)
            else:
                src_start = 0
                dst_start = -shift
                valid_len = max(0, action_horizon + shift)
            if valid_len <= 0:
                continue
            shifted[batch_idx, dst_start : dst_start + valid_len] = action_tensor[
                batch_idx, src_start : src_start + valid_len
            ]
            action_is_pad[batch_idx, dst_start : dst_start + valid_len] = False
        return shifted, action_is_pad

    def _predict_stream_noise(
        self,
        *,
        noisy_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        video_seq_len: int,
        video_tokens_per_frame: int,
    ) -> torch.Tensor:
        attention_mask = self.build_joint_attention_mask(
            video_seq_len=int(video_seq_len),
            action_seq_len=noisy_action.shape[1],
            video_tokens_per_frame=int(video_tokens_per_frame),
            device=noisy_action.device,
        )
        return self._predict_action_noise_with_cache(
            latents_action=noisy_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=int(video_seq_len),
        )

    def _loss_stream_full_chunk(
        self,
        *,
        pred_action: torch.Tensor,
        target_noise: torch.Tensor,
        timestep_action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        action_loss_token = F.mse_loss(pred_action.float(), target_noise.float(), reduction="none").mean(dim=2)
        token_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            device=action_loss_token.device,
            dtype=action_loss_token.dtype,
        )
        if token_weight.ndim == 1:
            token_weight = token_weight.unsqueeze(1)
        if action_mask is not None:
            action_mask = action_mask.to(device=action_loss_token.device, dtype=torch.bool)
            if tuple(action_mask.shape) != tuple(action_loss_token.shape):
                raise ValueError(
                    f"`action_mask` shape mismatch: expected {tuple(action_loss_token.shape)}, got {tuple(action_mask.shape)}."
                )
            token_weight = token_weight * action_mask.to(dtype=action_loss_token.dtype)
        weight_sum = token_weight.sum(dim=1).clamp(min=1.0)
        loss_stream_per_sample = (action_loss_token * token_weight).sum(dim=1) / weight_sum
        loss_stream = loss_stream_per_sample.mean()
        if return_per_sample:
            return loss_stream, loss_stream_per_sample
        return loss_stream

    def _extract_streaming_episode_batch(self, sample) -> Optional[dict[str, torch.Tensor]]:
        shared_required = {
            "target_action",
            "proprio_t",
            "context",
            "context_mask",
        }
        legacy_required = {"obs_prev", "obs_cur", "obs_next", "obs_next2"}
        has_single_obs = "obs" in sample
        has_legacy_obs = legacy_required.issubset(sample.keys())
        if not shared_required.issubset(sample.keys()) or (not has_single_obs and not has_legacy_obs):
            return None

        def _to_image(x: torch.Tensor) -> torch.Tensor:
            if x.ndim != 4:
                raise ValueError(f"Expected batched image tensor [B,3,H,W], got {tuple(x.shape)}")
            return x.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        context = sample["context"]
        context_mask = sample["context_mask"]
        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}."
            )
        batch = {
            "target_action": sample["target_action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": sample.get("action_is_pad", None).to(device=self.device, dtype=torch.bool, non_blocking=True)
            if sample.get("action_is_pad", None) is not None
            else None,
            "proprio_t": sample["proprio_t"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "context": context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "context_mask": context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        }
        if has_single_obs:
            batch["obs"] = _to_image(sample["obs"])
        else:
            batch["obs_prev"] = _to_image(sample["obs_prev"])
            batch["obs_cur"] = _to_image(sample["obs_cur"])
            batch["obs_next"] = _to_image(sample["obs_next"])
            batch["obs_next2"] = _to_image(sample["obs_next2"])
        return batch

    @staticmethod
    def _slice_cache_batch(
        cache_layers: list[dict[str, torch.Tensor]],
        start: int,
        end: int,
    ) -> list[dict[str, torch.Tensor]]:
        sliced: list[dict[str, torch.Tensor]] = []
        for layer in cache_layers:
            source_delta = layer.get("source_delta", 0)
            if torch.is_tensor(source_delta):
                source_delta = source_delta[start:end]
            else:
                source_delta = int(source_delta)
            sliced.append(
                {
                    "k": layer["k"][start:end],
                    "v": layer["v"][start:end],
                    "source_delta": source_delta,
                }
            )
        return sliced

    @staticmethod
    def _cache_key_to_source_delta(cache_key: str) -> int:
        mapping = {
            "prev": -1,
            "cur": 0,
            "next": 1,
            "next2": 2,
        }
        if str(cache_key) not in mapping:
            raise ValueError(f"Unsupported cache key for source delta: {cache_key}")
        return int(mapping[str(cache_key)])

    def _build_selected_video_cache_payload(
        self,
        episode_batch: dict[str, torch.Tensor],
        required_cache_keys: list[str],
    ) -> dict[str, Any]:
        if len(required_cache_keys) == 0:
            raise ValueError("`required_cache_keys` cannot be empty.")
        valid_keys = ("prev", "cur", "next", "next2")
        unique_required = sorted(set(str(k) for k in required_cache_keys))
        for key in unique_required:
            if key not in valid_keys:
                raise ValueError(f"Unsupported cache key: {key}")
        ordered_keys = [key for key in valid_keys if key in unique_required]
        batch_size = int(episode_batch["obs_cur"].shape[0])
        key_to_obs = {
            "prev": episode_batch["obs_prev"],
            "cur": episode_batch["obs_cur"],
            "next": episode_batch["obs_next"],
            "next2": episode_batch["obs_next2"],
        }
        obs_all = torch.cat([key_to_obs[key] for key in ordered_keys], dim=0)
        context_all = torch.cat([episode_batch["context"]] * len(ordered_keys), dim=0)
        context_mask_all = torch.cat([episode_batch["context_mask"]] * len(ordered_keys), dim=0)
        payload = self.build_streaming_video_cache_from_input_image(
            input_image=obs_all,
            context=context_all,
            context_mask=context_mask_all,
        )
        full_cache = payload["video_kv_cache"]
        output: dict[str, Any] = {
            "video_seq_len": int(payload["video_seq_len"]),
            "tokens_per_frame": int(payload["video_pre"]["meta"]["tokens_per_frame"]),
        }
        for idx, key in enumerate(ordered_keys):
            start = idx * batch_size
            end = (idx + 1) * batch_size
            sliced_cache = self._slice_cache_batch(full_cache, start, end)
            source_delta = self._cache_key_to_source_delta(key)
            for layer in sliced_cache:
                layer["source_delta"] = int(source_delta)
            output[key] = sliced_cache
        return output

    @staticmethod
    def _normalize_full_cache_mode(mode: str) -> str:
        mode = str(mode).strip().lower()
        valid_modes = {"full_prev", "full_cur", "full_next", "full_next2"}
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported `streaming_train.cache_mode`: {mode}. "
                f"Expected one of: {sorted(valid_modes)}."
            )
        return mode

    def _resolve_streaming_train_cache_mode(self) -> str:
        return self._normalize_full_cache_mode(self.streaming_train_cfg.get("cache_mode", "full_cur"))

    @staticmethod
    def _cache_key_from_full_mode(mode: str) -> str:
        normalized_mode = StreamingBackbone._normalize_full_cache_mode(mode)
        return normalized_mode[len("full_") :]

    def _compute_streaming_action_ft_loss(
        self,
        sample,
        *,
        tiled: bool = False,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del tiled
        episode_batch = self._extract_streaming_episode_batch(sample)
        if episode_batch is None:
            required_keys = (
                "obs + target_action + proprio_t + context + context_mask",
                "or obs_prev/obs_cur/obs_next/obs_next2 + target_action + proprio_t + context + context_mask",
            )
            present_keys = sorted(str(key) for key in sample.keys())
            raise ValueError(
                "Streaming action FT now requires episode-style samples and no longer falls back to pair mode. "
                f"Missing one or more required keys: {required_keys}. "
                f"Present keys: {present_keys}"
            )

        target_action = episode_batch["target_action"]
        resolved_context, resolved_context_mask = self._resolve_streaming_condition_inputs(
            prompt=None,
            context=episode_batch["context"],
            context_mask=episode_batch["context_mask"],
            proprio=episode_batch["proprio_t"],
        )
        episode_batch = dict(episode_batch)
        episode_batch["context"] = resolved_context
        episode_batch["context_mask"] = resolved_context_mask

        timestep_action, noisy_action, target_noise = self._sample_noisy_triplet(target_action)
        cache_mode = self._resolve_streaming_train_cache_mode()
        cache_key = self._cache_key_from_full_mode(cache_mode)
        video_ctx = nullcontext() if not self.freeze_video_expert else torch.no_grad()
        with video_ctx:
            if "obs" in episode_batch:
                if cache_key != "cur":
                    raise ValueError(
                        "Single-observation streaming training only supports "
                        "`streaming_train.cache_mode=full_cur`."
                    )
                selected_caches = self.build_streaming_video_cache_from_input_image(
                    input_image=episode_batch["obs"],
                    context=episode_batch["context"],
                    context_mask=episode_batch["context_mask"],
                )
                selected_cache = selected_caches["video_kv_cache"]
                video_seq_len = int(selected_caches["video_seq_len"])
                tokens_per_frame = int(selected_caches["video_pre"]["meta"]["tokens_per_frame"])
            else:
                selected_caches = self._build_selected_video_cache_payload(
                    episode_batch,
                    required_cache_keys=[cache_key],
                )
                selected_cache = selected_caches[cache_key]
                video_seq_len = int(selected_caches["video_seq_len"])
                tokens_per_frame = int(selected_caches["tokens_per_frame"])
        pred_action = self._predict_stream_noise(
            noisy_action=noisy_action,
            timestep_action=timestep_action,
            context=episode_batch["context"],
            context_mask=episode_batch["context_mask"],
            video_kv_cache=selected_cache,
            video_seq_len=video_seq_len,
            video_tokens_per_frame=tokens_per_frame,
        )
        loss_stream = self._loss_stream_full_chunk(
            pred_action=pred_action,
            target_noise=target_noise,
            timestep_action=timestep_action,
        )
        metrics = {
            "loss_streaming_action": float(loss_stream.detach().item()),
            "cache_mode_id": float(("prev", "cur", "next", "next2").index(cache_key)),
            "token_t_min": float(timestep_action.detach().float().min().item()),
            "token_t_max": float(timestep_action.detach().float().max().item()),
            "token_t_mean": float(timestep_action.detach().float().mean().item()),
        }
        return loss_stream, metrics

    def training_loss_streaming_action_ft(self, sample, tiled: bool = False):
        return self._compute_streaming_action_ft_loss(sample, tiled=tiled)
