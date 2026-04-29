from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .streaming_cache import stitch_prefix_cache


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
        batch_size = target_action.shape[0]
        noise_action = torch.randn_like(target_action)
        timestep_action = self._sample_batchwise_training_t(batch_size=batch_size, dtype=target_action.dtype)
        noisy_action = self.train_action_scheduler.add_noise(target_action, noise_action, timestep_action)
        target_noise = self.train_action_scheduler.training_target(target_action, noise_action, timestep_action)
        return timestep_action, noisy_action, target_noise

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
        proprio: Optional[torch.Tensor],
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
            proprio=proprio,
        )

    def _loss_stream_full_chunk(
        self,
        *,
        pred_action: torch.Tensor,
        target_noise: torch.Tensor,
        timestep_action: torch.Tensor,
        action_is_pad: Optional[torch.Tensor],
        position_weight: Optional[torch.Tensor] = None,
        return_per_sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        action_loss_token = F.mse_loss(pred_action.float(), target_noise.float(), reduction="none").mean(dim=2)
        token_weight = torch.ones_like(action_loss_token)
        if position_weight is not None:
            if position_weight.shape != action_loss_token.shape:
                raise ValueError(
                    f"`position_weight` shape mismatch: expected {tuple(action_loss_token.shape)}, "
                    f"got {tuple(position_weight.shape)}."
                )
            token_weight = position_weight.to(device=action_loss_token.device, dtype=action_loss_token.dtype)
        if action_is_pad is not None:
            valid = (~action_is_pad.to(device=action_loss_token.device, dtype=torch.bool)).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            token_weight = token_weight * valid
        weight_sum = token_weight.sum(dim=1).clamp(min=1.0)
        loss_stream = (action_loss_token * token_weight).sum(dim=1) / weight_sum
        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            loss_stream.device,
            dtype=loss_stream.dtype,
        )
        loss_stream_per_sample = loss_stream * action_weight
        loss_stream = loss_stream_per_sample.mean()
        if return_per_sample:
            return loss_stream, loss_stream_per_sample
        return loss_stream

    @staticmethod
    def _metric_mode_name(mode: str) -> str:
        mode = str(mode).strip()
        if not mode:
            mode = "unknown"
        return re.sub(r"[^0-9A-Za-z_]+", "_", mode)

    def _extract_streaming_episode_batch(self, sample) -> Optional[dict[str, torch.Tensor]]:
        required_keys = {
            "obs_prev",
            "obs_cur",
            "obs_next",
            "obs_next2",
            "target_action",
            "proprio_t",
            "context",
            "context_mask",
        }
        if not required_keys.issubset(sample.keys()):
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
        return {
            "obs_prev": _to_image(sample["obs_prev"]),
            "obs_cur": _to_image(sample["obs_cur"]),
            "obs_next": _to_image(sample["obs_next"]),
            "obs_next2": _to_image(sample["obs_next2"]),
            "target_action": sample["target_action"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "action_is_pad": sample.get("action_is_pad", None).to(device=self.device, dtype=torch.bool, non_blocking=True)
            if sample.get("action_is_pad", None) is not None
            else None,
            "proprio_t": sample["proprio_t"].to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "context": context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True),
            "context_mask": context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True),
        }

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

    def _get_streaming_infer_timesteps(self, infer_num_inference_steps: int) -> torch.Tensor:
        infer_num_inference_steps = int(infer_num_inference_steps)
        if infer_num_inference_steps not in self._streaming_infer_timestep_cache:
            timesteps, _ = self.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=infer_num_inference_steps,
                device=self.device,
                dtype=torch.float32,
            )
            self._streaming_infer_timestep_cache[infer_num_inference_steps] = timesteps.detach().float()
        return self._streaming_infer_timestep_cache[infer_num_inference_steps]

    def _map_training_timestep_to_bucket(self, timestep_action: torch.Tensor) -> int:
        infer_num_inference_steps = int(self.streaming_train_cfg.get("infer_num_inference_steps", 10))
        infer_timesteps = self._get_streaming_infer_timesteps(infer_num_inference_steps)
        tau_scalar = float(timestep_action.detach().float().mean().item())
        bucket = int(torch.argmin(torch.abs(infer_timesteps - tau_scalar)).item())
        return bucket

    def _sample_batchwise_training_t(self, batch_size: int, dtype: torch.dtype) -> torch.Tensor:
        sampled = self.train_action_scheduler.sample_training_t(
            batch_size=1,
            device=self.device,
            dtype=dtype,
        )
        return sampled.expand(int(batch_size))


    def _get_streaming_distribution_entries(self, bucket: int) -> list[dict[str, Any]]:
        cfg_dist = self.streaming_train_cfg.get("distribution", None)
        entries = cfg_dist.get(bucket)
        if entries is None:
            entries = cfg_dist.get(str(bucket))
        return [dict(entry) for entry in entries]

    def _load_real_schedule_pool(self) -> Optional[dict[int, list[dict[str, Any]]]]:
        schedule_path = self.streaming_train_cfg.get("schedule_path", None)
        if schedule_path is None:
            return None
        resolved_path = str(Path(str(schedule_path)).expanduser().resolve())
        cached_path = getattr(self, "_streaming_schedule_pool_path", None)
        cached_pool = getattr(self, "_streaming_schedule_pool", None)
        if cached_pool is not None and cached_path == resolved_path:
            return cached_pool

        payload = torch.load(resolved_path, map_location="cpu")
        schedules = payload.get("schedules", payload) if isinstance(payload, dict) else payload
        if not isinstance(schedules, list) or len(schedules) == 0:
            raise ValueError(f"Schedule file has no schedule traces: {resolved_path}")

        pool: dict[int, list[dict[str, Any]]] = {}
        for schedule in schedules:
            trigger_obs_index = int(schedule.get("trigger_obs_index", 0))
            steps = sorted(list(schedule.get("steps", [])), key=lambda row: int(row["denoise_step"]))
            for step in steps:
                layer_obs_indices = [int(v) for v in list(step.get("layer_obs_indices", []))]
                if len(layer_obs_indices) != int(self.mot.num_layers):
                    continue
                raw_source_offsets = [int(v) - int(trigger_obs_index) for v in layer_obs_indices]
                layer_cache_keys = [self._cache_key_from_offset(offset) for offset in raw_source_offsets]
                bucket = int(step["denoise_step"])
                pool.setdefault(bucket, []).append(
                    {
                        "mode": str(step.get("mode", "")),
                        "frontier": int(step.get("frontier", self.mot.num_layers)),
                        "layer_cache_keys": layer_cache_keys,
                    }
                )
        if len(pool) == 0:
            raise ValueError(f"Schedule file has no valid per-step entries: {resolved_path}")
        self._streaming_schedule_pool_path = resolved_path
        self._streaming_schedule_pool = pool
        return pool


    def _sample_cache_from_real_schedule(
        self,
        bucket: int,
        *,
        batch_size: int,
    ) -> Optional[tuple[list[str], str, Optional[int]]]:
        pool = self._load_real_schedule_pool()
        if pool is None:
            return None
        entries = pool.get(int(bucket), None)
        if not entries:
            available = sorted(int(key) for key, value in pool.items() if value)
            schedule_path = self.streaming_train_cfg.get("schedule_path", None)
            raise ValueError(
                f"Streaming schedule `{schedule_path}` has no entries for bucket {bucket}. "
                f"Available buckets: {available}. "
                "Please regenerate the schedule file with matching inference-step settings."
            )
        choice = int(torch.randint(len(entries), (1,), device=self.device).item())
        entry = entries[choice]
        row = ",".join(str(v) for v in entry["layer_cache_keys"])
        return [row] * int(batch_size), str(entry.get("mode", "")), int(entry.get("frontier", self.mot.num_layers))

    def _sample_cache_distribution(self, bucket: int) -> tuple[str, Optional[int]]:
        entries = self._get_streaming_distribution_entries(bucket)
        weights = torch.tensor(
            [float(entry.get("prob", 1.0)) for entry in entries],
            device=self.device,
            dtype=torch.float32,
        )
        if float(weights.sum().item()) <= 0.0:
            raise ValueError(f"Distribution weights for bucket {bucket} must sum to > 0.")
        choice = int(torch.multinomial(weights / weights.sum(), 1).item())
        entry = entries[choice]
        mode = self._normalize_streaming_cache_mode(str(entry["mode"]))
        if mode.startswith("full_"):
            return mode, None
        frontier_min = int(entry.get("frontier_min", 0))
        frontier_max = int(entry.get("frontier_max", self.mot.num_layers))
        frontier_min = max(0, min(self.mot.num_layers, frontier_min))
        frontier_max = max(frontier_min, min(self.mot.num_layers, frontier_max))
        frontier = int(torch.randint(frontier_min, frontier_max + 1, (1,), device=self.device).item())
        return mode, frontier

    def _compose_distribution_cache(
        self,
        caches: dict[str, Any],
        mode: str,
        frontier: Optional[int],
    ) -> list[dict[str, torch.Tensor]]:
        normalized_mode = self._normalize_streaming_cache_mode(mode)
        if normalized_mode.startswith("full_"):
            offset = self._streaming_cache_label_to_offset(normalized_mode[len("full_") :])
            return caches[self._cache_key_from_offset(offset)]
        if "_to_" in normalized_mode:
            old_label, new_label = normalized_mode.split("_to_", maxsplit=1)
            old_offset = self._streaming_cache_label_to_offset(old_label)
            new_offset = self._streaming_cache_label_to_offset(new_label)
            return stitch_prefix_cache(
                caches[self._cache_key_from_offset(new_offset)],
                caches[self._cache_key_from_offset(old_offset)],
                int(frontier),
            )
        raise ValueError(f"Unsupported streaming cache mode: {normalized_mode}")

    @staticmethod
    def _distribution_mode_to_id(mode: str) -> float:
        mode = StreamingBackbone._normalize_streaming_cache_mode(mode)
        mapping = {
            "full_prev": 0.0,
            "prev_to_cur": 1.0,
            "full_cur": 2.0,
            "cur_to_next": 3.0,
            "full_next": 4.0,
            "next_to_next2": 5.0,
            "full_next2": 6.0,
        }
        if mode not in mapping:
            raise ValueError(f"Unsupported streaming cache mode: {mode}")
        return mapping[mode]

    @staticmethod
    def _distribution_mode_to_id_safe(mode: str) -> float:
        try:
            return StreamingBackbone._distribution_mode_to_id(mode)
        except Exception:
            return -1.0

    def _compose_replay_layer_cache(
        self,
        caches: dict[str, Any],
        layer_cache_keys: list[str],
    ) -> list[dict[str, torch.Tensor]]:
        rows = [str(row).split(",") for row in layer_cache_keys]
        if any(len(row) != self.mot.num_layers for row in rows):
            raise ValueError(
                f"Each replay layer-cache row must contain {self.mot.num_layers} comma-separated keys."
            )
        merged: list[dict[str, torch.Tensor]] = []
        for layer_idx in range(self.mot.num_layers):
            k_rows = []
            v_rows = []
            source_deltas = []
            for sample_idx, row in enumerate(rows):
                key = row[layer_idx]
                if key not in caches:
                    raise ValueError(f"Replay cache key `{key}` unavailable. Available: {sorted(caches)}")
                layer = caches[key][layer_idx]
                k_rows.append(layer["k"][sample_idx : sample_idx + 1])
                v_rows.append(layer["v"][sample_idx : sample_idx + 1])
                source_deltas.append(self._cache_key_to_source_delta(key))
            merged.append(
                {
                    "k": torch.cat(k_rows, dim=0),
                    "v": torch.cat(v_rows, dim=0),
                    "source_delta": torch.as_tensor(
                        source_deltas,
                        device=k_rows[0].device,
                        dtype=torch.int64,
                    ),
                }
            )
        return merged

    def _layer_cache_keys_to_mean_offset(
        self,
        layer_cache_keys: list[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        offsets = []
        for row in layer_cache_keys:
            row_offsets = [float(self._cache_key_to_source_delta(key.strip())) for key in str(row).split(",")]
            offsets.append(sum(row_offsets) / float(max(len(row_offsets), 1)))
        return torch.as_tensor(offsets, device=device, dtype=dtype)

    @staticmethod
    def _streaming_cache_label_to_offset(label: str) -> int:
        label = str(label).strip()
        if label == "prev":
            return -1
        if label == "cur":
            return 0
        if label == "next":
            return 1
        if label.startswith("prev"):
            suffix = label[len("prev") :]
            if suffix.isdigit():
                return -int(suffix)
        if label.startswith("next"):
            suffix = label[len("next") :]
            if suffix.isdigit():
                return int(suffix)
        raise ValueError(f"Unsupported streaming cache label: {label}")

    @staticmethod
    def _streaming_cache_offset_to_label(offset: int) -> str:
        if int(offset) <= -1:
            return "prev"
        if int(offset) == 0:
            return "cur"
        if int(offset) == 1:
            return "next"
        return "next2"

    @staticmethod
    def _cache_key_from_offset(offset: int) -> str:
        if int(offset) <= -1:
            return "prev"
        if int(offset) == 0:
            return "cur"
        if int(offset) == 1:
            return "next"
        return "next2"

    @staticmethod
    def _required_cache_keys_for_mode(mode: str) -> list[str]:
        normalized_mode = StreamingBackbone._normalize_streaming_cache_mode(mode)
        required = set()
        if normalized_mode.startswith("full_"):
            label = normalized_mode[len("full_") :]
            offset = StreamingBackbone._streaming_cache_label_to_offset(label)
            required.add(StreamingBackbone._cache_key_from_offset(offset))
        elif "_to_" in normalized_mode:
            old_label, new_label = normalized_mode.split("_to_", maxsplit=1)
            old_offset = StreamingBackbone._streaming_cache_label_to_offset(old_label)
            new_offset = StreamingBackbone._streaming_cache_label_to_offset(new_label)
            required.add(StreamingBackbone._cache_key_from_offset(old_offset))
            required.add(StreamingBackbone._cache_key_from_offset(new_offset))
        else:
            raise ValueError(f"Unsupported streaming cache mode: {normalized_mode}")
        return [key for key in ("prev", "cur", "next", "next2") if key in required]

    @staticmethod
    def _normalize_streaming_cache_mode(mode: str) -> str:
        mode = str(mode).strip()
        if mode.startswith("full_"):
            raw_label = mode[len("full_") :]
            raw_offset = StreamingBackbone._streaming_cache_label_to_offset(raw_label)
            normalized_label = StreamingBackbone._streaming_cache_offset_to_label(raw_offset)
            return f"full_{normalized_label}"
        if "_to_" in mode:
            old_label, new_label = mode.split("_to_", maxsplit=1)
            old_offset = StreamingBackbone._streaming_cache_label_to_offset(old_label)
            new_offset = StreamingBackbone._streaming_cache_label_to_offset(new_label)
            old_label_norm = StreamingBackbone._streaming_cache_offset_to_label(old_offset)
            new_label_norm = StreamingBackbone._streaming_cache_offset_to_label(new_offset)
            if old_label_norm == new_label_norm:
                return f"full_{new_label_norm}"
            if StreamingBackbone._streaming_cache_label_to_offset(old_label_norm) >= StreamingBackbone._streaming_cache_label_to_offset(new_label_norm):
                return f"full_{new_label_norm}"
            return f"{old_label_norm}_to_{new_label_norm}"
        raise ValueError(f"Unsupported streaming cache mode: {mode}")

    def _mode_to_mean_offset(self, mode: str) -> float:
        normalized_mode = self._normalize_streaming_cache_mode(mode)
        if normalized_mode.startswith("full_"):
            return float(self._streaming_cache_label_to_offset(normalized_mode[len("full_") :]))
        old_label, new_label = normalized_mode.split("_to_", maxsplit=1)
        old_offset = float(self._streaming_cache_label_to_offset(old_label))
        new_offset = float(self._streaming_cache_label_to_offset(new_label))
        return 0.5 * (old_offset + new_offset)

    def _offset_to_chunk_anchor(
        self,
        offset: torch.Tensor,
        *,
        action_horizon: int,
    ) -> torch.Tensor:
        if offset.ndim != 1:
            raise ValueError(f"`offset` must be [B], got shape {tuple(offset.shape)}.")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")
        anchor = offset * float(self._get_obs_stride())
        return anchor.clamp(min=0.0, max=float(max(action_horizon - 1, 0)))

    def _build_position_decay_weight(
        self,
        *,
        anchor: torch.Tensor,
        action_horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if anchor.ndim != 1:
            raise ValueError(f"`anchor` must be [B], got shape {tuple(anchor.shape)}.")
        tau = float(self.streaming_train_cfg.get("position_decay_tau", 12.0))
        power = float(self.streaming_train_cfg.get("position_decay_power", 2.0))
        if tau <= 0.0:
            raise ValueError(f"`position_decay_tau` must be positive, got {tau}.")
        if power <= 0.0:
            raise ValueError(f"`position_decay_power` must be positive, got {power}.")
        positions = torch.arange(action_horizon, device=device, dtype=dtype).unsqueeze(0)
        delta = positions - anchor.to(device=device, dtype=dtype).unsqueeze(1)
        decay = torch.exp(-torch.pow(torch.clamp(delta, min=0.0) / tau, power))
        return decay * (delta >= 0.0).to(dtype=dtype)

    def training_loss_streaming_action_ft(self, sample, tiled: bool = False):
        del tiled
        episode_batch = self._extract_streaming_episode_batch(sample)
        if episode_batch is None:
            required_keys = (
                "obs_prev",
                "obs_cur",
                "obs_next",
                "obs_next2",
                "target_action",
                "proprio_t",
                "context",
                "context_mask",
            )
            present_keys = sorted(str(key) for key in sample.keys())
            raise ValueError(
                "Streaming action FT now requires episode-style samples and no longer falls back to pair mode. "
                f"Missing one or more required keys: {required_keys}. "
                f"Present keys: {present_keys}"
            )

        target_action = episode_batch["target_action"]
        timestep_action, noisy_action, target_noise = self._sample_noisy_triplet(target_action)
        bucket = self._map_training_timestep_to_bucket(timestep_action)
        required_cache_keys = ["prev", "cur", "next", "next2"]
        video_ctx = nullcontext() if not self.freeze_video_expert else torch.no_grad()
        with video_ctx:
            selected_caches = self._build_selected_video_cache_payload(
                episode_batch,
                required_cache_keys=required_cache_keys,
            )
        sampled_real_schedule = self._sample_cache_from_real_schedule(
            bucket,
            batch_size=int(target_action.shape[0]),
        )
        if sampled_real_schedule is not None:
            layer_cache_keys, sampled_mode, sampled_frontier = sampled_real_schedule
            stitched_cache = self._compose_replay_layer_cache(
                caches=selected_caches,
                layer_cache_keys=layer_cache_keys,
            )
            mean_offset = self._layer_cache_keys_to_mean_offset(
                layer_cache_keys,
                device=target_action.device,
                dtype=target_action.dtype,
            )
        else:
            sampled_mode, sampled_frontier = self._sample_cache_distribution(bucket)
            stitched_cache = self._compose_distribution_cache(
                caches=selected_caches,
                mode=sampled_mode,
                frontier=sampled_frontier,
            )
            mean_offset = torch.full(
                (int(target_action.shape[0]),),
                fill_value=float(self._mode_to_mean_offset(sampled_mode)),
                device=target_action.device,
                dtype=target_action.dtype,
            )
        position_anchor = self._offset_to_chunk_anchor(
            mean_offset,
            action_horizon=int(target_action.shape[1]),
        )
        position_weight = self._build_position_decay_weight(
            anchor=position_anchor,
            action_horizon=int(target_action.shape[1]),
            device=target_action.device,
            dtype=target_action.dtype,
        )
        pred_action = self._predict_stream_noise(
            noisy_action=noisy_action,
            timestep_action=timestep_action,
            context=episode_batch["context"],
            context_mask=episode_batch["context_mask"],
            video_kv_cache=stitched_cache,
            video_seq_len=int(selected_caches["video_seq_len"]),
            video_tokens_per_frame=int(selected_caches["tokens_per_frame"]),
            proprio=episode_batch["proprio_t"] if self.streaming_proprio_to_action_only else None,
        )
        loss_stream = self._loss_stream_full_chunk(
            pred_action=pred_action,
            target_noise=target_noise,
            timestep_action=timestep_action,
            action_is_pad=episode_batch["action_is_pad"],
            position_weight=position_weight,
        )

        if bool(self.streaming_train_cfg.get("mix_with_base_loss", False)):
            raise ValueError("`mix_with_base_loss=true` is unsupported for episode-based streaming dataset.")

        return loss_stream, {
            "loss_streaming_action": float(loss_stream.detach().item()),
            "bucket": float(bucket),
            "mode_id": self._distribution_mode_to_id_safe(sampled_mode),
            "frontier": -1.0 if sampled_frontier is None else float(sampled_frontier),
            "position_anchor_mean": float(position_anchor.detach().float().mean().item()),
        }
