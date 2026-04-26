from __future__ import annotations

from contextlib import nullcontext
import re
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .streaming_cache import stitch_prefix_cache


class StreamingBackbone:
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
        return_per_sample: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        action_loss_token = F.mse_loss(pred_action.float(), target_noise.float(), reduction="none").mean(dim=2)
        if action_is_pad is not None:
            valid = (~action_is_pad.to(device=action_loss_token.device, dtype=torch.bool)).to(
                device=action_loss_token.device,
                dtype=action_loss_token.dtype,
            )
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            loss_stream = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            loss_stream = action_loss_token.mean(dim=1)
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

    def _configured_replay_modes(self) -> list[str]:
        cfg_dist = self.streaming_train_cfg.get("distribution", None)
        modes: set[str] = set()
        if cfg_dist is not None and hasattr(cfg_dist, "values"):
            for entries in cfg_dist.values():
                if entries is None:
                    continue
                for entry in entries:
                    if hasattr(entry, "get") and entry.get("mode", None) is not None:
                        modes.add(str(entry.get("mode")))
        modes.add("other")
        return sorted(modes)

    def _replay_mode_loss_metrics(
        self,
        *,
        loss_per_sample: torch.Tensor,
        replay_modes: Any,
    ) -> dict[str, float]:
        if isinstance(replay_modes, str):
            replay_modes = [replay_modes]
        else:
            replay_modes = [str(mode) for mode in list(replay_modes)]
        if len(replay_modes) != int(loss_per_sample.shape[0]):
            raise ValueError(
                "`replay_mode` batch size mismatch: "
                f"got {len(replay_modes)} modes for loss shape {tuple(loss_per_sample.shape)}."
            )

        configured_modes = self._configured_replay_modes()
        configured_set = set(configured_modes)
        mode_to_indices: dict[str, list[int]] = {mode: [] for mode in configured_modes}
        for idx, mode in enumerate(replay_modes):
            mode_to_indices[mode if mode in configured_set else "other"].append(idx)

        metrics: dict[str, float] = {}
        detached_loss = loss_per_sample.detach().float()
        for mode in configured_modes:
            metric_mode = self._metric_mode_name(mode)
            indices = mode_to_indices[mode]
            if indices:
                index_tensor = torch.as_tensor(indices, device=detached_loss.device, dtype=torch.long)
                mode_loss_sum = detached_loss.index_select(0, index_tensor).sum()
                mode_count = float(len(indices))
            else:
                mode_loss_sum = torch.zeros((), device=detached_loss.device, dtype=detached_loss.dtype)
                mode_count = 0.0
            metrics[f"replay_mode/{metric_mode}_loss_sum"] = float(mode_loss_sum.item())
            metrics[f"replay_mode/{metric_mode}_count"] = mode_count
        return metrics

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
            "replay_x_t": sample.get("replay_x_t", None).to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if sample.get("replay_x_t", None) is not None
            else None,
            "replay_timestep": sample.get("replay_timestep", None).to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            if sample.get("replay_timestep", None) is not None
            else None,
            "replay_layer_cache_keys": sample.get("replay_layer_cache_keys", None),
            "replay_denoise_step": sample.get("replay_denoise_step", None).to(device=self.device, dtype=torch.float32, non_blocking=True)
            if sample.get("replay_denoise_step", None) is not None
            else None,
            "replay_mode": sample.get("replay_mode", None),
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

    def _default_streaming_distribution(self) -> dict[int, list[dict[str, Any]]]:
        return {
            0: [{"mode": "full_prev", "prob": 1.0}],
            1: [{"mode": "prev_to_cur", "prob": 1.0, "frontier_min": 8, "frontier_max": 26}],
            2: [
                {"mode": "full_cur", "prob": 0.7},
                {"mode": "prev_to_cur", "prob": 0.3, "frontier_min": 20, "frontier_max": 30},
            ],
            3: [{"mode": "full_cur", "prob": 1.0}],
            4: [{"mode": "cur_to_next", "prob": 1.0, "frontier_min": 8, "frontier_max": 24}],
            5: [
                {"mode": "full_next", "prob": 0.7},
                {"mode": "cur_to_next", "prob": 0.3, "frontier_min": 20, "frontier_max": 30},
            ],
            6: [{"mode": "full_next", "prob": 1.0}],
            7: [
                {"mode": "full_next", "prob": 0.7},
                {"mode": "next_to_next2", "prob": 0.3, "frontier_min": 8, "frontier_max": 24},
            ],
            8: [{"mode": "next_to_next2", "prob": 1.0, "frontier_min": 20, "frontier_max": 30}],
            9: [{"mode": "full_next2", "prob": 1.0}],
        }

    def _get_streaming_distribution_entries(self, bucket: int) -> list[dict[str, Any]]:
        cfg_dist = self.streaming_train_cfg.get("distribution", None)
        if cfg_dist is None:
            cfg_dist = self._default_streaming_distribution()
        entries = cfg_dist.get(bucket)
        if entries is None:
            entries = cfg_dist.get(str(bucket))
        if entries is None:
            entries = [{"mode": "full_cur", "prob": 1.0}]
        return [dict(entry) for entry in entries]

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

    def _summarize_replay_layer_cache_keys(self, layer_cache_keys: list[str]) -> dict[str, float]:
        counts = {"prev": 0, "cur": 0, "next": 0, "next2": 0}
        total = 0
        offset_sum = 0.0
        for row in layer_cache_keys:
            for key in str(row).split(","):
                key = key.strip()
                counts[key] = counts.get(key, 0) + 1
                total += 1
                offset_sum += float(self._cache_key_to_source_delta(key))
        denom = float(max(total, 1))
        return {
            "replay_source_offset": offset_sum / denom,
            "replay_prev_ratio": float(counts.get("prev", 0)) / denom,
            "replay_cur_ratio": float(counts.get("cur", 0)) / denom,
            "replay_next_ratio": float(counts.get("next", 0)) / denom,
            "replay_next2_ratio": float(counts.get("next2", 0)) / denom,
        }

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
        has_replay_xt = (
            episode_batch.get("replay_x_t", None) is not None
            and episode_batch.get("replay_timestep", None) is not None
            and episode_batch.get("replay_layer_cache_keys", None) is not None
        )
        if not has_replay_xt:
            present_keys = sorted(str(key) for key in sample.keys())
            raise ValueError(
                "Streaming action FT now only supports replay x_t training. "
                "Expected `replay_x_t`, `replay_timestep`, and `replay_layer_cache_keys`. "
                f"Present keys: {present_keys}"
            )

        timestep_action = episode_batch["replay_timestep"]
        if timestep_action.ndim == 0:
            timestep_action = timestep_action.expand(target_action.shape[0])
        noisy_action = episode_batch["replay_x_t"]
        sigma = (timestep_action / float(self.train_action_scheduler.num_train_timesteps)).to(
            device=target_action.device,
            dtype=target_action.dtype,
        )
        sigma = sigma.clamp(min=float(self.train_action_scheduler.eps))
        target_noise = (noisy_action - target_action) / sigma.view(-1, *([1] * (target_action.ndim - 1)))
        jitter = float(self.streaming_train_cfg.get("xt_timestep_jitter", 0.0))
        if jitter > 0.0:
            sigma_delta = torch.empty_like(sigma).uniform_(-jitter, jitter) / float(
                int(self.streaming_train_cfg.get("infer_num_inference_steps", 10))
            )
            sigma = (sigma + sigma_delta).clamp(
                min=float(self.train_action_scheduler.eps),
                max=1.0,
            )
            timestep_action = sigma * float(self.train_action_scheduler.num_train_timesteps)
            noisy_action = target_action + sigma.view(-1, *([1] * (target_action.ndim - 1))) * target_noise
        required_cache_keys = ["prev", "cur", "next", "next2"]
        video_ctx = nullcontext() if not self.freeze_video_expert else torch.no_grad()
        with video_ctx:
            selected_caches = self._build_selected_video_cache_payload(
                episode_batch,
                required_cache_keys=required_cache_keys,
            )
        layer_cache_keys = episode_batch["replay_layer_cache_keys"]
        if isinstance(layer_cache_keys, str):
            layer_cache_keys = [layer_cache_keys]
        layer_cache_keys = [str(v) for v in layer_cache_keys]
        stitched_cache = self._compose_replay_layer_cache(
            caches=selected_caches,
            layer_cache_keys=layer_cache_keys,
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
        loss_stream, loss_per_sample = self._loss_stream_full_chunk(
            pred_action=pred_action,
            target_noise=target_noise,
            timestep_action=timestep_action,
            action_is_pad=episode_batch["action_is_pad"],
            return_per_sample=True,
        )

        if bool(self.streaming_train_cfg.get("mix_with_base_loss", False)):
            raise ValueError("`mix_with_base_loss=true` is unsupported for episode-based streaming dataset.")

        metrics = {}
        replay_denoise_step = episode_batch.get("replay_denoise_step", None)
        if replay_denoise_step is not None:
            replay_denoise_step_f = replay_denoise_step.detach().float()
            metrics["replay_denoise_step_std"] = float(replay_denoise_step_f.std(unbiased=False).item())
        replay_modes = episode_batch.get("replay_mode", None)
        if replay_modes is not None:
            metrics.update(
                self._replay_mode_loss_metrics(
                    loss_per_sample=loss_per_sample,
                    replay_modes=replay_modes,
                )
            )
        return loss_stream, metrics
