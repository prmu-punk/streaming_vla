from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
import torch.nn.functional as F

from .streaming_cache import stitch_prefix_cache


class StreamingBackbone:
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
    ) -> torch.Tensor:
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
        loss_stream = (loss_stream * action_weight).mean()
        return loss_stream

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
            sliced.append(
                {
                    "k": layer["k"][start:end],
                    "v": layer["v"][start:end],
                }
            )
        return sliced

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
            output[key] = self._slice_cache_batch(full_cache, start, end)
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
        episode_batch = self._extract_streaming_episode_batch(sample)
        if episode_batch is not None:
            target_action = episode_batch["target_action"]
            timestep_action, noisy_action, target_noise = self._sample_noisy_triplet(target_action)

            bucket = self._map_training_timestep_to_bucket(timestep_action)
            mode, frontier = self._sample_cache_distribution(bucket)
            required_cache_keys = self._required_cache_keys_for_mode(mode)
            video_ctx = nullcontext() if not self.freeze_video_expert else torch.no_grad()
            with video_ctx:
                selected_caches = self._build_selected_video_cache_payload(
                    episode_batch,
                    required_cache_keys=required_cache_keys,
                )
            stitched_cache = self._compose_distribution_cache(
                caches=selected_caches,
                mode=mode,
                frontier=frontier,
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
            )
            if bool(self.streaming_train_cfg.get("mix_with_base_loss", False)):
                raise ValueError("`mix_with_base_loss=true` is unsupported for episode-based streaming dataset.")
            return loss_stream, {
                "loss_streaming_action": float(loss_stream.detach().item()),
                "bucket": float(bucket),
                "mode_id": self._distribution_mode_to_id(mode),
                "frontier": -1.0 if frontier is None else float(frontier),
            }

        del tiled
        pair = self._sample_streaming_batch_pair(sample)
        target_action = pair["target_action"]
        timestep_action, noisy_action, target_noise = self._sample_noisy_triplet(target_action)

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
        pred_action = self._predict_stream_noise(
            noisy_action=noisy_action,
            timestep_action=timestep_action,
            context=pair["context"],
            context_mask=pair["context_mask"],
            video_kv_cache=stitched_cache,
            video_seq_len=int(cache_new_payload["video_seq_len"]),
            video_tokens_per_frame=int(cache_new_payload["video_pre"]["meta"]["tokens_per_frame"]),
            proprio=pair["proprio_new"] if self.streaming_proprio_to_action_only else None,
        )
        loss_stream = self._loss_stream_full_chunk(
            pred_action=pred_action,
            target_noise=target_noise,
            timestep_action=timestep_action,
            action_is_pad=pair["action_is_pad"],
        )

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
