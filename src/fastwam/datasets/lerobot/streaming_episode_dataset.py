from __future__ import annotations

from collections import OrderedDict
from typing import Any, Optional

import torch

from .robot_video_dataset import DEFAULT_PROMPT, RobotVideoDataset


_TRAJECTORY_REPLAY_REGISTRY: dict[str, list[dict[str, Any]]] = {}


def register_trajectory_replay_records(key: str, records: list[dict[str, Any]]) -> None:
    _TRAJECTORY_REPLAY_REGISTRY[str(key)] = records


class StreamingRobotEpisodeDataset(RobotVideoDataset):
    def __init__(
        self,
        *args,
        effective_obs_stride: int = 3,
        history_obs: int = 1,
        future_obs: int = 2,
        trigger_every_n_obs: int = 3,
        keep_trigger_phase: bool = True,
        action_horizon: int = 32,
        episode_cache_size: int = 8,
        trajectory_replay_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if effective_obs_stride <= 0:
            raise ValueError(f"`effective_obs_stride` must be positive, got {effective_obs_stride}.")
        if history_obs < 0:
            raise ValueError(f"`history_obs` must be non-negative, got {history_obs}.")
        if future_obs < 0:
            raise ValueError(f"`future_obs` must be non-negative, got {future_obs}.")
        if trigger_every_n_obs <= 0:
            raise ValueError(f"`trigger_every_n_obs` must be positive, got {trigger_every_n_obs}.")
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}.")

        self.effective_obs_stride = int(effective_obs_stride)
        self.history_obs = int(history_obs)
        self.future_obs = int(future_obs)
        self.trigger_every_n_obs = int(trigger_every_n_obs)
        self.keep_trigger_phase = bool(keep_trigger_phase)
        self.action_horizon = int(action_horizon)
        self.episode_cache_size = max(int(episode_cache_size), 0)
        self.trajectory_replay_key = None if trajectory_replay_key is None else str(trajectory_replay_key)
        self._episode_cache: OrderedDict[int, dict[str, Any]] = OrderedDict()

        # We query exact frames ourselves from the underlying LeRobot datasets.
        self.lerobot_dataset._set_return_images(False)
        self.sample_index = self._build_sample_index()
        if not self.sample_index:
            raise ValueError("No valid streaming episode samples were constructed. Check stride/horizon settings.")
        self.trajectory_replay_records = self._load_trajectory_replay_records()

    def _load_trajectory_replay_records(self) -> list[dict[str, Any]]:
        if self.trajectory_replay_key is not None:
            return _TRAJECTORY_REPLAY_REGISTRY[self.trajectory_replay_key]
        return []

    def __len__(self):
        if self.trajectory_replay_records:
            return len(self.trajectory_replay_records)
        return len(self.sample_index)

    def _build_sample_index(self) -> list[tuple[int, int, int]]:
        sample_index: list[tuple[int, int, int]] = []
        ep_from = self.lerobot_dataset.episode_data_index["from"]
        ep_to = self.lerobot_dataset.episode_data_index["to"]
        num_episodes = int(ep_from.shape[0])
        for episode_idx in range(num_episodes):
            raw_num_obs = int(ep_to[episode_idx].item() - ep_from[episode_idx].item())
            if raw_num_obs <= 1:
                continue
            effective_obs_count = ((raw_num_obs - 1) // self.effective_obs_stride) + 1
            trigger_start = self.history_obs
            trigger_end = effective_obs_count - self.future_obs
            for trigger_obs_idx in range(trigger_start, trigger_end):
                if self.keep_trigger_phase and ((trigger_obs_idx + 1) % self.trigger_every_n_obs != 0):
                    continue
                raw_action_start = trigger_obs_idx * self.effective_obs_stride
                if raw_action_start + self.action_horizon > raw_num_obs - 1:
                    continue
                sample_index.append((episode_idx, trigger_obs_idx, raw_action_start))
        return sample_index

    def _resolve_episode_owner(self, episode_idx: int):
        local_idx = int(episode_idx)
        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            if local_idx < dataset.num_episodes:
                return dataset, local_idx
            local_idx -= dataset.num_episodes
        raise IndexError(f"Episode index {episode_idx} out of bounds.")

    def _load_episode_cache(self, episode_idx: int) -> dict[str, Any]:
        episode_idx = int(episode_idx)
        if episode_idx in self._episode_cache:
            payload = self._episode_cache.pop(episode_idx)
            self._episode_cache[episode_idx] = payload
            return payload

        dataset, local_episode_idx = self._resolve_episode_owner(episode_idx)
        episode_raw = self.lerobot_dataset.multi_dataset.get_episode_data(episode_idx)
        episode_raw = self.lerobot_dataset._split_lerobot_sample(episode_raw)

        state_raw = {
            meta["key"]: self.lerobot_dataset._get_state(meta, episode_raw).float()
            for meta in self.lerobot_dataset.state_meta
        }
        action_raw = {
            meta["key"]: self.lerobot_dataset._get_action(meta, episode_raw).float()
            for meta in self.lerobot_dataset.action_meta
        }
        task_index = int(episode_raw["task_index"][0].item())
        instruction = dataset.meta.tasks[task_index]
        payload = {
            "dataset": dataset,
            "local_episode_idx": int(local_episode_idx),
            "state_raw": state_raw,
            "action_raw": action_raw,
            "instruction": instruction,
        }
        if self.episode_cache_size > 0:
            self._episode_cache[episode_idx] = payload
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return payload

    def _query_episode_images(self, dataset, local_episode_idx: int, raw_frame_indices: list[int]) -> torch.Tensor:
        query_timestamps = {
            meta["lerobot_key"]: [float(frame_idx) / float(dataset.fps) for frame_idx in raw_frame_indices]
            for meta in self.lerobot_dataset.image_meta
            if self.camera_key is None or meta["key"] == self.camera_key
        }
        decoded = dataset._query_videos(query_timestamps, int(local_episode_idx))

        processed_cameras = []
        for meta in self.lerobot_dataset.image_meta:
            if self.camera_key is not None and meta["key"] != self.camera_key:
                continue
            image = decoded[meta["lerobot_key"]]
            if image.ndim == 3:
                image = image.unsqueeze(0)
            image = (image * 255).to(torch.uint8)
            processed_cameras.append(image)

        video = torch.stack(processed_cameras, dim=0)  # [num_cameras, T, C, H, W]
        num_cameras, t_video, c_dim, height, width = video.shape

        if self.concat_multi_camera == "robotwin":
            if num_cameras != 3:
                raise ValueError(f"`concat_multi_camera='robotwin'` requires 3 cameras, got {num_cameras}.")
            cam_top = torch.nn.functional.interpolate(
                video[0].float(), size=[256, 320], mode="bilinear", align_corners=False
            ).to(video.dtype)
            cam_left = torch.nn.functional.interpolate(
                video[1].float(), size=[128, 160], mode="bilinear", align_corners=False
            ).to(video.dtype)
            cam_right = torch.nn.functional.interpolate(
                video[2].float(), size=[128, 160], mode="bilinear", align_corners=False
            ).to(video.dtype)
            bottom = torch.cat([cam_left, cam_right], dim=-1)
            video = torch.cat([cam_top, bottom], dim=-2)
        elif num_cameras > 1:
            if self.concat_multi_camera == "horizontal":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-1)
            elif self.concat_multi_camera == "vertical":
                video = torch.cat([video[i] for i in range(num_cameras)], dim=-2)
            else:
                raise ValueError(
                    f"Invalid concat_multi_camera: {self.concat_multi_camera}. "
                    "Expected one of: horizontal, vertical, robotwin."
                )
        else:
            video = video.squeeze(0)

        video = self.resize_transform(video)
        video = self.crop_transform(video)
        video = self.normalize_transform(video)
        if video.shape != (t_video, c_dim, self.video_size[0], self.video_size[1]):
            # multi-camera concat changes the final width; only validate T/C/H.
            if video.shape[0] != t_video or video.shape[1] != c_dim or video.shape[2] != self.video_size[0]:
                raise ValueError(f"Unexpected processed video shape: {tuple(video.shape)}")
        return video

    def _normalize_action_and_state(
        self,
        action_raw: dict[str, torch.Tensor],
        state_raw: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processor = self.lerobot_dataset.processor
        if processor is None:
            raise ValueError("Streaming episode dataset requires initialized `processor` with normalizer.")

        batch = {
            "action": {k: v.clone() for k, v in action_raw.items()},
            "state": {k: v.clone() for k, v in state_raw.items()},
        }
        batch = processor.action_state_transform(batch)
        batch = processor.normalizer.forward(batch)
        batch = processor.action_state_merger.forward(batch)
        return batch["action"], batch["state"]

    def __getitem__(self, idx):
        replay_record = None
        if self.trajectory_replay_records:
            replay_record = self.trajectory_replay_records[int(idx)]
            idx = int(replay_record["dataset_index"])
        episode_idx, trigger_obs_idx, raw_action_start = self.sample_index[idx]
        payload = self._load_episode_cache(episode_idx)

        raw_frame_indices = [
            (trigger_obs_idx - 1) * self.effective_obs_stride,
            trigger_obs_idx * self.effective_obs_stride,
            (trigger_obs_idx + 1) * self.effective_obs_stride,
            (trigger_obs_idx + 2) * self.effective_obs_stride,
        ]
        video = self._query_episode_images(
            dataset=payload["dataset"],
            local_episode_idx=payload["local_episode_idx"],
            raw_frame_indices=raw_frame_indices,
        )

        action_raw = {
            key: tensor[raw_action_start : raw_action_start + self.action_horizon]
            for key, tensor in payload["action_raw"].items()
        }
        state_raw = {
            key: tensor[raw_action_start : raw_action_start + 1]
            for key, tensor in payload["state_raw"].items()
        }
        target_action, proprio_t = self._normalize_action_and_state(
            action_raw=action_raw,
            state_raw=state_raw,
        )

        instruction = payload["instruction"]
        if self.override_instruction is not None:
            instruction = self.override_instruction
        prompt = DEFAULT_PROMPT.format(task=instruction)
        context, context_mask = self._get_cached_text_context(prompt)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)

        sample = {
            "obs_prev": video[0],
            "obs_cur": video[1],
            "obs_next": video[2],
            "obs_next2": video[3],
            "target_action": target_action,
            "action_is_pad": torch.zeros((target_action.shape[0],), dtype=torch.bool),
            "proprio_t": proprio_t.squeeze(0),
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "trigger_obs_idx": int(trigger_obs_idx),
            "raw_action_start": int(raw_action_start),
            "episode_idx": int(episode_idx),
        }
        if replay_record is not None:
            sample.update(
                {
                    "replay_x_t": replay_record["x_t"].float(),
                    "replay_timestep": torch.as_tensor(float(replay_record["timestep"]), dtype=torch.float32),
                    "replay_layer_cache_keys": ",".join(str(v) for v in replay_record["layer_cache_keys"]),
                    "replay_denoise_step": torch.as_tensor(int(replay_record["denoise_step"]), dtype=torch.long),
                    "replay_mode": str(replay_record.get("mode", "unknown")),
                }
            )
        return sample
