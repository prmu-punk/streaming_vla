from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .replay_buffer import ReplayBuffer


class LiberoEpisodeDataset(Dataset[Dict[str, Any]]):
    """
    Episode-level dataset backed by OAT ReplayBuffer.

    One item = one full episode:
    - images[T,H,W,3]
    - states[T,S]
    - actions[T,A]
    - prompt(str)
    """

    def __init__(
        self,
        zarr_path: str,
        *,
        image_key: str = "agentview_rgb",
        extra_image_keys: Sequence[str] = (),
        action_key: str = "action",
        state_keys: Sequence[str] = (
            "robot0_joint_pos",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        prompt_key: str = "prompt",
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        self.buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")
        self.image_key = image_key
        self.extra_image_keys = list(extra_image_keys)
        self.action_key = action_key
        self.state_keys = list(state_keys)
        self.prompt_key = prompt_key

        episode_ends = np.asarray(self.buffer.episode_ends[:], dtype=np.int64)
        n_episodes = int(len(episode_ends))
        if n_episodes == 0:
            raise ValueError(f"No episodes found in dataset: {zarr_path}")
        if max_episodes is not None:
            n_episodes = min(n_episodes, int(max_episodes))

        self._episodes: List[Tuple[int, int]] = []
        prev_end = 0
        for i in range(n_episodes):
            end = int(episode_ends[i])
            self._episodes.append((prev_end, end))
            prev_end = end

        action_arr = self.buffer[self.action_key]
        if len(action_arr.shape) != 2:
            raise ValueError(
                f"Expected action array rank 2 [T, D], got shape {tuple(action_arr.shape)}"
            )
        self.action_dim = int(action_arr.shape[1])
        self.state_dim = int(self._build_states_slice(slice(0, 1)).shape[-1])

    def compute_action_stats(self, episode_indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        total = np.zeros((self.action_dim,), dtype=np.float64)
        total_sq = np.zeros((self.action_dim,), dtype=np.float64)
        count = 0
        for idx in episode_indices:
            ep_start, ep_end = self.get_episode_bounds(int(idx))
            arr = np.asarray(self.buffer[self.action_key][slice(ep_start, ep_end)], dtype=np.float64)
            total += arr.sum(axis=0)
            total_sq += np.square(arr).sum(axis=0)
            count += int(arr.shape[0])
        if count <= 0:
            raise ValueError("No action samples found for normalization.")
        mean = total / float(count)
        var = np.maximum(total_sq / float(count) - np.square(mean), 1.0e-6)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(var).astype(np.float32))

    def compute_state_stats(self, episode_indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
        total = np.zeros((self.state_dim,), dtype=np.float64)
        total_sq = np.zeros((self.state_dim,), dtype=np.float64)
        count = 0
        for idx in episode_indices:
            ep_start, ep_end = self.get_episode_bounds(int(idx))
            arr = self._build_states_slice(slice(ep_start, ep_end)).astype(np.float64)
            total += arr.sum(axis=0)
            total_sq += np.square(arr).sum(axis=0)
            count += int(arr.shape[0])
        if count <= 0:
            raise ValueError("No state samples found for normalization.")
        mean = total / float(count)
        var = np.maximum(total_sq / float(count) - np.square(mean), 1.0e-6)
        return torch.from_numpy(mean.astype(np.float32)), torch.from_numpy(np.sqrt(var).astype(np.float32))

    def __len__(self) -> int:
        return len(self._episodes)

    def get_episode_bounds(self, idx: int) -> Tuple[int, int]:
        return self._episodes[int(idx)]

    def get_episode_length(self, idx: int) -> int:
        ep_start, ep_end = self.get_episode_bounds(idx)
        return int(ep_end - ep_start)

    def get_prompt(self, idx: int) -> str:
        ep_start, _ = self.get_episode_bounds(idx)
        return self._normalize_prompt(self.buffer[self.prompt_key][ep_start])

    def _build_states_slice(self, sl: slice) -> np.ndarray:
        pieces: List[np.ndarray] = []
        for key in self.state_keys:
            arr = np.asarray(self.buffer[key][sl], dtype=np.float32)
            pieces.append(arr.reshape(arr.shape[0], -1))
        return np.concatenate(pieces, axis=1)

    @staticmethod
    def _normalize_prompt(value: Any) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.ndarray):
            if value.shape == ():
                return LiberoEpisodeDataset._normalize_prompt(value.item())
            if value.size == 0:
                return ""
            return LiberoEpisodeDataset._normalize_prompt(value.reshape(-1)[0])
        return str(value)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ep_start, ep_end = self.get_episode_bounds(idx)
        sl = slice(ep_start, ep_end)

        images = np.asarray(self.buffer[self.image_key][sl], dtype=np.uint8)
        extra_images = {
            key: torch.from_numpy(np.asarray(self.buffer[key][sl], dtype=np.uint8))
            for key in self.extra_image_keys
        }
        actions = np.asarray(self.buffer[self.action_key][sl], dtype=np.float32)
        states = self._build_states_slice(sl)

        return {
            "images": torch.from_numpy(images),
            "extra_images": extra_images,
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "prompt": self.get_prompt(idx),
            "episode_len": torch.tensor(ep_end - ep_start, dtype=torch.long),
        }
