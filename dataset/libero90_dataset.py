from __future__ import annotations

import math
import pathlib
import sys
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# Reuse OAT replay buffer loader to read the zarr layout produced by their conversion scripts.
_OAT_ROOT = pathlib.Path(__file__).resolve().parents[1] / "lingbot-va" / "oat"
if str(_OAT_ROOT) not in sys.path:
    sys.path.append(str(_OAT_ROOT))

from oat.oat.common.replay_buffer import ReplayBuffer  # type: ignore


class LiberoChunkDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    Chunk-aligned dataset:
    - sample starts at t with stride=chunk_horizon
    - target is actions[t:t+H]
    - tail chunk shorter than H is padded, with valid_len provided
    """

    def __init__(
        self,
        zarr_path: str,
        chunk_horizon: int,
        stride: int,
        image_key: str = "agentview_rgb",
        action_key: str = "action",
        state_keys: Sequence[str] = (
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ),
        max_episodes: int | None = None,
    ) -> None:
        super().__init__()
        if chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}.")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}.")

        self.buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[action_key, image_key, *state_keys],
        )
        self.image_key = image_key
        self.action_key = action_key
        self.state_keys = list(state_keys)
        self.chunk_horizon = int(chunk_horizon)
        self.stride = int(stride)

        episode_ends = np.asarray(self.buffer.episode_ends[:], dtype=np.int64)
        n_episodes = int(len(episode_ends))
        if n_episodes == 0:
            raise ValueError(f"No episodes found in dataset: {zarr_path}")
        if max_episodes is not None:
            n_episodes = min(n_episodes, int(max_episodes))

        self._episodes: List[Tuple[int, int]] = []
        prev_end = 0
        for episode_idx in range(n_episodes):
            end = int(episode_ends[episode_idx])
            self._episodes.append((prev_end, end))
            prev_end = end

        self._indices: List[Tuple[int, int]] = []
        for episode_idx, (start, end) in enumerate(self._episodes):
            ep_len = end - start
            if ep_len <= 0:
                continue
            for local_t in range(0, ep_len, self.stride):
                self._indices.append((episode_idx, local_t))

        if not self._indices:
            raise ValueError(
                "No chunk samples produced. Check dataset length / chunk_horizon / stride."
            )

        action_arr = self.buffer[self.action_key]
        if len(action_arr.shape) != 2:
            raise ValueError(
                f"Expected action array rank 2 [T, D], got shape {tuple(action_arr.shape)}"
            )
        self.action_dim = int(action_arr.shape[1])

        example_state = self._build_state(0)
        self.state_dim = int(example_state.shape[0])

    def __len__(self) -> int:
        return len(self._indices)

    def _global_t(self, episode_idx: int, local_t: int) -> int:
        ep_start, _ = self._episodes[episode_idx]
        return ep_start + local_t

    def _build_state(self, global_t: int) -> np.ndarray:
        pieces: List[np.ndarray] = []
        for key in self.state_keys:
            value = np.asarray(self.buffer[key][global_t], dtype=np.float32).reshape(-1)
            pieces.append(value)
        return np.concatenate(pieces, axis=0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        episode_idx, local_t = self._indices[idx]
        ep_start, ep_end = self._episodes[episode_idx]
        global_t = ep_start + local_t

        image_t = np.asarray(self.buffer[self.image_key][global_t], dtype=np.uint8)
        state_t = self._build_state(global_t)

        # GT chunk: actions[t:t+H], padded to H for batching convenience.
        action_seq = np.asarray(
            self.buffer[self.action_key][global_t : global_t + self.chunk_horizon],
            dtype=np.float32,
        )
        valid_len = int(action_seq.shape[0])
        if valid_len <= 0:
            raise RuntimeError(f"Internal error: zero-length chunk at idx={idx}, global_t={global_t}")

        gt_chunk = np.zeros((self.chunk_horizon, self.action_dim), dtype=np.float32)
        gt_chunk[:valid_len] = action_seq

        return {
            "image_t": torch.from_numpy(image_t),
            "state_t": torch.from_numpy(state_t),
            "gt_chunk": torch.from_numpy(gt_chunk),
            "valid_len": torch.tensor(valid_len, dtype=torch.long),
            "episode_idx": torch.tensor(episode_idx, dtype=torch.long),
            "start_t": torch.tensor(local_t, dtype=torch.long),
            "episode_len": torch.tensor(ep_end - ep_start, dtype=torch.long),
        }


class LiberoEpisodeDataset(Dataset[Dict[str, Any]]):
    """
    Episode-level dataset:
    - one item = one full episode
    - fields: images[T,H,W,3], states[T,S], actions[T,A], prompt(str)
    """

    def __init__(
        self,
        zarr_path: str,
        *,
        image_key: str = "agentview_rgb",
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
        self.buffer = ReplayBuffer.copy_from_path(
            zarr_path,
            keys=[image_key, action_key, *state_keys, prompt_key],
        )
        self.image_key = image_key
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

    def __len__(self) -> int:
        return len(self._episodes)

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
        ep_start, ep_end = self._episodes[idx]
        sl = slice(ep_start, ep_end)

        images = np.asarray(self.buffer[self.image_key][sl], dtype=np.uint8)
        actions = np.asarray(self.buffer[self.action_key][sl], dtype=np.float32)
        states = self._build_states_slice(sl)

        prompt_arr = self.buffer[self.prompt_key][sl]
        prompt = self._normalize_prompt(prompt_arr[0]) if len(prompt_arr) > 0 else ""

        return {
            "images": torch.from_numpy(images),
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "prompt": prompt,
            "episode_len": torch.tensor(ep_end - ep_start, dtype=torch.long),
        }
