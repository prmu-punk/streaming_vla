from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .libero90_dataset import LiberoEpisodeDataset


@dataclass(frozen=True)
class AnchorMeta:
    episode_idx: int
    anchor_t: int


class LiberoOfflineContextDataset(Dataset[Dict[str, Any]]):
    """
    One sample:
    - variable-length history of complete steps, backtracked from anchor_t
    - history step gaps are sampled from [step_dt_min_ms, step_dt_max_ms] on the source timeline
    - one anchor step at anchor_t
    - action chunks are contiguous source actions starting at each step time
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
        source_dt_ms: int = 50,
        step_dt_min_ms: int = 200,
        step_dt_max_ms: int = 300,
        num_frames: int = 6,
        chunk_horizon: int = 5,
        anchor_stride_steps: int = 1,
        max_context_len: int = 10_000,
        fixed_action_tokens: int = 8,
        max_episodes: int | None = None,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if source_dt_ms <= 0:
            raise ValueError(f"source_dt_ms must be positive, got {source_dt_ms}")
        if step_dt_min_ms <= 0 or step_dt_max_ms <= 0:
            raise ValueError(
                f"step_dt_min_ms/step_dt_max_ms must be positive, got {step_dt_min_ms}/{step_dt_max_ms}"
            )
        if step_dt_min_ms > step_dt_max_ms:
            raise ValueError(
                f"step_dt_min_ms must be <= step_dt_max_ms, got {step_dt_min_ms}>{step_dt_max_ms}"
            )
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")
        if chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}")
        if anchor_stride_steps <= 0:
            raise ValueError(f"anchor_stride_steps must be positive, got {anchor_stride_steps}")
        if max_context_len <= 0:
            raise ValueError(f"max_context_len must be positive, got {max_context_len}")
        if fixed_action_tokens <= 0:
            raise ValueError(f"fixed_action_tokens must be positive, got {fixed_action_tokens}")

        self.source_dt_ms = int(source_dt_ms)
        self.step_dt_min_ms = int(step_dt_min_ms)
        self.step_dt_max_ms = int(step_dt_max_ms)
        self.num_frames = int(num_frames)
        self.chunk_horizon = int(chunk_horizon)
        self.anchor_stride_steps = int(anchor_stride_steps)
        self.max_context_len = int(max_context_len)
        self.fixed_action_tokens = int(fixed_action_tokens)

        stride_min = max(1, int(round(float(step_dt_min_ms) / float(source_dt_ms))))
        stride_max = max(1, int(round(float(step_dt_max_ms) / float(source_dt_ms))))
        self.step_strides = list(range(stride_min, stride_max + 1))
        self._estimated_prompt_tokens = 64
        self._estimated_video_tokens_per_frame = 16
        self._estimated_user_overhead = 14
        self._estimated_assistant_overhead = 6
        self._estimated_state_tokens = 1

        self.base = LiberoEpisodeDataset(
            zarr_path=zarr_path,
            image_key=image_key,
            action_key=action_key,
            state_keys=state_keys,
            prompt_key=prompt_key,
            max_episodes=max_episodes,
        )
        self.state_dim = self.base.state_dim
        self.action_dim = self.base.action_dim

        if episode_indices is None:
            episode_indices = list(range(len(self.base)))

        self._anchors: List[AnchorMeta] = []
        for ep_idx in episode_indices:
            ep = self.base[int(ep_idx)]
            t_len = int(ep["actions"].shape[0])
            anchor_end = t_len - self.chunk_horizon + 1
            if anchor_end <= 0:
                continue
            for anchor_t in range(0, anchor_end, self.anchor_stride_steps):
                self._anchors.append(AnchorMeta(episode_idx=int(ep_idx), anchor_t=int(anchor_t)))

        if not self._anchors:
            raise ValueError("No valid anchor samples produced. Check dataset settings.")

    def __len__(self) -> int:
        return len(self._anchors)

    def _make_rng(self, episode_idx: int, anchor_t: int) -> np.random.Generator:
        seed = int((episode_idx + 1) * 1_000_003 + anchor_t * 97 + self.source_dt_ms * 17)
        return np.random.default_rng(seed)

    def _video_window_indices(self, t_idx: int) -> List[int]:
        start = int(t_idx) - self.num_frames + 1
        return [max(0, start + i) for i in range(self.num_frames)]

    def _history_step_times(self, *, anchor_t: int, episode_idx: int) -> List[int]:
        rng = self._make_rng(episode_idx=episode_idx, anchor_t=anchor_t)
        times_rev: List[int] = []
        cursor = int(anchor_t)
        while True:
            stride = int(rng.choice(self.step_strides))
            prev_t = cursor - stride
            if prev_t < 0:
                break
            times_rev.append(int(prev_t))
            cursor = int(prev_t)
        return list(reversed(times_rev))

    def _estimate_anchor_tokens(self) -> int:
        return (
            self._estimated_user_overhead
            + self._estimated_state_tokens
            + self.num_frames * self._estimated_video_tokens_per_frame
            + self._estimated_assistant_overhead
            + self.fixed_action_tokens
        )

    def _estimate_history_step_tokens(self) -> int:
        return (
            self._estimated_user_overhead
            + self._estimated_state_tokens
            + self.num_frames * self._estimated_video_tokens_per_frame
            + self._estimated_assistant_overhead
            + self.fixed_action_tokens
            + 1
        )

    def _truncate_history_by_budget(self, history_t: List[int]) -> List[int]:
        budget = self.max_context_len
        used = self._estimated_prompt_tokens + self._estimate_anchor_tokens()
        keep_rev: List[int] = []
        hist_step_tokens = self._estimate_history_step_tokens()
        for t_idx in reversed(history_t):
            if used + hist_step_tokens > budget:
                break
            keep_rev.append(int(t_idx))
            used += hist_step_tokens
        return list(reversed(keep_rev))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._anchors[idx]
        ep = self.base[meta.episode_idx]

        anchor_t = int(meta.anchor_t)
        history_t = self._history_step_times(anchor_t=anchor_t, episode_idx=meta.episode_idx)
        history_t = self._truncate_history_by_budget(history_t)

        images = ep["images"]
        states = ep["states"]
        actions = ep["actions"]

        context_videos = []
        for t_idx in history_t:
            frame_ids = self._video_window_indices(int(t_idx))
            context_videos.append(images[torch.as_tensor(frame_ids, dtype=torch.long)])
        if context_videos:
            context_videos_t = torch.stack(context_videos, dim=0)
            context_states_t = states[torch.as_tensor(history_t, dtype=torch.long)]
        else:
            context_videos_t = torch.empty(
                (0, self.num_frames, *images.shape[1:]),
                dtype=images.dtype,
            )
            context_states_t = torch.empty((0, states.shape[-1]), dtype=states.dtype)

        history_chunks: List[torch.Tensor] = []
        for t_idx in history_t:
            chunk_ids = list(range(int(t_idx), int(t_idx) + self.chunk_horizon))
            history_chunks.append(actions[torch.as_tensor(chunk_ids, dtype=torch.long)])
        if history_chunks:
            context_action_chunks_t = torch.stack(history_chunks, dim=0)
        else:
            context_action_chunks_t = torch.empty(
                (0, self.chunk_horizon, actions.shape[-1]),
                dtype=actions.dtype,
            )

        anchor_frame_ids = self._video_window_indices(anchor_t)
        anchor_video = images[torch.as_tensor(anchor_frame_ids, dtype=torch.long)]
        anchor_state = states[anchor_t]

        target_t = list(range(anchor_t, anchor_t + self.chunk_horizon))
        target_chunk = actions[torch.as_tensor(target_t, dtype=torch.long)]

        return {
            "prompt": ep["prompt"],
            "context_videos": context_videos_t,
            "context_states": context_states_t,
            "context_action_chunks": context_action_chunks_t,
            "context_time_indices": torch.tensor(history_t, dtype=torch.long),
            "anchor_video": anchor_video,
            "anchor_state": anchor_state,
            "anchor_time_idx": torch.tensor(anchor_t, dtype=torch.long),
            "target_chunk": target_chunk,
            "target_time_indices": torch.tensor(target_t, dtype=torch.long),
            "episode_idx": torch.tensor(meta.episode_idx, dtype=torch.long),
        }


def offline_context_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch
