from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from torch.utils.data import Dataset

from .libero90_dataset import LiberoEpisodeDataset


@dataclass(frozen=True)
class AnchorMeta:
    episode_idx: int
    anchor_pos: int


class LiberoOfflineContextDataset(Dataset[Dict[str, Any]]):
    """
    One sample:
    - fixed-length history of K steps
    - each history step has a GT next action chunk (teacher-forced in replay)
    - one anchor step to predict the target chunk
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
        step_dt_ms: int = 100,
        context_steps: int = 8,
        chunk_horizon: int = 5,
        anchor_stride_steps: int = 1,
        max_episodes: int | None = None,
        episode_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if source_dt_ms <= 0 or step_dt_ms <= 0:
            raise ValueError(f"source_dt_ms/step_dt_ms must be positive, got {source_dt_ms}/{step_dt_ms}")
        if context_steps <= 0:
            raise ValueError(f"context_steps must be positive, got {context_steps}")
        if chunk_horizon <= 0:
            raise ValueError(f"chunk_horizon must be positive, got {chunk_horizon}")
        if anchor_stride_steps <= 0:
            raise ValueError(f"anchor_stride_steps must be positive, got {anchor_stride_steps}")

        self.source_dt_ms = int(source_dt_ms)
        self.step_dt_ms = int(step_dt_ms)
        self.step_stride = max(1, int(round(float(step_dt_ms) / float(source_dt_ms))))
        self.context_steps = int(context_steps)
        self.chunk_horizon = int(chunk_horizon)
        self.anchor_stride_steps = int(anchor_stride_steps)

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

        self._time_indices_by_episode: Dict[int, List[int]] = {}
        self._anchors: List[AnchorMeta] = []
        for ep_idx in episode_indices:
            ep = self.base[int(ep_idx)]
            t_len = int(ep["actions"].shape[0])
            time_indices = list(range(0, t_len, self.step_stride))
            if not time_indices:
                continue
            self._time_indices_by_episode[int(ep_idx)] = time_indices

            # For each anchor position p we need:
            # - K previous history steps: [p-K, ..., p-1]
            # - one target chunk: [p, ..., p+H-1]
            anchor_start = self.context_steps
            anchor_end = len(time_indices) - self.chunk_horizon + 1
            if anchor_end <= anchor_start:
                continue
            for anchor_pos in range(anchor_start, anchor_end, self.anchor_stride_steps):
                self._anchors.append(AnchorMeta(episode_idx=int(ep_idx), anchor_pos=anchor_pos))

        if not self._anchors:
            raise ValueError("No valid anchor samples produced. Check context/chunk settings.")

    def __len__(self) -> int:
        return len(self._anchors)

    @staticmethod
    def _video_window_indices(t_idx: int) -> List[int]:
        # 3-frame causal history window: [t-2, t-1, t].
        return [max(0, t_idx - 2), max(0, t_idx - 1), t_idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        meta = self._anchors[idx]
        ep = self.base[meta.episode_idx]
        time_indices = self._time_indices_by_episode[meta.episode_idx]

        anchor_pos = int(meta.anchor_pos)
        context_positions = list(range(anchor_pos - self.context_steps, anchor_pos))
        context_t = [time_indices[p] for p in context_positions]
        anchor_t = int(time_indices[anchor_pos])
        target_t = time_indices[anchor_pos : anchor_pos + self.chunk_horizon]

        images = ep["images"]
        states = ep["states"]
        actions = ep["actions"]

        context_videos = []
        for t_idx in context_t:
            frame_ids = self._video_window_indices(int(t_idx))
            context_videos.append(images[torch.as_tensor(frame_ids, dtype=torch.long)])
        context_videos = torch.stack(context_videos, dim=0)
        context_states = states[torch.as_tensor(context_t, dtype=torch.long)]

        history_chunks: List[torch.Tensor] = []
        for p in context_positions:
            chunk_indices = time_indices[p : p + self.chunk_horizon]
            history_chunks.append(actions[torch.as_tensor(chunk_indices, dtype=torch.long)])
        context_action_chunks = torch.stack(history_chunks, dim=0)

        anchor_frame_ids = self._video_window_indices(anchor_t)
        anchor_video = images[torch.as_tensor(anchor_frame_ids, dtype=torch.long)]
        anchor_state = states[anchor_t]
        target_chunk = actions[torch.as_tensor(target_t, dtype=torch.long)]

        return {
            "prompt": ep["prompt"],
            "context_videos": context_videos,
            "context_states": context_states,
            "context_action_chunks": context_action_chunks,
            "context_time_indices": torch.tensor(context_t, dtype=torch.long),
            "anchor_video": anchor_video,
            "anchor_state": anchor_state,
            "anchor_time_idx": torch.tensor(anchor_t, dtype=torch.long),
            "target_chunk": target_chunk,
            "target_time_indices": torch.tensor(target_t, dtype=torch.long),
            "episode_idx": torch.tensor(meta.episode_idx, dtype=torch.long),
            "anchor_pos": torch.tensor(anchor_pos, dtype=torch.long),
        }


def offline_context_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Keep as list because each sample contains variable-length multimodal internals.
    return batch
