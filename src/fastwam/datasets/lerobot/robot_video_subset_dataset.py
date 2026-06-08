from typing import List, Optional

import torch

from .robot_video_dataset import RobotVideoDataset


class RobotVideoSubsetDataset(RobotVideoDataset):
    """Thin wrapper over RobotVideoDataset that only exposes selected episodes.

    This intentionally keeps the original RobotVideoDataset/BaseLerobotDataset
    construction path untouched. It only remaps dataset indices after the
    normal train/val split has been built, which makes it suitable for quick
    task-subset experiments without changing the underlying loader behavior.
    """

    def __init__(
        self,
        *args,
        episode_indices: Optional[List[int]] = None,
        episode_ranges: Optional[List[List[int]]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.episode_indices = episode_indices
        self.episode_ranges = episode_ranges
        self._subset_indices = self._build_subset_indices()

    def _episode_is_selected(self, episode_index: int) -> bool:
        if self.episode_indices is None and self.episode_ranges is None:
            return True

        if self.episode_indices is not None and int(episode_index) in {int(i) for i in self.episode_indices}:
            return True

        if self.episode_ranges is not None:
            for start, end in self.episode_ranges:
                if int(start) <= int(episode_index) < int(end):
                    return True

        return False

    def _build_subset_indices(self) -> torch.Tensor:
        selected = []
        local_episode_offset = 0

        for dataset in self.lerobot_dataset.multi_dataset._datasets:
            for local_episode_idx, original_episode_idx in enumerate(dataset.episodes):
                if not self._episode_is_selected(int(original_episode_idx)):
                    continue

                global_episode_idx = local_episode_offset + local_episode_idx
                start = int(self.lerobot_dataset.episode_data_index["from"][global_episode_idx].item())
                end = int(self.lerobot_dataset.episode_data_index["to"][global_episode_idx].item())
                selected.extend(range(start, end))

            local_episode_offset += dataset.num_episodes

        if not selected:
            raise ValueError(
                "Episode subset produced no samples. "
                f"episode_indices={self.episode_indices}, episode_ranges={self.episode_ranges}"
            )

        return torch.tensor(selected, dtype=torch.long)

    def __len__(self):
        return int(self._subset_indices.numel())

    def _get(self, idx):
        raw_idx = int(self._subset_indices[int(idx)].item())
        return super()._get(raw_idx)
