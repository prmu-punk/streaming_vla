from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import zarr
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "libero90_async_dataset requires zarr. Install with `pip install zarr`."
    ) from exc


def _open_zarr_root(zarr_path: str) -> Any:
    """只读打开 zarr 根组，作为后续键解析入口。"""
    root = zarr.open_group(zarr_path, mode="r")
    return root


def _resolve_array(root: Any, key: str) -> Any:
    """解析数据键并兼容 `{key}`/`data/{key}` 两种命名接口。"""
    if key in root:
        return root[key]
    data_key = f"data/{key}"
    if data_key in root:
        return root[data_key]
    raise KeyError(f"Array key not found in zarr: '{key}' (also tried '{data_key}')")


def _resolve_episode_ends(root) -> Any:
    """读取 episode 边界数组，兼容 `episode_ends` 与 `meta/episode_ends`。"""
    for candidate in ("episode_ends", "meta/episode_ends"):
        if candidate in root:
            return np.asarray(root[candidate][:], dtype=np.int64)
    raise KeyError("episode_ends not found in zarr root (tried 'episode_ends' and 'meta/episode_ends').")


class LiberoEpisodeDataset(Dataset[Dict[str, Any]]):
    """
    OAT 解耦版 episode-level dataset。

    zarr 接口约定（可调）：
    - episode ends: `episode_ends` 或 `meta/episode_ends`
    - 数据键：优先 `{key}`，其次 `data/{key}`
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
        """构建 episode 级别 LIBERO 数据集。

        参数:
            zarr_path: zarr 数据集路径。
            image_key: 图像数组键名。
            action_key: 动作数组键名。
            state_keys: 状态字段键名序列，读取后按顺序拼接。
            prompt_key: 语言指令键名。
            max_episodes: 可选截断 episode 数量上限。

        接口对应:
            `__getitem__` 返回 `images/states/actions/prompt/episode_len`，供离线采样与在线匹配复用。
        """
        super().__init__()
        self.root = _open_zarr_root(zarr_path)
        self.image_arr = _resolve_array(self.root, image_key)
        self.action_arr = _resolve_array(self.root, action_key)
        self.state_arrs = [_resolve_array(self.root, key) for key in state_keys]
        self.prompt_arr = _resolve_array(self.root, prompt_key)

        self.image_key = image_key
        self.action_key = action_key
        self.state_keys = list(state_keys)
        self.prompt_key = prompt_key

        episode_ends = _resolve_episode_ends(self.root)
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

        if len(self.action_arr.shape) != 2:
            raise ValueError(
                f"Expected action array rank 2 [T, D], got shape {tuple(self.action_arr.shape)}"
            )
        self.action_dim = int(self.action_arr.shape[1])
        self.state_dim = int(self._build_states_slice(slice(0, 1)).shape[-1])

    def __len__(self) -> int:
        """返回 episode 数量，作为 DataLoader 采样上界。"""
        return len(self._episodes)

    def _build_states_slice(self, sl: slice) -> np.ndarray:
        """按时间切片读取并拼接多路状态字段，输出二维状态矩阵。"""
        pieces: List[np.ndarray] = []
        for arr in self.state_arrs:
            value = np.asarray(arr[sl], dtype=np.float32)
            pieces.append(value.reshape(value.shape[0], -1))
        return np.concatenate(pieces, axis=1)

    @staticmethod
    def _normalize_prompt(value: Any) -> str:
        """将 zarr 中可能的 bytes/ndarray 标量统一转换为字符串 prompt。"""
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
        """按 episode 索引读取样本并返回标准字段字典。

        参数:
            idx: episode 索引。

        返回:
            包含 `images/states/actions/prompt/episode_len` 的样本字典，
            作为离线训练与在线检索的统一数据接口。
        """
        ep_start, ep_end = self._episodes[idx]
        sl = slice(ep_start, ep_end)

        images = np.asarray(self.image_arr[sl], dtype=np.uint8)
        actions = np.asarray(self.action_arr[sl], dtype=np.float32)
        states = self._build_states_slice(sl)

        prompt_seq = self.prompt_arr[sl]
        prompt = self._normalize_prompt(prompt_seq[0]) if len(prompt_seq) > 0 else ""

        return {
            "images": torch.from_numpy(images),
            "states": torch.from_numpy(states),
            "actions": torch.from_numpy(actions),
            "prompt": prompt,
            "episode_len": torch.tensor(ep_end - ep_start, dtype=torch.long),
        }
