from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import pathlib
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .libero90_async_dataset import LiberoEpisodeDataset
from model.template_qwen3_vla import IM_END, build_prompt_prefill_text, build_step_user_prefix, build_video_text


@dataclass(frozen=True)
class AnchorMeta:
    episode_idx: int
    anchor_t: int


@dataclass(frozen=True)
class SamplePlan:
    episode_idx: int
    anchor_t: int
    history_t: tuple[int, ...]
    sample_length: int


class LiberoOfflineContextDataset(Dataset[Dict[str, Any]]):
    """
    async 训练专用离线 context dataset（OAT 解耦版）。

    说明：
    - 样本规划与 bucket 组织已对齐主干，但上下文中不包含 action token。
    """

    def __init__(
        self,
        zarr_path: str,
        *,
        image_key: str = "agentview_rgb",
        aux_image_key: str | None = None,
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
        episode_cache_size: int = 8,
        max_episodes: int | None = None,
        episode_indices: Sequence[int] | None = None,
        processor: Any,
    ) -> None:
        """构建离线 context 采样数据集，用于 rtc_async 训练。

        参数:
            zarr_path: LIBERO zarr 数据路径。
            image_key: 图像键名。
            action_key: 动作键名。
            state_keys: 状态键序列，按顺序拼接成状态向量。
            prompt_key: 文本指令键名。
            source_dt_ms: 原始数据时间步（毫秒）。
            step_dt_min_ms: 历史采样最小步距（毫秒）。
            step_dt_max_ms: 历史采样最大步距（毫秒）。
            num_frames: 每个 step 的视频窗口帧数。
            chunk_horizon: 动作 chunk 长度。
            anchor_stride_steps: anchor 枚举步长。
            max_context_len: 上下文 token 长度上限。
            episode_cache_size: worker-local episode LRU cache 容量，0 表示关闭。
            max_episodes: 可选 episode 数量上限。
            episode_indices: 可选子集 episode 索引。
            processor: 用于真实估算多模态输入 token 长度的 processor。
        接口对应:
            `__getitem__` 产出的 `context_* / anchor_* / target_chunk`
            直接对应 `Qwen3RTCVLAEncoder.forward_offline_context_batch` 输入契约。
        """
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

        self.source_dt_ms = int(source_dt_ms)
        self.step_dt_min_ms = int(step_dt_min_ms)
        self.step_dt_max_ms = int(step_dt_max_ms)
        self.num_frames = int(num_frames)
        self.chunk_horizon = int(chunk_horizon)
        self.anchor_stride_steps = int(anchor_stride_steps)
        self.max_context_len = int(max_context_len)
        self.episode_cache_size = max(0, int(episode_cache_size))

        stride_min = max(1, int(round(float(step_dt_min_ms) / float(source_dt_ms))))
        stride_max = max(1, int(round(float(step_dt_max_ms) / float(source_dt_ms))))
        self.step_strides = list(range(stride_min, stride_max + 1))

        self.base = LiberoEpisodeDataset(
            zarr_path=zarr_path,
            image_key=image_key,
            extra_image_keys=([aux_image_key] if aux_image_key is not None else []),
            action_key=action_key,
            state_keys=state_keys,
            prompt_key=prompt_key,
            max_episodes=max_episodes,
        )
        self.aux_image_key = aux_image_key
        self.state_dim = self.base.state_dim
        self.action_dim = self.base.action_dim

        if episode_indices is None:
            episode_indices = list(range(len(self.base)))

        self._anchors: List[AnchorMeta] = []
        self._plans: List[SamplePlan] = []

        self._planning_processor = processor
        self._prompt_length_cache: Dict[int, int] = {}
        self._step_length_cache: Dict[int, int] = {}
        self._episode_cache: OrderedDict[int, Dict[str, Any]] = OrderedDict()
        self._action_mean: torch.Tensor | None = None
        self._action_std: torch.Tensor | None = None
        self._state_mean: torch.Tensor | None = None
        self._state_std: torch.Tensor | None = None
        self._plan_cache_path = self._build_plan_cache_path(
            zarr_path=zarr_path,
            episode_indices=episode_indices,
        )

        if self._plan_cache_path.exists():
            self._load_plans_from_cache()
        else:
            for ep_idx in episode_indices:
                ep = self._get_episode(int(ep_idx))
                t_len = int(ep["actions"].shape[0])
                anchor_end = t_len - self.chunk_horizon + 1
                if anchor_end <= 0:
                    continue
                for anchor_t in range(0, anchor_end, self.anchor_stride_steps):
                    meta = AnchorMeta(episode_idx=int(ep_idx), anchor_t=int(anchor_t))
                    self._anchors.append(meta)
                    self._plans.append(self._build_sample_plan(meta))
            self._save_plans_to_cache()

        if not self._anchors:
            raise ValueError("No valid anchor samples produced. Check dataset settings.")

        self._planning_processor = None
        self._prompt_length_cache.clear()
        self._step_length_cache.clear()
        self._episode_cache.clear()

    def __len__(self) -> int:
        """返回可用 anchor 样本数，作为离线 context 训练采样空间。"""
        return len(self._plans)

    def get_estimated_length(self, idx: int) -> int:
        return int(self._plans[idx].sample_length)

    def sample_indices_for_episodes(self, episode_indices: Sequence[int]) -> List[int]:
        keep = set(int(x) for x in episode_indices)
        return [
            idx for idx, plan in enumerate(self._plans) if int(plan.episode_idx) in keep
        ]

    def set_normalization_stats(
        self,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> None:
        self._action_mean = action_mean.detach().float().cpu()
        self._action_std = action_std.detach().float().cpu()
        self._state_mean = state_mean.detach().float().cpu()
        self._state_std = state_std.detach().float().cpu()

    def _make_rng(self, episode_idx: int, anchor_t: int) -> np.random.Generator:
        """为指定 episode-anchor 对生成可复现随机数源。"""
        seed = int((episode_idx + 1) * 1_000_003 + anchor_t * 97 + self.source_dt_ms * 17)
        return np.random.default_rng(seed)

    def _build_plan_cache_path(
        self,
        *,
        zarr_path: str,
        episode_indices: Sequence[int],
    ) -> pathlib.Path:
        cache_dir = pathlib.Path(__file__).resolve().parents[1] / ".cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_key = {
            "zarr_path": str(zarr_path),
            "source_dt_ms": self.source_dt_ms,
            "step_dt_min_ms": self.step_dt_min_ms,
            "step_dt_max_ms": self.step_dt_max_ms,
            "num_frames": self.num_frames,
            "chunk_horizon": self.chunk_horizon,
            "anchor_stride_steps": self.anchor_stride_steps,
            "max_context_len": self.max_context_len,
            "episode_indices": [int(x) for x in episode_indices],
        }
        digest = hashlib.sha1(json.dumps(cache_key, sort_keys=True).encode("utf-8")).hexdigest()[:16]
        return cache_dir / f"libero_async_plan_{digest}.pt"

    def _load_plans_from_cache(self) -> None:
        payload = torch.load(self._plan_cache_path, map_location="cpu")
        self._anchors = [AnchorMeta(**item) for item in payload["anchors"]]
        self._plans = [
            SamplePlan(
                episode_idx=int(item["episode_idx"]),
                anchor_t=int(item["anchor_t"]),
                history_t=tuple(int(x) for x in item["history_t"]),
                sample_length=int(item["sample_length"]),
            )
            for item in payload["plans"]
        ]

    def _save_plans_to_cache(self) -> None:
        payload = {
            "anchors": [
                {"episode_idx": int(meta.episode_idx), "anchor_t": int(meta.anchor_t)}
                for meta in self._anchors
            ],
            "plans": [
                {
                    "episode_idx": int(plan.episode_idx),
                    "anchor_t": int(plan.anchor_t),
                    "history_t": [int(x) for x in plan.history_t],
                    "sample_length": int(plan.sample_length),
                }
                for plan in self._plans
            ],
        }
        torch.save(payload, self._plan_cache_path)

    def _get_episode(self, episode_idx: int) -> Dict[str, Any]:
        cached = self._episode_cache.get(int(episode_idx), None)
        if cached is not None:
            self._episode_cache.move_to_end(int(episode_idx))
            return cached

        ep = self.base[int(episode_idx)]
        if self.episode_cache_size > 0:
            self._episode_cache[int(episode_idx)] = ep
            self._episode_cache.move_to_end(int(episode_idx))
            while len(self._episode_cache) > self.episode_cache_size:
                self._episode_cache.popitem(last=False)
        return ep

    def _video_window_indices(self, t_idx: int) -> List[int]:
        """计算某时刻对应的视频窗口索引，缺失前缀用 0 对齐。"""
        start = int(t_idx) - self.num_frames + 1
        return [max(0, start + i) for i in range(self.num_frames)]

    def _history_step_times(self, *, anchor_t: int, episode_idx: int) -> List[int]:
        """基于随机步距向后回溯历史时刻序列。"""
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

    def _build_step_text(self, *, ts_ms: int | None, video_token: str, has_aux: bool) -> str:
        return (
            build_step_user_prefix(
                ts_ms=ts_ms,
                video_token=build_video_text(video_token=video_token, has_aux=has_aux),
                close_previous_assistant=False,
            )
            + IM_END
            + "\n"
        )

    def _make_video_tensor(self, frames: np.ndarray | torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        if isinstance(frames, np.ndarray):
            frames_t = torch.from_numpy(frames)
        else:
            frames_t = frames
        target_frames = self.num_frames if num_frames is None else int(num_frames)
        if frames_t.dim() == 3:
            frames_t = frames_t.unsqueeze(0)
        if frames_t.shape[-1] == 3:
            frames_t = frames_t.permute(0, 3, 1, 2)
        if frames_t.shape[0] < target_frames:
            repeat = target_frames - frames_t.shape[0]
            frames_t = torch.cat([frames_t, frames_t[-1:].repeat(repeat, 1, 1, 1)], dim=0)
        elif frames_t.shape[0] > target_frames:
            frames_t = frames_t[:target_frames]
        return frames_t

    def _prompt_length(self, episode_idx: int) -> int:
        cached = self._prompt_length_cache.get(int(episode_idx), None)
        if cached is not None:
            return int(cached)
        prompt = str(self._get_episode(int(episode_idx))["prompt"])
        prompt_text = build_prompt_prefill_text(prompt)
        encoded = self._planning_processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        prompt_len = int(len(encoded["input_ids"]))
        self._prompt_length_cache[int(episode_idx)] = prompt_len
        return prompt_len

    def _step_length(self, *, episode_idx: int) -> int:
        cached = self._step_length_cache.get(int(episode_idx), None)
        if cached is not None:
            return int(cached)

        ep = self._get_episode(int(episode_idx))
        aux_images = ep.get("extra_images", {})
        aux_stack = aux_images.get(self.aux_image_key, None) if self.aux_image_key is not None else None
        has_aux = aux_stack is not None
        images = ep["images"]
        t_idx = max(0, int(images.shape[0] // 2))
        ts_ms = int(t_idx) * int(self.source_dt_ms)
        step_text = self._build_step_text(
            ts_ms=ts_ms,
            video_token=self._planning_processor.video_token,
            has_aux=has_aux,
        )
        frame_ids = self._video_window_indices(int(t_idx))
        videos = [[self._make_video_tensor(images[torch.as_tensor(frame_ids, dtype=torch.long)])]]
        if has_aux:
            videos[0].append(self._make_video_tensor(aux_stack[int(t_idx)].unsqueeze(0), 1))
        proc = self._planning_processor(
            text=[step_text],
            videos=videos,
            padding=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
        step_len = int(proc["input_ids"].shape[1])
        self._step_length_cache[int(episode_idx)] = step_len
        return step_len

    def _build_sample_plan(self, meta: AnchorMeta) -> SamplePlan:
        full_history_t = self._history_step_times(anchor_t=int(meta.anchor_t), episode_idx=int(meta.episode_idx))
        prompt_len = self._prompt_length(int(meta.episode_idx))
        step_len = self._step_length(episode_idx=int(meta.episode_idx))
        base_len = prompt_len + step_len
        available_len = max(self.max_context_len - base_len, 0)
        max_history_steps = available_len // max(step_len, 1)
        kept = full_history_t[-max_history_steps:] if max_history_steps > 0 else []
        total_len = base_len + len(kept) * step_len

        history_t = tuple(int(t) for t in kept)
        return SamplePlan(
            episode_idx=int(meta.episode_idx),
            anchor_t=int(meta.anchor_t),
            history_t=history_t,
            sample_length=int(total_len),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """按 anchor 索引构造离线 context 训练样本。

        参数:
            idx: anchor 样本索引。

        返回:
            包含 `prompt/context_* /anchor_* /target_chunk` 的字典，
            与 `Qwen3RTCVLAEncoder.forward_offline_context_batch` 输入接口对齐。
        """
        plan = self._plans[idx]
        ep = self._get_episode(plan.episode_idx)

        anchor_t = int(plan.anchor_t)
        history_t = list(plan.history_t)

        images = ep["images"]
        aux_images = ep.get("extra_images", {})
        aux_stack = aux_images.get(self.aux_image_key, None) if self.aux_image_key is not None else None
        states = ep["states"]
        actions = ep["actions"]

        context_videos = []
        context_aux_videos = []
        for t_idx in history_t:
            frame_ids = self._video_window_indices(int(t_idx))
            context_videos.append(images[torch.as_tensor(frame_ids, dtype=torch.long)])
            if aux_stack is not None:
                context_aux_videos.append(aux_stack[int(t_idx)].unsqueeze(0))
        if context_videos:
            context_videos_t = torch.stack(context_videos, dim=0)
        else:
            context_videos_t = torch.empty(
                (0, self.num_frames, *images.shape[1:]),
                dtype=images.dtype,
            )
        if context_aux_videos:
            context_aux_videos_t = torch.stack(context_aux_videos, dim=0)
        else:
            context_aux_videos_t = torch.empty(
                (0, 1, *images.shape[1:]),
                dtype=images.dtype,
            )

        anchor_frame_ids = self._video_window_indices(anchor_t)
        anchor_video = images[torch.as_tensor(anchor_frame_ids, dtype=torch.long)]
        anchor_aux_video = aux_stack[anchor_t].unsqueeze(0) if aux_stack is not None else torch.empty(
            (0, *images.shape[1:]),
            dtype=images.dtype,
        )
        anchor_state = states[anchor_t]

        target_t = list(range(anchor_t, anchor_t + self.chunk_horizon))
        target_chunk = actions[torch.as_tensor(target_t, dtype=torch.long)]

        if self._state_mean is not None and self._state_std is not None:
            anchor_state = (anchor_state - self._state_mean) / self._state_std
        if self._action_mean is not None and self._action_std is not None:
            target_chunk = (target_chunk - self._action_mean) / self._action_std

        return {
            "prompt": ep["prompt"],
            "context_videos": context_videos_t,
            "context_aux_videos": context_aux_videos_t,
            "context_time_indices": torch.tensor(history_t, dtype=torch.long),
            "anchor_video": anchor_video,
            "anchor_aux_video": anchor_aux_video,
            "anchor_state": anchor_state,
            "anchor_time_idx": torch.tensor(anchor_t, dtype=torch.long),
            "target_chunk": target_chunk,
            "episode_idx": torch.tensor(plan.episode_idx, dtype=torch.long),
        }


def offline_context_collate(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """保留样本字典列表结构的 collate 接口。

    接口对应:
        与 `forward_offline_context_batch(samples=...)` 的输入类型一致，
        不做张量堆叠，交由编码器内部按样本异构长度处理。
    """
    return batch
