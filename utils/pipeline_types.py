from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class StepPacket:
    step_id: int
    frames: ArrayLike
    aux_frames: Optional[ArrayLike]
    state: torch.Tensor
    ts: Optional[int]
    num_frames: int


@dataclass
class EncodedStepPacket:
    step_id: int
    input_ids: torch.LongTensor
    attention_mask: torch.Tensor
    pixel_values_videos: Optional[torch.FloatTensor]
    video_grid_thw: torch.LongTensor
    state_tokens: torch.Tensor
    precomputed_video_outputs: Any = None


@dataclass
class TokenPacket:
    step_id: int
    action_token_ids: torch.LongTensor
    ended_by_eos: bool


@dataclass
class ChunkPacket:
    step_id: int
    action_token_ids: torch.LongTensor
    action_chunk: torch.Tensor
