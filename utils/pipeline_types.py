from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]


@dataclass
class StepPacket:
    step_id: int
    frames: ArrayLike
    state: torch.Tensor
    ts: Optional[int]
    num_frames: int


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
