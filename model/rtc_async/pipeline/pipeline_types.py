from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch


ArrayLike = Union[np.ndarray, torch.Tensor]
KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class StepPacket:
    step_id: int
    frames: ArrayLike
    aux_frames: Optional[ArrayLike]
    state: torch.Tensor
    ts_ms: Optional[int]
    num_frames: int


@dataclass
class ContextPacket:
    step_id: int
    state: torch.Tensor
    ts_ms: Optional[int]
    kv_cache: KVCache
    attention_mask: torch.Tensor
    prompt_mask: torch.Tensor
    step_mask: torch.Tensor


@dataclass
class ActionPacket:
    step_id: int
    ts_ms: Optional[int]
    step_delay_steps: int
    action_chunk: torch.Tensor


@dataclass
class ExecutePacket:
    step_id: int
    ts_ms: Optional[int]
    step_delay_steps: int
    prefix_len: int
    action_chunk: torch.Tensor
    stitched_chunk: torch.Tensor
    execute_chunk: torch.Tensor
