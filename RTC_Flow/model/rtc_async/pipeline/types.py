from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
import torch

ArrayLike = Union[NDArray[np.generic], torch.Tensor]


@dataclass
class RTCStepPacket:
    """封装单步输入样本，供流水线排队与后续动作生成使用。"""

    step_id: int
    frames: ArrayLike
    state: torch.Tensor
    ts: Optional[int]
    num_frames: int


@dataclass
class RTCChunkPacket:
    """封装某一步的动作 chunk 与可执行子段。"""

    step_id: int
    action_chunk: torch.Tensor
    execute_chunk: torch.Tensor
    inference_delay: int
    execute_horizon: int
