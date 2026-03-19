from __future__ import annotations

import torch
import torch.nn as nn


class StateConditionAdapter(nn.Module):
    """状态条件适配器。"""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """将原始 state 条件映射到动作专家所需隐藏维度。"""

        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向映射 state 条件张量。"""

        return self.net(x)
