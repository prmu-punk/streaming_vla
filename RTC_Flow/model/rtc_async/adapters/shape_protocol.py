from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionShapeSpec:
    """动作张量协议，约束 horizon 与 action_dim。"""

    horizon: int
    action_dim: int


@dataclass(frozen=True)
class StateShapeSpec:
    """状态张量协议，约束 state_dim。"""

    state_dim: int


@dataclass(frozen=True)
class KVShapeSpec:
    """KV 缓存协议，约束导出层数。"""

    n_layers: int
