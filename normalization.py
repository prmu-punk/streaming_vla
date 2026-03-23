from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


def _to_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    return x.to(device=ref.device, dtype=ref.dtype)


@dataclass
class RTCNormalizer:
    action_mean: torch.Tensor
    action_std: torch.Tensor
    state_mean: torch.Tensor
    state_std: torch.Tensor

    @classmethod
    def from_stats(
        cls,
        *,
        action_mean: torch.Tensor,
        action_std: torch.Tensor,
        state_mean: torch.Tensor,
        state_std: torch.Tensor,
    ) -> "RTCNormalizer":
        return cls(
            action_mean=action_mean.detach().float().cpu(),
            action_std=action_std.detach().float().cpu(),
            state_mean=state_mean.detach().float().cpu(),
            state_std=state_std.detach().float().cpu(),
        )

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RTCNormalizer":
        return cls.from_stats(
            action_mean=payload["action_mean"],
            action_std=payload["action_std"],
            state_mean=payload["state_mean"],
            state_std=payload["state_std"],
        )

    def to_payload(self) -> dict[str, torch.Tensor]:
        return {
            "action_mean": self.action_mean.detach().cpu(),
            "action_std": self.action_std.detach().cpu(),
            "state_mean": self.state_mean.detach().cpu(),
            "state_std": self.state_std.detach().cpu(),
        }

    def normalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = _to_like(self.action_mean, action)
        std = _to_like(self.action_std, action)
        return (action - mean) / std

    def unnormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        mean = _to_like(self.action_mean, action)
        std = _to_like(self.action_std, action)
        return action * std + mean

    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        mean = _to_like(self.state_mean, state)
        std = _to_like(self.state_std, state)
        return (state - mean) / std
