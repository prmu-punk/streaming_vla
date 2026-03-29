from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

import torch
import torch.nn as nn

from .diffusion_head import DiffusionKVCache, euler_sample_actions
from .model import ActionExpertBackbone, ActionExpertConfig

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class ActionExpertRunnerConfig:
    """动作专家运行配置。"""

    state_dim: int
    action_dim: int
    horizon: int
    cond_dim: int
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    norm_eps: float = 1e-6
    ffn_multiple_of: int = 256
    ffn_dim_multiplier: float | None = None
    num_inference_steps: int = 5


class ActionExpertRunner(nn.Module):
    """动作专家统一训练/采样封装。"""

    def __init__(self, config: ActionExpertRunnerConfig) -> None:
        """构建动作专家主干与 KV 缓存容器。"""

        super().__init__()
        self.config = config
        self.kv_cache_store = DiffusionKVCache()
        self.backbone = ActionExpertBackbone(
            ActionExpertConfig(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                horizon=config.horizon,
                cond_dim=config.cond_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                num_heads=config.num_heads,
                mlp_ratio=config.mlp_ratio,
                norm_eps=config.norm_eps,
                ffn_multiple_of=config.ffn_multiple_of,
                ffn_dim_multiplier=config.ffn_dim_multiplier,
            )
        )

    def forward(
        self,
        *,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        step_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """训练路径前向，输出速度场预测。"""

        return self.backbone(
            noisy_action=noisy_action,
            state=state,
            time=time,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )

    @torch.inference_mode()
    def sample(
        self,
        *,
        state: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        step_mask: torch.Tensor | None = None,
        known_action: torch.Tensor | None = None,
        known_mask: torch.Tensor | None = None,
        kv_cache_key: Hashable | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """推理路径采样动作 chunk，并可复用 KV 条件缓存。"""

        return euler_sample_actions(
            model=self,
            state=state,
            horizon=self.config.horizon,
            action_dim=self.config.action_dim,
            num_steps=self.config.num_inference_steps,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
            known_action=known_action,
            known_mask=known_mask,
            kv_cache_store=self.kv_cache_store,
            kv_cache_key=kv_cache_key,
            generator=generator,
        )
