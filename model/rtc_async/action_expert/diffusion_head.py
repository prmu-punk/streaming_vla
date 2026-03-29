from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Protocol

import torch

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class VelocityModel(Protocol):

    def __call__(
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
        ...


@dataclass
class DiffusionKVCache:
    """为扩散采样缓存跨步复用的 VLM KV 条件。"""

    store: dict[Hashable, KVCache] = field(default_factory=dict)
    attention_mask_store: dict[Hashable, torch.Tensor] = field(default_factory=dict)
    prompt_mask_store: dict[Hashable, torch.Tensor] = field(default_factory=dict)
    step_mask_store: dict[Hashable, torch.Tensor] = field(default_factory=dict)

    def put(
        self,
        key: Hashable,
        kv_cache: KVCache | None,
        attention_mask: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        step_mask: torch.Tensor | None = None,
    ) -> None:
        """写入指定键对应的 KV 快照。"""

        if kv_cache is None:
            return
        self.store[key] = [(k.detach(), v.detach()) for k, v in kv_cache]
        if attention_mask is not None:
            self.attention_mask_store[key] = attention_mask.detach()
        if prompt_mask is not None:
            self.prompt_mask_store[key] = prompt_mask.detach()
        if step_mask is not None:
            self.step_mask_store[key] = step_mask.detach()

    def get(
        self, key: Hashable
    ) -> tuple[KVCache | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """读取指定键对应的 KV 快照。"""

        return (
            self.store.get(key),
            self.attention_mask_store.get(key),
            self.prompt_mask_store.get(key),
            self.step_mask_store.get(key),
        )

    def clear(self) -> None:
        """清空缓存。"""

        self.store.clear()
        self.attention_mask_store.clear()
        self.prompt_mask_store.clear()
        self.step_mask_store.clear()


def euler_sample_actions(
    *,
    model: VelocityModel,
    state: torch.Tensor,
    horizon: int,
    action_dim: int,
    num_steps: int,
    kv_cache: KVCache | None = None,
    attention_mask: torch.Tensor | None = None,
    prompt_mask: torch.Tensor | None = None,
    step_mask: torch.Tensor | None = None,
    known_action: torch.Tensor | None = None,
    known_mask: torch.Tensor | None = None,
    kv_cache_store: DiffusionKVCache | None = None,
    kv_cache_key: Hashable | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    使用欧拉积分生成连续动作 chunk。

    kv_cache 参数与 qwen3_stream.export_selected_kv_cache 的返回值直接对接；
    若提供 kv_cache_store + kv_cache_key，则在多次采样间自动读写缓存。
    """

    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    batch_size = state.shape[0]
    device = state.device
    dtype = state.dtype
    if kv_cache is None and kv_cache_store is not None and kv_cache_key is not None:
        kv_cache, cached_attention_mask, cached_prompt_mask, cached_step_mask = kv_cache_store.get(kv_cache_key)
        if attention_mask is None:
            attention_mask = cached_attention_mask
        if prompt_mask is None:
            prompt_mask = cached_prompt_mask
        if step_mask is None:
            step_mask = cached_step_mask
    elif kv_cache is not None and kv_cache_store is not None and kv_cache_key is not None:
        kv_cache_store.put(
            kv_cache_key,
            kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )
    if known_action is not None:
        known_action = known_action.to(device=device, dtype=dtype)
        if known_action.shape != (batch_size, horizon, action_dim):
            raise ValueError(
                f"known_action must be [B, H, D], got {tuple(known_action.shape)}, "
                f"expected {(batch_size, horizon, action_dim)}"
            )
    if known_mask is not None:
        known_mask = known_mask.to(device=device, dtype=torch.bool)
        if known_mask.shape != (batch_size, horizon):
            raise ValueError(f"known_mask must be [B, H], got {tuple(known_mask.shape)}")
    if (known_action is None) != (known_mask is None):
        raise ValueError("known_action and known_mask must be provided together")
    x_t = torch.randn((batch_size, horizon, action_dim), device=device, dtype=dtype, generator=generator)
    if known_action is not None and known_mask is not None:
        x_t = torch.where(known_mask.unsqueeze(-1), known_action, x_t)
    t = torch.zeros((batch_size,), device=device, dtype=dtype)
    dt = 1.0 / float(num_steps)
    for _ in range(num_steps):
        time_chunk = t[:, None].expand(batch_size, horizon)
        if known_action is not None and known_mask is not None:
            time_chunk = torch.where(known_mask, torch.ones_like(time_chunk), time_chunk)
        v_t = model(
            noisy_action=x_t,
            state=state,
            time=time_chunk,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
            prompt_mask=prompt_mask,
            step_mask=step_mask,
        )
        x_t = x_t + dt * v_t
        if known_action is not None and known_mask is not None:
            x_t = torch.where(known_mask.unsqueeze(-1), known_action, x_t)
        t = t + dt
    return x_t
