from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Protocol

import torch

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


class VelocityModel(Protocol):
    """速度场模型协议，约束采样器调用签名。"""

    def __call__(
        self,
        *,
        noisy_action: torch.Tensor,
        state: torch.Tensor,
        time: torch.Tensor,
        kv_cache: KVCache | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """根据当前噪声状态预测速度场。"""

        ...


@dataclass
class DiffusionKVCache:
    """为扩散采样缓存跨步复用的 VLM KV 条件。"""

    store: dict[Hashable, KVCache] = field(default_factory=dict)
    attention_mask_store: dict[Hashable, torch.Tensor] = field(default_factory=dict)

    def put(
        self,
        key: Hashable,
        kv_cache: KVCache | None,
        attention_mask: torch.Tensor | None = None,
    ) -> None:
        """写入指定键对应的 KV 快照。"""

        if kv_cache is None:
            return
        self.store[key] = [(k.detach(), v.detach()) for k, v in kv_cache]
        if attention_mask is not None:
            self.attention_mask_store[key] = attention_mask.detach()

    def get(self, key: Hashable) -> tuple[KVCache | None, torch.Tensor | None]:
        """读取指定键对应的 KV 快照。"""

        return self.store.get(key), self.attention_mask_store.get(key)

    def clear(self) -> None:
        """清空缓存。"""

        self.store.clear()
        self.attention_mask_store.clear()


def euler_sample_actions(
    *,
    model: VelocityModel,
    state: torch.Tensor,
    horizon: int,
    action_dim: int,
    num_steps: int,
    kv_cache: KVCache | None = None,
    attention_mask: torch.Tensor | None = None,
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
        kv_cache, cached_attention_mask = kv_cache_store.get(kv_cache_key)
        if attention_mask is None:
            attention_mask = cached_attention_mask
    elif kv_cache is not None and kv_cache_store is not None and kv_cache_key is not None:
        kv_cache_store.put(kv_cache_key, kv_cache, attention_mask)
    x_t = torch.randn((batch_size, horizon, action_dim), device=device, dtype=dtype, generator=generator)
    t = torch.zeros((batch_size,), device=device, dtype=dtype)
    dt = 1.0 / float(num_steps)
    for _ in range(num_steps):
        v_t = model(
            noisy_action=x_t,
            state=state,
            time=t,
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
        x_t = x_t + dt * v_t
        t = t + dt
    return x_t
