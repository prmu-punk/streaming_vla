from __future__ import annotations

from typing import Iterable

import torch

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


def _cache_to_layer_list(past_key_values: object) -> KVCache:
    """将模型 cache 结构标准化为 layer-wise (K, V) 列表。"""

    if past_key_values is None:
        return []
    layers = []
    for layer in past_key_values:
        if not isinstance(layer, tuple) or len(layer) < 2:
            raise ValueError("Unexpected cache layer format.")
        k, v = layer[0], layer[1]
        if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
            raise ValueError("Cache k/v must be tensors.")
        layers.append((k, v))
    return layers


def export_selected_kv_cache(
    *,
    past_key_values: object,
    selected_layers: Iterable[int],
    clone: bool = True,
) -> KVCache:
    """
    导出指定层 KV cache。

    返回值协议与 action_expert 的 kv_cache 入参保持一致，可直接注入。
    """

    layers = _cache_to_layer_list(past_key_values)
    if not layers:
        return []
    selected = []
    total = len(layers)
    for idx in selected_layers:
        if idx < 0 or idx >= total:
            raise ValueError(f"selected layer index out of range: {idx}, total={total}")
        k, v = layers[idx]
        if clone:
            k = k.clone()
            v = v.clone()
        selected.append((k, v))
    return selected
