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


def _kv_seq_dim(x: torch.Tensor) -> int:
    if x.dim() == 4:
        return 2 if x.shape[1] < x.shape[2] else 1
    if x.dim() == 3:
        return 1
    if x.dim() == 2:
        return 1
    raise ValueError(f"Unsupported KV rank: {x.dim()}, shape={tuple(x.shape)}")


def _index_select_seq(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    return x.index_select(_kv_seq_dim(x), positions)


def _align_bool_mask(mask: torch.Tensor, *, kv_len: int, device: torch.device) -> torch.Tensor:
    if mask.dim() != 2:
        raise ValueError(f"mask must be [B, S], got {tuple(mask.shape)}")
    if mask.shape[0] != 1:
        raise ValueError(f"Only batch size 1 is supported for compact KV export, got {mask.shape[0]}")
    out = mask.to(device=device)
    if out.dtype != torch.bool:
        out = out > 0
    seq_len = out.shape[1]
    if seq_len > kv_len:
        out = out[:, -kv_len:]
    elif seq_len < kv_len:
        pad = torch.zeros((out.shape[0], kv_len - seq_len), device=device, dtype=torch.bool)
        out = torch.cat([pad, out], dim=1)
    return out


def export_compact_selected_kv_cache(
    *,
    past_key_values: object,
    selected_layers: Iterable[int],
    prompt_mask: torch.Tensor,
    step_mask: torch.Tensor,
    clone: bool = True,
) -> tuple[KVCache, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = export_selected_kv_cache(
        past_key_values=past_key_values,
        selected_layers=selected_layers,
        clone=False,
    )
    if not selected:
        empty_mask = torch.zeros((1, 0), dtype=torch.bool, device=prompt_mask.device)
        empty_attn = torch.zeros((1, 0), dtype=torch.long, device=prompt_mask.device)
        return [], empty_attn, empty_mask, empty_mask.clone()

    kv_len = int(selected[0][0].shape[_kv_seq_dim(selected[0][0])])
    device = selected[0][0].device
    aligned_prompt_mask = _align_bool_mask(prompt_mask, kv_len=kv_len, device=device)
    aligned_step_mask = _align_bool_mask(step_mask, kv_len=kv_len, device=device)
    keep_mask = aligned_prompt_mask | aligned_step_mask
    positions = torch.nonzero(keep_mask[0], as_tuple=False).flatten()
    compact_attention_mask = torch.ones((1, int(positions.numel())), device=device, dtype=torch.long)
    compact_prompt_mask = aligned_prompt_mask.index_select(1, positions)
    compact_step_mask = aligned_step_mask.index_select(1, positions)

    compact_kv: KVCache = []
    for k, v in selected:
        layer_kv_len = int(k.shape[_kv_seq_dim(k)])
        if layer_kv_len != kv_len:
            raise ValueError(f"Inconsistent KV lengths across layers: expected {kv_len}, got {layer_kv_len}")
        compact_k = _index_select_seq(k, positions)
        compact_v = _index_select_seq(v, positions)
        if clone:
            compact_k = compact_k.clone()
            compact_v = compact_v.clone()
        compact_kv.append((compact_k, compact_v))
    return compact_kv, compact_attention_mask, compact_prompt_mask, compact_step_mask
