from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PreTrainedConfig


class DynamicLayer(CacheLayerMixin):
    """
    A cache layer that grows dynamically as more tokens are generated.
    It stores the key and value states as tensors of shape [batch_size, num_heads, seq_len, head_dim].
    """

    is_sliding = False

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.keys = torch.cat([self.keys, key_states], dim=-2)
        self.values = torch.cat([self.values, value_states], dim=-2)
        return self.keys, self.values

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        if not self.is_initialized or self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)

        if self.get_seq_length() <= max_length:
            return

        self.keys = self.keys[..., :max_length, :]
        self.values = self.values[..., :max_length, :]

    def drop_left(self, num_tokens: int) -> None:
        if not self.is_initialized or num_tokens <= 0:
            return
        seq_len = self.get_seq_length()
        if num_tokens >= seq_len:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]
            return
        self.keys = self.keys[..., num_tokens:, :]
        self.values = self.values[..., num_tokens:, :]

    def select_indices(self, keep_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        if keep_idx.numel() == 0:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]
            return
        self.keys = self.keys.index_select(-2, keep_idx.to(self.keys.device))
        self.values = self.values.index_select(-2, keep_idx.to(self.values.device))


class DynamicSlidingWindowLayer(DynamicLayer):
    """
    A cache layer that grows dynamically as more tokens are generated, up until the sliding window size.
    It stores the key and value states as tensors of shape [batch_size, num_heads, min(seq_len, sliding_window), head_dim].
    """

    is_sliding = True

    def __init__(self, sliding_window: int):
        super().__init__()
        self.sliding_window = sliding_window
        self.cumulative_length = 0
        self._sliding_window_tensor = torch.tensor(self.sliding_window, dtype=torch.long)

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        super().lazy_initialization(key_states, value_states)
        self._sliding_window_tensor = self._sliding_window_tensor.to(self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        self.cumulative_length += key_states.shape[-2]

        full_key_states = torch.cat([self.keys, key_states], dim=-2)
        full_value_states = torch.cat([self.values, value_states], dim=-2)
        self.keys = full_key_states[:, :, -self.sliding_window + 1 :, :]
        self.values = full_value_states[:, :, -self.sliding_window + 1 :, :]

        return full_key_states, full_value_states

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        query_length = cache_position.shape[0]
        is_full = self.cumulative_length >= self.sliding_window

        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length

        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sliding_window

    def crop(self, max_length: int) -> None:
        if self.get_seq_length() >= self.sliding_window:
            raise ValueError(
                "Cannot `crop` a `DynamicSlidingWindowLayer` after it has seen more tokens than its"
                "sliding window (otherwise some states are lost)"
            )
        super().crop(max_length)
        self.cumulative_length = self.keys.shape[-2]

    def drop_left(self, num_tokens: int) -> None:
        if not self.is_initialized or num_tokens <= 0:
            return
        self.cumulative_length = max(self.cumulative_length - num_tokens, 0)
        seq_len = self.keys.shape[-2] if self.keys is not None else 0
        if seq_len == 0:
            return
        if num_tokens >= seq_len:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]
            return
        self.keys = self.keys[..., num_tokens:, :]
        self.values = self.values[..., num_tokens:, :]

    def select_indices(self, keep_idx: torch.LongTensor) -> None:
        if not self.is_initialized:
            return
        if keep_idx.numel() == 0:
            self.keys = self.keys[..., :0, :]
            self.values = self.values[..., :0, :]
            self.cumulative_length = 0
            return
        self.keys = self.keys.index_select(-2, keep_idx.to(self.keys.device))
        self.values = self.values.index_select(-2, keep_idx.to(self.values.device))
        self.cumulative_length = self.keys.shape[-2]


class DynamicCache(Cache):
    """
    A cache that grows dynamically as more tokens are generated. This is the default for generative models.
    It stores the key and value states as a list of CacheLayer, one for each layer.
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor | None, ...]] | None = None,
        config: PreTrainedConfig | None = None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        layers: list[CacheLayerMixin] = []
        if config is not None:
            decoder_config = config.get_text_config(decoder=True)
            sliding_window = getattr(decoder_config, "sliding_window", None) or getattr(
                decoder_config, "attention_chunk_size", None
            )
            layer_types = getattr(decoder_config, "layer_types", None)
            if layer_types is None:
                layer_types = [
                    "sliding_attention" if sliding_window is not None else "full_attention"
                    for _ in range(decoder_config.num_hidden_layers)
                ]
            if hasattr(decoder_config, "num_kv_shared_layers"):
                layer_types = layer_types[: -decoder_config.num_kv_shared_layers]

            for layer_type in layer_types:
                if layer_type in ("sliding_attention", "chunked_attention"):
                    layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                else:
                    layers.append(DynamicLayer())

        if ddp_cache_data is not None:
            for layer_idx, kv_and_optional_sliding in enumerate(ddp_cache_data):
                if config is None:
                    sliding_window_tensor = kv_and_optional_sliding[2] if len(kv_and_optional_sliding) == 3 else None
                    if sliding_window_tensor is not None:
                        sliding_window = sliding_window_tensor[0].item()
                        layers.append(DynamicSlidingWindowLayer(sliding_window=sliding_window))
                    else:
                        layers.append(DynamicLayer())
                _, _ = layers[layer_idx].update(kv_and_optional_sliding[0], kv_and_optional_sliding[1])

        if len(layers) == 0:
            super().__init__(
                layer_class_to_replicate=DynamicLayer,
                offloading=offloading,
                offload_only_non_sliding=offload_only_non_sliding,
            )
        else:
            super().__init__(layers=layers, offloading=offloading, offload_only_non_sliding=offload_only_non_sliding)

    def __iter__(self):
        for layer in self.layers:
            yield layer.keys, layer.values, getattr(layer, "_sliding_window_tensor", None)

    def drop_left(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            return
        for layer in self.layers:
            drop_fn = getattr(layer, "drop_left", None)
            if drop_fn is not None:
                drop_fn(num_tokens)

    def select_indices(self, keep_idx: torch.LongTensor) -> None:
        if keep_idx.numel() == 0:
            for layer in self.layers:
                select_fn = getattr(layer, "select_indices", None)
                if select_fn is not None:
                    select_fn(keep_idx)
            return
        for layer in self.layers:
            select_fn = getattr(layer, "select_indices", None)
            if select_fn is None:
                raise ValueError(f"Layer {layer} does not support select_indices")
            select_fn(keep_idx)

    def select_mask(self, keep_mask: torch.BoolTensor) -> None:
        if keep_mask.numel() == 0:
            self.select_indices(keep_mask.new_zeros((0,), dtype=torch.long))
            return
        keep_idx = torch.nonzero(keep_mask, as_tuple=False).flatten()
        self.select_indices(keep_idx)

    def drop_indices(self, drop_idx: torch.LongTensor, seq_len: int) -> None:
        if drop_idx.numel() == 0:
            return
        device = drop_idx.device
        keep_mask = torch.ones(seq_len, dtype=torch.bool, device=device)
        keep_mask[drop_idx] = False
        self.select_mask(keep_mask)


__all__ = ["Cache", "DynamicCache", "DynamicLayer", "DynamicSlidingWindowLayer"]
