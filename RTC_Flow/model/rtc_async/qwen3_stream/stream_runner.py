from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from model.qwen3_vl.stream_runner import Qwen3VLStreamRunner

from .kv_export import export_selected_kv_cache

KVCache = list[tuple[torch.Tensor, torch.Tensor]]


@dataclass
class StreamConditionSnapshot:
    """动作专家采样所需的流式条件快照。"""

    kv_cache: KVCache
    attention_mask: torch.Tensor | None


class RTCQwen3StreamAdapter:
    """将原始流式 runner 适配为动作专家可消费的条件导出器。"""

    def __init__(self, runner: Qwen3VLStreamRunner) -> None:
        """包装原始 stream runner，提供 RTC 友好的条件导出接口。"""

        self.runner = runner

    def export_condition_snapshot(
        self,
        *,
        selected_layers: Iterable[int],
        clone: bool = True,
    ) -> StreamConditionSnapshot:
        """导出与动作专家对齐的 KV/attention 条件。"""

        kv_cache = export_selected_kv_cache(
            past_key_values=self.runner.state.past_key_values,
            selected_layers=selected_layers,
            clone=clone,
        )
        attention_mask = self.runner.state.attention_mask
        if clone:
            if attention_mask is not None:
                attention_mask = attention_mask.clone()
        return StreamConditionSnapshot(
            kv_cache=kv_cache,
            attention_mask=attention_mask,
        )
