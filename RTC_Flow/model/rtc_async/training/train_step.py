from __future__ import annotations

import torch


def build_token_rtc_label_mask(
    *,
    labels: torch.LongTensor,
    action_token_spans: torch.LongTensor,
    max_delay_tokens: int,
    generator: torch.Generator | None = None,
) -> torch.LongTensor:
    """遗留兼容接口：token 版 RTC mask 在 kv-cache 连续动作范式下已废弃。"""

    if labels.dim() != 2:
        raise ValueError(f"labels must be [B, L], got {tuple(labels.shape)}")
    if action_token_spans.dim() != 2 or action_token_spans.shape[1] != 2:
        raise ValueError(
            f"action_token_spans must be [B,2] with [start,end), got {tuple(action_token_spans.shape)}"
        )
    if max_delay_tokens <= 0:
        return labels

    raise RuntimeError(
        "build_token_rtc_label_mask is deprecated for rtc_async kv-cache conditioning. "
        "Use action-space RTC training: build_rtc_inpainting_batch + rtc_velocity_loss."
    )
