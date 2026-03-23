from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RTCInpaintingBatch:
    """训练期 RTC 样本包，包含去噪输入、监督目标与有效损失掩码。"""

    x_t: torch.Tensor
    u_t: torch.Tensor
    loss_mask: torch.Tensor
    delay: torch.Tensor
    time: torch.Tensor


def _sample_time(
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z = torch.randn((batch_size,), device=device, dtype=dtype, generator=generator)
    return torch.sigmoid(z)


def build_rtc_inpainting_batch(
    *,
    action: torch.Tensor,
    delay: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> RTCInpaintingBatch:
    """构造训练时 action inpainting 批次，前缀强制已知、后缀参与学习。"""

    if action.dim() != 3:
        raise ValueError(f"action must be [B, H, D], got {tuple(action.shape)}")
    batch_size, horizon, _ = action.shape
    device = action.device
    dtype = action.dtype

    noise = torch.randn(action.shape, device=device, dtype=dtype, generator=generator)
    base_time = _sample_time(
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    u_t = action - noise

    if delay is None:
        delay = torch.zeros((batch_size,), device=device, dtype=torch.long)
    else:
        delay = delay.to(device=device, dtype=torch.long)
    if delay.shape != (batch_size,):
        raise ValueError(f"delay must be [B], got {tuple(delay.shape)}")
    delay = delay.clamp(min=0, max=max(horizon - 1, 0))

    pos = torch.arange(horizon, device=device)[None, :]
    prefix_mask = pos < delay[:, None]
    time = base_time[:, None].expand(batch_size, horizon)
    time = torch.where(prefix_mask, torch.ones_like(time), time)
    x_t = (1 - time[:, :, None]) * noise + time[:, :, None] * action
    loss_mask = ~prefix_mask

    return RTCInpaintingBatch(
        x_t=x_t,
        u_t=u_t,
        loss_mask=loss_mask,
        delay=delay,
        time=time,
    )


def rtc_velocity_loss(
    *,
    pred_u_t: torch.Tensor,
    batch: RTCInpaintingBatch,
) -> torch.Tensor:
    """在 RTC 有效位置上聚合速度场 MSE 损失。"""

    if pred_u_t.shape != batch.u_t.shape:
        raise ValueError(f"pred_u_t shape mismatch: {tuple(pred_u_t.shape)} vs {tuple(batch.u_t.shape)}")
    per_elem = (pred_u_t - batch.u_t).pow(2).mean(dim=-1)
    denom = batch.loss_mask.sum().clamp_min(1).to(per_elem.dtype)
    return (per_elem * batch.loss_mask.to(per_elem.dtype)).sum() / denom
