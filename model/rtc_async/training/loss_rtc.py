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


def _sample_delay(
    *,
    batch_size: int,
    simulated_delay: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """按指数衰减分布采样每个样本的 simulated delay。"""

    if simulated_delay <= 0:
        return torch.zeros((batch_size,), device=device, dtype=torch.long)
    weights = torch.exp(torch.arange(simulated_delay, 0, -1, device=device, dtype=torch.float32))
    weights = weights / weights.sum()
    return torch.multinomial(weights, num_samples=batch_size, replacement=True, generator=generator)


def build_rtc_inpainting_batch(
    *,
    action: torch.Tensor,
    simulated_delay: int | None,
    generator: torch.Generator | None = None,
) -> RTCInpaintingBatch:
    """构造训练时 action inpainting 批次，前缀强制已知、后缀参与学习。"""

    if action.dim() != 3:
        raise ValueError(f"action must be [B, H, D], got {tuple(action.shape)}")
    batch_size, horizon, _ = action.shape
    device = action.device
    dtype = action.dtype

    noise = torch.randn(action.shape, device=device, dtype=dtype, generator=generator)
    time = torch.rand((batch_size,), device=device, dtype=dtype, generator=generator)
    u_t = action - noise

    if simulated_delay is None or simulated_delay <= 0:
        time_chunk = time[:, None, None]
        x_t = (1 - time_chunk) * noise + time_chunk * action
        loss_mask = torch.ones((batch_size, horizon), device=device, dtype=torch.bool)
        delay = torch.zeros((batch_size,), device=device, dtype=torch.long)
        return RTCInpaintingBatch(
            x_t=x_t,
            u_t=u_t,
            loss_mask=loss_mask,
            delay=delay,
            time=time,
        )

    delay = _sample_delay(
        batch_size=batch_size,
        simulated_delay=simulated_delay,
        device=device,
        generator=generator,
    )
    pos = torch.arange(horizon, device=device)[None, :]
    prefix_mask = pos < delay[:, None]

    time_expanded = time[:, None].expand(batch_size, horizon)
    time_expanded = torch.where(prefix_mask, torch.ones_like(time_expanded), time_expanded)
    x_t = (1 - time_expanded[:, :, None]) * noise + time_expanded[:, :, None] * action
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
