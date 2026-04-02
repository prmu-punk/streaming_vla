from __future__ import annotations

from dataclasses import dataclass

import torch

@dataclass
class RTCInpaintingBatch:
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
    return torch.rand((batch_size,), device=device, dtype=dtype, generator=generator)

def build_rtc_inpainting_batch(
    *,
    action: torch.Tensor,
    delay: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> RTCInpaintingBatch:

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
    time_per_pos = base_time[:, None].expand(batch_size, horizon)
    time_per_pos = torch.where(prefix_mask, torch.ones_like(time_per_pos), time_per_pos)
    x_t = (1 - time_per_pos[:, :, None]) * noise + time_per_pos[:, :, None] * action
    loss_mask = ~prefix_mask

    return RTCInpaintingBatch(
        x_t=x_t,
        u_t=u_t,
        loss_mask=loss_mask,
        delay=delay,
        time=time_per_pos,
    )

def rtc_velocity_loss(
    *,
    pred_u_t: torch.Tensor,
    batch: RTCInpaintingBatch,
) -> torch.Tensor:

    if pred_u_t.shape != batch.u_t.shape:
        raise ValueError(f"pred_u_t shape mismatch: {tuple(pred_u_t.shape)} vs {tuple(batch.u_t.shape)}")
    per_elem = (pred_u_t - batch.u_t).pow(2).mean(dim=-1)
    denom = batch.loss_mask.sum().clamp_min(1).to(per_elem.dtype)
    return (per_elem * batch.loss_mask.to(per_elem.dtype)).sum() / denom


def rtc_denoise_mse(
    *,
    pred_u_t: torch.Tensor,
    batch: RTCInpaintingBatch,
) -> torch.Tensor:

    if pred_u_t.shape != batch.u_t.shape:
        raise ValueError(f"pred_u_t shape mismatch: {tuple(pred_u_t.shape)} vs {tuple(batch.u_t.shape)}")
    if batch.time.dim() != 2:
        raise ValueError(f"batch.time must be [B, H], got {tuple(batch.time.shape)}")
    one_minus_t = 1.0 - batch.time[:, :, None]
    pred_action = batch.x_t + one_minus_t * pred_u_t
    target_action = batch.x_t + one_minus_t * batch.u_t
    per_elem = (pred_action - target_action).pow(2).mean(dim=-1)
    denom = batch.loss_mask.sum().clamp_min(1).to(per_elem.dtype)
    return (per_elem * batch.loss_mask.to(per_elem.dtype)).sum() / denom
