from __future__ import annotations

import numpy as np
import torch


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def make_video_tensor(
    frames: np.ndarray | torch.Tensor,
    num_frames: int,
) -> torch.Tensor:
    if isinstance(frames, np.ndarray):
        frames_t = torch.from_numpy(frames)
    else:
        frames_t = frames
    if frames_t.dim() == 3:
        frames_t = frames_t.unsqueeze(0)
    if frames_t.shape[-1] == 3:
        frames_t = frames_t.permute(0, 3, 1, 2)
    if frames_t.shape[0] < num_frames:
        repeat = num_frames - frames_t.shape[0]
        frames_t = torch.cat([frames_t, frames_t[-1:].repeat(repeat, 1, 1, 1)], dim=0)
    elif frames_t.shape[0] > num_frames:
        frames_t = frames_t[:num_frames]
    return frames_t


def decode_token_ids(tokenizer, token_ids: torch.LongTensor) -> str:
    token_ids = token_ids.detach().to("cpu")
    return tokenizer.decode(
        token_ids.tolist(),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
