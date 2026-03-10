# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from functools import lru_cache
from typing import List, Literal, Optional

from oat.model.common.misc import is_power_of_two, powers_of_two


__all__ = ["MaskedNestedDropout"]


@lru_cache(maxsize=128)
def _compute_power_biased_weights(sample_horizon: int, power: float, device_str: str) -> torch.Tensor:
    """Compute power-biased weights: P(k) = k^power / sum(1^power..N^power)
    
    Higher power gives stronger bias toward full tokens:
    - power=1.0 (linear): ~40% for full tokens when N=4
    - power=2.0 (quadratic): ~53% for full tokens when N=4
    - power=3.0 (cubic): ~70% for full tokens when N=4
    
    Args:
        sample_horizon: Number of tokens
        power: Exponent for the power law (1.0=linear, 2.0=quadratic, 3.0=cubic)
        device_str: Device string (e.g., 'cuda:0', 'cpu') - must be string for caching
    
    Returns:
        Weights tensor on the specified device
    """
    weights = torch.arange(1, sample_horizon + 1, dtype=torch.float32, device=device_str) ** power
    weights = weights / weights.sum()
    return weights


class MaskedNestedDropout(nn.Module):
    """
    Module that randomly drops tokens of the given tensors in a nested fashion, i.e.
    performs nested dropout / Matryoshka sampling. Simply replaces dropped tokens with
    a learnable mask token.

    Args:
        dim: Dimension size of the mask token.
        size_sampling_mode: Method to sample the number of tokens to randomly drop.
            - "disable": No dropout
            - "uniform": Uniform probability across all token counts
            - "pow2": Sample only power-of-2 token counts
            - "uniform_pow2": Uniform sampling rounded up to nearest power of 2
            - "{power}_biased": Power-law biased sampling toward full tokens
    """

    def __init__(
        self,
        dim: int,
        size_sampling_mode: str = "uniform",
    ):
        super().__init__()
        self.dim = dim
        self.size_sampling_mode = size_sampling_mode

        if self.size_sampling_mode != "disable":
            self.dropout_mask_token = nn.Parameter(torch.randn(self.dim), requires_grad=True)
            trunc_normal_(self.dropout_mask_token, std=0.02)

    def sample_keep_k(self, batch_size: int, sample_horizon: int, device: torch.device):
        if self.size_sampling_mode == "uniform":
            keep_ks = torch.randint(
                low=1, high=sample_horizon + 1, size=(batch_size,), device=device
            )

        elif self.size_sampling_mode == "pow2":
            assert is_power_of_two(sample_horizon)
            pow2_vals = torch.tensor(powers_of_two(1, sample_horizon), device=device)
            idx = torch.randint(0, len(pow2_vals), (batch_size,), device=device)
            keep_ks = pow2_vals[idx]

        elif self.size_sampling_mode == "uniform_pow2":
            ks = torch.randint(1, sample_horizon + 1, (batch_size,), device=device)
            # Round each up to next power of two
            keep_ks = torch.where(
                (ks & (ks - 1)) == 0,
                ks,
                1 << (ks - 1).bit_length(),
            )
        
        elif self.size_sampling_mode.endswith("_biased"):
            power_map = {
                "linear_biased": 1.0,
                "quadratic_biased": 2.0,
                "cubic_biased": 3.0,
            }
            power = power_map[self.size_sampling_mode]
            weights = _compute_power_biased_weights(sample_horizon, power, str(device))
            indices = torch.multinomial(weights, batch_size, replacement=True)
            keep_ks = indices + 1  # +1 because indices are 0-based but we want 1-based
            
        else:
            raise ValueError(f"size_sampling_mode {self.size_sampling_mode} not defined.")

        return keep_ks

    @torch.compiler.disable
    def forward(self, 
        x: torch.Tensor,
        eval_keep_k: Optional[List[int]] = None
    ) -> torch.Tensor:
        if self.size_sampling_mode == "disable":
            return x

        B, N, _ = x.shape
        if self.training:
            keep_ks = self.sample_keep_k(B, N, x.device)
            mask = keep_ks.unsqueeze(1) <= torch.arange(N, device=x.device).unsqueeze(0)  # (B, N)
            x[mask] = self.dropout_mask_token
        else:
            if eval_keep_k is not None:
                mask = torch.tensor(
                    eval_keep_k, device=x.device
                ).unsqueeze(1) <= torch.arange(N, device=x.device).unsqueeze(0)  # (B, N)
                x[mask] = self.dropout_mask_token
        return x
