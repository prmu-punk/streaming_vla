# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

import torch
import torch.nn as nn
from typing import Any, Dict, Optional

from oat.model.common.misc import get_autocast_context, str_to_dtype

__all__ = ["LinearLayer"]


class LinearLayer(nn.Module):
    """Linear layer module.
    CANNOT be used as readout when using μP - use LinearHead instead. For that reason,
    the `dim` argument referes to the output dimension and can be used for setting
    muP base shapes.

    Args:
        dim_in: Input dimension size.
        dim_out: Output dimension size.
        weight_init_style: Initialization style for weights ('xavier', 'zero', or 'trunc_normal').
        dtype_override: Optional string to override the tensor data type. If None, the
            current mixed precision state is not changed.
        proj_bias: Whether to learn a bias in the linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        weight_init_style: str = "xavier",
        dtype_override: Optional[str] = None,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.dim_in, self.dim_out = dim_in, dim_out
        self.dtype_override = str_to_dtype(dtype_override)

        self.proj = nn.Linear(self.dim_in, self.dim_out, bias=proj_bias)

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        if self.weight_init_style == "zero":
            nn.init.constant_(self.proj.weight, 0)
        elif self.weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)
        elif self.weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(self.proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        # Bias
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with get_autocast_context(x, self.dtype_override):
            x = self.proj(x)
        return x
