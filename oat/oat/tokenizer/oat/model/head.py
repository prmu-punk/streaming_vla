# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from functools import partial
import einops
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Callable, Dict, Optional, Union

from oat.model.common.misc import get_autocast_context, str_to_dtype
from oat.tokenizer.oat.model.norm import Fp32LayerNorm

__all__ = ["LinearHead", "MLPHead"]


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def expand_emb_from_ps(emb, ps):
    return torch.cat(
        [einops.repeat(emb_i, "d -> 1 n d", n=shape.numel()) for emb_i, shape in zip(emb, ps)],
        dim=1,
    )


class LinearHead(nn.Module):
    """
    Linear head module with optional adaLN modulation.

    Args:
        dim: Input dimension size.
        dim_out: Output dimension size.
        weight_init_style: Initialization style for weights ('zero', 'xavier', or 'trunc_normal').
        norm_layer: Optional normalization layer applied before the projection.
        dtype_override: Optional string to override the tensor data type.
        use_adaLN: Whether to use adaLN-Zero modulation.
        adaLN_bias: Whether to use bias in the adaLN-Zero modulation layer.
        proj_bias: Whether to use bias in the projection layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        weight_init_style: str = "zero",
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        use_adaLN: bool = False,
        adaLN_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        self.dim_in, self.dim_out = dim, dim_out
        self.dtype_override = str_to_dtype(dtype_override)

        # Optional LayerNorm
        self.norm = norm_layer(self.dim_in) if norm_layer is not None else nn.Identity()

        # Optional adaLN-Zero
        if use_adaLN:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(self.dim_in, 2 * self.dim_in, bias=adaLN_bias)
            )

        # Linear projection head.
        self.proj = nn.Linear(self.dim_in, self.dim_out, bias=proj_bias)

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if self.weight_init_style == "zero" or "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_default(self, x: torch.Tensor) -> torch.Tensor:
        with get_autocast_context(x, self.dtype_override):
            x = self.proj(self.norm(x))
        return x

    def forward_adaLN(
        self, x: torch.Tensor, adaLN_emb: torch.Tensor, adaLN_packing_fn: Callable
    ) -> torch.Tensor:
        with get_autocast_context(x, self.dtype_override):
            x = self.norm(x)
            shift, scale = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb)).chunk(2, dim=-1)
            x = modulate(x, shift, scale)
            x = self.proj(x)
        return x

    @torch.compiler.disable
    def forward(self, 
        x: torch.Tensor,
        adaLN_emb: Optional[torch.Tensor] = None,
        adaLN_packing_fn: Optional[Callable] = None,
    ) -> torch.Tensor:

        if adaLN_emb is not None:
            if adaLN_packing_fn is None:
                adaLN_packing_fn = partial(expand_emb_from_ps, ps=[torch.Size([x.shape[1]])])
            x = self.forward_adaLN(x, adaLN_emb, adaLN_packing_fn)
        else:
            x = self.forward_default(x)

        return x


class MLPHead(nn.Module):
    """
    MLP head module.

    Args:
        dim: Input dimension size.
        dim_out: Output dimension size.
        num_layers: Number of MLP layers
        dim_hidden_ratio: MLP hidden dimension ratio.
        act_layer: Activation layer used in the MLP.
        weight_init_style: Initialization style for weights ('xavier', or 'trunc_normal').
        zero_init_out_proj: Whether or not to zero-init the final out projection layer.
        norm_layer: Optional normalization layer applied before the projection.
        dtype_override: Optional string to override the tensor data type.
        bias: Whether to use bias in the linear layers.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        num_layers: int,
        dim_hidden_ratio: int = 4.0,
        act_layer: nn.Module = nn.SiLU,
        weight_init_style: str = "xavier",
        zero_init_out_proj: bool = True,
        norm_layer: Optional[Union[partial, nn.Module]] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        dtype_override: Optional[str] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.dim_in, self.dim_out = dim, dim_out
        self.dim_hidden = int(dim_hidden_ratio * dim)
        self.num_layers = num_layers
        self.act_layer = act_layer
        self.dtype_override = str_to_dtype(dtype_override)
        self.bias = bias

        mlp_layers = []

        # Optional LayerNorm
        if norm_layer is not None:
            mlp_layers.append(norm_layer(self.dim_in))

        # Input projection
        mlp_layers.append(nn.Linear(self.dim_in, self.dim_hidden, bias=bias))
        mlp_layers.append(act_layer())

        # Hidden layers
        assert num_layers >= 2
        for _ in range(num_layers - 2):
            mlp_layers.append(nn.Linear(self.dim_hidden, self.dim_hidden, bias=bias))
            mlp_layers.append(act_layer())

        # Output projection head. Using custom layer for μP.
        self.out_proj = nn.Linear(self.dim_hidden, self.dim_out, bias=bias)

        # Full MLP without output projection
        self.mlp = nn.Sequential(*mlp_layers)

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_out_proj = zero_init_out_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "out_proj" in name:
                    nn.init.constant_(m.weight, 0)
                elif self.weight_init_style == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                elif self.weight_init_style == "trunc_normal":
                    nn.init.trunc_normal_(m.weight, std=0.02)
                else:
                    raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
                # Bias
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # LayerNorm
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        with get_autocast_context(x, self.dtype_override):
            x = self.out_proj(self.mlp(x))
        return x
