# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from typing import Optional, Union, Callable

from oat.tokenizer.oat.model.norm import Fp32LayerNorm
from oat.tokenizer.oat.model.block import (
    Block,
    BlockAdaLN,
)

__all__ = ["Transformer"]


def get_from_dict(data_dict, key, default=None):
    return data_dict[key] if key is not None else default


class Transformer(nn.Module):
    """
    Transformer module using FlexAttention.

    Args:
        dim: Dimension of the input and output features.
        depth: Number of Transformer blocks in the model.
        head_dim: Dimension of each attention head.
        mlp_ratio: Ratio of the hidden dimension size to the input dimension size in the MLP layers.
        qkv_bias: Whether to use bias in the Q, K, V projections of the attention layers.
        proj_bias: Whether to use bias in the projection layers of the attention.
        mlp_bias: Whether to use bias in the MLP layers.
        drop: Dropout rate applied to attention and MLP layers.
        drop_path_rate: Dropout rate for stochastic depth (drop path).
        act_layer: Activation layer used in the MLPs.
        norm_layer: Normalization layer used before attention and MLP layers.
        gated_mlp: Whether to use gated MLP layers in the transformer blocks.
        qk_norm: Whether to apply normalization to the Q and K projections.
        weight_init_style: Style of weight initialization ('xavier', 'trunc_normal').
        zero_init_query_proj: Whether to zero-initialize the query projection layer.
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
    """

    def __init__(
        self,
        dim: int = 768,
        depth: int = 12,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = False,
        mlp_bias: bool = False,
        drop: float = 0.0,
        drop_path_rate: float = 0.0,
        act_layer: nn.Module = nn.SiLU,
        norm_layer: Union[nn.Module, partial] = partial(
            Fp32LayerNorm, bias=False, elementwise_affine=False
        ),
        gated_mlp: bool = True,
        qk_norm: bool = True,
        weight_init_style: str = "xavier",
        zero_init_query_proj: bool = False,
        use_adaLN: bool = False,
        adaLN_expansion: int = 1,
    ):
        super().__init__()

        block_fn = BlockAdaLN if use_adaLN else Block

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                block_fn(
                    dim=dim,
                    head_dim=head_dim,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    mlp_bias=mlp_bias,
                    drop=drop,
                    drop_path=dpr[i],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    gated_mlp=gated_mlp,
                    qk_norm=qk_norm,
                    adaLN_expansion=adaLN_expansion,
                )
                for i in range(depth)
            ]
        )

        # Weight init
        self.weight_init_style = weight_init_style
        self.zero_init_query_proj = zero_init_query_proj
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        for name, m in self.named_modules():
            # Linear
            if isinstance(m, nn.Linear):
                # Weight
                if "adaLN_modulation" in name:
                    nn.init.constant_(m.weight, 0)
                elif "wq" in name and self.zero_init_query_proj:
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

    def forward(self, 
        x: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
        adaLN_emb: Optional[torch.Tensor] = None,
        adaLN_packing_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                block_mask=block_mask,
                adaLN_emb=adaLN_emb,
                adaLN_packing_fn=adaLN_packing_fn,
            )
        return x

