# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from functools import partial

import torch.nn as nn
import torch.nn.functional as F

from oat.tokenizer.oat.model.attention import SelfAttention
from oat.tokenizer.oat.model.drop_path import DropPath
from oat.tokenizer.oat.model.mlp import GatedMlp, Mlp
from oat.tokenizer.oat.model.norm import Fp32LayerNorm

__all__ = ["Block", "BlockAdaLN"]


def modulate(x, shift, scale):
    return x * (1.0 + scale) + shift


def expand_to_padded_seq(emb, padded_seq):
    N_emb, N_seq = emb.shape[1], padded_seq.shape[1]
    num_padding_tokens = N_seq - N_emb
    if num_padding_tokens == 0:
        return emb
    else:
        return F.pad(emb, (0, 0, 0, num_padding_tokens))


class Block(nn.Module):
    """Transformer block.

    Args:
        dim: Transformer dimension size.
        num_heads: Number of attention heads (overrides head_dim if specified).
        head_dim: Dimension size per attention head.
        mlp_ratio: Ratio of hidden dimension size to input dimension size in the MLP.
        qkv_bias: Whether to use bias in Q, K, V projections.
        proj_bias: Whether to use bias in the output projection layer.
        mlp_bias: Whether to use bias in the MLP layer.
        drop: Dropout rate for attention and MLP linear layers.
        drop_path: Stochastic depth drop rate.
        act_layer: Activation layer used in the MLP.
        norm_layer: Pre-normalization layer.
        gated_mlp: Whether to use a gated MLP layer, e.g. for SwiGLU.
        qk_norm: Whether to apply QK normalization.
    """

    def __init__(
        self,
        dim,
        num_heads=None,
        head_dim=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        proj_bias=False,
        mlp_bias=False,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
        gated_mlp=False,
        qk_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = norm_layer(dim)
        num_heads = num_heads or dim // head_dim

        self.attn = SelfAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            proj_drop=drop,
            qk_norm=qk_norm,
            norm_layer=norm_layer,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        mlp_layer = GatedMlp if gated_mlp else Mlp
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            bias=mlp_bias,
            drop=drop,
        )

    def forward(self, x, block_mask=None, **kwargs):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                block_mask=block_mask,
            )
        )
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class BlockAdaLN(Block):
    """Transformer block with adaLN-zero modulation.
    See Block for arguments.
    Args:
        adaLN_expansion: Expansion factor for adaLN modulation, e.g. for learning separate
            shift and scale parameters for patches and registers.
    """

    def __init__(self, adaLN_expansion: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.adaLN_expansion = adaLN_expansion
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(self.dim, adaLN_expansion * 6 * self.dim, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(
        self,
        x,
        block_mask=None,
        adaLN_emb=None,
        adaLN_packing_fn=None,
        **kwargs,
    ):
        # Embed and expand adaLN_embs: B x (exp*6*D) -> sum(N_i) x (6*D)
        adaLN_emb_packed = adaLN_packing_fn(self.adaLN_modulation(adaLN_emb))
        adaLN_emb_packed = expand_to_padded_seq(adaLN_emb_packed, x)
        gate_msa, gate_mlp, shift_msa, scale_msa, shift_mlp, scale_mlp = adaLN_emb_packed.chunk(
            6, dim=-1
        )

        x = x + gate_msa * self.drop_path(
            self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa),
                block_mask=block_mask,
            )
        )
        x = x + gate_mlp * self.drop_path(self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x
