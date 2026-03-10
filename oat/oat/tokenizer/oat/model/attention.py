# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from functools import partial

import einops

import torch
import torch.nn as nn
import torch.nn.functional as F

from oat.tokenizer.oat.model.norm import Fp32LayerNorm

__all__ = ["Attention", "SelfAttention", "CrossAttention"]


class Attention(nn.Module):
    """Multi-head attention module with optional normalization, scaling, and dropout.

    Args:
        dim: Transformer dimension size.
        num_heads: Number of attention heads.
        qkv_bias: Whether to use bias in Q, K, V projections.
        proj_bias: Whether to use bias in the output projection layer.
        proj_drop: Dropout rate for the projection layer. 0.0 = no dropout.
        qk_norm: Whether to apply QK normalization.
        norm_layer: Normalization layer when using QK-norm.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        proj_bias=False,
        proj_drop=0.0,
        qk_norm=True,
        norm_layer=partial(Fp32LayerNorm, bias=False, elementwise_affine=False),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        if qk_norm:
            self.q_norm = norm_layer(self.head_dim)
            self.k_norm = norm_layer(self.head_dim)
        else:
            self.q_norm = self.k_norm = nn.Identity()

    def forward(self, xq, xk, xv, block_mask=None):
        """Forward pass of the FlexAttention module.

        Args:
            xq: Input tensor of shape [B, N_q, D] to compute queries.
            xk: Input tensor of shape [B, N_kv, D] to compute keys.
            xv: Input tensor of shape [B, N_kv, D] to compute values.
            block_mask: attn mask.

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """

        # q, k, v each of shape [batch_size, sequence_length, num_heads*head_dim]
        q, k, v = self.wq(xq), self.wk(xk), self.wv(xv)

        # Separate heads of q, k, v into [batch_size, num_heads, sequence_length, head_dim]
        q = einops.rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = einops.rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = einops.rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        # Optional QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # use block_mask as the attention mask
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=block_mask
        )
        # Combine heads back to [batch_size, sequence_length, num_heads*head_dim]
        x = einops.rearrange(x, "b h n d -> b n (h d)")

        # Project and apply optional dropout
        x = self.proj_drop(self.proj(x))
        return x


class SelfAttention(Attention):
    """Multi-head self-attention module with optional normalization, scaling, and dropout.
    See Attention for arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, block_mask=None, **kwargs):
        """Forward pass of the SelfAttention module.

        Args:
            x: Input tensor of shape [B, ..., D] to compute queries, keys, and values from.
            block_mask: attn mask.

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """
        return super().forward(
            xq=x,
            xk=x,
            xv=x,
            block_mask=block_mask,
        )


class CrossAttention(Attention):
    """Multi-head cross-attention module with optional normalization, scaling, and dropout.
    See Attention for arguments.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, context, block_mask=None, **kwargs):
        """Forward pass of the CrossAttention module.

        Args:
            x: Input tensor of shape [B, ..., D] to compute queries from.
            context: Context tensor of shape [B, ..., D] to compute keys, and values from.
            block_mask: attn mask

        Returns:
            Output tensor after applying attention, projection, and dropout.
        """
        return super().forward(
            xq=x,
            xk=context,
            xv=context,
            block_mask=block_mask,
        )
