import torch
import torch.nn as nn
from typing import Optional

class SampleEmbedder(nn.Module):
    """
    Module for patch embedding of tensors across arbitrary dimensions, projecting
    patches to a desired output dimension.

    Args:
        channels_in: Input feature dimension of each tensor.
        dim: Optional output feature dimension after projection; if None, no projection is applied.
        weight_init_style: Initialization style for weights ('xavier', 'zero', or 'trunc_normal').
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: Optional[int] = None,
        weight_init_style: str = "xavier",
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        if self.dim_out is not None:
            self.patch_proj = nn.Linear(self.dim_in, self.dim_out, bias=True)
        else:
            self.dim_out = dim_in
            self.patch_proj = nn.Identity()

        # Weight init
        self.weight_init_style = weight_init_style
        self.init_weights_sp()

    def init_weights_sp(self):
        """SP weight initialization scheme"""
        if self.dim_out is None:
            return

        if self.weight_init_style == "zero":
            nn.init.constant_(self.patch_proj.weight, 0)
        elif self.weight_init_style == "xavier":
            nn.init.xavier_uniform_(self.patch_proj.weight)
        elif self.weight_init_style == "trunc_normal":
            nn.init.trunc_normal_(self.patch_proj.weight, std=0.02)
        else:
            raise ValueError(f"Unsupported weight init: {self.weight_init_style}")
        # Bias
        if self.patch_proj.bias is not None:
            nn.init.constant_(self.patch_proj.bias, 0)

    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_proj(x)
        return x
