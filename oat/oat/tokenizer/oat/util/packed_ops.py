# --------------------------------------------------------------------
# Copyright (C) 2024–2025 EPFL & Apple Inc.
# Licensed under the EPFL–Apple Sample Code License (Non-Commercial)
# 
# This file is adapted from the original implementation.
# Modifications by Chaoqi Liu for research purposes.
# --------------------------------------------------------------------

from typing import Callable, List, Union

from einops import pack, unpack

import torch
import torch.nn as nn

__all__ = ["packed_proj", "packed_call", "element_wise_call"]

# Type aliases
TensorOrTensorList = Union[torch.Tensor, List[torch.Tensor]]


def packed_proj(x_list: List[torch.Tensor], proj: nn.Module) -> List[torch.Tensor]:
    """
    Applies a projection to a list of tensors by packing them into one tensor, applying the projection,
    and then unpacking the result.

    Args:
        x_list: List of tensors to project.
        proj: Projection function or module.

    Returns:
        List of projected tensors.
    """
    x_packed, ps = pack(x_list, "b * d")
    x_packed = proj(x_packed)
    return unpack(x_packed, ps, "b * d")


def packed_call(fn: Callable, x: TensorOrTensorList):
    """
    Applies a function to a tensor or a list of tensors. When given a list, the tensors are concatenated,
    processed, and split back into their original shapes.

    Args:
        fn: Function to apply.
        x: Tensor or list of tensors.

    Returns:
        Processed tensor(s) matching the input structure. Handles functions that return multiple values.
    """
    if isinstance(x, torch.Tensor):
        return fn(x)

    # x_list is a list of (B, D, H, W) or (B, N, D) tensors
    x_list = x
    x_seqlens = [x.shape[1] for x in x_list]

    # Pack into one large sequence and call the function
    x_packed = torch.cat(x_list, dim=1)
    results_packed = fn(x_packed)

    results: List[Union[torch.Tensor, List[torch.Tensor]]] = []
    # Handle variable number of return values
    if isinstance(results_packed, tuple):  # Function returned multiple values
        for res_i in results_packed:
            if (
                isinstance(res_i, torch.Tensor)
                and res_i.ndim > 1
                and res_i.shape[1] == sum(x_seqlens)
            ):
                # Split back into individual samples
                res = list(torch.split(res_i, x_seqlens, dim=1))
                results.append(res)
            else:
                results.append(res_i)

    else:  # Function returned a single value
        # Split back into individual samples
        results = list(torch.split(results_packed, x_seqlens, dim=1))

    return results


def element_wise_call(fn: Callable, x: TensorOrTensorList) -> TensorOrTensorList:
    """
    Applies a function element-wise to a tensor or each tensor in a list.

    Args:
        fn: Function to apply.
        x: Tensor or list of tensors.

    Returns:
        Tensor or list of tensors with the function applied.
    """
    return [fn(xi) for xi in x] if isinstance(x, list) else fn(x)
