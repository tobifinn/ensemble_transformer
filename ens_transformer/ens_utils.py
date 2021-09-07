#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def ens_to_batch(in_tensor: torch.Tensor) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(-1, *in_tensor.shape[2:])
    except RuntimeError:
        out_tensor = in_tensor.reshape(-1, *in_tensor.shape[2:]).contiguous()
    return out_tensor


def split_batch_ens(
        in_tensor: torch.Tensor,
        like_tensor: torch.Tensor
) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(
            *like_tensor.shape[:2], *in_tensor.shape[1:]
        )
    except RuntimeError:
        out_tensor = in_tensor.reshape(
            *like_tensor.shape[:2], *in_tensor.shape[1:]
        ).contiguous()
    return out_tensor


def split_mean_perts(
        in_tensor: torch.Tensor,
        dim: int = 1
) -> Tuple[torch.Tensor, torch.Tensor]:
    mean_tensor = in_tensor.mean(dim=dim, keepdims=True)
    perts_tensor = in_tensor - mean_tensor
    return mean_tensor, perts_tensor
