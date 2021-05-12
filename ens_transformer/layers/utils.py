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

# External modules
import torch

# Internal modules


logger = logging.getLogger(__name__)


def ens_to_batch(in_tensor: torch.Tensor) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(-1, *in_tensor.shape[-3:])
    except RuntimeError:
        out_tensor = in_tensor.reshape(-1, *in_tensor.shape[-3:]).contiguous()
    return out_tensor


def split_batch_ens(
        in_tensor: torch.Tensor,
        like_tensor: torch.Tensor
) -> torch.Tensor:
    try:
        out_tensor = in_tensor.view(
            *like_tensor.shape[:-3], *in_tensor.shape[-3:]
        )
    except RuntimeError:
        out_tensor = in_tensor.reshape(
            *like_tensor.shape[:-3], *in_tensor.shape[-3:]
        ).contiguous()
    return out_tensor


class EnsembleWrapper(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, in_tensor: torch.Tensor):
        in_tensor_batched = ens_to_batch(in_tensor)
        modified_tensor = self.base_layer(in_tensor_batched)
        out_tensor = split_batch_ens(modified_tensor, in_tensor)
        return out_tensor
