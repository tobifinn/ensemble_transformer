#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.01.21
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
from .utils import ens_to_batch, split_batch_ens


logger = logging.getLogger(__name__)


class EarthPadding(torch.nn.Module):
    """
    Padding for ESMs with periodic and zero-padding in longitudinal and
    latitudinal direction, respectively.
    """
    def __init__(self, pad_size: int = 1):
        super().__init__()
        self.pad_size = pad_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lon_left = x[..., -self.pad_size:]
        lon_right = x[..., :self.pad_size]
        lon_padded = torch.cat([lon_left, x, lon_right], dim=-1)
        lat_zeros = torch.zeros_like(lon_padded[..., -self.pad_size:, :])
        lat_padded = torch.cat([lat_zeros, lon_padded, lat_zeros], dim=-2)
        return lat_padded


class EnsConv2d(torch.nn.Module):
    """
    Added viewing for ensemble-based 2d convolutions.
    """
    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int = 5
    ):
        super().__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels,
                                      kernel_size=kernel_size)

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_tensor_batched = ens_to_batch(in_tensor)
        convolved_tensor = self.conv2d(in_tensor_batched)
        out_tensor = split_batch_ens(convolved_tensor, in_tensor)
        return out_tensor
