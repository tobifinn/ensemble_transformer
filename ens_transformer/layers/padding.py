#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 13.05.21
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
