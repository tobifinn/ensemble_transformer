#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 20.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Iterable

# External modules
import torch.nn

from hydra.utils import get_class

# Internal modules
from .conv import EnsConv2d
from .padding import EarthPadding


logger = logging.getLogger(__name__)


class ModelEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            n_channels: Iterable = (64, 64, 64),
            kernel_size: int = 5,
            activation: str = 'torch.nn.SELU'
    ):
        super().__init__()
        modules = []
        old_channels = in_channels
        for curr_channel in n_channels:
            modules.append(EarthPadding((kernel_size-1)//2))
            modules.append(EnsConv2d(
                old_channels, curr_channel, kernel_size=kernel_size
            ))
            modules.append(get_class(activation)(inplace=True))
            old_channels = curr_channel
        self.n_channels = n_channels
        self.activation = activation
        self.kernel_size = kernel_size
        self.net = torch.nn.Sequential(
            *modules
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.net(in_tensor)
        return out_tensor
