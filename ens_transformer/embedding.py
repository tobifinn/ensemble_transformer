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
from .layers import EnsConv2d, EarthPadding


logger = logging.getLogger(__name__)


class ModelEmbedding(torch.nn.Module):
    def __init__(
            self,
            n_channels: Iterable = (64, 64, 64),
            filter_sizes: Iterable = (5, 5, 5),
            activation: str = 'torch.nn.SELU'
    ):
        super().__init__()
        modules = []
        old_feat = 3
        for idx, curr_channel in n_channels:
            modules.append(EarthPadding((filter_sizes[idx]-1)//2))
            modules.append(EnsConv2d(
                old_feat, curr_channel, kernel_size=filter_sizes[idx]
            ))
            modules.append(get_class(activation)(inplace=True))
            old_feat = curr_channel
        self.n_channels = n_channels
        self.activation = activation
        self.filter_size = filter_sizes
        self.net = torch.nn.Sequential(
            *modules
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        out_tensor = self.net(in_tensor)
        return out_tensor
