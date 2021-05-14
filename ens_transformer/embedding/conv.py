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
from typing import Tuple

# External modules
import torch
from hydra.utils import get_class

# Internal modules
from ..utils import EnsembleWrapper
from ..layers import EnsConv2d, EarthPadding


logger = logging.getLogger(__name__)


class ConvEmbedding(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            n_conv_layers: int = 6,
            n_init_features: int = 4,
            embedding_size: int = 256,
            activation: str = 'torch.nn.SELU',
            dim_sizes: Tuple[int, int] = (32, 64)
    ):
        super().__init__()
        self.conv_layers, conv_out_size = self._init_conv_layers(
            in_channels=in_channels,
            n_conv_layers=n_conv_layers,
            n_init_features=n_init_features,
            activation=activation,
            dim_sizes=dim_sizes
        )
        self.output_layer = torch.nn.Linear(
            in_features=conv_out_size,
            out_features=embedding_size,
            bias=True
        )

    @staticmethod
    def _init_conv_layers(
            in_channels: int = 3,
            n_conv_layers: int = 6,
            n_init_features: int = 4,
            activation: str = 'torch.nn.SELU',
            dim_sizes: Tuple[int, int] = (32, 64)
    ) -> Tuple[torch.nn.Sequential, int]:
        curr_in_channels = in_channels
        curr_out_channels = n_init_features
        conv_layers = []
        for idx in range(n_conv_layers):
            if idx > 0:
                conv_layers.append(
                    EnsembleWrapper(torch.nn.MaxPool2d(3, padding=1))
                )
            conv_layers.append(
                torch.nn.Sequential(
                    EarthPadding(pad_size=2),
                    EnsConv2d(
                        in_channels=curr_in_channels,
                        out_channels=curr_out_channels,
                        kernel_size=5
                    ),
                    get_class(activation)(inplace=True)
                )
            )
            curr_in_channels = curr_out_channels
            curr_out_channels = curr_out_channels * 2
        conv_layers = torch.nn.Sequential(*conv_layers)
        reduce_factor = 2 ** (n_conv_layers-1)
        out_size = dim_sizes[0] * dim_sizes[1] // reduce_factor // reduce_factor
        return conv_layers, out_size

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        curr_tensor = in_tensor
        for curr_layer in self.conv_layers:
            curr_tensor = curr_layer(curr_tensor)
        curr_tensor = curr_tensor.view(
            *curr_tensor.shape[:2], -1
        )
        embedded_tensor = self.output_layer(curr_tensor)
        return embedded_tensor
