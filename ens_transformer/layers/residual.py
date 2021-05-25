#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 25.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union

# External modules
import torch
from hydra.utils import get_class

# Internal modules
from . import EarthPadding, EnsConv2d


logger = logging.getLogger(__name__)


class ResidualLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int = 64,
            out_channels: int = 64,
            kernel_size: int = 5,
            branch_activation: str = 'torch.nn.ReLU',
            activation: str = 'torch.nn.ReLU',
            n_residuals: int = 1
    ):
        super().__init__()
        if kernel_size > 1:
            self.padding = EarthPadding(pad_size=(kernel_size - 1) // 2)
        else:
            self.padding = torch.nn.Sequential()
        self.branch_activation = get_class(branch_activation)(inplace=True)
        self.activation = get_class(activation)(inplace=True)

        self.conv_1 = EnsConv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=0, bias=False
        )
        self.conv_1.conv2d.base_layer.weight.data = \
            self.conv_1.conv2d.base_layer.weight.data * (n_residuals ** -0.5)
        self.conv_2 = EnsConv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=kernel_size, padding=0, bias=False
        )
        torch.nn.init.zeros_(self.conv_2.conv2d.base_layer.weight)
        if in_channels != out_channels:
            self.proj_conv = torch.nn.Sequential(
                self.padding,
                EnsConv2d(
                    in_channels=in_channels, out_channels=out_channels,
                    kernel_size=kernel_size, padding=0, bias=False
                )
            )
        else:
            self.proj_conv = torch.nn.Sequential()
        self.bias_before_conv_1 = torch.nn.Parameter(torch.zeros(1))
        self.bias_before_branch_activation = torch.nn.Parameter(torch.zeros(1))
        self.bias_before_conv_2 = torch.nn.Parameter(torch.zeros(1))
        self.bias_before_activation = torch.nn.Parameter(torch.zeros(1))
        self.multiplier = torch.nn.Parameter(torch.ones(1))

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        branch_tensor = self.bias_before_conv_1 + in_tensor
        branch_tensor = self.padding(branch_tensor)
        branch_tensor = self.conv_1(branch_tensor)
        branch_tensor = self.bias_before_branch_activation + branch_tensor
        branch_tensor = self.branch_activation(branch_tensor)
        branch_tensor = self.bias_before_conv_2 + branch_tensor
        branch_tensor = self.padding(branch_tensor)
        branch_tensor = self.conv_2(branch_tensor)
        branch_tensor = self.multiplier * branch_tensor
        branch_tensor = self.bias_before_activation + branch_tensor
        projected_tensor = self.proj_conv(in_tensor)
        out_tensor = projected_tensor + branch_tensor
        out_tensor = self.activation(out_tensor)
        return out_tensor
