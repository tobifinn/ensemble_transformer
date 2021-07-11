#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05.10.20
#
# Created for ensemble-transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
from typing import Union

# External modules
import torch.nn
from hydra.utils import get_class

# Internal modules
from ..layers import EnsConv2d
from .softmax_transformer import SoftmaxTransformer


logger = logging.getLogger(__name__)


class SoftmaxFNNTransformer(SoftmaxTransformer):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
            n_fnn_channels: int = 128,
            activation: Union[None, str] = 'torch.nn.SELU',
            key_activation: Union[None, str] = 'torch.nn.SELU',
            value_layer: bool = True,
            same_key_query: bool = False,
            layer_norm: bool = False,
    ):
        super().__init__(
            n_channels=n_channels,
            n_heads=n_heads,
            activation=activation,
            key_activation=key_activation,
            value_layer=value_layer,
            same_key_query=same_key_query,
            layer_norm=layer_norm
        )
        self.fnn_res_layer = torch.nn.Sequential(
            EnsConv2d(
                in_channels=n_channels,
                out_channels=n_fnn_channels,
                kernel_size=1,
                bias=False
            ),
            get_class(activation)(inplace=True),
            EnsConv2d(
                in_channels=n_fnn_channels,
                out_channels=n_channels,
                kernel_size=1,
                bias=False
            )
        )

    def forward(
            self,
            in_tensor: torch.Tensor
    ) -> torch.Tensor:
        out_attention_tensor = super().forward(in_tensor=in_tensor)
        out_norm_tensor = self.layer_norm(out_attention_tensor)
        residual_tensor = self.fnn_res_layer(out_norm_tensor)
        output_tensor = out_attention_tensor + residual_tensor
        return output_tensor
