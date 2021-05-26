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
import abc
from typing import Union, Tuple

# External modules
import torch
import numpy as np

from hydra.utils import get_class

# Internal modules
from ..layers import EnsConv2d, EarthPadding


logger = logging.getLogger(__name__)


__all__ = [
    'BaseTransformer'
]


class BaseTransformer(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
            activation: Union[None, str] = 'torch.nn.SELU',
            key_activation: Union[None, str] = 'torch.nn.SELU',
            value_layer: bool = True,
            same_key_query: bool = False,
    ):
        super().__init__()
        if activation is not None:
            self.activation = get_class(activation)(inplace=True)
        else:
            self.activation = None
        self.value_layer = self._construct_value_layer(
            n_channels=n_channels,
            n_heads=n_heads,
            value_layer=value_layer
        )
        self.key_layer = self._construct_branch_layer(
            n_channels=n_channels,
            n_heads=n_heads,
            key_activation=key_activation,
        )
        if same_key_query:
            self.query_layer = self.key_layer
        else:
            self.query_layer = self._construct_branch_layer(
                n_channels=n_channels,
                n_heads=n_heads,
                key_activation=key_activation,
            )
        self.out_layer = EnsConv2d(
            in_channels=n_heads, out_channels=n_channels, kernel_size=1,
            padding=0
        )
        torch.nn.init.zeros_(self.out_layer.conv2d.base_layer.weight)

    @staticmethod
    def _construct_value_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            value_layer: bool = True,
    ) -> torch.nn.Sequential:
        layers = []
        if value_layer:
            conv_layer = EnsConv2d(
                in_channels=n_channels,
                out_channels=n_heads,
                kernel_size=1,
                bias=False
            )
            layers.append(conv_layer)
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _construct_branch_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            key_activation: Union[None, str] = None,
    ) -> torch.nn.Sequential:
        conv_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=1,
            bias=False
        )
        if key_activation == 'torch.nn.SELU':
            lecun_stddev = np.sqrt(1/n_channels)
            torch.nn.init.normal_(
                conv_layer.conv2d.base_layer.weight,
                std=lecun_stddev
            )
        layers = [conv_layer]
        if key_activation:
            layers.append(get_class(key_activation)(inplace=True))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _dot_product(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('bichw, bjchw->bcij', x, y)

    @abc.abstractmethod
    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        pass

    @staticmethod
    def _apply_weights(
            value_tensor: torch.Tensor,
            weights_tensor: torch.Tensor
    ) -> torch.Tensor:
        value_mean = value_tensor.mean(dim=1, keepdim=True)
        value_perts = value_tensor-value_mean
        transformed_perts = torch.einsum(
            'bcij, bichw->bjchw', weights_tensor, value_perts
        )
        transformed_tensor = value_mean + transformed_perts
        return transformed_tensor

    def _apply_layers(
            self,
            in_tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_layer(in_tensor)
        key = self.key_layer(in_tensor)
        query = self.query_layer(in_tensor)
        return value, key, query

    def forward(
            self,
            in_tensor: torch.Tensor
    ) -> torch.Tensor:
        value, key, query = self._apply_layers(
            in_tensor=in_tensor
        )
        weights = self._get_weights(key=key, query=query)
        id_matrix = torch.eye(weights.shape[-1])[None, :, :]
        weights = weights + id_matrix.to(weights)

        transformed = self._apply_weights(value, weights)
        out_tensor = in_tensor + self.out_layer(transformed)
        if self.activation is not None:
            out_tensor = self.activation(out_tensor)
        return out_tensor
