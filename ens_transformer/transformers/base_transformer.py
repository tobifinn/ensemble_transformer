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
import torch.nn.functional as F

from hydra.utils import get_class

# Internal modules
from ..layers import EnsConv2d, EarthPadding
from ..utils import ens_to_batch, split_batch_ens


logger = logging.getLogger(__name__)


__all__ = [
    'BaseTransformer'
]


class BaseTransformer(torch.nn.Module):
    def __init__(
            self,
            channels: int,
            activation: Union[None, str] = None,
            key_activation: Union[None, str] = 'torch.nn.ReLU',
            value_layer: bool = True,
            same_key_query: bool = False,
            ens_mems: int = 50
    ):
        super().__init__()
        if activation is not None:
            self.activation = get_class(activation)(inplace=True)
        else:
            self.activation = None
        self.value_layer = self._construct_value_layer(
            channels=channels,
            value_layer=value_layer
        )
        self.key_layer = self._construct_branch_layer(
            channels=channels,
            key_activation=key_activation,
        )
        if same_key_query:
            self.query_layer = self.key_layer
        else:
            self.query_layer = self._construct_branch_layer(
                channels=channels,
                key_activation=key_activation,
            )
        self.identity = torch.nn.Parameter(torch.eye(ens_mems),
                                           requires_grad=False)

    @staticmethod
    def _construct_value_layer(
            channels: int,
            value_layer: bool = True,
    ) -> torch.nn.Sequential:
        layers = []
        if value_layer:
            conv_layer = EnsConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=5
            )
            torch.nn.init.kaiming_normal(conv_layer.conv2d.base_layer.weight)
            torch.nn.init.constant(conv_layer.conv2d.base_layer.bias, 0.)
            layers = [EarthPadding(pad_size=2), conv_layer]
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _construct_branch_layer(
            channels: int,
            key_activation: Union[None, str] = None,
    ) -> torch.nn.Sequential:
        conv_layer = EnsConv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=5
        )
        torch.nn.init.kaiming_normal(conv_layer.conv2d.base_layer.weight)
        torch.nn.init.constant(conv_layer.conv2d.base_layer.bias, 0.)
        layers = [EarthPadding(pad_size=2), conv_layer]
        if key_activation:
            layers.append(get_class(key_activation)(inplace=True))
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _dot_product(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('bichw, bjchw->bijc', x, y)

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
            'bijc, bichw->bjchw', weights_tensor, value_perts
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
        transformed = self._apply_weights(value, weights)
        if self.activation is not None:
            transformed = self.activation(transformed)
        return transformed
