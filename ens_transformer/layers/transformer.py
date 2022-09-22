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
from . import EnsConv2d, EarthPadding


logger = logging.getLogger(__name__)


__all__ = [
    'SoftmaxTransformer'
]


class SoftmaxTransformer(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
            layer_norm: bool = False,
            n_mixing: int = 64
    ):
        super().__init__()
        if layer_norm is not None:
            self.layer_norm = torch.nn.LayerNorm([n_channels, 32, 64])
        else:
            self.layer_norm = torch.nn.Sequential()

        self.value_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=1,
            bias=False
        )
        self.key_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=1,
            bias=False,
            padding=0
        )
        self.query_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=1,
            bias=False,
            padding=0
        )
        self.out_layer = EnsConv2d(
            in_channels=n_heads, out_channels=n_channels, kernel_size=1,
            bias=False, padding=0
        )
        torch.nn.init.zeros_(self.out_layer.conv2d.base_layer.weight)
        self.mixing_layer = torch.nn.Sequential(
            EnsConv2d(
                in_channels=n_channels, out_channels=n_mixing,
                kernel_size=1, padding=0
            ),
            torch.nn.GELU(),
            EnsConv2d(
                in_channels=n_mixing, out_channels=n_channels,
                kernel_size=1, padding=0
            )
        )

    @staticmethod
    def _dot_product(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('bichw, bjchw->bcij', x, y)

    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        gram_mat = self._dot_product(key, query)
        gram_mat = gram_mat / np.sqrt(key.shape[-2]*key.shape[-1])
        weights = torch.softmax(gram_mat, dim=-2)
        return weights

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
        transformed_tensor = value_tensor + transformed_perts
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
        pre_norm_tensor = self.layer_norm(in_tensor)
        value, key, query = self._apply_layers(
            in_tensor=pre_norm_tensor
        )
        weights = self._get_weights(key=key, query=query)

        transformed = self._apply_weights(value, weights)
        after_transformed_tensor = in_tensor + self.out_layer(transformed)
        out_tensor = after_transformed_tensor + self.mixing_layer(
            after_transformed_tensor
        )
        return out_tensor
