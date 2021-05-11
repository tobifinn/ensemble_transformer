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

# Internal modules
from .conv import EnsConv2d, EarthPadding
from .utils import ens_to_batch, split_batch_ens


logger = logging.getLogger(__name__)


class BaseTransformer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            value_activation: bool = True,
            embedding_size: int = 256,
            n_key_neurons: int = 1,
            coarsening_factor: int = 1,
            interpolation_mode: str = 'bilinear',
            grid_dims: Tuple[int, int] = (32, 64),
            same_key_query: bool = False
    ):
        super().__init__()
        self.coarsening_factor = coarsening_factor
        self.interpolation_mode = interpolation_mode
        self.grid_dims = grid_dims
        self.n_key_neurons = n_key_neurons

        self.value_layer = self._construct_value_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            activation=value_activation
        )
        self.key_layer, self.query_layer = self._construct_key_query_layer(
            embedding_size=embedding_size,
            n_key_neurons=n_key_neurons,
            same_key_query=same_key_query
        )

    @property
    def local_grid_dims(self) -> Tuple[int, int]:
        local_dims = (
            self.grid_dims[0] // self.coarsening_factor,
            self.grid_dims[1] // self.coarsening_factor
        )
        return local_dims

    @staticmethod
    def _construct_value_layer(
            in_channels: int,
            out_channels: int,
            activation: bool = True
    ) -> torch.nn.Sequential:
        layers = [
            EarthPadding(pad_size=2),
            EnsConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5
            )
        ]
        if activation:
            layers.append(torch.nn.SELU(inplace=True))
        return torch.nn.Sequential(*layers)

    def _construct_key_query_layer(
            self,
            embedding_size: int = 256,
            n_key_neurons: int = 1,
            same_key_query: bool = False
    ):
        out_features = self.local_grid_dims[0] * self.local_grid_dims[1]
        out_features *= n_key_neurons
        key_layer = torch.nn.Linear(
            in_features=embedding_size,
            out_features=out_features,
            bias=False
        )
        if same_key_query:
            query_layer = key_layer
        else:
            query_layer = torch.nn.Linear(
                in_features=embedding_size,
                out_features=out_features,
                bias=False
            )
        return key_layer, query_layer

    def interpolate(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_tensor_batched = ens_to_batch(input_tensor)
        padded_tensor = torch.cat(
            [input_tensor_batched[..., -1:], input_tensor_batched,
             input_tensor_batched[..., :1]], dim=-1
        )
        interpolated_tensor = F.interpolate(
            padded_tensor,
            scale_factor=self.coarsening_factor,
            mode=self.interpolation_mode,
            align_corners=False
        )
        interp_slice = slice(self.coarsening_factor, -self.coarsening_factor)
        interpolated_tensor = interpolated_tensor[..., interp_slice]
        output_tensor = split_batch_ens(interpolated_tensor, input_tensor)
        return output_tensor

    @abc.abstractmethod
    def _dot_product(
            self,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('binhw, bjnhw->bijhw', x, y)

    @abc.abstractmethod
    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def _apply_weights(
            self,
            value_tensor: torch.Tensor,
            weights_tensor: torch.Tensor
    ) -> torch.Tensor:
        value_mean = value_tensor.mean(dim=1, keepdim=True)
        value_perts = value_tensor-value_mean
        transformed_perts = torch.einsum(
            'bijhw, bichw->bjchw', weights_tensor, value_perts
        )
        transformed_tensor = value_mean + transformed_perts
        return transformed_tensor

    def _apply_layers(
            self,
            value_base: torch.Tensor,
            key_base: torch.Tensor,
            query_base: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_layer(value_base)
        key = self.key_layer(key_base)
        query = self.query_layer(query_base)
        return value, key, query

    def forward(
            self,
            value_base: torch.Tensor,
            key_base: Union[torch.Tensor, None] = None,
            query_base: Union[torch.Tensor, None] = None
    ):
        if key_base is None:
            key_base = value_base
        if query_base is None:
            query_base = value_base
        value, key, query = self._apply_layers(
            value_base=value_base, key_base=key_base, query_base=query_base
        )
        weights = self._get_weights(key=key, query=query)
        weights = self.interpolate(weights)
        transformed = self._apply_weights(value, weights)
        return transformed
