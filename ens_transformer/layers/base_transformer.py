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


__all__ = [
    'SELUKernel',
    'BaseTransformer'
]


class SELUKernel(torch.nn.SELU):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        activated_tensor = super().forward(input_tensor)
        activated_tensor = activated_tensor+1
        return activated_tensor


_activations = {
    'selu': torch.nn.SELU,
    'relu': torch.nn.ReLU,
    'selu_kernel': SELUKernel
}


class BaseTransformer(torch.nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            value_activation: Union[None, str] = None,
            embedding_size: int = 256,
            n_key_neurons: int = 1,
            coarsening_factor: int = 1,
            key_activation: Union[None, str] = None,
            interpolation_mode: str = 'bilinear',
            same_key_query: bool = False,
            grid_dims: Tuple[int, int] = (32, 64),
            ens_mems: int = 50
    ):
        super().__init__()
        self.coarsening_factor = coarsening_factor
        self.interpolation_mode = interpolation_mode
        self.grid_dims = grid_dims
        self.n_key_neurons = n_key_neurons

        self.value_layer = self._construct_value_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            value_activation=value_activation
        )
        self.key_layer, self.query_layer = self._construct_key_query_layer(
            embedding_size=embedding_size,
            n_key_neurons=n_key_neurons,
            key_activation=key_activation,
            same_key_query=same_key_query
        )
        self.identity = torch.nn.Parameter(torch.eye(ens_mems),
                                           requires_grad=False)

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
            value_activation: Union[None, str] = None,
    ) -> torch.nn.Sequential:
        layers = [
            EarthPadding(pad_size=2),
            EnsConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=5
            )
        ]
        if value_activation is not None:
            layers.append(_activations[value_activation](inplace=True))
        return torch.nn.Sequential(*layers)

    def _construct_key_query_layer(
            self,
            embedding_size: int = 256,
            n_key_neurons: int = 1,
            key_activation: Union[None, str] = None,
            same_key_query: bool = False
    ):
        out_features = self.local_grid_dims[0] * self.local_grid_dims[1]
        out_features *= n_key_neurons
        key_layer = torch.nn.Linear(
            in_features=embedding_size,
            out_features=out_features,
            bias=False
        )
        key_layer.weight.data = torch.ones_like(
            key_layer.weight.data
        )
        if key_activation is not None:
            key_layer = torch.nn.Sequential(
                key_layer,
                _activations[key_activation](inplace=True)
            )
        if same_key_query:
            query_layer = key_layer
        else:
            query_layer = torch.nn.Linear(
                in_features=embedding_size,
                out_features=out_features,
                bias=False
            )
            query_layer.weight.data = torch.ones_like(
                query_layer.weight.data
            )
            if key_activation is not None:
                query_layer = torch.nn.Sequential(
                    query_layer,
                    _activations[key_activation](inplace=True)
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
        )
        interp_slice = slice(self.coarsening_factor, -self.coarsening_factor)
        interpolated_tensor = interpolated_tensor[..., interp_slice]
        output_tensor = split_batch_ens(interpolated_tensor, input_tensor)
        return output_tensor

    @staticmethod
    def _dot_product(
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

    @staticmethod
    def _apply_weights(
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
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_layer(in_tensor)
        key = self.key_layer(embedding)
        query = self.query_layer(embedding)
        key = key.view(*key.shape[:-1], self.n_key_neurons,
                       *self.local_grid_dims)
        query = query.view(*query.shape[:-1], self.n_key_neurons,
                           *self.local_grid_dims)
        return value, key, query

    def forward(
            self,
            in_tensor: torch.Tensor,
            embedding: torch.Tensor
    ) -> torch.Tensor:
        value, key, query = self._apply_layers(
            in_tensor=in_tensor, embedding=embedding
        )
        weights = self._get_weights(key=key, query=query)
        weights = weights + self.identity.view(1, 50, 50, 1, 1)
        weights = self.interpolate(weights)
        transformed = self._apply_weights(value, weights)
        return transformed
