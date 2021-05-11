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

# Internal modules


logger = logging.getLogger(__name__)


class BaseTransformer(torch.nn.Module):
    def __init__(
            self,
            value_layer: torch.nn.Module,
            key_layer: torch.nn.Module,
            query_layer: torch.nn.Module
    ):
        super().__init__()
        self.value_layer = value_layer
        self.key_layer = key_layer
        self.query_layer = query_layer

    @abc.abstractmethod
    def _dot_product(
            self,
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        pass

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
            value: torch.Tensor,
            weights: torch.Tensor
    ) -> torch.Tensor:
        pass

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
        transformed = self._apply_weights(value, weights)
        return transformed
