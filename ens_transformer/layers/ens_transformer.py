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
from typing import Union, Tuple

# External modules
import torch
import torch.nn.functional as F

# Internal modules
from .base_transformer import BaseTransformer


logger = logging.getLogger(__name__)


class EnsTransformer(BaseTransformer):
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
            grid_dims: Tuple[int, int] = (32, 64),
            same_key_query: bool = False
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            value_activation=value_activation,
            embedding_size=embedding_size,
            n_key_neurons=n_key_neurons,
            coarsening_factor=coarsening_factor,
            key_activation=key_activation,
            interpolation_mode=interpolation_mode,
            grid_dims=grid_dims,
            same_key_query=same_key_query
        )
        self.reg_value = torch.nn.Parameter(torch.ones(1))

    def _solve_lin(
            self,
            hessian: torch.Tensor,
            moment_matrix: torch.Tensor
    ) -> torch.Tensor:
        hessian_moved = hessian.moveaxis(1, -1).moveaxis(1, -1)
        moment_moved = moment_matrix.moveaxis(1, -1).moveaxis(1, -1)
        reg_lam = F.softplus(self.reg_value)
        hessian_reg = hessian_moved + reg_lam * self.identity
        weights, _ = torch.solve(moment_moved, hessian_reg)
        weights = weights.moveaxis(-1, 1).moveaxis(-1, 1)
        return weights

    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        hessian = self._dot_product(key, key)
        moment_matrix = self._dot_product(key, query)
        weights = self._solve_lin(hessian, moment_matrix)
        return weights
