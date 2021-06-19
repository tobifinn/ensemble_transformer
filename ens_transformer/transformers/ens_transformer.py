#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05.10.20
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}

# System modules
import logging
from typing import Union

# External modules
import torch
import torch.nn.functional as F
import numpy as np

# Internal modules
from .base_transformer import BaseTransformer


logger = logging.getLogger(__name__)


class EnsTransformer(BaseTransformer):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
            activation: Union[None, str] = 'torch.nn.SELU',
            key_activation: Union[None, str] = 'torch.nn.SELU',
            value_layer: bool = True,
            same_key_query: bool = False,
            layer_norm: bool = False
    ):
        super().__init__(
            n_channels=n_channels,
            n_heads=n_heads,
            activation=activation,
            value_layer=value_layer,
            key_activation=key_activation,
            same_key_query=same_key_query,
            layer_norm=layer_norm
        )
        self.reg_value = torch.nn.Parameter(torch.ones(n_heads))

    def _solve_lin(
            self,
            hessian: torch.Tensor,
            moment_matrix: torch.Tensor
    ) -> torch.Tensor:
        reg_lam = F.softplus(self.reg_value)[:, None, None]
        id_matrix = torch.eye(hessian.shape[-1])[None, :, :]
        reg_lam = reg_lam * id_matrix.to(hessian)
        hessian_reg = hessian + reg_lam
        weights, _ = torch.solve(moment_matrix, hessian_reg)
        return weights

    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        norm_factor = 1. / np.sqrt(key.shape[-2] * key.shape[-1])
        hessian = self._dot_product(key, key) / norm_factor
        moment_matrix = self._dot_product(key, query) / norm_factor
        weights = self._solve_lin(hessian, moment_matrix)
        return weights
