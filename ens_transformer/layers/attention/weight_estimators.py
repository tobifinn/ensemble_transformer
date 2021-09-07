#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 29.07.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
import abc


# External modules
import torch
import torch.nn.functional as F
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


class WeightEstimator(torch.nn.Module):
    @staticmethod
    def _dot_product(
            x: torch.Tensor,
            y: torch.Tensor
    ) -> torch.Tensor:
        return torch.einsum('bic..., bjc...->bcij', x, y)

    @staticmethod
    def _estimate_norm_factor(estimate_from: torch.Tensor) -> float:
        projection_size = np.prod(estimate_from.shape[3:])
        norm_factor = 1 / np.sqrt(projection_size)
        return norm_factor

    @abc.abstractmethod
    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        pass


class SoftmaxWeights(WeightEstimator):
    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        gram_mat = self._dot_product(key, query)
        norm_factor = self._estimate_norm_factor(key)
        gram_mat = gram_mat * norm_factor
        weights = torch.softmax(gram_mat, dim=-2)
        return weights


class KernelWeights(WeightEstimator):
    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        gram_mat = self._dot_product(key, query)
        norm_factor = self._estimate_norm_factor(key)
        gram_mat = gram_mat * norm_factor
        weights = gram_mat / gram_mat.sum(dim=-2, keepdim=True)
        return weights


class GaussianProcessWeights(WeightEstimator):
    def __init__(self, n_heads: int):
        super().__init__()
        self.reg_value = torch.nn.Parameter(torch.ones(n_heads))

    def _add_regularization(self, input_tensor: torch.Tensor) -> torch.Tensor:
        reg_lam = F.softplus(self.reg_value)[:, None, None]
        id_matrix = torch.eye(input_tensor.shape[-1])[None, :, :]
        reg_lam = reg_lam * id_matrix.to(input_tensor)
        input_regularized = input_tensor + reg_lam
        return input_regularized

    @staticmethod
    def _solve_lin(
            hessian: torch.Tensor,
            moment_matrix: torch.Tensor
    ) -> torch.Tensor:
        solution, _ = torch.solve(moment_matrix, hessian)
        return solution

    def forward(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        norm_factor = self._estimate_norm_factor(key)
        hessian = self._dot_product(key, key) * norm_factor
        hessian = self._add_regularization(hessian)
        moment_matrix = self._dot_product(key, query) * norm_factor
        weights = self._solve_lin(hessian, moment_matrix)
        return weights
