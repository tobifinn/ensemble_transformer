#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 19.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch
import torch.nn.functional as F
import numpy as np

# Internal modules
from .ens_transformer import EnsTransformer

logger = logging.getLogger(__name__)


class ApproxTransformer(EnsTransformer):
    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        key = key-key.mean(dim=1, keepdim=True)
        query = query-query.mean(dim=1, keepdim=True)
        query_pred = torch.einsum('bichw,ij->bjchw', key, self.identity[0, 0])
        resid = query - query_pred
        norm_factor = 1. / np.sqrt(key.shape[-2] * key.shape[-1])
        hessian = key.pow(2).sum(dim=(-2, -1)) / norm_factor
        hessian = torch.diag_embed(hessian.transpose(-1, -2))
        hessian = hessian + F.softplus(self.reg_value)
        moment_matrix = self._dot_product(key, resid) / norm_factor
        delta_weights = moment_matrix / hessian
        weights = self.identity - delta_weights
        return weights
