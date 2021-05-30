#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11.05.21
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

# Internal modules
from .base_transformer import BaseTransformer


logger = logging.getLogger(__name__)


class KernelTransformer(BaseTransformer):
    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        weights = self._dot_product(key, query)
        weights = weights / weights.sum(dim=-2, keepdim=True)
        return weights
