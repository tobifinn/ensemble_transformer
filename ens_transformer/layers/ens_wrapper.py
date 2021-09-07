#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 30.08.21
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
import torch.nn

# Internal modules
from ..ens_utils import ens_to_batch, split_batch_ens


logger = logging.getLogger(__name__)


class EnsembleWrapper(torch.nn.Module):
    def __init__(self, base_layer: torch.nn.Module):
        super().__init__()
        self.base_layer = base_layer

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_tensor_batched = ens_to_batch(in_tensor)
        modified_tensor = self.base_layer(in_tensor_batched)
        out_tensor = split_batch_ens(modified_tensor, in_tensor)
        return out_tensor
