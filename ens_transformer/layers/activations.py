#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 12.05.21
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


logger = logging.getLogger(__name__)


class SELUKernel(torch.nn.SELU):
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        activated_tensor = super().forward(input_tensor)
        activated_tensor = activated_tensor+1
        return activated_tensor


avail_activations = {
    'selu': torch.nn.SELU,
    'relu': torch.nn.ReLU,
    'selu_kernel': SELUKernel
}
