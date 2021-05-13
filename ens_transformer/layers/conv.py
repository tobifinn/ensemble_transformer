#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.01.21
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
from ..utils import EnsembleWrapper


logger = logging.getLogger(__name__)


class EnsConv2d(torch.nn.Module):
    """
    Added viewing for ensemble-based 2d convolutions.
    """
    def __init__(
            self, *args, **kwargs
    ):
        super().__init__()
        self.conv2d = EnsembleWrapper(
            torch.nn.Conv2d(*args, **kwargs)
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        convolved_tensor = self.conv2d(in_tensor)
        return convolved_tensor
