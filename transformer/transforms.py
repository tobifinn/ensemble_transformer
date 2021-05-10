#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 09.01.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Any, Dict

# External modules
import torch
import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


class Normalizer(object):
    def __init__(
            self,
            mean: torch.Tensor,
            std: torch.Tensor,
            eps: float = 1E-9
    ):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(
            self,
            input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        centered_tensor = input_tensor - self.mean
        normalized_tensor = (centered_tensor + self.eps) / (self.std + self.eps)
        return normalized_tensor


def to_tensor(input_array: np.ndarray) -> torch.Tensor:
    transformed_tensor = torch.from_numpy(input_array).float()
    return transformed_tensor
