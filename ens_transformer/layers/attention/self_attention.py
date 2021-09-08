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
from typing import Tuple

# External modules
import torch

# Internal modules
from .reweighters import Reweighter
from .weight_estimators import WeightEstimator


logger = logging.getLogger(__name__)


class SelfAttentionModule(torch.nn.Module):
    def __init__(
            self,
            value_projector: torch.nn.Module,
            key_projector: torch.nn.Module,
            query_projector: torch.nn.Module,
            output_projector: torch.nn.Module,
            activation: torch.nn.Module,
            weight_estimator: WeightEstimator,
            reweighter: Reweighter
     ):
        super().__init__()
        self.value_projector = value_projector
        self.key_projector = key_projector
        self.query_projector = query_projector
        self.output_projector = output_projector
        self.activation = activation
        self.weight_estimator = weight_estimator
        self.reweighter = reweighter

    def project_input(
            self,
            in_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        value = self.value_projector(in_tensor)
        key = self.key_projector(in_tensor)
        query = self.query_projector(in_tensor)
        return value, key, query

    def forward(self, in_tensor: torch.Tensor):
        value, key, query = self.project_input(in_tensor)
        weights = self.weight_estimator(key, query)
        transformed_values = self.reweighter(value, weights)
        output_tensor = self.output_projector(transformed_values)
        output_tensor += in_tensor
        activated_tensor = self.activation(output_tensor)
        return activated_tensor
