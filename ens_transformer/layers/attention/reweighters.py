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

# Internal modules
from ens_transformer.ens_utils import split_mean_perts


logger = logging.getLogger(__name__)


class Reweighter(torch.nn.Module):
    @staticmethod
    def _apply_weights(
            apply_weights_to: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        weighted_tensor = torch.einsum(
            'bcij, bic...->bjc...', weight_tensor, apply_weights_to
        )
        return weighted_tensor

    @abc.abstractmethod
    def forward(
            self,
            state_tensor: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        pass


class StateReweighter(Reweighter):
    def forward(
            self,
            state_tensor: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        _, perts_tensor = split_mean_perts(state_tensor, dim=1)
        weighted_perts = self._apply_weights(perts_tensor, weight_tensor)
        output_tensor = state_tensor + weighted_perts
        return output_tensor


class MeanReweighter(Reweighter):
    def forward(
            self,
            state_tensor: torch.Tensor,
            weight_tensor: torch.Tensor
    ) -> torch.Tensor:
        mean_tensor, perts_tensor = split_mean_perts(state_tensor, dim=1)
        weighted_perts = self._apply_weights(perts_tensor, weight_tensor)
        output_tensor = mean_tensor + weighted_perts
        return output_tensor
