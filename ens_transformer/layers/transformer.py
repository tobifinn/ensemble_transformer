#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 27.09.22
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
import torch.nn

# Internal modules
from .self_attention import EnsembleSelfAttention
from .mixin import EnsembleMixinLayer


logger = logging.getLogger(__name__)


class EnsembleTransformerLayer(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
            n_mixin: int = 64,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.self_attention = EnsembleSelfAttention(
            n_channels=n_channels,
            n_heads=n_heads,
            layer_scale_init_value=layer_scale_init_value
        )
        self.mixin = EnsembleMixinLayer(
            n_channels=n_channels,
            n_mixin=n_mixin,
            layer_scale_init_value=layer_scale_init_value
        )

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        transformed_tensor = self.self_attention(input_tensor)
        mixed_tensor = self.mixin(transformed_tensor)
        return mixed_tensor
