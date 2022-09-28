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
from einops import rearrange

# Internal modules


logger = logging.getLogger(__name__)


class EnsembleMixinLayer(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_mixin: int = 64,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.normaliser = torch.nn.LayerNorm([n_channels, 32, 64])
        self.layer_in = torch.nn.Linear(n_channels, n_mixin)
        self.activation = torch.nn.GELU()
        self.layer_out = torch.nn.Linear(n_mixin, n_channels)
        self.gamma = torch.nn.Parameter(
            torch.full((n_channels,), layer_scale_init_value),
            requires_grad=True
        ) if layer_scale_init_value is not None else 1.

    def forward(self, in_tensor: torch.Tensor):
        residual_tensor = self.normaliser(in_tensor)
        residual_tensor = rearrange(
            residual_tensor,
            pattern="b e c h w -> b e h w c"
        )
        residual_tensor = self.layer_in(residual_tensor)
        residual_tensor = self.activation(residual_tensor)
        residual_tensor = self.layer_out(residual_tensor) * self.gamma
        residual_tensor = rearrange(
            residual_tensor,
            pattern="b e h w c -> b e c h w"
        )
        out_tensor = in_tensor + residual_tensor
        return out_tensor
