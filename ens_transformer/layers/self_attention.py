#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 07.09.22
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Tuple
from math import sqrt

# External modules
import torch.nn
from einops import rearrange

# Internal modules


logger = logging.getLogger(__name__)


class EnsembleSelfAttention(torch.nn.Module):
    def __init__(
            self,
            n_channels: int = 64,
            n_heads: int = 64,
    ):
        super().__init__()
        self.normaliser = torch.nn.LayerNorm([n_channels, 32, 64])
        self.value_layer = torch.nn.Linear(n_channels, n_heads, bias=False)
        self.query_layer = torch.nn.Linear(n_channels, n_heads, bias=False)
        self.key_layer = torch.nn.Linear(n_channels, n_heads, bias=False)
        self.out_layer = torch.nn.Linear(n_heads, n_channels, bias=False)
        self.scale = (32 * 64) ** -0.5

    def project_tensor(
            self,
            in_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_tensor = self.query_layer(in_tensor)
        key_tensor = self.key_layer(in_tensor)
        value_tensor = self.value_layer(in_tensor)
        return value_tensor, key_tensor, query_tensor

    def estimate_attention(self, key: torch.Tensor, query: torch.Tensor):
        dot_product = torch.einsum('bihwc, bjhwc->bijc', key, query)
        dot_product = dot_product * self.scale
        attention = torch.softmax(dot_product, dim=-3)
        return attention

    def forward(self, in_tensor):
        normalised_tensor = self.normaliser(in_tensor)
        normalised_channels_last = rearrange(
            normalised_tensor, pattern="bechw->behwc"
        )
        v_tensor, k_tensor, q_tensor = self.project_tensor(
            normalised_channels_last
        )
        attention = self.estimate_attention(k_tensor, q_tensor)
        transformed_v = torch.einsum(
            'bijc,bihwc->bjhwc', attention, v_tensor
        )
        out_tensor = self.out_layer(transformed_v)
        out_tensor = rearrange(out_tensor, pattern="behwc->bechw")
        out_tensor = in_tensor + out_tensor
        return out_tensor
