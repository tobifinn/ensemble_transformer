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
import torch.nn

# Internal modules


logger = logging.getLogger(__name__)


class LinearEmbedding(torch.nn.Module):
    def __init__(
            self,
            input_size: int = 6144,
            embedding_size: int = 256
    ):
        super().__init__()
        self.input_size = input_size
        self.embedding_layer = torch.nn.Linear(
            in_features=input_size, out_features=embedding_size, bias=False
        )

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        in_embed_tensor = in_tensor.view(
            *in_tensor.shape[:2], self.input_size
        )
        output_tensor = self.embedding_layer(in_embed_tensor)
        return output_tensor
