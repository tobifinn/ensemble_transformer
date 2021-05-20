#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 20.05.21
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
from omegaconf import DictConfig
from hydra.utils import get_class

# Internal modules
from .base_net import BaseNet
from ..layers import EarthPadding


logger = logging.getLogger(__name__)


class PPNNet(BaseNet):
    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            embedded_channels: int = 64,
            hidden_channels: int = 64,
            n_transformers: int = 1
    ) -> torch.nn.Sequential:
        transformer_list = []
        in_channels = embedded_channels + 2
        for idx in range(n_transformers):
            submodule = torch.nn.Sequential(
                EarthPadding((cfg.kernel_size - 1) // 2),
                torch.nn.Conv2d(
                    in_channels, hidden_channels,
                    kernel_size=cfg.kernel_size, padding=0
                ),
                get_class(cfg.activation)(inplace=True)
            )
            in_channels = hidden_channels
            transformer_list.append(submodule)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers

    @staticmethod
    def _estimate_mean_std(
            output_ensemble: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_mean = output_ensemble[:, 0]
        output_stddev = torch.exp(0.5 * output_ensemble[:, 1])
        return output_mean, output_stddev

    def forward(
            self, input_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded_tensor = self.embedding(input_tensor)
        mean_embedding = embedded_tensor.mean(dim=1)
        in_mean = input_tensor[..., [0], :, :].mean(dim=1)
        in_std = input_tensor[..., [0], :, :].std(dim=1)
        in_embedded = torch.cat([mean_embedding, in_mean, in_std], dim=-3)
        transformed_tensor = self.transformers(in_embedded)
        output_tensor = self.output_layer(transformed_tensor)
        output_tensor = output_tensor.view(-1, 2, 32, 64)
        embedded_tensor = embedded_tensor.view(*embedded_tensor.shape[:2], -1)
        return output_tensor, embedded_tensor