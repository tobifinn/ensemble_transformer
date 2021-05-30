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
from ..layers import ResidualLayer

logger = logging.getLogger(__name__)


class DirectNet(BaseNet):
    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            embedded_channels: int = 64,
            n_transformers: int = 1
    ) -> Tuple[torch.nn.Sequential, int]:
        transformer_list = []
        for idx in range(n_transformers):
            residual_module: ResidualLayer = get_class(cfg._target_)(
                in_channels=embedded_channels,
                out_channels=embedded_channels,
                kernel_size=cfg.kernel_size,
                branch_activation=cfg.branch_activation,
                activation=cfg.activation,
                n_residuals=n_transformers
            )
            transformer_list.append(residual_module)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers, embedded_channels
