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

# External modules
import torch
from omegaconf import DictConfig
from hydra.utils import get_class

# Internal modules
from .base_net import BaseNet
from ..layers import EnsConv2d, EarthPadding

logger = logging.getLogger(__name__)


class DirectNet(BaseNet):
    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            in_channels: int = 64,
            hidden_channels: int = 64,
            n_transformers: int = 1
    ) -> torch.nn.Sequential:
        transformer_list = []
        curr_channels = in_channels
        for idx in range(n_transformers):
            submodule = torch.nn.Sequential(
                EarthPadding((cfg.kernel_size-1)//2),
                EnsConv2d(curr_channels, hidden_channels,
                          kernel_size=cfg.kernel_size, padding=0),
                get_class(cfg.activation)(inplace=True)
            )
            curr_channels = hidden_channels
            transformer_list.append(submodule)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers
