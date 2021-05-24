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
            embedded_channels: int = 64,
            n_transformers: int = 1
    ) -> torch.nn.Sequential:
        transformer_list = []
        for idx in range(n_transformers):
            curr_transformer = []
            if cfg.kernel_size > 1:
                curr_transformer.append(
                    EarthPadding(pad_size=(cfg.kernel_size-1) // 2)
                )
            curr_transformer.append(
                EnsConv2d(embedded_channels, embedded_channels,
                          kernel_size=cfg.kernel_size, padding=0),
            )
            curr_transformer.append(get_class(cfg.activation)(inplace=True))
            submodule = torch.nn.Sequential(*curr_transformer)
            transformer_list.append(submodule)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers
