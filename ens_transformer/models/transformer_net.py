#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 11.05.21
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
from hydra.utils import instantiate
from omegaconf import DictConfig

# Internal modules
from .base_net import BaseNet


logger = logging.getLogger(__name__)


class TransformerNet(BaseNet):
    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            embedded_channels: int = 64,
            n_transformers: int = 1
    ) -> Tuple[torch.nn.Sequential, int]:
        transformer_list = []
        for idx in range(n_transformers):
            curr_transformer = instantiate(cfg, n_channels=embedded_channels)
            transformer_list.append(curr_transformer)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers, embedded_channels
