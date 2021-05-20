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

# External modules
import torch
from hydra.utils import get_class
from omegaconf import DictConfig

# Internal modules
from .base_net import BaseNet


logger = logging.getLogger(__name__)


class TransformerNet(BaseNet):
    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            hidden_channels: int = 64,
            n_transformers: int = 1
    ) -> torch.nn.ModuleList:
        transformer_list = torch.nn.ModuleList()
        for idx in range(n_transformers):
            curr_transformer = get_class(cfg._target_)(
                channels=hidden_channels,
                activation=cfg.activation,
                key_activation=cfg.key_activation,
                same_key_query=cfg.same_key_query,
                value_layer=cfg.value_layer
            )
            transformer_list.append(curr_transformer)
        return transformer_list
