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
from typing import Tuple, Union

# External modules
import torch
from hydra.utils import instantiate, get_class
from omegaconf import DictConfig

import numpy as np

# Internal modules
from .base_net import BaseNet
from ens_transformer.layers import EnsConv2d
from ens_transformer.layers import attention as attention_layers


logger = logging.getLogger(__name__)


class TransformerNet(BaseNet):
    @staticmethod
    def _construct_value_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            value_layer: bool = True,
    ) -> torch.nn.Sequential:
        layers = []
        if value_layer:
            conv_layer = EnsConv2d(
                in_channels=n_channels,
                out_channels=n_heads,
                kernel_size=1,
                bias=False
            )
            layers.append(conv_layer)
        return torch.nn.Sequential(*layers)

    @staticmethod
    def _construct_branch_layer(
            n_channels: int = 64,
            n_heads: int = 64,
            key_activation: Union[None, str] = None,
    ) -> torch.nn.Sequential:
        conv_layer = EnsConv2d(
            in_channels=n_channels,
            out_channels=n_heads,
            kernel_size=1,
            bias=False
        )
        if key_activation == 'torch.nn.SELU':
            lecun_stddev = np.sqrt(1/n_channels)
            torch.nn.init.normal_(
                conv_layer.conv2d.base_layer.weight,
                std=lecun_stddev
            )
        layers = [conv_layer]
        if key_activation:
            layers.append(get_class(key_activation)(inplace=True))
        return torch.nn.Sequential(*layers)

    def _construct_attention_module(
            self,
            cfg: DictConfig,
            n_channels: int = 64
    ) -> torch.nn.Module:
        value_layer = self._construct_value_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            value_layer=cfg['value_layer']
        )
        key_layer = self._construct_branch_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            key_activation=cfg['key_activation']
        )
        query_layer = self._construct_branch_layer(
            n_channels=n_channels, n_heads=cfg['n_heads'],
            key_activation=cfg['key_activation']
        )
        if cfg['layer_norm']:
            layer_norm = torch.nn.LayerNorm([n_channels, 32, 64])
            value_layer = torch.nn.Sequential(layer_norm, value_layer)
            key_layer = torch.nn.Sequential(layer_norm, key_layer)
            query_layer = torch.nn.Sequential(layer_norm, query_layer)
            
        out_layer = EnsConv2d(
            in_channels=cfg['n_heads'], out_channels=n_channels, kernel_size=1,
            padding=0
        )
        torch.nn.init.zeros_(out_layer.conv2d.base_layer.weight)
        
        module = attention_layers.SelfAttentionModule(
            value_projector=value_layer,
            key_projector=key_layer,
            query_projector=query_layer,
            output_projector=out_layer,
            activation=cfg['activation'],
            reweighter=instantiate(cfg['reweighter']),
            weight_estimator=instantiate(cfg['zeight_estimator'])
        )
        return module

    def _init_transformers(
            self,
            cfg: DictConfig,
            embedded_channels: int = 64,
            n_transformers: int = 1
    ) -> Tuple[torch.nn.Sequential, int]:
        transformer_list = []
        for idx in range(n_transformers):
            curr_transformer = self._construct_attention_module(
                cfg=cfg, n_channels=embedded_channels
            )
            transformer_list.append(curr_transformer)
        transformers = torch.nn.Sequential(*transformer_list)
        return transformers, embedded_channels
