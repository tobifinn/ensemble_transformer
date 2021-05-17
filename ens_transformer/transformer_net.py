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
from typing import Tuple, Union, Dict, Any

# External modules
import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

import numpy as np

# Internal modules
from .layers import EnsConv2d, EarthPadding
from .measures import crps_loss, WeightedScore


logger = logging.getLogger(__name__)


class TransformerNet(pl.LightningModule):
    def __init__(
            self,
            optimizer: DictConfig,
            scheduler: DictConfig,
            transformer: DictConfig,
            in_channels: int = 3,
            hidden_channels: int = 64,
            n_transformers: int = 1,
            learning_rate: float = 1E-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.randn(
            1, 50, in_channels, 32, 64
        )
        self.transformers = self._init_transformers(
            transformer,
            hidden_channels=hidden_channels,
            n_transformers=n_transformers
        )
        self.in_layer = torch.nn.Sequential(
            EarthPadding(3),
            EnsConv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=7
            )
        )
        self.output_layer = EnsConv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1
        )

        self.metrics = torch.nn.ModuleDict({
            'crps': WeightedScore(
                lambda prediction, target: crps_loss(
                    prediction[0], prediction[1], target[:, 2]
                ),
            ),
            'mse': WeightedScore(
                lambda prediction, target: (prediction[0]-target[:, 2]).pow(2),
            ),
            'var': WeightedScore(
                lambda prediction, target: prediction[1].pow(2),
            )
        })
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.save_hyperparameters()

    @property
    def in_size(self):
        return self.example_input_array[0, 0].numel()

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

    def configure_optimizers(
            self
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        if self.scheduler_cfg is not None:
            scheduler = instantiate(self.scheduler_cfg, optimizer=optimizer)
            optimizer = {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'eval_loss',
                    'interval': 'epoch',
                    'frequency': 1,
                }
            }
        return optimizer

    def forward(self, input_tensor) -> torch.Tensor:
        transformed_tensor = self.in_layer(input_tensor)
        for transformer in self.transformers:
            transformed_tensor = transformer(
                in_tensor=transformed_tensor,
            )
        output_tensor = self.output_layer(transformed_tensor).squeeze(dim=-3)
        return output_tensor

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        prediction = (output_mean, output_std)
        loss = self.metrics['crps'](prediction, target_tensor).mean()
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        prediction = (output_mean, output_std)
        crps = self.metrics['crps'](prediction, target_tensor).mean()
        rmse = self.metrics['mse'](prediction, target_tensor).mean().sqrt()
        spread = self.metrics['var'](prediction, target_tensor).mean().sqrt()
        self.log('eval_loss', crps, prog_bar=True)
        self.log('eval_rmse', rmse, prog_bar=True)
        self.log('eval_spread', spread, prog_bar=True)
        self.log('hp_metric', crps)
        return crps
