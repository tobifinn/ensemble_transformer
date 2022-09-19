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
from abc import abstractmethod

# External modules
import pytorch_lightning as pl
import torch
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig

import numpy as np

# Internal modules
from ..layers import EnsConv2d
from ..measures import crps_loss, WeightedScore


logger = logging.getLogger(__name__)


class BaseNet(pl.LightningModule):
    def __init__(
            self,
            optimizer: DictConfig,
            transformer: DictConfig,
            embedding: DictConfig,
            in_channels: int = 3,
            output_channels: int = 1,
            n_transformers: int = 1,
            learning_rate: float = 1E-3,
            loss_str: str = 'crps'
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.randn(
            1, 50, in_channels, 32, 64
        )
        self.embedding = instantiate(embedding, in_channels=in_channels)
        self.transformers, out_channels = self._init_transformers(
            transformer,
            embedded_channels=embedding.n_channels[-1],
            n_transformers=n_transformers
        )
        self.output_layer = EnsConv2d(
            in_channels=out_channels,
            out_channels=output_channels,
            kernel_size=1
        )

        self.metrics = torch.nn.ModuleDict({
            'crps': WeightedScore(
                lambda prediction, target: crps_loss(
                    prediction[0], prediction[1], target
                ),
            ),
            'mse': WeightedScore(
                lambda prediction, target: (prediction[0]-target).pow(2),
            ),
            'var': WeightedScore(
                lambda prediction, target: prediction[1].pow(2),
            )
        })
        self.loss_function = self.metrics[loss_str]
        self.optimizer_cfg = optimizer
        self.save_hyperparameters()

    @property
    def in_size(self):
        return self.example_input_array[0, 0].numel()

    @staticmethod
    @abstractmethod
    def _init_transformers(
            cfg: DictConfig,
            embedded_channels: int = 64,
            n_transformers: int = 1
    ) -> Tuple[torch.nn.Sequential, int]:
        pass

    @staticmethod
    def _estimate_mean_std(
            output_ensemble: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        output_std = output_std.clamp(min=1E-6, max=1E6)
        return output_mean, output_std

    def configure_optimizers(
            self
    ) -> Union[torch.optim.Optimizer, Dict[str, Any]]:
        optimizer = instantiate(self.optimizer_cfg, self.parameters())
        return optimizer

    def forward(self, input_tensor) -> torch.Tensor:
        embedded_tensor = self.embedding(input_tensor)
        transformed_tensor = self.transformers(embedded_tensor)
        output_tensor = self.output_layer(transformed_tensor).squeeze(dim=-3)
        return output_tensor

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        output_mean, output_std = self._estimate_mean_std(output_ensemble)
        prediction = (output_mean, output_std)
        loss = self.loss_function(prediction, target_tensor).mean()
        self.log('loss', loss, prog_bar=True)
        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble = self(in_tensor)
        output_mean, output_std = self._estimate_mean_std(output_ensemble)
        prediction = (output_mean, output_std)
        loss = self.loss_function(prediction, target_tensor).mean()
        crps = self.metrics['crps'](prediction, target_tensor).mean()
        rmse = self.metrics['mse'](prediction, target_tensor).mean().sqrt()
        spread = self.metrics['var'](prediction, target_tensor).mean().sqrt()
        self.log('eval_loss', loss, prog_bar=True)
        self.log('eval_crps', crps, prog_bar=True)
        self.log('eval_rmse', rmse, prog_bar=True)
        self.log('eval_spread', spread, prog_bar=True)
        self.log('hp_metric', loss)
        return crps
