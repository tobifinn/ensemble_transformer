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
from .layers.conv import EnsConv2d
from .measures import crps_loss, WeightedScore


logger = logging.getLogger(__name__)


class TransformerNet(pl.LightningModule):
    def __init__(
            self,
            embedding: DictConfig,
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
        self.embedding = self._init_embedding(embedding)
        self.transformers = self._init_transformers(
            transformer,
            hidden_channels=hidden_channels,
            n_transformers=n_transformers
        )
        self.first_shortcut = EnsConv2d(
            in_channels=1,
            out_channels=hidden_channels,
            kernel_size=1
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
    def _init_embedding(
            embedding_cfg: DictConfig
    ) -> torch.nn.Module:
        embedding = instantiate(embedding_cfg)
        return embedding

    @staticmethod
    def _init_transformers(
            cfg: DictConfig,
            hidden_channels: int = 64,
            n_transformers: int = 1
    ) -> torch.nn.ModuleList:
        in_channels = 1
        transformer_list = torch.nn.ModuleList()
        for idx in range(n_transformers):
            curr_transformer = get_class(cfg._target_)(
                in_channels=in_channels,
                out_channels=hidden_channels,
                value_activation=cfg.value_activation,
                embedding_size=cfg.embedding_size,
                n_key_neurons=cfg.n_key_neurons,
                coarsening_factor=cfg.coarsening_factor,
                key_activation=cfg.key_activation,
                interpolation_mode=cfg.interpolation_mode,
                grid_dims=cfg.grid_dims,
                same_key_query=cfg.same_key_query
            )
            in_channels = hidden_channels
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

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        in_embed_tensor = input_tensor.view(
            *input_tensor.shape[:2], self.in_size
        )
        embedding_tensor = self.embedding(in_embed_tensor)
        transformed_tensor = input_tensor[..., [0], :, :]
        shortcut_tensor = self.first_shortcut(transformed_tensor)
        for transformer in self.transformers:
            transformed_tensor = transformer(
                in_tensor=transformed_tensor,
                embedding=embedding_tensor
            )
            transformed_tensor = transformed_tensor + shortcut_tensor
            shortcut_tensor = transformed_tensor
        output_tensor = self.output_layer(transformed_tensor).squeeze(dim=-3)
        return output_tensor, embedding_tensor

    def training_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble, _ = self(in_tensor)
        output_mean = output_ensemble.mean(dim=1)
        output_std = output_ensemble.std(dim=1, unbiased=True)
        prediction = (output_mean, output_std)
        loss = self.metrics['crps'](prediction, target_tensor).mean()
        return loss

    def validation_step(
            self,
            batch: Tuple[torch.Tensor, torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        in_tensor, target_tensor = batch
        output_ensemble, embedded_ens = self(in_tensor)
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
        if batch_idx == 0:
            labels = torch.arange(self.hparams['batch_size'],
                                  device=self.device)
            labels = labels.view(self.hparams['batch_size'], 1)
            labels = torch.ones_like(embedded_ens[..., 0]) * labels
            embedded_ens = embedded_ens.view(
                -1, self.hparams['embedding']['embedding_size']
            )
            labels = labels.view(-1)
            self.logger.experiment.add_embedding(
                embedded_ens, metadata=labels, tag='weather_embedding',
                global_step=self.global_step
            )
        return crps
