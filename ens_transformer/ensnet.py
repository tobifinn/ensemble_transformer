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
import pytorch_lightning as pl
import torch

import numpy as np

# Internal modules
from .layers import EnsConv2d, avail_transformers
from .measures import crps_loss, WeightedScore


logger = logging.getLogger(__name__)


class EnsNet(pl.LightningModule):
    def __init__(
            self,
            embedding_size: int = 256,
            embedding_hidden: int = 0,
            in_channels: int = 3,
            learning_rate: float = 1E-3,
            n_transformers: int = 1,
            transformer_name: str = 'ensemble',
            n_transform_channels: int = 64,
            value_activation: Union[None, str] = None,
            n_key_neurons: int = 1,
            coarsening_factor: int = 1,
            key_activation: Union[None, str] = None,
            interpolation_mode: str = 'bilinear',
            same_key_query: bool = False,
            grid_dims: Tuple[int, int] = (32, 64),
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.randn(
            1, 50, in_channels, 32, 64
        )
        self.embedding = self._init_embedding(
            embedding_size=embedding_size,
            embedding_hidden=embedding_hidden
        )
        self.first_shortcut = EnsConv2d(
            in_channels=1,
            out_channels=n_transform_channels,
            kernel_size=1
        )
        self.transform_layers = self._init_transformers(
            out_channels=n_transform_channels,
            n_transformers=n_transformers,
            transformer_name=transformer_name,
            embedding_size=embedding_size,
            value_activation=value_activation,
            n_key_neurons=n_key_neurons,
            coarsening_factor=coarsening_factor,
            key_activation=key_activation,
            interpolation_mode=interpolation_mode,
            grid_dims=grid_dims,
            same_key_query=same_key_query
        )

        self.output_layer = EnsConv2d(
            in_channels=n_transform_channels,
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
        self.save_hyperparameters(
            'embedding_size',
            'embedding_hidden',
            'learning_rate',
            'in_channels',
            'n_transformers',
            'transformer_name',
            'n_transform_channels',
            'value_activation',
            'n_key_neurons',
            'coarsening_factor',
            'key_activation',
            'interpolation_mode',
            'same_key_query'
        )

    @property
    def in_size(self):
        return self.example_input_array[:, 0].numel()

    def _init_embedding(
            self,
            embedding_size: int = 256,
            embedding_hidden: int = 0
    ) -> torch.nn.Module:
        if embedding_hidden > 0:
            embedding = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self.in_size,
                    out_features=embedding_hidden,
                    bias=True
                ),
                torch.nn.SELU(inplace=True),
                torch.nn.Linear(
                    in_features=embedding_hidden,
                    out_features=embedding_size,
                    bias=True
                )
            )
        else:
            embedding = torch.nn.Linear(
                in_features=self.in_size,
                out_features=embedding_size,
                bias=True
            )
        return embedding

    @staticmethod
    def _init_transformers(
            out_channels: int = 64,
            n_transformers: int = 1,
            transformer_name: str = 'ensemble',
            embedding_size: int = 256,
            value_activation: Union[None, str] = 'selu',
            n_key_neurons: int = 1,
            coarsening_factor: int = 1,
            key_activation: Union[None, str] = 'relu',
            interpolation_mode: str = 'bilinear',
            grid_dims: Tuple[int, int] = (32, 64),
            same_key_query: bool = False
    ) -> torch.nn.ModuleList:
        in_channels = 1
        transformer_list = torch.nn.ModuleList()
        for idx in range(n_transformers):
            curr_transformer = avail_transformers[transformer_name](
                in_channels=in_channels,
                out_channels=out_channels,
                value_activation=value_activation,
                embedding_size=embedding_size,
                n_key_neurons=n_key_neurons,
                coarsening_factor=coarsening_factor,
                key_activation=key_activation,
                interpolation_mode=interpolation_mode,
                grid_dims=grid_dims,
                same_key_query=same_key_query
            )
            in_channels = out_channels
            transformer_list.append(curr_transformer)
        return transformer_list

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def forward(self, input_tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        in_embed_tensor = input_tensor.view(
            *input_tensor.shape[:2], self.in_size
        )
        embedding_tensor = self.embedding(in_embed_tensor)
        transformed_tensor = input_tensor[..., [0], :, :]
        shortcut_tensor = self.first_shortcut(transformed_tensor)
        for transformer in self.transform_layers:
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
            embedded_ens = embedded_ens.view(-1, self.hparams['embedding_size'])
            labels = labels.view(-1)
            self.logger.experiment.add_embedding(
                embedded_ens, metadata=labels, tag='weather_embedding',
                global_step=self.global_step
            )
        return crps
