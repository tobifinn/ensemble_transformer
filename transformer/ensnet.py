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
from typing import Tuple, Callable

# External modules
import pytorch_lightning as pl
import torch

import numpy as np

# Internal modules
from .layers.conv import EnsConv2d
from .measures import crps_loss, WeightedScore


logger = logging.getLogger(__name__)


class EnsNet(pl.LightningModule):
    def __init__(
            self,
            lats: np.ndarray,
            embedding_size: int = 256,
            embedding_hidden: int = 0,
            in_channels: int = 3,
            learning_rate: float = 1E-3,
            n_transformers: int = 1,
            coarsening_factor: int = 1,
            n_transform_channels: int = 64,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.example_input_array = torch.randn(
            1, in_channels, 32, 64
        )
        self.embedding = self._init_embedding(
            embedding_size=embedding_size,
            embedding_hidden=embedding_hidden
        )
        self.transform_layers = self._init_transformers()
        self.output_layer = EnsConv2d(
            in_channels=n_transform_channels,
            out_channels=1,
            kernel_size=1
        )
        self.metrics = {
            'crps': WeightedScore(
                lambda prediction, target: crps_loss(
                    prediction[0], prediction[1], target[:, 2]
                ),
                lats=lats
            ),
            'mse': WeightedScore(
                lambda prediction, target: (prediction[0]-target[:, 2]).pow(2),
                lats=lats
            ),
            'var': WeightedScore(
                lambda prediction, target: prediction[1].pow(2),
                lats=lats
            )
        }

    @property
    def in_size(self):
        return self.example_input_array.numel

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

    def _init_transformers(self) -> torch.nn.ModuleList:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate
        )
        return optimizer

    def forward(self, input_tensor) -> torch.Tensor:
        in_embed_tensor = input_tensor.view(
            *input_tensor.shape[:2], self.in_size
        )
        embedding_tensor = self.embedding(in_embed_tensor)
        transformed_tensor = input_tensor[..., 0, :, :]
        for transformer in self.transform_layers:
            transformed_tensor = transformer(
                in_tensor=transformed_tensor,
                embedding=embedding_tensor
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
        self.log('loss', loss, prog_bar=True, logger=True)
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
        self.log("hp_metric", crps)
        return crps
