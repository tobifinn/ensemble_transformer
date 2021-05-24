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
import os
import logging

# External modules
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

import matplotlib.figure

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase

# Internal modules


main_logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='config')
def main_train(cfg: DictConfig) -> None:
    os.chdir(get_original_cwd())
    pl.seed_everything(cfg.seed, workers=True)

    data_module: pl.LightningDataModule = instantiate(cfg.data.data_module)
    data_module.setup()

    network: pl.LightningModule = instantiate(
        cfg.model,
        in_channels=len(cfg.data.include_vars),
        learning_rate=cfg.learning_rate
    )
    network.hparams['batch_size'] = cfg.batch_size

    if cfg.callbacks is not None:
        callbacks = []
        for _, callback_cfg in cfg.callbacks.items():
            curr_callback: pl.callbacks.Callback = instantiate(callback_cfg)
            callbacks.append(curr_callback)
    else:
        callbacks = None

    training_logger = None
    if cfg.logger is not None:
        for _, logger_cfg in cfg.logger.items():
            training_logger: LightningLoggerBase = instantiate(logger_cfg)

    trainer: pl.Trainer = instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=training_logger
    )

    trainer.fit(model=network, datamodule=data_module)


if __name__ == '__main__':
    main_train()
