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
import argparse

# External modules
import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig

import pytorch_lightning as pl

# Internal modules


main_logger = logging.getLogger(__name__)


@hydra.main(config_path='../configs', config_name='config')
def main_train(
        cfg: DictConfig
) -> None:
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

    trainer: pl.Trainer = instantiate(
        cfg.trainer
    )
    if hasattr(cfg, 'tune_batch_size'):
        max_batch_size = trainer.tuner.scale_batch_size(
            model=network, datamodule=data_module
        )
        print('Maximum batch size: {0}'.format(max_batch_size))

    if hasattr(cfg, 'tune_lr'):
        lr_finder = trainer.tuner.lr_find(
            model=network, datamodule=data_module, num_training=500
        )
        fig = lr_finder.plot(suggest=True)
        fig.savefig('learning_rate.png', dpi=300)
        fig.show()
        print('LR Suggestion: {0}'.format(lr_finder.suggestion()))


if __name__ == '__main__':
    main_train()
