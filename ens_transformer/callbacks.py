#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 23.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging

# External modules
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer import Trainer

# Internal modules


logger = logging.getLogger(__name__)


class ValidationOnStartCallback(Callback):
    # Based on https://github.com/PyTorchLightning/pytorch-lightning/issues/1715#issuecomment-642480058
    def __init__(self):
        pass

    def on_train_start(self, trainer: Trainer, pl_module):
        return trainer.run_evaluation()
