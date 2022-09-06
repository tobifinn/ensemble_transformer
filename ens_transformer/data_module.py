#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 10.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Optional, Union, Tuple, Callable
import os

# External modules
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

# Internal modules
from .dataset_ifs_era import IFSERADataset
from .transforms import to_tensor


logger = logging.getLogger(__name__)


class IFSERADataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = '../data/processed/dataset',
            batch_size: int = 64,
            subsample_size: Union[None, int] = None,
            pin_memory: bool = True,
            num_workers: int = 4
    ):
        super().__init__()
        self._split_perc = 0.1
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.subsample_size = subsample_size

        self.ds_train = None
        self.ds_eval = None
        self.ds_test = None

    @staticmethod
    def _init_transforms(
            normalizer_path: Union[None, str] = None
    ) -> Tuple[Callable, Callable]:
        if normalizer_path is not None:
            normalizers = torch.load(normalizer_path)
            input_transform = Compose([to_tensor, normalizers['ifs']])
        else:
            input_transform = to_tensor
        target_transform = to_tensor
        return input_transform, target_transform

    def setup(self, stage: Optional[str] = None) -> None:
        self.ds_train = IFSERADataset(
            dataset_path=os.path.join(self.data_dir, 'train'),
            subsample_size=self.subsample_size
        )
        self.ds_eval = IFSERADataset(
            dataset_path=os.path.join(self.data_dir, 'eval'),
            subsample_size=None
        )
        self.ds_test = IFSERADataset(
            dataset_path=os.path.join(self.data_dir, 'test'),
            subsample_size=None
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_train,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=True,
            batch_size=self.batch_size
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_eval,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.ds_test,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            batch_size=self.batch_size
        )
