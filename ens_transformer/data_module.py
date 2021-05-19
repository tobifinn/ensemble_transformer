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
from typing import Optional, Union, Tuple, Callable, Iterable
import os

# External modules
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose

# Internal modules
from .dataset_ifs_era import IFSERADataset
from .transforms import to_tensor


logger = logging.getLogger(__name__)


class IFSERADataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = '../data/processed',
            batch_size: int = 64,
            normalizer_path: Union[None, str] =
                '../data/interim/normalizers.pt',
            include_vars: Union[None, Iterable[str]] = None,
            subsample_size: Union[None, int] = None,
            num_workers: int = 4,
            pin_memory: bool = True
    ):
        super().__init__()
        self._split_perc = 0.1
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.normalizer_path = normalizer_path
        self.include_vars = include_vars
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subsample_size = subsample_size

    @staticmethod
    def _init_transforms(
            normalizer_path: Union[None, str] = None
    ) -> Tuple[Callable, Callable]:
        if normalizer_path is not None:
            normalizers = torch.load(normalizer_path)
            input_transform = Compose([to_tensor, normalizers['ifs']])
            target_transform = Compose([to_tensor, normalizers['era']])
        else:
            input_transform = target_transform = to_tensor
        return input_transform, target_transform

    def setup(self, stage: Optional[str] = None) -> None:
        input_transform, target_transform = self._init_transforms(
            self.normalizer_path
        )
        self.ds_test = IFSERADataset(
            ifs_path=os.path.join(self.data_dir, 'ifs', 'ds_test'),
            era_path=os.path.join(self.data_dir, 'ifs', 'ds_test'),
            include_vars=self.include_vars,
            input_transform=input_transform,
            target_transform=target_transform,
            subsample_size=None
        )
        self.lats = self.ds_test.ifs['latitude'].values
        train_full = IFSERADataset(
            ifs_path=os.path.join(self.data_dir, 'ifs', 'ds_train'),
            era_path=os.path.join(self.data_dir, 'ifs', 'ds_train'),
            include_vars=self.include_vars,
            input_transform=input_transform,
            target_transform=target_transform,
            subsample_size=self.subsample_size
        )
        len_eval = int(len(train_full) * self._split_perc)
        len_train = len(train_full)-len_eval
        self.ds_train, self.ds_eval = random_split(
            train_full, [len_train, len_eval]
        )
        val_full = IFSERADataset(
            ifs_path=os.path.join(self.data_dir, 'ifs', 'ds_train'),
            era_path=os.path.join(self.data_dir, 'ifs', 'ds_train'),
            include_vars=self.include_vars,
            input_transform=input_transform,
            target_transform=target_transform,
            subsample_size=None
        )
        self.ds_eval.dataset = val_full

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
