#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 08.01.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Iterable, Tuple, Callable

# External modules
from torch.utils.data.dataset import Dataset
import xarray as xr
import numpy as np
import torch

# Internal modules


logger = logging.getLogger(__name__)


class IFSERADataset(Dataset):
    def __init__(
            self,
            ifs_path: str,
            era_path: str,
            include_vars: Union[None, Iterable[str]] = None,
            input_transform: Union[None, Callable] = None,
            target_transform: Union[None, Callable] = None,
            subsample_size: Union[None, int] = 20,
    ):
        super().__init__()
        self.ifs_path = ifs_path
        self.era_path = era_path
        self.include_vars = include_vars
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.subsample_size = subsample_size
        self.era5 = self.get_era5()
        self.ifs = self.get_ifs()

    def get_era5(self) -> xr.DataArray:
        ds_era = xr.open_zarr(self.era_path)['t2m']
        ds_era = ds_era - 273.15
        return ds_era

    def get_ifs(self) -> xr.DataArray:
        ds_ifs = xr.open_zarr(self.ifs_path)
        if self.include_vars is not None:
            ds_ifs = ds_ifs[list(self.include_vars)]
        ds_ifs = ds_ifs.to_array('var_name')
        ds_ifs = ds_ifs.transpose('time', 'ensemble', 'var_name', 'latitude',
                                  'longitude')
        return ds_ifs

    def __len__(self) -> int:
        return len(self.era5)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[Union[np.ndarray, torch.Tensor],
               Union[np.ndarray, torch.Tensor]]:
        ifs_tensor = self.ifs[idx].values
        if self.subsample_size is not None:
            ens_idx = np.random.choice(
                ifs_tensor.shape[0], size=self.subsample_size, replace=False
            )
            ifs_tensor = ifs_tensor[ens_idx]
        if self.input_transform is not None:
            ifs_tensor = self.input_transform(ifs_tensor)

        era_tensor = self.era5[idx].values
        if self.target_transform is not None:
            era_tensor = self.target_transform(era_tensor)
        return ifs_tensor, era_tensor
