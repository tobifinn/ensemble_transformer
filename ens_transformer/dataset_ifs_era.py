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
from typing import Tuple, Union

# External modules
from torch.utils.data.dataset import Dataset
import numpy as np
import torch
import zarr

# Internal modules


logger = logging.getLogger(__name__)


def to_tensor(in_array: "np.ndarray") -> torch.Tensor:
    in_tensor = torch.from_numpy(in_array)
    out_tensor = in_tensor.float()
    return out_tensor


class IFSERADataset(Dataset):
    def __init__(
            self,
            dataset_path: str,
            subsample_size: Union[None, int] = 20,
    ):
        super().__init__()
        self.dataset: zarr.Group = None
        self._dataset_path = None
        self.dataset_path = dataset_path
        self.subsample_size = subsample_size

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @dataset_path.setter
    def dataset_path(self, new_path: str) -> None:
        self.dataset = zarr.open(new_path, mode='r')
        self._dataset_path = new_path

    def __len__(self) -> int:
        return self.dataset["input"].shape[0]

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_tensor = self.dataset["input"][idx]
        target_tensor = self.dataset["target"][idx]

        if self.subsample_size is not None:
            ens_idx = np.random.choice(
                input_tensor.shape[0], size=self.subsample_size,
                replace=False
            )
            input_tensor = input_tensor[ens_idx]
        input_tensor = to_tensor(input_tensor)
        target_tensor = to_tensor(target_tensor)
        return input_tensor, target_tensor
