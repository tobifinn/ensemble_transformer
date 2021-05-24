#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 24.05.21
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2021}  {Tobias Sebastian Finn}


# System modules
import logging
import argparse
import os

# External modules
from hydra import initialize, compose
from hydra.utils import instantiate

import torch
import numpy as np
import xarray as xr

import pytorch_lightning as pl

from tqdm.autonotebook import tqdm

# Internal modules
from ens_transformer.data_module import IFSERADataModule
from ens_transformer.models import PPNNet


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Predict a dataset with trained '
                                             'models')
parser.add_argument(
    '--data_dir', type=str, default='../data/processed',
    help='The trainings data is stored in this directory, default: '
         '../data/processed'
)
parser.add_argument(
    '--model_path', type=str, default='../data/models',
    help='The path to the trained models, default: ../data/models'
)
parser.add_argument(
    '--exp_name', type=str, required=True,
    help='The experiment name that is used to create the prediction dataset '
         'and to load the model data'
)
parser.add_argument(
    '--store_path', type=str, default='../data/processed/prediction',
    help='The path were the prediction within a netCDF file should be stored, '
         'default: ../data/processed/prediction'
)
parser.add_argument(
    '--batch_size', type=int, default=64,
    help='The batch size for the prediction'
)


def predict_dataset(args: argparse.Namespace):
    device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    model_path = os.path.join(args.model_path, args.exp_name)
    with initialize(config_path=os.path.join(model_path, 'hydra')):
        cfg = compose('config.yaml')

    pl.seed_everything(42)

    ens_net = instantiate(
        cfg.model,
        in_channels=len(cfg.data.include_vars),
        learning_rate=cfg.learning_rate
    )
    ens_net = ens_net.load_from_checkpoint(
        os.path.join(model_path, 'last.ckpt'),
        map_location=device,
    )
    ens_net = ens_net.to(device)

    data_module: IFSERADataModule = instantiate(
        cfg['data']['data_module'],
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        pin_memory=True if torch.cuda.is_available() else False
    )
    data_module.setup()

    prediction = []
    test_dataloader = data_module.test_dataloader()
    for ifs_batch, _ in tqdm(iter(test_dataloader), total=len(test_dataloader)):
        ifs_batch = ifs_batch.to(device)
        with torch.no_grad():
            prediction.append(ens_net(ifs_batch).cpu())
    prediction = torch.cat(prediction, dim=0)

    try:
        norm_mean = data_module.ds_test.target_transform.transforms[1].mean
        norm_std = data_module.ds_test.target_transform.transforms[1].std
    except AttributeError:
        norm_mean = 0.
        norm_std = 1.

    if isinstance(ens_net, PPNNet):
        output_mean, output_std = ens_net._estimate_mean_std(prediction)
        output_mean = output_mean * norm_std + norm_mean
        output_std = output_std * norm_std
        output_dataset = xr.Dataset(
            {
                'mean': data_module.ds_test.era5.copy(data=output_mean.numpy()),
                'stddev': data_module.ds_test.era5.copy(data=output_std.numpy())
            }
        )
    else:
        prediction = prediction * norm_std + norm_mean
        template_ds = data_module.ds_test.era5.expand_dims(
            'ensemble', axis=1
        )
        output_dataset = xr.Dataset(
            {
                'mean': data_module.ds_test.era5.copy(
                    data=prediction.numpy().mean(axis=1)
                ),
                'stddev': data_module.ds_test.era5.copy(
                    data=prediction.numpy().std(axis=1, ddof=1)
                ),
                'particles': template_ds.copy(data=prediction.numpy())
            }
        )
        save_path = os.path.join(args.store_path,
                                 '{0:s}.nc'.format(args.exp_name))
        output_dataset.to_netcdf(save_path)

    save_path = os.path.join(args.store_path, '{0:s}.nc'.format(args.exp_name))
    output_dataset.to_netcdf(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments namespace: {0}'.format(str(args)))
    predict_dataset(args)
