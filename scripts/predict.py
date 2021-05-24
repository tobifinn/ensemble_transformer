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


def predict_dataset(args: argparse.Namespace):
    model_path = os.path.join(args.model_path, args.exp_name)
    with initialize(config_path=os.path.join(model_path, 'hydra')):
        cfg = compose('config.yaml')
    ens_net = instantiate(
        cfg.model,
        in_channels=len(cfg.data.include_vars),
        learning_rate=cfg.learning_rate
    )
    ens_net = ens_net.load_from_checkpoint(
        os.path.join(model_path, 'last.ckpt'),
    )

    data_module: IFSERADataModule = instantiate(
        cfg['data']['data_module'],
        data_dir=args.data_dir,
    )
    data_module.setup()


    try:
        norm_mean = data_module.ds_test.target_transform[1].mean
        norm_std = data_module.ds_test.target_transform[1].std
    except AttributeError:
        norm_mean = 0.
        norm_std = 1.

    parametric = False
    if isinstance(ens_net, PPNNet):
        parametric = True

    prediction = []
    data_iter = iter(data_module.test_dataloader())
    for ifs_data, _ in tqdm(data_iter):
        with torch.no_grad():
            net_output, _ = ens_net(ifs_data)
            if parametric:
                output_mean, output_std = ens_net._estimate_mean_std(net_output)
                output_mean = output_mean * norm_std + norm_mean
                output_std = output_std * norm_std
                net_output = torch.stack((output_mean, output_std), dim=1)
            else:
                net_output = net_output * norm_std + norm_mean
            prediction.append(net_output.detach().cpu().numpy())
    prediction = np.concatenate(prediction, axis=0)
    if parametric:
        output_dataset = xr.Dataset(
            {
                'mean': data_module.ds_test.era.copy(data=prediction[:, 0]),
                'stddev': data_module.ds_test.era.copy(data=prediction[:, 1])
            }
        )

    else:
        template_ds = data_module.ds_test.era.expand_dims(
            'ensemble', axis=1
        )
        output_dataset = xr.Dataset(
            {
                'mean': data_module.ds_test.era.copy(
                    data=prediction.mean(axis=1)
                ),
                'stddev': data_module.ds_test.era.copy(
                    data=prediction.std(axis=1, ddof=1)
                ),
                'particles': template_ds.copy(data=prediction)
            }
        )
    save_path = os.path.join(args.store_path, '{0:s}.nc'.format(args.exp_name))
    output_dataset.to_netcdf(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments namespace: {0}'.format(str(args)))
    predict_dataset(args)
