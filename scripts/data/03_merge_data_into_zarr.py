#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 06.09.22
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.finn@enpc.fr
#
#    Copyright (C) {2022}  {Tobias Sebastian Finn}


# System modules
import logging
import argparse
import os

# External modules
import xarray as xr
import numpy as np

# Internal modules
import utils


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Merge and regrid data from ERA5')
parser.add_argument(
    '--era5_path', type=str,
    help='The raw ERA5 data is stored within this directory'
)
parser.add_argument(
    "--ifs_path", type=str,
    help="the raw IFS data is stored within this directory"
)
parser.add_argument(
    '--data_path', type=str, help='The prepared data is stored in this files'
)
parser.add_argument(
    '--eval_fraction', type=float, default=0.1,
    help="The fraction of evaluation data"
)
parser.add_argument("--n_workers", type=int, default=4)
parser.add_argument("--cluster_address", type=str, default=None)
parser.add_argument(
    "--seed", type=int, default=42,
)


def process_ifs_data(data_path: str) -> xr.DataArray:
    ds_ifs_pl = xr.open_mfdataset(
        os.path.join(data_path, "pl_*.nc"),
        parallel=True, chunks={'time': 2}
    )
    ds_ifs_t = ds_ifs_pl['t'].to_dataset('level')
    ds_ifs_t = ds_ifs_t.rename({500: 't_500', 850: 't_850'})
    ds_ifs_gh = ds_ifs_pl['gh'].to_dataset('level')
    ds_ifs_gh = ds_ifs_gh.rename({500: 'gh_500', 850: 'gh_850'})

    ds_ifs_sfc = xr.open_mfdataset(
        os.path.join(data_path, "sfc_*.nc"),
        parallel=True, chunks={'time': 2}
    )
    ds_ifs_merged = xr.merge([ds_ifs_sfc, ds_ifs_t, ds_ifs_gh])
    ds_ifs_merged = ds_ifs_merged[["t2m", "t_850", "gh_500"]]
    ds_ifs_merged = ds_ifs_merged.isel(latitude=slice(None, None, -1))
    ds_ifs_merged = ds_ifs_merged.rename({'number': 'ensemble'})
    ds_ifs_merged = ds_ifs_merged.to_array(dim="in_channels")
    ds_ifs_merged = ds_ifs_merged.transpose(
        "time", "ensemble", "in_channels", "latitude", "longitude"
    )
    logger.info("Loaded the IFS data")
    return ds_ifs_merged


def process_era_data(data_path: str) -> xr.DataArray:
    ds_era5 = xr.open_dataset(
        os.path.join(data_path, 'regridded_merged.nc'),
        chunks=({'time': 2})
    )
    ds_era5 = ds_era5[['t2m']]
    ds_era5 = ds_era5.rename({'lat': 'latitude', 'lon': 'longitude'})
    ds_era5 = ds_era5.to_array(dim="out_channels")
    ds_era5 = ds_era5.transpose(
        "time", "out_channels", "latitude", "longitude"
    )
    logger.info("Loaded the ERA5 data")
    return ds_era5


def main(args: argparse.Namespace):
    rnd = np.random.default_rng(args.seed)
    logger.info("Initialised random seed")

    ds_ifs = process_ifs_data(args.ifs_path)
    ds_era = process_era_data(args.era5_path)
    ds_merged = xr.Dataset({
        "input": ds_ifs,
        "target": ds_era
    })
    ds_merged = ds_merged.sel(
        time=slice('2017-01-01 00:00', '2019-12-31 12:00')
    )
    ds_merged = ds_merged.chunk({
        "time": 1, "ensemble": -1, "in_channels": -1, "out_channels": -1,
        "latitude": -1, "longitude": -1
    })
    logger.info("Loaded the data")

    ds_train_eval = ds_merged.sel(
        time=slice('2017-01-01 00:00', '2018-12-31 12:00')
    )
    len_train_eval = len(ds_train_eval.time)
    eval_idx = rnd.choice(
        len_train_eval,
        size=int(len_train_eval*args.eval_fraction),
        replace=False
    )
    ds_train = ds_train_eval.drop_isel(time=eval_idx)
    ds_eval = ds_train_eval.isel(time=eval_idx)
    ds_test = ds_merged.sel(time=slice('2019-01-01 00:00', '2019-12-31 12:00'))
    logger.info("Splitted the dataset into train, eval, and test")

    # Normalise data
    norm_mean = ds_train.mean(
        ["time", "ensemble", "latitude", "longitude"]
    ).compute()
    norm_std = ds_train.std(
        ["time", "ensemble", "latitude", "longitude"], ddof=1
    ).compute()
    logger.info("Estimated the normaliser from training dataset")

    norm_mean.to_netcdf(os.path.join(args.data_path, "normaliser_mean.nc"))
    norm_std.to_netcdf(os.path.join(args.data_path, "normaliser_std.nc"))
    logger.info("Stored the normaliser climatology")

    ds_train = (ds_train-norm_mean) / norm_std
    ds_eval = (ds_eval-norm_mean) / norm_std
    ds_test = (ds_test-norm_mean) / norm_std
    logger.info("Normalised the data")

    encoding = {
        'input': {'dtype': 'float32', 'scale_factor': 1.0, 'add_offset': 0.0},
        'target': {'dtype': 'float32', 'scale_factor': 1.0, 'add_offset': 0.0},
    }
    ds_train.to_zarr(
        os.path.join(args.data_path, "train"), mode="w",
        encoding=encoding, compute=True
    )
    logger.info("Stored the training data")
    ds_eval.to_zarr(
        os.path.join(args.data_path, "eval"), mode="w",
        encoding=encoding, compute=True
    )
    logger.info("Stored the evaluation data")
    ds_test.to_zarr(
        os.path.join(args.data_path, "test"), mode="w",
        encoding=encoding, compute=True
    )
    logger.info("Stored the testing data")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    _ = utils.initialize_cluster_client(
        args.n_workers, args.cluster_address
    )
    main(args)
