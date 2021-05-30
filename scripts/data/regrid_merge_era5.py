#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import os.path
import argparse
from typing import Any, Union

# External modules
import xarray as xr
from tqdm import tqdm

# Internal modules


logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='Merge and regrid data from ERA5')
parser.add_argument('era5_dir', type=str,
                    help='The raw ERA5 data is stored within this directory')
parser.add_argument('save_path', type=str,
                    help='The regridded data is saved to this file path')



def get_regridder(
        grid_in: xr.Dataset, out_res: float, method='bilinear'
) -> Any:
    import xesmf as xe
    grid_out = xr.Dataset(
        {
            'lat': (['lat'], np.arange(-90+out_res/2, 90, out_res)),
            'lon': (['lon'], np.arange(0, 360, out_res)),
        }
    )
    regridder = xe.Regridder(grid_in, grid_out, method=method,
                             periodic=True, reuse_weights=True)
    return regridder


def _regrid_loop(
        regridder: Any,
        ds: xr.Dataset,
        chunk_size: Union[int, None] = None
) -> xr.Dataset:
    if chunk_size is None:
        chunk_size = len(ds['time'])
    ds_regridded = []
    regrid_pbar = tqdm(range(0, len(ds['time']), chunk_size))
    for chunk in regrid_pbar:
        tmp_regridded = regridder(
            ds.isel(time=slice(chunk, chunk+chunk_size))
        )
        ds_regridded.append(tmp_regridded)
    ds_regridded = xr.concat(ds_regridded, dim='time')
    for var in ds_regridded:
        ds_regridded[var].attrs = ds[var].attrs
    ds_regridded.attrs = ds.attrs
    return ds_regridded


def regrid_ds(
        ds_ens: Union[xr.Dataset, None],
        chunk_size: Union[int, None] = None
) -> Union[xr.Dataset, None]:
    if ds_ens is not None:
        grid_in = ds_ens[['lat', 'lon']]
        regridder = get_regridder(grid_in, out_res=5.625)
        ds_regridded = _regrid_loop(regridder, ds_ens, chunk_size)
    else:
        ds_regridded = None
    return ds_regridded


def load_merge_data(era5_dir: str) -> xr.Dataset:
    ds_t2m = xr.open_dataset(
        os.path.join(era5_dir, 't2m_raw.nc')
    ).sel(expver=1) .chunk({'time': 10})
    ds_t850 = xr.open_dataset(
        os.path.join(era5_dir, 't850_raw.nc')
    ).sel(expver=1) .chunk({'time': 10})
    ds_z500 = xr.open_dataset(
        os.path.join(era5_dir, 'z500_raw.nc')
    ).sel(expver=1) .chunk({'time': 10})
    print('Got data')
    merged_ds = xr.merge([ds_z500, ds_t850, ds_t2m])
    print('Merged datasets')
    merged_ds = merged_ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    print('Cleaned data')
    return merged_ds


def regrid_merge_era5(era5_dir: str, save_path: str):
    merged_ds = load_merge_data(era5_dir).load()
    print('Loaded data')
    regridded_ds = regrid_ds(merged_ds, chunk_size=100)
    print('Regridded data')
    regridded_ds.to_netcdf(save_path)


if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments namespace: {0}'.format(str(args)))
    regrid_merge_era5(args.era5_dir, args.save_path)
