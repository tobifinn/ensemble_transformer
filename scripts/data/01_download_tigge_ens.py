#!/bin/env python
# -*- coding: utf-8 -*-

# System modules
import logging
import os
from copy import deepcopy
from typing import List, Tuple
import argparse

# External modules

import pandas as pd
from pandas.tseries.offsets import MonthEnd

from tqdm import tqdm

from ecmwfapi import ECMWFDataServer

# Internal modules


logger = logging.getLogger(__name__)


default_request_dict = {
    "class": "ti",
    "dataset": "tigge",
    "expver": "prod",
    "number": "1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/41/42/43/44/45/46/47/48/49/50",
    "origin": "ecmf",
    "step": "48",
    "time": "00:00:00/12:00:00",
    "type": "pf",
    "interpolation": "bilinear",
    "grid": "5.625/5.625",
    "area": "87.1875/0/-87.1875/354.375",
    "format": "netcdf"
}


param_dict = {
    'sfc': {
        'param': '167',
        'levtype': 'sfc',
        'filename': 'sfc_%Y%m.nc'
    },
    'pl': {
        'param': '130/156',
        'levtype': 'pl',
        'levelist': '500/850',
        'filename': 'pl_%Y%m.nc'
    },
}


parser = argparse.ArgumentParser(description='Download ECMWF EPS data from '
                                             'tigge archive')
parser.add_argument('save_dir', type=str,
                    help='The data is saved to this directory')
parser.add_argument('start', type=str,
                    help='The start date in format %Y-%m (e.g. 2008-01)')
parser.add_argument('end', type=str,
                    help='The end date in format %Y-%m (e.g. 2008-03)')
parser.add_argument('--var_names', type=str, nargs='+',
                    help='These variables are downloaded')


def _gen_date_list(start_date: str, end_date: str) -> List[Tuple[pd.Timestamp]]:
    start_date = pd.to_datetime(start_date, format='%Y-%m')
    end_date = pd.to_datetime(end_date, format='%Y-%m')
    period_range = pd.period_range(start_date, end_date, freq='M',)
    start_range = period_range.to_timestamp()
    end_range = start_range + MonthEnd(1)
    zipped_range = list(zip(start_range, end_range))
    return zipped_range


def download_ens_tigge(save_dir: str, start_date: str, end_date: str,
                       variables: List[str]):
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    server = ECMWFDataServer()
    download_dict = deepcopy(default_request_dict)

    date_list = _gen_date_list(start_date, end_date)
    var_pbar = tqdm(variables, leave=False)
    for var in var_pbar:
        var_download_dict = deepcopy(download_dict)
        var_param_dict = deepcopy(param_dict[var])
        file_name_template = var_param_dict.pop('filename')
        var_download_dict.update(var_param_dict)

        date_pbar = tqdm(date_list)
        for start, end in date_pbar:
            var_filename = start.strftime(file_name_template)
            var_download_dict['target'] = os.path.join(save_dir, var_filename)
            var_download_dict["date"] = '{0:s}/to/{1:s}'.format(
                start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
            )
            if not os.path.isfile(var_download_dict['target']):
                server.retrieve(var_download_dict)
            else:
                date_pbar.write(
                    '{0:s} was already downloaded'.format(
                        var_download_dict['target']
                    )
                )
        var_pbar.close()


if __name__ == '__main__':
    args = parser.parse_args()
    print('Arguments namespace: {0}'.format(str(args)))
    download_ens_tigge(args.save_dir, args.start, args.end, args.var_names)
