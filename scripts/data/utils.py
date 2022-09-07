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
from typing import Union

# External modules
import distributed

# Internal modules


logger = logging.getLogger(__name__)


def initialize_cluster_client(
        n_workers: int,
        cluster_address: Union[str, None] = None,
        memory_limit: str = "auto",
) -> distributed.Client:
    """
    Initialize a cluster client.

    Parameters
    ----------
    n_workers : int
        The number of workers.
    memory_limit : str
        The memory limit for the workers. The default is set to auto.

    Returns
    -------
    Client
        The initialized cluster client.
    """
    if cluster_address is None:
        cluster = distributed.LocalCluster(
            n_workers=n_workers, threads_per_worker=1,
            local_directory='/tmp/distributed', memory_limit=memory_limit
        )
        client = distributed.Client(cluster)
        logger.info("Initialised new client")
        logger.info("Dashboard link: {}".format(client.dashboard_link))
    else:
        client = distributed.get_client(cluster_address)
        logger.info(f"Loaded client from {cluster_address:s}")
    return client
