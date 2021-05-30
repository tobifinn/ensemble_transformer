#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.09.20
#
# Created for ensemble_transformer
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}


# System modules
import logging
from typing import Union, Tuple, Callable

# External modules

import torch
import torch.nn
import torch.utils.data

import numpy as np

# Internal modules


_frac_sqrt_pi = 1 / np.sqrt(np.pi)
_normal_dist = torch.distributions.Normal(0., 1.)
_default_lats = np.linspace(-87.1875, 87.1875, num=32)


def crps_loss(
        pred_mean: torch.Tensor,
        pred_stddev: torch.Tensor,
        target: torch.Tensor,
        eps: Union[int, float] = 1E-12
) -> torch.Tensor:
    normed_diff = (pred_mean - target + eps) / (pred_stddev + eps)
    try:
        cdf = _normal_dist.cdf(normed_diff)
        pdf = _normal_dist.log_prob(normed_diff).exp()
    except ValueError:
        print(normed_diff)
        raise ValueError
    crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - _frac_sqrt_pi)
    return crps


class WeightedScore(torch.nn.Module):
    """
    This weighted score wraps a given evaluation function to weight the score
    in latitudinal directions.
    The assumption is that the second last dimension represent the
    latitudinal dimension.

    Parameters
    ----------
    base_score : Callable
        This base score function evaluates a given prediction and target and
        output a non-rediced scored.
    lats : numpy.ndarray or None, optional (default = None)
        This array contains the latitudinal coordinates in degrees.
        If none the default latitudinal values for 32 grid points are used.
    """
    def __init__(
            self,
            base_score: Callable,
            lats: Union[np.ndarray, None] = None
    ):
        super().__init__()
        self._weights = None
        self.base_score = base_score
        self.weights = torch.nn.Parameter(
            self.estimate_weights(lats), requires_grad=False
        )

    @staticmethod
    def estimate_weights(lats: Union[np.ndarray, None]) -> torch.Tensor:
        """
        This method estimates the weights based on set latitudinal coordinates.
        After the first iteration, the weights are stored and simply returned if
        the type was not diff_meanchanged.

        Parameters
        ----------
        lats : numpy.ndarray
            This array contains the latitudinal coordinates in degrees.

        Returns
        -------
        weights : torch.Tensor
            The estimated weights.
        """
        if lats is None:
            lats = _default_lats
        weights = np.cos(np.deg2rad(lats))
        weights = weights / weights.mean()
        weights = torch.from_numpy(weights[:, None])
        return weights

    def forward(self, *args, **kwargs) -> torch.Tensor:
        non_reduced_score = self.base_score(*args, **kwargs)
        weighted_score = self.weights * non_reduced_score
        return weighted_score
