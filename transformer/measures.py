#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 04.09.20
#
# Created for 20_neurips_vaeda
#
# @author: Tobias Sebastian Finn, tobias.sebastian.finn@uni-hamburg.de
#
#    Copyright (C) {2020}  {Tobias Sebastian Finn}
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

# System modules
import logging
from typing import Union, Tuple, Callable

# External modules

import torch
import torch.nn
import torch.utils.data

import numpy as np

# Internal modules


logger = logging.getLogger(__name__)


frac_sqrt_pi = 1 / np.sqrt(np.pi)


def crps_loss(
        prediction: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        eps: Union[int, float] = 1E-10
) -> torch.Tensor:
    pred_mean, pred_stddev = prediction
    normal_distribution = torch.distributions.Normal(
        torch.zeros_like(pred_mean), 1.
    )
    normed_diff = (pred_mean - target + eps) / (pred_stddev + eps)
    cdf = normal_distribution.cdf(normed_diff)
    pdf = normal_distribution.log_prob(normed_diff).exp()
    crps = pred_stddev * (normed_diff * (2 * cdf - 1) + 2 * pdf - frac_sqrt_pi)
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
    lats : numpy.ndarray
        This array contains the latitudinal coordinates in degrees.
    """
    def __init__(
            self,
            base_score: Callable,
            lats: np.ndarray
    ):
        super().__init__()
        self._weights = None
        self.base_score = base_score
        self.weights = torch.nn.Parameter(
            self.estimate_weights(lats), requires_grad=False
        )

    @staticmethod
    def estimate_weights(lats: np.ndarray) -> torch.Tensor:
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
        weights = np.cos(np.deg2rad(lats))
        weights = weights / weights.mean()
        weights = torch.from_numpy(weights[:, None])
        return weights

    def forward(self, *args, **kwargs) -> torch.Tensor:
        non_reduced_score = self.base_score(*args, **kwargs)
        weighted_score = self.weights * non_reduced_score
        return weighted_score
