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
from typing import Union, Tuple

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


class WeightedScore(object):
    def __init__(self, base_score, lat):
        self._weights = None
        self.base_score = base_score
        self.lat = lat

    def get_weights(self, as_tensor):
        if self._weights is None:
            weights_lat = np.cos(np.deg2rad(self.lat.values))
            weights_lat = weights_lat / weights_lat.mean()
            self._weights = torch.from_numpy(weights_lat[:, None])
        if self._weights.type() != as_tensor.type():
            self._weights = self._weights.to(as_tensor)
        return self._weights

    def __call__(self, prediction, target):
        non_reduced_score = self.base_score(prediction, target)
        weighted_score = self.get_weights(non_reduced_score) * non_reduced_score
        return weighted_score
