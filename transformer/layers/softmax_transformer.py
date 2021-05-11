#!/bin/env python
# -*- coding: utf-8 -*-
#
# Created on 05.10.20
#
# Created for ensemble-transformer
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

# External modules
import torch.nn

import numpy as np

# Internal modules
from .base_transformer import BaseTransformer


logger = logging.getLogger(__name__)


class SoftmaxTransformer(BaseTransformer):
    def _get_weights(
            self,
            key: torch.Tensor,
            query: torch.Tensor
    ) -> torch.Tensor:
        gram_mat = self._dot_product(key, query)
        gram_mat = gram_mat / np.sqrt(self.n_key_neurons)
        weights = torch.softmax(gram_mat, dim=1)
        return weights
