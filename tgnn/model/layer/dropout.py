# Copyright (c) 2025, Tencent Inc. All rights reserved.
# Data: 2025/1/9 15:58
# Author: chenchenqin
from functools import partialmethod
from typing import Union, List

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample
    """

    def __init__(self, p=0.):
        super(DropPath, self).__init__()
        self.p = p

    def forward(self, hidden_state: torch.Tensor, dim=0):
        """

        Args:
            hidden_state: [bs, *]

        Returns:
            output: [bs, *]
        """
        if self.p == 0. or not self.training:
            return hidden_state

        keep_prob = 1 - self.p
        shape = [1, ] * hidden_state.ndim
        shape[dim] = hidden_state.shape[dim]
        random_mask = keep_prob + torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_mask.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_mask

        return output


class DropoutAxiswise(nn.Dropout):
    """
    Implementation of dropout with the ability to share the dropout mask
    along a particular dimension.
    """

    def __init__(self,
                 p: float,
                 batch_dim: Union[int, List[int]],
                 inplace: bool = False):
        super(DropoutAxiswise, self).__init__(p=p, inplace=inplace)
        if type(batch_dim) == int:
            batch_dim = [batch_dim]
        self.batch_dim = batch_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = list(x.shape)
        if self.batch_dim is not None:
            for bd in self.batch_dim:
                shape[bd] = 1
        mask = super().forward(x.new_ones(shape))
        return x * mask