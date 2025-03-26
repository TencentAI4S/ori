# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/11/1 17:18
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GELU(nn.Module):
    def __init__(self, approximate: str = 'none') -> None:
        super().__init__()
        assert approximate in ['none', 'tanh', 'sigmoid', 'erf']
        self.approximate = approximate

    def forward(self, x):
        if self.approximate == 'sigmoid':
            return torch.sigmoid(1.702 * x) * x
        elif self.approximate == 'erf':
            return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        else:
            return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return 'approximate={}'.format(self.approximate)
